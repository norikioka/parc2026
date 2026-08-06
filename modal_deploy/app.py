"""PARC2026: 本番相当(GPU/CUDA13.0/Python3.10)環境での提出物検証をModal上で行う。

Colabの度重なるランタイム切断・セッション不安定性を避けるため、CLIデプロイで
完結するModal(https://modal.com)に切り替えた(2026-08-06)。公式配布リポジトリの
Dockerfileを、READMEの「本番相当(GPU)を手元で再現する場合」の指示通りに改造した
Dockerfile.gpuをベースイメージとして使う。

使い方:
    modal run app.py::offline_check       # ネットワーク遮断下でのオフライン起動確認
    modal run app.py::pipeline_smoke_test # LIBERO-Plus実環境での疎通確認(track1, 2episodes)
"""

from __future__ import annotations

from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parent.parent  # ~/projects/academic/parc2026
SRC = LOCAL_ROOT / "src" / "parc2026"
MODEL_DIR_LOCAL = LOCAL_ROOT / "1st" / "smolvla_libero_plus_spatial_lora_merged"

app = modal.App("parc2026-gpu-verify")

image = (
    modal.Image.from_dockerfile(str(Path(__file__).resolve().parent / "Dockerfile.gpu"))
    # Modal自身がこのFunctionを実行するpython環境にも、検証スクリプト側で使うパッケージが要る
    # (venv/bin/pythonで動くpolicy_server.py側とは別物。曖昧さを無くすため明示的にインストールする)
    .pip_install("numpy", "msgpack", "requests")
    .add_local_file(str(SRC / "requirements_smolvla.txt"), "/workspace/submission_template/requirements.txt", copy=True)
    .run_commands("/workspace/venv/bin/pip install -q -r /workspace/submission_template/requirements.txt")
    .add_local_file(str(SRC / "policy_server_smolvla_full.py"), "/workspace/submission_template/policy_server.py", copy=True)
    .add_local_dir(str(SRC / "vendor"), "/workspace/submission_template/vendor", copy=True)
    .add_local_dir(str(MODEL_DIR_LOCAL), "/workspace/submission_template/model_weights", copy=True)
    .add_local_dir(str(SRC / "vlm_assets"), "/workspace/submission_template/model_weights/vlm", copy=True)
    # 【2026-08-06】build_and_validate_submission(arm_name=...)でmodel_weightsを丸ごと
    # 差し替える際、vlmアセットも一緒に消えるため、差し替え不要な場所に複製しておく
    .add_local_dir(str(SRC / "vlm_assets"), "/workspace/vlm_assets_backup", copy=True)
)


train_volume = modal.Volume.from_name("parc2026-train-outputs", create_if_missing=True)


@app.function(image=image, gpu="L4", timeout=3600, volumes={"/train_vol": train_volume})
def evaluate_arm(arm_name: str, n_episodes: int = 10, max_steps: int = 600):
    """【2026-08-06 改善ループ用】train_app.py::trainで学習したarmを、既存の検証用
    policy_server.pyに読み込ませてtrack1で評価する。結果を構造化して返す
    (呼び出し側でdocs/step4_experiments.csvに追記し、arm間の比較を蓄積できるようにする)。"""
    import json
    import os
    import re
    import shutil
    import subprocess
    import time

    train_volume.reload()

    arm_model_src = Path(f"/train_vol/{arm_name}/merged")
    assert arm_model_src.is_dir(), f"{arm_model_src} が見つかりません。train_app.py::trainを先に実行してください。"

    model_dir_name = f"model_weights_{arm_name}"
    model_dst = Path(f"/workspace/submission_template/{model_dir_name}")
    shutil.rmtree(model_dst, ignore_errors=True)
    shutil.copytree(arm_model_src, model_dst)
    # VLMアセット(Hub参照回避用)は全arm共通なので、既にイメージに焼き込み済みのものを流用する
    shutil.copytree(
        "/workspace/submission_template/model_weights/vlm", model_dst / "vlm", dirs_exist_ok=True
    )

    log_path = f"/tmp/policy_server_eval_{arm_name}.log"
    log_file = open(log_path, "w")
    server_env = os.environ.copy()
    server_env["PARC_MODEL_DIR"] = model_dir_name
    proc = subprocess.Popen(
        ["/workspace/venv/bin/python", "policy_server.py", "--port", "8000"],
        cwd="/workspace/submission_template",
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=server_env,
    )
    time.sleep(5)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/LIBERO-plus:/workspace:/workspace/compe"
    t0 = time.time()
    result = subprocess.run(
        ["/workspace/venv/bin/python", "-m", "pipeline",
         "--server-url", "http://localhost:8000", "--track", "track1",
         "--n-episodes", str(n_episodes), "--max-steps", str(max_steps)],
        cwd="/workspace", env=env, capture_output=True, text=True, timeout=7200,
    )
    elapsed = time.time() - t0
    proc.terminate()
    log_file.close()

    print(f"=== arm={arm_name} n_episodes={n_episodes} elapsed={elapsed:.1f}s ===")
    print(result.stdout[-3000:])

    overall_match = re.search(r"総合スコア\s+([0-9.]+)", result.stdout)
    overall = float(overall_match.group(1)) if overall_match else None
    task_scores = dict(re.findall(r"^\s{4}(\S+):\s+([0-9.]+)%", result.stdout, re.MULTILINE))

    return {
        "arm_name": arm_name,
        "n_episodes": n_episodes,
        "overall_local_success_rate": overall,
        "per_task": task_scores,
        "elapsed_sec": round(elapsed, 1),
        "returncode": result.returncode,
    }


@app.local_entrypoint()
def run_evaluate_arm(arm_name: str, n_episodes: int = 10):
    """評価を実行し、結果をdocs/step4_experiments.csvに追記する(改善ループの記録)。"""
    import csv
    import datetime

    result = evaluate_arm.remote(arm_name, n_episodes)
    print(result)

    csv_path = LOCAL_ROOT / "docs" / "step4_experiments.csv"
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "arm_name", "n_episodes", "overall_local_success_rate",
                              "elapsed_sec", "returncode"])
        writer.writerow([
            datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            result["arm_name"], result["n_episodes"], result["overall_local_success_rate"],
            result["elapsed_sec"], result["returncode"],
        ])
    print(f"記録: {csv_path}")


@app.function(image=image, gpu="L4", timeout=600)
def offline_check():
    """ネットワーク遮断下(HF_HUB_OFFLINE=1)でpolicy_server.pyが起動し、
    /health -> /reset -> /act まで正常応答するかを確認する。"""
    import os
    import subprocess
    import time

    import msgpack
    import numpy as np
    import requests

    env = os.environ.copy()
    env["HF_HOME"] = "/tmp/empty_hf_cache"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    os.makedirs(env["HF_HOME"], exist_ok=True)

    proc = subprocess.Popen(
        ["/workspace/venv/bin/python", "policy_server.py", "--port", "8000"],
        cwd="/workspace/submission_template",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    health_ok = False
    for _ in range(60):
        if proc.poll() is not None:
            break
        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200:
                health_ok = True
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    if not health_ok:
        out, _ = proc.communicate(timeout=10)
        print("=== SERVER LOG (起動失敗) ===")
        print(out)
        print("=== OFFLINE CHECK: FAIL (health) ===")
        return

    print("health OK")

    r = requests.post("http://localhost:8000/reset", json={"instruction": "pick up the red mug"})
    print("reset:", r.status_code, r.text)

    obs = {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    packed = {k: {"dtype": str(v.dtype), "shape": list(v.shape), "data": v.tobytes()} for k, v in obs.items()}
    body = msgpack.packb(packed, use_bin_type=True)

    t0 = time.time()
    r = requests.post("http://localhost:8000/act", data=body, headers={"Content-Type": "application/x-msgpack"})
    latency = time.time() - t0
    action = np.frombuffer(msgpack.unpackb(r.content, raw=False)["data"], dtype=np.float32)
    print(f"act status={r.status_code} latency={latency:.3f}s action={action}")

    proc.terminate()

    print("=== torch/cuda確認 ===")
    subprocess.run(
        ["/workspace/venv/bin/python", "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"],
        check=False,
    )

    if action.shape == (7,) and health_ok:
        print("=== OFFLINE CHECK: PASS ===")
    else:
        print("=== OFFLINE CHECK: FAIL ===")


@app.function(image=image, gpu="L4", timeout=600, volumes={"/train_vol": train_volume})
def build_and_validate_submission(arm_name: str | None = None):
    """submission_template/一式をzip化し、公式validate_submission.pyで検証する。
    PASSしたzipのバイト列を返す(呼び出し側でローカルに保存する)。
    【2026-08-06 改善ループ用】arm_nameを指定すると、Modal Volume上のtrain_app.py::train
    出力に採用モデルを一時的に差し替えてからzip化する(未指定時は現行提出物のまま、挙動不変)。"""
    import shutil
    import subprocess

    if arm_name is not None:
        train_volume.reload()
        arm_model_src = Path(f"/train_vol/{arm_name}/merged")
        assert arm_model_src.is_dir(), f"{arm_model_src} が見つかりません。train_app.py::trainを先に実行してください。"
        model_dst = Path("/workspace/submission_template/model_weights")
        shutil.rmtree(model_dst)
        shutil.copytree(arm_model_src, model_dst)
        # VLMアセット(Hub参照回避用)は元のmodel_weights/vlmから退避しておいたものを戻す必要があるが、
        # 上のrmtreeで消えるため、イメージビルド時にvlm_assets単体でも別途コピーしておく(下記image定義参照)
        shutil.copytree("/workspace/vlm_assets_backup", model_dst / "vlm")
        print(f"submission_template/model_weights を arm={arm_name} に差し替えました")

    zip_base = "/tmp/submission"
    shutil.make_archive(zip_base, "zip", root_dir="/workspace/submission_template")
    zip_path = f"{zip_base}.zip"

    result = subprocess.run(
        ["/workspace/venv/bin/python", "validate_submission.py", zip_path],
        cwd="/workspace",
        capture_output=True,
        text=True,
        timeout=300,
    )
    print("=== validate_submission.py stdout ===")
    print(result.stdout)
    print("=== validate_submission.py stderr ===")
    print(result.stderr)
    print("returncode:", result.returncode)

    with open(zip_path, "rb") as f:
        zip_bytes = f.read()
    print(f"zip size: {len(zip_bytes) / 1024 / 1024:.1f} MB")

    return result.returncode, zip_bytes


@app.local_entrypoint()
def save_submission(out_name: str = "submission.zip", arm_name: str = ""):
    """【2026-08-06 planner設計】出力ファイル名を固定していたため、実験のたびに
    0.09397を出した実物を上書きしかねなかった。呼び出し時に名前を指定できるようにする。
    arm_nameを指定すると、そのarmのモデルでzipを作る(未指定なら現行提出物のまま)。
    例: modal run app.py::save_submission --out-name submission_treatment_b.zip --arm-name treatment_b"""
    returncode, zip_bytes = build_and_validate_submission.remote(arm_name or None)
    out_path = LOCAL_ROOT / out_name
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} は既に存在します。実物の上書きを防ぐため、別名を指定してください。"
        )
    with open(out_path, "wb") as f:
        f.write(zip_bytes)
    print(f"保存先: {out_path} ({len(zip_bytes) / 1024 / 1024:.1f} MB)")
    if returncode == 0:
        print("=== validate_submission.py: PASS ===")
    else:
        print("=== validate_submission.py: FAIL (returncode != 0) ===")


@app.function(image=image, gpu="L4", timeout=1200)
def n_action_steps_ab_test(n_action_steps: int):
    """n_action_steps(50 vs 10)をtrack1 exampleタスクで比較する(提出不要)。
    【2026-08-06 critic Opusレビュー】成功率を上げる意図の変更(50→10)が、滑らかさスコア
    (jerk/SPARC)を悪化させている可能性が未検証だったため、A/Bで確認する。"""
    import os
    import subprocess
    import time

    log_path = f"/tmp/policy_server_n{n_action_steps}.log"
    log_file = open(log_path, "w")
    server_env = os.environ.copy()
    server_env["PARC_N_ACTION_STEPS"] = str(n_action_steps)
    proc = subprocess.Popen(
        ["/workspace/venv/bin/python", "policy_server.py", "--port", "8000"],
        cwd="/workspace/submission_template",
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=server_env,
    )
    time.sleep(5)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/LIBERO-plus:/workspace:/workspace/compe"
    result = subprocess.run(
        ["/workspace/venv/bin/python", "-m", "pipeline",
         "--server-url", "http://localhost:8000", "--track", "track1", "--n-episodes", "2", "--max-steps", "600"],
        cwd="/workspace",
        env=env,
        capture_output=True,
        text=True,
        timeout=1000,
    )
    print(f"=== n_action_steps={n_action_steps}: pipeline stdout ===")
    print(result.stdout[-4000:])
    print("returncode:", result.returncode)

    proc.terminate()
    log_file.close()


@app.function(image=image, gpu="L4", timeout=1200)
def pipeline_smoke_test():
    """LIBERO-Plus実環境(GPU)でtrack1を2エピソード回し、クラッシュしないか確認する。"""
    import os
    import subprocess
    import time

    # 【2026-08-06修正】stdout=PIPEのまま誰も読まないと、出力が溜まった時点で
    # policy_server側が書き込みブロックする恐れがある(タイムアウトの原因調査を妨げる)。
    # ログファイルに書き出し、実行後にまとめて読む方式にする。
    log_path = "/tmp/policy_server_pipeline_test.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        ["/workspace/venv/bin/python", "policy_server.py", "--port", "8000"],
        cwd="/workspace/submission_template",
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    time.sleep(5)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/LIBERO-plus:/workspace:/workspace/compe"
    t0 = time.time()
    result = subprocess.run(
        ["/workspace/venv/bin/python", "-m", "pipeline",
         "--server-url", "http://localhost:8000", "--track", "track1", "--n-episodes", "2", "--max-steps", "600"],
        cwd="/workspace",
        env=env,
        capture_output=True,
        text=True,
        timeout=1000,
    )
    elapsed = time.time() - t0
    print("=== pipeline stdout ===")
    print(result.stdout[-4000:])
    print("=== pipeline stderr ===")
    print(result.stderr[-2000:])
    print("returncode:", result.returncode, "elapsed:", f"{elapsed:.1f}s")

    proc.terminate()
    log_file.close()
    print("=== policy_server.py 自身のログ ===")
    with open(log_path) as f:
        print(f.read())

    proc.terminate()
