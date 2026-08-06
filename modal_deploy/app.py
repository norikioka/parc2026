"""PARC2026: 本番相当(GPU/CUDA13.0/Python3.10)環境での提出物検証をModal上で行う。

Colabの度重なるランタイム切断・セッション不安定性を避けるため、CLIデプロイで
完結するModal(https://modal.com)に切り替えた(2026-08-06)。公式配布リポジトリの
Dockerfileを、READMEの「本番相当(GPU)を手元で再現する場合」の指示通りに改造した
Dockerfile.gpuをベースイメージとして使う。

使い方:
    modal run app.py::offline_check       # ネットワーク遮断下でのオフライン起動確認
    modal run app.py::pipeline_smoke_test # LIBERO-Plus実環境での疎通確認(track1, 2episodes)
"""

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
)


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


@app.function(image=image, gpu="L4", timeout=600)
def build_and_validate_submission():
    """submission_template/一式をzip化し、公式validate_submission.pyで検証する。
    PASSしたzipのバイト列を返す(呼び出し側でローカルに保存する)。"""
    import shutil
    import subprocess

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
def save_submission():
    returncode, zip_bytes = build_and_validate_submission.remote()
    out_path = LOCAL_ROOT / "submission.zip"
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
