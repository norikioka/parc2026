"""PARC2026: オフライン(held-out)評価。

【2026-08-07 ユーザー指摘で着手】これまでのローカル評価(シミュレータで実際に動かして
成功率を見る方式)は、公開4タスク・40エピソードしか使えず、しかもそのタスクは学習データにも
含まれているため、「汎化しているか」ではなく「見た画像にどれだけ強く適合したか」を測って
しまっていた可能性が高い。broad_unfrozen(ローカル最高90%・本番最低0.03065)がその典型例。

lerobot本体の`make_train_eval_datasets`(datasets/factory.py)が、まさに欲しかった
「タスクごとに末尾のepisodeをheld-outとして自動分割する」機能を持っていたので、
これと同じロジック・同じ`policy.forward()`によるvalidation loss計算を、学習後の
任意のモデルに対して事後実行できるようにしたもの。lerobot_train.py内のeval実装
(622-634行目)と完全に同じ手順を再現している。

使い方:
    modal run offline_eval.py::evaluate --arm-name broad
"""

from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("parc2026-offline-eval")

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "cmake", "ffmpeg")
    .run_commands(
        "git clone --quiet --depth 1 --branch v0.6.0 "
        "https://github.com/huggingface/lerobot.git /opt/lerobot"
    )
    .run_commands(
        "python -m pip install --quiet --upgrade -e '/opt/lerobot[training,smolvla,peft]'"
    )
    .run_commands("python -m pip uninstall -y torchao")
)

volume = modal.Volume.from_name("parc2026-train-outputs", create_if_missing=True)

DATASET_REPO = "lerobot/libero_plus"
DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"


@app.function(image=train_image, gpu="L4", timeout=7200, volumes={"/vol": volume})
def evaluate(arm_name: str, eval_split: float = 0.05, max_eval_per_task: int = 6, batch_size: int = 8):
    """各タスクの末尾(eval_split割合、最大max_eval_per_task本)をheld-outとして、
    そのモデルの学習で使ったepisode_indicesと重複しないことを確認した上で、
    validation lossを計算する。"""
    import json

    import torch
    from torch.utils.data import DataLoader

    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.collate import lerobot_collate_fn

    model_dir = Path(f"/vol/{arm_name}/merged")
    assert model_dir.is_dir(), f"{model_dir} が見つかりません。"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PreTrainedConfig.from_pretrained(model_dir)
    config.device = device
    policy = SmolVLAPolicy.from_pretrained(model_dir, config=config)
    policy.to(device)
    policy.eval()
    preprocessor, _postprocessor = make_pre_post_processors(config, pretrained_path=model_dir)
    print(f"モデルロード完了: {arm_name}")

    # --- held-outエピソードの選定: 各タスクの末尾(lerobot本体のmake_train_eval_datasetsと同じロジック) ---
    dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, revision=DATASET_REVISION)
    task_to_episodes: dict[str, list[int]] = {}
    for episode_index, task_cell in enumerate(dataset_metadata.episodes["tasks"]):
        name = task_cell[0] if not isinstance(task_cell, str) and len(task_cell) else task_cell
        task_to_episodes.setdefault(str(name), []).append(int(episode_index))

    import math

    trained_indices: set[int] = set()
    record_path = model_dir / "train_args_record.json"
    if record_path.is_file():
        with open(record_path) as f:
            trained_indices = set(json.load(f).get("episode_indices", []))
        print(f"学習済みエピソード{len(trained_indices)}件を除外対象に確認")

    eval_episodes: list[int] = []
    leaked = 0
    for task, eps in task_to_episodes.items():
        eps_sorted = sorted(eps)
        n_eval = min(max_eval_per_task, math.ceil(len(eps_sorted) * eval_split))
        candidates = eps_sorted[len(eps_sorted) - n_eval:]
        for idx in candidates:
            if idx in trained_indices:
                leaked += 1
                continue
            eval_episodes.append(idx)

    print(f"held-outエピソード数: {len(eval_episodes)}（{len(task_to_episodes)}タスク、"
          f"学習セットとの重複除外{leaked}件）")

    # --- データセットロード(lerobot_train.pyのeval_datasetと同じ構成) ---
    delta_timestamps = resolve_delta_timestamps(config, dataset_metadata)
    eval_dataset = LeRobotDataset(
        DATASET_REPO,
        revision=DATASET_REVISION,
        episodes=eval_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        video_backend="torchcodec",
        return_uint8=True,
    )
    collate_fn = lerobot_collate_fn if eval_dataset.meta.has_language_columns else None
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False, collate_fn=collate_fn,
    )

    # --- validation loss計算(lerobot_train.py 618-634行目と同一手順) ---
    eval_loss_sum = 0.0
    n_eval_batches = 0
    with torch.no_grad():
        for eval_batch in eval_dataloader:
            eval_batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in eval_batch.items()}
            for cam_key in eval_dataset.meta.camera_keys:
                if cam_key in eval_batch and eval_batch[cam_key].dtype == torch.uint8:
                    eval_batch[cam_key] = eval_batch[cam_key].to(dtype=torch.float32) / 255.0
            eval_batch = preprocessor(eval_batch)
            loss, _ = policy.forward(eval_batch)
            eval_loss_sum += loss.item()
            n_eval_batches += 1

    eval_loss = eval_loss_sum / max(n_eval_batches, 1)
    print(f"=== arm={arm_name} held_out_episodes={len(eval_episodes)} "
          f"n_batches={n_eval_batches} eval_loss={eval_loss:.5f} ===")
    result = {
        "arm_name": arm_name, "held_out_episodes": len(eval_episodes),
        "n_eval_batches": n_eval_batches, "eval_loss": eval_loss, "leaked_excluded": leaked,
    }
    # 【2026-08-07】ローカル接続が途中で切れても結果を失わないよう、Volume側にも直接保存する
    # (local_entrypointの戻り値受信だけに依存しない)
    result_path = Path(f"/vol/{arm_name}/offline_eval_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    print(f"結果をVolumeに保存: {result_path}")
    return result


@app.function(volumes={"/vol": volume})
def fetch_result(arm_name: str):
    """接続切れ等でrun_evaluateがCSV記録できなかった場合に、Volumeに保存済みの
    結果を後から取得する。"""
    import json
    volume.reload()
    result_path = Path(f"/vol/{arm_name}/offline_eval_result.json")
    if not result_path.is_file():
        return None
    with open(result_path) as f:
        return json.load(f)


@app.local_entrypoint()
def run_evaluate(arm_name: str, eval_split: float = 0.05, max_eval_per_task: int = 6):
    import csv
    import datetime

    result = evaluate.remote(arm_name, eval_split, max_eval_per_task)
    print(result)

    csv_path = LOCAL_ROOT / "docs" / "step4_offline_eval.csv"
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "arm_name", "held_out_episodes", "n_eval_batches",
                              "eval_loss", "leaked_excluded"])
        writer.writerow([
            datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            result["arm_name"], result["held_out_episodes"], result["n_eval_batches"],
            result["eval_loss"], result["leaked_excluded"],
        ])
    print(f"記録: {csv_path}")
