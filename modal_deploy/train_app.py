"""PARC2026: SmolVLA LoRA学習をModal上で行う(ステップ4 Phase 1〜2)。

Colabの`examples/smolvla_libero_spatial_lora.ipynb`(現在の提出物の学習元)を移植したもの。
学習側はlerobot v0.6.0をPython 3.12上でそのままpip installする(推論側のPython 3.10.12
互換パッチとは無関係、別環境)。

使い方:
    # Phase 1: 疎通確認(極小学習、5分程度)
    modal run train_app.py::train --steps 50 --arm-name smoke_test

    # Phase 2: 本学習(control: 現行と同一設定の再現)
    modal run train_app.py::train --steps 3000 --arm-name control

    # Phase 2: 本学習(treatment B: image_transforms有効化)
    modal run train_app.py::train --steps 3000 --arm-name treatment_b --image-transforms

    # Phase 2: 本学習(treatment C: エピソード数増、任意)
    modal run train_app.py::train --steps 3000 --arm-name treatment_c --episodes-per-task 20
"""

from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("parc2026-train")

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


@app.function(
    image=train_image,
    gpu="L4",
    timeout=3600 * 4,
    volumes={"/vol": volume},
)
def train(
    arm_name: str,
    steps: int = 3000,
    episodes_per_task: int = 5,
    image_transforms: bool = False,
    batch_size: int = 1,
    seed: int = 42,
):
    import json
    import os
    import re
    import subprocess
    from collections import defaultdict, deque

    import torch

    BASE_MODEL_REPO = "lerobot/smolvla_libero_plus"
    BASE_MODEL_REVISION = "7bb70aa5bc92b82c9239142775d3a173103567ff"
    VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    DATASET_REPO = "lerobot/libero_plus"
    DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"

    SPATIAL_TASK_NAMES = [
        "pick up the black bowl from table center and place it on the plate",
        "pick up the black bowl next to the cookie box and place it on the plate",
        "pick up the black bowl next to the plate and place it on the plate",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "pick up the black bowl on the cookie box and place it on the plate",
        "pick up the black bowl on the ramekin and place it on the plate",
        "pick up the black bowl on the stove and place it on the plate",
        "pick up the black bowl on the wooden cabinet and place it on the plate",
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
    ]

    LEARNING_RATE = 3e-4
    FINAL_LEARNING_RATE = 3e-5
    WARMUP_STEPS = min(100, steps // 10 or 1)
    LORA_R = 16
    LORA_ALPHA = 16
    LOG_FREQ = 10

    OUTPUT_DIR = Path(f"/vol/{arm_name}/outputs")
    MERGED_MODEL_DIR = Path(f"/vol/{arm_name}/merged")

    print(f"=== arm={arm_name} steps={steps} episodes_per_task={episodes_per_task} "
          f"image_transforms={image_transforms} ===")

    # --- 1. 基底モデルのダウンロード ---
    from huggingface_hub import snapshot_download

    base_model_local = Path(snapshot_download(
        repo_id=BASE_MODEL_REPO,
        revision=BASE_MODEL_REVISION,
        allow_patterns=[
            "config.json", "model.safetensors", "train_config.json",
            "policy_preprocessor.json", "policy_preprocessor*.safetensors",
            "policy_postprocessor.json", "policy_postprocessor*.safetensors",
        ],
        ignore_patterns=["README.md", "eval/**"],
    ))
    assert (base_model_local / "model.safetensors").is_file(), "Base model not found."
    print("Base model ready.")

    # --- 2. 学習エピソードの選定(タスクごとに等間隔でepisodes_per_task本) ---
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    def normalize_task_name(value: str) -> str:
        value = value.lower().replace("_", " ")
        value = re.sub(r"[^a-z0-9 ]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def choose_evenly_spaced(episode_indices: list[int], count: int) -> list[int]:
        if count >= len(episode_indices):
            return episode_indices
        positions = [round(i * (len(episode_indices) - 1) / (count - 1)) for i in range(count)]
        return [episode_indices[p] for p in positions]

    dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, revision=DATASET_REVISION)
    task_to_episodes: dict[str, list[int]] = defaultdict(list)
    for episode_index, task_cell in enumerate(dataset_metadata.episodes["tasks"]):
        name = task_cell[0] if not isinstance(task_cell, str) and len(task_cell) else task_cell
        task_to_episodes[str(name)].append(int(episode_index))

    available_by_normalized = {normalize_task_name(t): t for t in task_to_episodes}
    selected_by_task = {}
    for task_name in SPATIAL_TASK_NAMES:
        actual_task = available_by_normalized.get(normalize_task_name(task_name))
        assert actual_task is not None, f"Spatial task not found: {task_name}"
        selected_by_task[actual_task] = choose_evenly_spaced(
            task_to_episodes[actual_task], episodes_per_task
        )
    episode_indices = sorted(i for v in selected_by_task.values() for i in v)
    print(f"Training data: 10 tasks x {episodes_per_task} episodes = {len(episode_indices)} episodes")

    # --- 3. lerobot-train実行 ---
    mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    episodes_json = "[" + ",".join(map(str, episode_indices)) + "]"

    command = [
        "lerobot-train",
        f"--policy.path={base_model_local}",
        f"--policy.vlm_model_name={VLM_REPO}",
        "--policy.push_to_hub=false",
        "--policy.repo_id=null",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.empty_cameras=0",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        f"--policy.optimizer_lr={LEARNING_RATE}",
        f"--policy.scheduler_decay_lr={FINAL_LEARNING_RATE}",
        f"--policy.scheduler_warmup_steps={WARMUP_STEPS}",
        f"--policy.scheduler_decay_steps={steps}",
        f"--dataset.repo_id={DATASET_REPO}",
        f"--dataset.revision={DATASET_REVISION}",
        f"--dataset.episodes={episodes_json}",
        "--dataset.use_imagenet_stats=false",
        "--dataset.video_backend=torchcodec",
        f"--dataset.image_transforms.enable={'true' if image_transforms else 'false'}",
        f"--output_dir={OUTPUT_DIR}",
        f"--job_name={arm_name}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        "--num_workers=0",
        "--persistent_workers=false",
        "--env_eval_freq=0",
        "--eval_steps=0",
        f"--seed={seed}",
        "--save_checkpoint=true",
        f"--save_freq={steps}",
        "--save_checkpoint_to_hub=false",
        f"--log_freq={LOG_FREQ}",
        "--wandb.enable=false",
        "--peft.method_type=LORA",
        f"--peft.r={LORA_R}",
        f"--peft.lora_alpha={LORA_ALPHA}",
    ]

    env = os.environ.copy()
    env["ACCELERATE_MIXED_PRECISION"] = mixed_precision
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TQDM_DISABLE"] = "1"

    import shutil
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    print("Starting training...")
    proc = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    recent = deque(maxlen=100)
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line:
            continue
        recent.append(line)
        if "step:" in line and "loss:" in line:
            print(line)
    rc = proc.wait()
    if rc != 0:
        print("\n".join(recent))
        raise RuntimeError(f"Training failed: {rc}")
    print("Training complete.")

    # --- 4. LoRAマージ ---
    import gc

    from peft import PeftModel
    from safetensors import safe_open
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint_dir = OUTPUT_DIR / "checkpoints" / f"{steps:06d}" / "pretrained_model"
    assert (checkpoint_dir / "adapter_model.safetensors").is_file(), "Final adapter not found."

    gc.collect()
    torch.cuda.empty_cache()

    merge_config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    merge_config.device = "cpu"
    merge_config.pretrained_path = base_model_local
    merge_config.use_peft = False

    base_policy = SmolVLAPolicy.from_pretrained(base_model_local, config=merge_config, strict=False)
    peft_policy = PeftModel.from_pretrained(base_policy, checkpoint_dir, is_trainable=False, torch_device="cpu")
    merged_policy = peft_policy.merge_and_unload(safe_merge=True)

    shutil.rmtree(MERGED_MODEL_DIR, ignore_errors=True)
    MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    merged_policy.config.use_peft = False
    merged_policy.config.pretrained_path = None
    merged_policy.config.push_to_hub = False
    merged_policy.config.repo_id = None
    merged_policy.config.device = None
    merged_policy.config.load_vlm_weights = False
    merged_policy.config.vlm_model_name = VLM_REPO
    merged_policy.save_pretrained(MERGED_MODEL_DIR)

    for pattern in [
        "policy_preprocessor.json", "policy_preprocessor*.safetensors",
        "policy_postprocessor.json", "policy_postprocessor*.safetensors",
    ]:
        for source_path in checkpoint_dir.glob(pattern):
            shutil.copy2(source_path, MERGED_MODEL_DIR / source_path.name)

    with safe_open(MERGED_MODEL_DIR / "model.safetensors", framework="pt", device="cpu") as w:
        assert not any("lora_" in k.lower() for k in w.keys()), "LoRA parameters remain after merge."

    # 学習時の設定を記録として残す(「引数を渡したから効いているはず」で済ませない)
    with open(MERGED_MODEL_DIR / "train_args_record.json", "w") as f:
        json.dump({
            "arm_name": arm_name, "steps": steps, "episodes_per_task": episodes_per_task,
            "image_transforms": image_transforms, "batch_size": batch_size, "seed": seed,
            "episode_indices": episode_indices,
        }, f, indent=2)

    volume.commit()
    print(f"Merged model ready at /vol/{arm_name}/merged (Modal Volume: parc2026-train-outputs)")


@app.local_entrypoint()
def download_merged(arm_name: str):
    """学習済みモデルをModal VolumeからローカルMacにダウンロードする。"""
    import subprocess
    dest = LOCAL_ROOT / f"{arm_name}_merged"
    dest.mkdir(exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "parc2026-train-outputs", f"{arm_name}/merged", str(dest), "--force"],
        check=True,
    )
    print(f"ダウンロード完了: {dest}")
