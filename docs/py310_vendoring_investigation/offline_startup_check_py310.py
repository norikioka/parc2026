import typing
import typing_extensions

# lerobot 0.6.0はPython>=3.12を要求しているが、実際に使っている新機能は
# typingモジュールの一部の名前(Self/Unpack等、3.11以降で追加)のみで、
# 純粋な構文(PEP695ジェネリクス)以外はtyping_extensionsのバックポートで代替できる。
# 本番の採点環境がPython 3.10.12固定(公式README確定)なため、この形で吸収する。
for _name in ("Self", "Unpack", "override", "TypeVarTuple", "ParamSpec", "Required", "NotRequired"):
    if not hasattr(typing, _name) and hasattr(typing_extensions, _name):
        setattr(typing, _name, getattr(typing_extensions, _name))
"""ネットワーク遮断下でSmolVLAPolicyがロードできるかをローカル(Mac/CPU)で確認する。

policy_server_smolvla_full.py の MyPolicy.__init__ と同じ手順を、
HF_HUB_OFFLINE=1 (ネットワークアクセスを完全に禁止する公式のオフラインモード)の下で再現する。
Colab上のGPU評価環境を待たずに、「VLM Hub参照の修正が本当に効いているか」だけを先に検証する。
"""
import os
import sys
import time
from pathlib import Path

# 評価環境のネットワーク遮断を模す。この時点でネットワークに触れようとするコードがあれば
# huggingface_hub側が例外を出すため、「気づかずHubにアクセスしていた」を機械的に検出できる。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve()
PARC_ROOT = Path("/Users/norikioka/projects/PARC")
model_dir = str((PARC_ROOT / "1st" / "smolvla_libero_plus_spatial_lora_merged").resolve())
vlm_local_dir = PARC_ROOT / "src" / "parc2026" / "vlm_assets"

print(f"[check] model_dir = {model_dir}")
print(f"[check] vlm_local_dir = {vlm_local_dir} (exists={vlm_local_dir.exists()})")
print(f"[check] HF_HUB_OFFLINE = {os.environ['HF_HUB_OFFLINE']}")

t0 = time.time()

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

device = "cpu"
config = PreTrainedConfig.from_pretrained(model_dir)
config.device = device

if vlm_local_dir.exists():
    config.vlm_model_name = str(vlm_local_dir.resolve())
    print(f"[check] config.vlm_model_name -> {config.vlm_model_name}")
else:
    print("[check] 警告: vlm_local_dirが存在しません。Hub参照のままロードを試みます。")

config.n_action_steps = 10

print("[check] SmolVLAPolicy.from_pretrained() 呼び出し開始...")
policy = SmolVLAPolicy.from_pretrained(model_dir, config=config)
policy.to(device)
policy.eval()
print(f"[check] policy ロード成功 ({time.time() - t0:.1f}s)")

preprocessor_overrides = {
    # このMacにはCUDAが無いためのローカル検証用の上書き。本番(policy_server_smolvla_full.py)には入れない。
    "device_processor": {"device": "cpu"},
}
if vlm_local_dir.exists():
    preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": str(vlm_local_dir.resolve())}
preprocessor, postprocessor = make_pre_post_processors(
    config, pretrained_path=model_dir, preprocessor_overrides=preprocessor_overrides
)
print("[check] preprocessor/postprocessor 構築成功")

# 実際に1推論回して、フォワードパス自体もネットワークなしで完走するか確認する
import numpy as np

front = np.zeros((3, 128, 128), dtype=np.float32)
wrist = np.zeros((3, 128, 128), dtype=np.float32)
state = np.zeros(8, dtype=np.float32)  # eef_pos(3) + axisangle(3) + gripper_qpos(2)

observation = {
    "observation.images.front": torch.from_numpy(front).unsqueeze(0).to(device),
    "observation.images.wrist": torch.from_numpy(wrist).unsqueeze(0).to(device),
    "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
    "task": ["pick up the object"],
}
observation = preprocessor(observation)
with torch.inference_mode():
    action = policy.select_action(observation)
action = postprocessor(action)
action = action.squeeze(0).to("cpu").numpy()
print(f"[check] 推論成功、action.shape={action.shape}, action={action.round(3)}")

print(f"\n[check] 全工程完走。所要時間: {time.time() - t0:.1f}秒")
print("[check] === OFFLINE STARTUP CHECK: PASS ===")
