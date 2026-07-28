"""環境変数・パス関連の共通設定。ローカル(Mac)とColab(Linux+GPU)の両方から import される。"""

import os
from pathlib import Path

# Colab上では /content/drive/MyDrive/PARC2026 、ローカルでは repo 直下の .local_data を既定にする
DRIVE_ROOT = Path(os.environ.get("PARC_DRIVE_ROOT", "/content/drive/MyDrive/PARC2026"))
LOCAL_FALLBACK_ROOT = Path(__file__).resolve().parents[2] / ".local_data"

DATA_ROOT = DRIVE_ROOT if DRIVE_ROOT.parent.exists() else LOCAL_FALLBACK_ROOT
CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
HF_CACHE_DIR = DATA_ROOT / "hf_cache"


def ensure_dirs() -> None:
    for d in (DATA_ROOT, CHECKPOINT_DIR, HF_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
