"""ポリシーサーバー（提出用テンプレート）

このファイルを編集して、自分のモデルを組み込んでください。
編集が必要なのは MyPolicy クラスの中身だけです。
それ以外のコード（サーバー部分、シリアライゼーション）は変更不可です。

ローカルテスト:
    pip install -r requirements.txt
    python policy_server.py                  # サーバー起動（port 8000）

    # 別ターミナルで評価実行
    python -m pipeline --server-url http://localhost:8000 --dry-run
"""

import argparse
from abc import ABC, abstractmethod

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response

# ============================================================
# ポリシーのインターフェース定義（変更不可）
# MyPolicy が満たすべき get_action() / reset() の仕様を定める。
# ============================================================


class BasePolicy(ABC):
    """ポリシーの基底クラス。get_action() と reset() を実装してください。"""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測からアクションを推論する。

        Args:
            obs: 環境からの観測。以下のキーが含まれる:
                - "agentview_image": (128, 128, 3) uint8
                - "robot0_eye_in_hand_image": (128, 128, 3) uint8
                - "robot0_joint_pos": (7,) float
                - "robot0_eef_pos": (3,) float
                - "robot0_eef_quat": (4,) float
                - "robot0_gripper_qpos": (2,) float

        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        ...

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時に呼ばれる。内部状態をリセットしてください。

        Args:
            instruction: タスクの言語指示（例: "pick up the red mug and place it on the shelf"）
        """
        ...


# ============================================================
# ここを編集する（MyPolicy の中身だけを自分のモデルに置き換える）
# ============================================================


class MyPolicy(BasePolicy):
    def __init__(self):
        from pathlib import Path

        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_dir = str((Path(__file__).parent / "model_weights").resolve())
        config = PreTrainedConfig.from_pretrained(model_dir)
        config.device = self.device

        # 【2026-08-04 planner設計・実ソース検証済み】評価中は外部ネットワークアクセスが禁止されているが、
        # SmolVLAPolicyはvlm_model_name(デフォルト"HuggingFaceTB/SmolVLM2-500M-Video-Instruct"という
        # HF Hub上の文字列参照)からAutoConfig.from_pretrained/AutoProcessor.from_pretrainedを
        # 必ず呼ぶ(smolvlm_with_expert.py:99,101、load_vlm_weightsの値に関わらずAutoProcessor側は無条件実行)。
        # モデル本体の重みは1st/.../model.safetensors自体に既に含まれている(500個中490個のテンソルが
        # vlm_with_expert.*)ため、Hubから取得すべきはconfig/tokenizer等の軽量ファイルのみ。
        # これをmodel_weights/vlm/にローカル同梱し、Hubアクセスを完全に回避する。
        vlm_local_dir = Path(__file__).parent / "model_weights" / "vlm"
        if vlm_local_dir.exists():
            config.vlm_model_name = str(vlm_local_dir.resolve())
        else:
            print(f"[MyPolicy] 警告: {vlm_local_dir} が見つかりません。"
                  f"vlm_model_nameはHF Hub参照のままのため、評価中のネットワーク遮断でサーバー起動が失敗する恐れがあります。")

        # 【2026-08-04 planner設計・実ソース検証済み】chunk_size(=50, モデルが1回のFlow Matchingで
        # 予測する長さ)とn_action_steps(=実際に使ってから再計画するまでの長さ)は独立しており、
        # select_action()は単純に先頭n_action_steps個をキューに積むだけ(modeling_smolvla.py:267)。
        # 既定の50だと600ステップのエピソード中わずか12回しか画像を見ずに開ループで動くことになり、
        # LIBERO-Plus型の摂動タスクでは途中でズレると修正できない。10に下げて再計画頻度を5倍にする
        # (計算コストは変わらない。実測レイテンシ平均0.29秒/最大0.86秒に対し10秒制約の余裕は十分)。
        config.n_action_steps = 10

        self.policy = SmolVLAPolicy.from_pretrained(model_dir, config=config)
        self.policy.to(self.device)
        self.policy.eval()

        # 【2026-08-05 ローカル(Mac/CPU)でHF_HUB_OFFLINE=1下の実行を検証して発見】
        # config.vlm_model_nameとは別に、model_dir/policy_preprocessor.jsonの
        # tokenizer_processorステップにもHub文字列("HuggingFaceTB/SmolVLM2-...")が
        # 独立してハードコードされており、make_pre_post_processors内部で
        # AutoTokenizer.from_pretrained(tokenizer_name)としてHubアクセスを試みる。
        # config.vlm_model_nameの修正だけでは防げないため、preprocessor_overridesで
        # 同じローカルパスに向ける必要がある(この修正がないと採点環境でここだけ起動失敗する)。
        preprocessor_overrides = {}
        if vlm_local_dir.exists():
            preprocessor_overrides["tokenizer_processor"] = {
                "tokenizer_name": str(vlm_local_dir.resolve())
            }

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config, pretrained_path=model_dir, preprocessor_overrides=preprocessor_overrides
        )

        self.instruction = ""
        self._step_count = 0  # 診断ログ用(A6): 100ステップごとにアクション統計を出力

    def reset(self, instruction: str = "") -> None:
        self.instruction = instruction
        self.policy.reset()  # action chunkキャッシュをクリア(エピソード間で使い回さないため必須)
        self._step_count = 0

    @staticmethod
    def _quat_to_axisangle(quat):
        import numpy as np

        quat = np.asarray(quat, dtype=np.float32)
        w = float(np.clip(quat[3], -1.0, 1.0))
        den = np.sqrt(max(1.0 - w * w, 0.0))
        if den <= 1e-10:
            return np.zeros(3, dtype=np.float32)
        angle = 2.0 * np.arccos(w)
        axis = quat[:3] / den
        return (axis * angle).astype(np.float32)

    @staticmethod
    def _preprocess_image(image_hwc_uint8):
        import numpy as np

        img = image_hwc_uint8.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = img[:, ::-1, ::-1]  # LeRobotの学習データのカメラ向き規約に合わせてH・W方向をflip
        return np.ascontiguousarray(img)

    def get_action(self, obs: dict) -> np.ndarray:
        # 予選はN=1(1試行の失敗がそのままTrackのスコア喪失に直結する)ため、
        # 想定外の例外でクラッシュするより、無難なゼロアクション(その場に留まる)を返す方が安全。
        # (2026-08-04 criticレビューMAJOR#5で指摘、対応)
        try:
            return self._get_action_impl(obs)
        except Exception as exc:  # noqa: BLE001 — 予選N=1のため意図的に広くキャッチしゼロ点回避を優先
            print(f"[MyPolicy] get_action failed, falling back to zero action: {exc}")
            return np.zeros(7, dtype=np.float32)

    def _get_action_impl(self, obs: dict) -> np.ndarray:
        import numpy as np

        front_img = self._preprocess_image(obs["agentview_image"])
        wrist_img = self._preprocess_image(obs["robot0_eye_in_hand_image"])

        eef_axisangle = self._quat_to_axisangle(obs["robot0_eef_quat"])
        state = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                eef_axisangle,
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            ]
        ).astype(np.float32)

        observation = {
            "observation.images.front": self.torch.from_numpy(front_img).unsqueeze(0).to(self.device),
            "observation.images.wrist": self.torch.from_numpy(wrist_img).unsqueeze(0).to(self.device),
            "observation.state": self.torch.from_numpy(state).unsqueeze(0).to(self.device),
            "task": [self.instruction],
        }

        observation = self.preprocessor(observation)
        with self.torch.inference_mode():
            action = self.policy.select_action(observation)
        action = self.postprocessor(action)

        action = action.squeeze(0).to("cpu").numpy().astype(np.float32)
        if action.shape != (7,):
            raise ValueError(f"unexpected action shape: {action.shape}")

        # 診断ログ(A6): 100ステップごとにグリッパー値等を出力し、
        # 「そもそも掴もうとしているか」を次回のGPU実行から確認できるようにする
        self._step_count += 1
        if self._step_count % 100 == 1:
            print(f"[MyPolicy] step={self._step_count} action={action.round(3)} "
                  f"gripper={action[6]:.3f}")

        return action


# ============================================================
# 以下は変更不可
# ============================================================


def deserialize_obs(data: bytes) -> dict[str, np.ndarray]:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb(
        {"data": action.astype(np.float32).tobytes()},
        use_bin_type=True,
    )


app = FastAPI(title="VLA Policy Server")
_policy: BasePolicy | None = None


def set_policy(policy: BasePolicy) -> None:
    global _policy
    _policy = policy


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        import json
        data = json.loads(body)
        instruction = data.get("instruction", "")
    _policy.reset(instruction=instruction)
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    action = _policy.get_action(obs)
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    set_policy(MyPolicy())
    print(f"Policy server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
