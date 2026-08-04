"""policy_server.py にそのまま貼り付ける自己完結版(parc2026パッケージへの依存なし)。

使い方: 以下の「ここから」〜「ここまで」を丸ごとコピーし、policy_server.py の
`class MyPolicy(BasePolicy):` の中身(__init__/get_action/reset)を置き換える。
(継承元 `BasePolicy` はテンプレート側にすでに定義されているのでそのまま)

観測変換ロジックの根拠は src/parc2026/libero_obs_processing.py および
src/parc2026/my_policy_smolvla.py のdocstringを参照(LeRobot公式ソースコードで検証済み)。
"""

# ============ ここから policy_server.py の MyPolicy に貼り付け ============

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

        self.policy = SmolVLAPolicy.from_pretrained(model_dir, config=config)
        self.policy.to(self.device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config, pretrained_path=model_dir
        )

        self.instruction = ""

    def reset(self, instruction: str = "") -> None:
        self.instruction = instruction
        self.policy.reset()  # action chunkキャッシュをクリア(エピソード間で使い回さないため必須)

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

    def get_action(self, obs: dict) -> "np.ndarray":
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
        assert action.shape == (7,), f"unexpected action shape: {action.shape}"
        return action

# ============ ここまで ============
