"""PARC2026 submission_template/policy_server.py の MyPolicy に貼り付けるSmolVLA実装。

## 検証済みの根拠(2026-08-04, LeRobot公式ソースコードを直接確認)

- 観測の前処理(LiberoProcessorStep相当)は`lerobot.processor.env_processor.LiberoProcessorStep`
  (https://github.com/huggingface/lerobot/blob/main/src/lerobot/processor/env_processor.py)を
  そのまま再現している。state=[eef_pos(3), quat_to_axisangle(eef_quat)(3), gripper_qpos(2)]、
  画像はH・W両方向flip。この部分は`src/parc2026/libero_obs_processing.py`でユニットテスト済み
- モデル自身が持つpreprocessor(`policy_preprocessor.json`)はrename/batch化/tokenize/device配置/
  正規化のみを行い、上記のLIBERO固有変換(state構築・画像flip)は含まれていない
  (`~/projects/PARC/1st/smolvla_libero_plus_spatial_lora_merged/policy_preprocessor.json`で確認済み)
  → そのため両方を自分で組み合わせる必要がある
- 画像のリサイズは不要: `SmolVLAPolicy.prepare_images()`が内部で`resize_with_pad()`により
  自動リサイズ・パディングする(`lerobot/policies/smolvla/modeling_smolvla.py`で確認済み)
- 呼び出し順序は`lerobot/scripts/lerobot_eval.py`の実際のロールアウトループで確認済み:
  `policy.reset()` → (毎ステップ) `observation = env_preprocessor(observation)` →
  `observation = preprocessor(observation)` → `action = policy.select_action(observation)` →
  `action = postprocessor(action)`
- `policy.select_action()`は内部でaction chunk(n_action_steps=50)をキャッシュするため、
  50回に1回だけ実際にモデル推論が走り、それ以外はキャッシュからpopするだけで高速
  (10秒/リクエスト制約に対して有利)

## 使い方

このファイルの`SmolVLAMyPolicy`クラスの中身を、`submission_template/policy_server.py`の
`MyPolicy`クラスにコピーする(継承元は`BasePolicy`のまま)。`model_weights/`に本リポジトリの
`1st/smolvla_libero_plus_spatial_lora_merged/`の中身一式を配置してzip化する。

## 未検証の部分(Colabでの実機確認が必要)

- `preprocessor(dict)`, `postprocessor(action)` の正確な入出力型(dictかtensorか)は
  lerobot_eval.pyの使用パターンから推測しているが、実際にこのモデルで動かして確認していない
- 推論速度が10秒/リクエスト以内に収まるか
"""

from pathlib import Path

import numpy as np
import torch

from parc2026.libero_obs_processing import build_observation_state, preprocess_image


class SmolVLAMyPolicy:
    """submission_template/policy_server.py の MyPolicy にそのまま貼り付ける実装。

    貼り付け時は `class MyPolicy(BasePolicy):` の中身をこのクラスの中身で置き換える
    (クラス名・継承元はテンプレート側の `MyPolicy(BasePolicy)` のまま変更しないこと)。
    """

    def __init__(self, model_dir: str = "model_weights"):
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_dir = str(Path(model_dir).resolve())
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

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        # 1. LIBERO固有の観測前処理(LiberoProcessorStep相当、自前実装)
        front_img = preprocess_image(obs["agentview_image"])
        wrist_img = preprocess_image(obs["robot0_eye_in_hand_image"])
        state = build_observation_state(
            eef_pos=obs["robot0_eef_pos"],
            eef_quat=obs["robot0_eef_quat"],
            gripper_qpos=obs["robot0_gripper_qpos"],
        )

        # 2. torch tensor化してバッチ次元を追加(B=1)
        observation = {
            "observation.images.front": torch.from_numpy(front_img).unsqueeze(0).to(self.device),
            "observation.images.wrist": torch.from_numpy(wrist_img).unsqueeze(0).to(self.device),
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device),
            "task": [self.instruction],
        }

        # 3. モデル自身のpreprocessor(rename/tokenize/device配置/正規化) → 推論 → postprocessor
        observation = self.preprocessor(observation)
        with torch.inference_mode():
            action = self.policy.select_action(observation)
        action = self.postprocessor(action)

        # 4. (7,) float32 numpy配列に変換して返す
        action = action.squeeze(0).to("cpu").numpy().astype(np.float32)
        assert action.shape == (7,), f"unexpected action shape: {action.shape}"
        return action
