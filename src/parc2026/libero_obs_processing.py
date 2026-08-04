"""PARC2026のTrack1観測(robosuite/LIBERO-plus形式)をSmolVLA(LeRobot経由)の入力形式に変換する。

変換ロジックはLeRobot公式実装 `lerobot.processor.env_processor.LiberoProcessorStep`
(https://github.com/huggingface/lerobot/blob/main/src/lerobot/processor/env_processor.py)
を正確に再現したもの。SmolVLAはこのプロセッサで前処理された観測で学習・評価されているため、
ここを間違えるとモデルは学習時と異なる入力を受け取り、性能が大きく劣化する。

- 状態ベクトル(observation.state, shape=(8,)): eef_pos(3) + eef_quatのaxis-angle変換(3) + gripper_qpos(2)
- 画像(observation.images.*): H・W両方向に180度回転(flip)してから目標解像度にリサイズ
  (LeRobotの学習データセット(HuggingFaceVLA/libero)のカメラ向き規約に合わせるため必須)
"""

import numpy as np


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """クォータニオン(x, y, z, w)を軸角度ベクトル(3,)に変換する。

    LeRobotの`LiberoProcessorStep._quat2axisangle`と同一のアルゴリズム。
    """
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape != (4,):
        raise ValueError(f"quat_to_axisangle expected shape (4,), got {quat.shape}")

    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(max(1.0 - w * w, 0.0))

    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * np.arccos(w)
    axis = quat[:3] / den
    return (axis * angle).astype(np.float32)


def build_observation_state(
    eef_pos: np.ndarray, eef_quat: np.ndarray, gripper_qpos: np.ndarray
) -> np.ndarray:
    """PARC2026の生観測(robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos)から
    SmolVLAが期待する8次元のobservation.stateを構築する。

    構成順序: [eef_pos(3), eef_axisangle(3), gripper_qpos(2)] = 8次元
    """
    eef_pos = np.asarray(eef_pos, dtype=np.float32)
    gripper_qpos = np.asarray(gripper_qpos, dtype=np.float32)
    if eef_pos.shape != (3,):
        raise ValueError(f"eef_pos expected shape (3,), got {eef_pos.shape}")
    if gripper_qpos.shape != (2,):
        raise ValueError(f"gripper_qpos expected shape (2,), got {gripper_qpos.shape}")

    eef_axisangle = quat_to_axisangle(eef_quat)
    state = np.concatenate([eef_pos, eef_axisangle, gripper_qpos]).astype(np.float32)
    assert state.shape == (8,)
    return state


def preprocess_image(image_hwc_uint8: np.ndarray) -> np.ndarray:
    """PARC2026の生画像(H, W, 3) uint8 をSmolVLAの入力形式(3, H, W) float32[0,1]に変換する。

    リサイズは不要: SmolVLAPolicy.prepare_images()が内部でresize_with_pad()により
    config.resize_imgs_with_padding(512x512)へ自動リサイズ・パディングするため
    (`lerobot/policies/smolvla/modeling_smolvla.py`で確認済み)。ここでは元解像度のまま、
    LeRobotの`LiberoProcessorStep`と同じ「H・W方向を反転(180度回転)」のみ行う
    (これはLeRobotの学習データセットのカメラ向き規約に合わせるための必須処理で、
    自動化されない)。
    """
    image_hwc_uint8 = np.asarray(image_hwc_uint8)
    if image_hwc_uint8.ndim != 3 or image_hwc_uint8.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got {image_hwc_uint8.shape}")

    # uint8 [0,255] HWC -> float32 [0,1] CHW
    img = image_hwc_uint8.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # (3, H, W)

    # H・W両方向のflip(180度回転) — LeRobotの学習データのカメラ向き規約に合わせる
    img = img[:, ::-1, ::-1]

    return np.ascontiguousarray(img)
