import numpy as np
import pytest

from parc2026.libero_obs_processing import (
    build_observation_state,
    preprocess_image,
    quat_to_axisangle,
)


def test_quat_identity_gives_zero_rotation():
    # (x, y, z, w) = (0, 0, 0, 1) は無回転
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    result = quat_to_axisangle(quat)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-6)


def test_quat_90deg_around_z():
    # z軸周り90度回転: (0, 0, sin(45deg), cos(45deg))
    angle = np.pi / 2
    quat = np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])
    result = quat_to_axisangle(quat)
    # 軸=z(0,0,1)、角度=90度 のベクトルになるはず
    np.testing.assert_allclose(result, [0.0, 0.0, angle], atol=1e-5)


def test_quat_180deg_around_x():
    # x軸周り180度回転: (sin(90deg), 0, 0, cos(90deg)) = (1, 0, 0, 0)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    result = quat_to_axisangle(quat)
    np.testing.assert_allclose(result, [np.pi, 0.0, 0.0], atol=1e-5)


def test_quat_to_axisangle_rejects_wrong_shape():
    with pytest.raises(ValueError):
        quat_to_axisangle(np.array([0.0, 0.0, 0.0]))


def test_build_observation_state_shape_and_order():
    eef_pos = np.array([0.1, 0.2, 0.3])
    eef_quat = np.array([0.0, 0.0, 0.0, 1.0])  # 無回転
    gripper_qpos = np.array([0.04, -0.04])

    state = build_observation_state(eef_pos, eef_quat, gripper_qpos)

    assert state.shape == (8,)
    assert state.dtype == np.float32
    # [eef_pos(3), axisangle(3)=0, gripper(2)] の順序
    np.testing.assert_allclose(state[:3], eef_pos, atol=1e-6)
    np.testing.assert_allclose(state[3:6], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(state[6:8], gripper_qpos, atol=1e-6)


def test_build_observation_state_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        build_observation_state(np.zeros(2), np.array([0, 0, 0, 1.0]), np.zeros(2))
    with pytest.raises(ValueError):
        build_observation_state(np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3))


def test_preprocess_image_shape_and_range():
    img = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
    result = preprocess_image(img)

    assert result.shape == (3, 128, 128)  # リサイズはモデル側(prepare_images)が行うため、ここでは元サイズのまま
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_image_flips_both_axes():
    # 左上(0,0)だけ白、他は黒の画像 -> flip後は右下(-1,-1)が白になるはず
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[0, 0, :] = 255

    result = preprocess_image(img)

    assert result[0, 0, 0] == pytest.approx(0.0)
    assert result[0, -1, -1] == pytest.approx(1.0)


def test_preprocess_image_rejects_wrong_shape():
    with pytest.raises(ValueError):
        preprocess_image(np.zeros((128, 128), dtype=np.uint8))
