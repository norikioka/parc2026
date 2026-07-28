import numpy as np

from parc2026.scoring import (
    collision_free,
    eef_rotation_total,
    jerk_rms,
    sparc,
    step_count,
    success_rate,
    summarize_episode,
    trajectory_total_distance,
)


def test_success_rate():
    assert success_rate([True, True, False, True]) == 0.75
    assert success_rate([]) == 0.0


def test_jerk_rms_smooth_vs_jerky():
    t = np.linspace(0, 1, 100)
    smooth = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    jerky = smooth.copy()
    jerky[::5, 0] += 0.1  # 不規則な揺れを混ぜる

    assert jerk_rms(smooth, dt=t[1] - t[0]) < jerk_rms(jerky, dt=t[1] - t[0])


def test_sparc_returns_non_positive():
    t = np.linspace(0, 1, 100)
    positions = np.stack([np.sin(t), np.cos(t), t], axis=1)
    value = sparc(positions, dt=t[1] - t[0])
    assert value <= 0.0


def test_eef_rotation_total_zero_when_constant():
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
    assert eef_rotation_total(quats) == 0.0


def test_step_count_and_distance():
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    assert step_count(positions) == 3
    assert trajectory_total_distance(positions) == 2.0


def test_collision_free():
    assert collision_free([False, False, False])
    assert not collision_free([False, True, False])


def test_summarize_episode_shape():
    t = np.linspace(0, 1, 20)
    positions = np.stack([t, t, t], axis=1)
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (20, 1))
    result = summarize_episode(
        eef_positions=positions,
        eef_quaternions=quats,
        collision_flags=[False] * 20,
        success=True,
        dt=t[1] - t[0],
    )
    assert result["success"] is True
    assert result["collision_free"] is True
    assert result["step_count"] == 20
