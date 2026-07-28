"""PARC2026説明会で示されたスコアリング4指標(タスク成功率/滑らかさ/実行効率/安全性)の
自己評価用ユーティリティ。重み付けは非公開のため、ここでは各指標を素朴に算出するのみで
最終スコアへの合成は行わない。LIBERO評価ログ(EEF軌道・衝突フラグ)を渡して使う想定。

滑らかさ指標のうちSPARC(Spectral Arc Length)は以下の文献の定義に基づく:
Balasubramanian et al., "On the analysis of movement smoothness", 2015.
"""

from itertools import pairwise

import numpy as np


def success_rate(successes: list[bool]) -> float:
    if not successes:
        return 0.0
    return float(np.mean(successes))


def jerk_rms(eef_positions: np.ndarray, dt: float) -> float:
    """EEF位置の時系列(shape: [T, 3])からジャーク(加速度の時間微分)のRMSを求める。"""
    if len(eef_positions) < 4:
        return 0.0
    velocity = np.diff(eef_positions, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    return float(np.sqrt(np.mean(np.sum(jerk**2, axis=1))))


def sparc(eef_positions: np.ndarray, dt: float, fc_cutoff: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral Arc Length。値は負で、0に近いほど滑らか。"""
    if len(eef_positions) < 4:
        return 0.0
    velocity = np.linalg.norm(np.diff(eef_positions, axis=0) / dt, axis=1)
    if np.allclose(velocity, 0):
        return 0.0

    n = len(velocity)
    freq = np.fft.rfftfreq(n, d=dt)
    spectrum = np.abs(np.fft.rfft(velocity))
    spectrum /= spectrum.max() + 1e-12

    mask = freq <= fc_cutoff
    freq, spectrum = freq[mask], spectrum[mask]

    above_th = np.where(spectrum >= amp_th)[0]
    if len(above_th) == 0:
        return 0.0
    freq, spectrum = freq[: above_th[-1] + 1], spectrum[: above_th[-1] + 1]

    arc_length = -np.sum(np.sqrt(np.diff(freq) ** 2 + np.diff(spectrum) ** 2))
    return float(arc_length)


def eef_rotation_total(eef_quaternions: np.ndarray) -> float:
    """連続する姿勢クォータニオン間の角度差の総和(ラジアン)。shape: [T, 4] (w,x,y,z)。"""
    if len(eef_quaternions) < 2:
        return 0.0
    total = 0.0
    for q1, q2 in pairwise(eef_quaternions):
        dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
        total += 2 * np.arccos(dot)
    return float(total)


def step_count(eef_positions: np.ndarray) -> int:
    return len(eef_positions)


def trajectory_total_distance(eef_positions: np.ndarray) -> float:
    if len(eef_positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(eef_positions, axis=0), axis=1)))


def collision_free(collision_flags: list[bool]) -> bool:
    """1つでも衝突があればFalse(安全性NG)。"""
    return not any(collision_flags)


def summarize_episode(
    eef_positions: np.ndarray,
    eef_quaternions: np.ndarray,
    collision_flags: list[bool],
    success: bool,
    dt: float,
) -> dict:
    """1エピソード分の生ログから4指標を算出してdictで返す。"""
    return {
        "success": success,
        "jerk_rms": jerk_rms(eef_positions, dt),
        "sparc": sparc(eef_positions, dt),
        "eef_rotation_total": eef_rotation_total(eef_quaternions),
        "step_count": step_count(eef_positions),
        "trajectory_total_distance": trajectory_total_distance(eef_positions),
        "collision_free": collision_free(collision_flags),
    }
