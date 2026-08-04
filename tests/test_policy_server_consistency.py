"""実際にColabにデプロイされる policy_server_smolvla_full.py の観測変換ロジックが、
テスト済みの libero_obs_processing.py と一致していることを検証する。

背景(2026-08-04 criticレビュー MAJOR#1): policy_server_smolvla_full.py は
parc2026パッケージに依存しない自己完結ファイルとして設計されており、
クォータニオン変換・画像前処理のロジックを libero_obs_processing.py から
「手動コピー」している。libero_obs_processing.py 側だけをテストしていても、
実際に提出される policy_server_smolvla_full.py 側が独自に劣化・乖離しても
テストがそれを検知できない、という指摘を受けて追加した。

このテストは両実装に同じ入力を与え、出力が完全一致することを確認することで、
「デプロイ版が参照実装からズレていないか」を継続的に保証する。
"""

import sys
from pathlib import Path

import numpy as np

from parc2026.libero_obs_processing import preprocess_image, quat_to_axisangle

# policy_server_smolvla_full.py は独立スクリプトとして書かれているため、
# パッケージ経由ではなくパスを通して直接importする
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "parc2026"))
import policy_server_smolvla_full as deployed


def test_deployed_quat_to_axisangle_matches_reference():
    test_quats = [
        [0.0, 0.0, 0.0, 1.0],  # 恒等回転
        [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)],  # z軸90度
        [1.0, 0.0, 0.0, 0.0],  # x軸180度
        [0.1, 0.2, 0.3, 0.9273618495495704],  # 適当な非正規化に近いケース
    ]
    for quat in test_quats:
        quat = np.asarray(quat)
        reference = quat_to_axisangle(quat)
        deployed_result = deployed.MyPolicy._quat_to_axisangle(quat)
        np.testing.assert_allclose(
            reference, deployed_result, atol=1e-6,
            err_msg=f"deployed版がquat={quat}で参照実装と不一致",
        )


def test_deployed_preprocess_image_matches_reference():
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)

    reference = preprocess_image(img)
    deployed_result = deployed.MyPolicy._preprocess_image(img)

    np.testing.assert_allclose(
        reference, deployed_result, atol=1e-6,
        err_msg="deployed版のpreprocess_imageが参照実装と不一致",
    )
