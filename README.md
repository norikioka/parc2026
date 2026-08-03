# PARC2026 参加リポジトリ

AIRoA × 東京大学松尾・岩澤研究室 Physical AI Robot Challenge 2026 (PARC2026) 参加用。

- 目標: まず予選（〜8/14提出締切）を確実に突破する
- 公式予選配布リポジトリ: https://github.com/matsuolab/PARC2026_pre （一次情報・最優先で参照）
- 採用モデル方針: SmolVLA（`lerobot/smolvla_libero_plus`ベースのLoRA追加学習、公式サンプルに準拠）
- **戦略の詳細は `docs/strategy.md`、環境構築の詳細手順は `docs/env_setup.md` を参照**
- 開催概要: `docs/` 内の説明会資料・要約テキストを参照

## 提出形式（公式仕様）

LeRobotの評価コマンドをそのまま提出するのではなく、**HTTPポリシーサーバー一式のzip**を提出する。
観測(128×128画像+関節角+EEF位置/姿勢+グリッパー)→7次元float32相対アクションを返すサーバーを実装し、
`submission_template/policy_server.py`の`MyPolicy`クラスだけを編集する。
**推論は1リクエスト10秒以内厳守**（超過でTrackが即0点）。詳細は`docs/strategy.md`参照。

## 環境構成（ローカル ⇔ RunPod / Colab 連携）

- **ローカル(Mac)**: コード編集・git管理専用（MuJoCoのGPU描画非対応のため実行環境としては使わない）
- **RunPod**: 配布リポジトリ`setup.sh`実行・`pipeline`での評価・ポリシーサーバー動作確認
  （本番採点環境=Python 3.10.12 / CUDA13.0 / NVIDIA L4 24GB / EGLレンダリング）
- **Colab**: SmolVLA LoRA学習ノートブック実行（公式が無料T4での完走を保証）

```
ローカル (Claude Code / Codex CLI で編集)
   │  git push
   ▼
GitHub リポジトリ (public: norikioka/parc2026)
   │
   ├─ clone/pull → RunPod (matsuolab/PARC2026_pre の setup.sh 実行、評価パイプライン)
   └─ clone/pull → Colab (SmolVLA LoRAノートブック実行)
```

### ローカルセットアップ

Python 3.12を`uv`で管理。ローカルはコード編集・軽量テスト専用。

```bash
cd ~/projects/PARC
uv sync              # pyproject.toml の依存関係をインストール
uv run pytest        # テスト実行
```

リポジトリ: https://github.com/norikioka/parc2026 （public）

### RunPodセットアップ

配布リポジトリ本体は本リポジトリには含めず、RunPod上で別途clone・構築する（`docs/env_setup.md`参照）。

```bash
git clone https://github.com/matsuolab/PARC2026_pre.git
cd PARC2026_pre
bash setup.sh     # 初回のみ、10〜20分
source env.sh      # 評価実行のたび毎回

# 動作確認（ランダムポリシーのまま疎通確認）
python submission_template/policy_server.py --port 8000 &
python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 2
```

Pi0.5(LeRobot直接eval)ルートで発生したOOM問題の詳細な経緯・回避策は`docs/env_setup.md`に残してある
（方針転換済みだが、同種の問題の参考として保持）。

## ディレクトリ構成

```
PARC/
├── docs/                    # コンペ説明会資料・戦略・進捗・環境構築メモ
├── src/parc2026/            # 共通コード（スコアリング等の自作ユーティリティ）
├── tests/                   # ローカルで軽量に回せるユニットテスト
└── pyproject.toml           # ローカル用依存関係（uv管理）
```

## 開発体制

- ローカル実装: Claude Code
- Codex CLI（`codex exec` / `codex review`）も連携可能。実装のセカンドオピニオンや並行作業に使う場合はその都度指示する
