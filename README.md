# PARC2026 参加リポジトリ

AIRoA × 東京大学松尾・岩澤研究室 Physical AI Robot Challenge 2026 (PARC2026) 参加用。

- 目標: まず予選（8月中旬〜）を確実に突破する
- 評価環境: LIBERO / LIBERO-Plus ベンチマーク
- 採用モデル方針: OpenVLA-OFT（保険としてNora）
- **戦略の詳細は `docs/strategy.md`、環境構築の詳細手順は `docs/env_setup.md` を参照**
- 開催概要: `docs/` 内の説明会資料・要約テキストを参照

## 環境構成（ローカル ⇔ Colab 連携）

計算資源はGoogle Colaboratory（無料/Pro）中心。ローカル(Apple Silicon Mac, GPUなし)は
コード編集・軽量テスト用、実際の学習・シミュレーション実行はColab GPU上で行う。

```
ローカル (Claude Code / Codex CLI で編集)
   │  git push
   ▼
GitHub リポジトリ
   │  git clone / pull
   ▼
Google Colab (GPU, notebooks/colab_bootstrap.ipynb から起動)
```

### ローカルセットアップ

Python 3.12を`uv`で管理（LeRobot最新版の要件に合わせている）。
ローカルはコード編集・軽量テスト専用 — MuJoCoのGPU描画がMacで動かないため、
実際のシミュレーション・学習はColab側で行う。

```bash
cd ~/projects/PARC
uv sync              # pyproject.toml の依存関係をインストール
uv run pytest        # テスト実行（あれば）
uv run jupyter lab   # ノートブックをローカルで開く場合
```

### Colabセットアップ

1. GitHubにこのリポジトリをpush（未作成の場合は要相談）
2. HuggingFaceでPaliGemmaの利用規約に同意し、トークンを発行。Colab Secretsに`HF_TOKEN`として登録
3. `notebooks/colab_bootstrap.ipynb` をColabで開き、ランタイムをGPUに設定
4. `GITHUB_REPO_URL` を実際のリポジトリURLに書き換えて上から順に実行
   - GPU確認 → Driveマウント（`HF_HOME`永続化・`MUJOCO_GL=egl`設定）→ 自分のリポジトリclone/pull →
     LeRobot本体を`pi,libero` extrasでインストール → HF認証 → （無印LIBERO検証後に）LIBERO-Plus導入

詳細な手順・つまずきポイントは `docs/env_setup.md` を参照。

## ディレクトリ構成

```
PARC/
├── docs/                    # コンペ説明会資料・調査メモ
├── notebooks/
│   └── colab_bootstrap.ipynb  # Colab起動用ノートブック
├── src/parc2026/            # 共通コード（ローカル・Colab両方から import）
├── tests/                   # ローカルで軽量に回せるユニットテスト
├── pyproject.toml           # ローカル用依存関係（uv管理）
└── requirements-colab.txt   # Colab用依存関係（LeRobot/robosuite/mujoco等）
```

## 開発体制

- ローカル実装: Claude Code
- Codex CLI（`codex exec` / `codex review`）も連携可能。実装のセカンドオピニオンや並行作業に使う場合はその都度指示する
