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

## 環境構成（Colab先行、RunPodは保険）

計算資源は**Colab無料枠を先行**させる方針（2026-08-04決定）。SmolVLA(約4.5億パラメータ)は
公式が「Colab T4で完走」を保証している軽量モデルのため、Pi0.5で発生したようなOOM問題は
起きにくいと想定している。詰まった場合のみRunPod（本番同様のLinux+GPU環境）へ切り替える。

- **ローカル(Mac)**: コード編集・git管理専用（MuJoCoのGPU描画非対応のため実行環境としては使わない）
- **Colab（第一候補）**: 配布リポジトリの環境構築・SmolVLA LoRA学習・疎通確認
  （`notebooks/parc2026_pre_setup.ipynb` から起動）
- **RunPod（保険）**: Colabで解決しないエラーが出た場合、または本番環境（Python3.10.12/CUDA13.0/
  NVIDIA L4 24GB/EGL）により近い環境で最終検証したい場合に使う

```
ローカル (Claude Code / Codex CLI で編集)
   │  git push
   ▼
GitHub リポジトリ (public: norikioka/parc2026)
   │
   └─ clone/pull → Colab (notebooks/parc2026_pre_setup.ipynb から matsuolab/PARC2026_pre を
                    clone・setup.sh実行・SmolVLA学習・疎通確認)
```

### ローカルセットアップ

Python 3.12を`uv`で管理。ローカルはコード編集・軽量テスト専用。

```bash
cd ~/projects/academic/parc2026
uv sync              # pyproject.toml の依存関係をインストール
uv run pytest        # テスト実行
```

リポジトリ: https://github.com/norikioka/parc2026 （public）

### Colabセットアップ

1. `notebooks/parc2026_pre_setup.ipynb` をColabで開き、ランタイムをGPUに設定
2. 上から順に実行（Python3.10確保 → `matsuolab/PARC2026_pre`clone → `setup.sh`実行 →
   ランダムポリシーでの疎通確認）
3. 成功したら`examples/smolvla_libero_spatial_lora.ipynb`（配布リポジトリ内、別途Colabで開く）で
   SmolVLAのLoRA学習に進む

詰まった場合は`docs/env_setup.md`の「つまずきポイント一覧」を確認・追記し、それでも解決しなければ
RunPodへの切り替えを検討する（Pi0.5時代のRunPod環境構築手順も同ファイルに残してある）。

## ディレクトリ構成

```
PARC/
├── docs/                              # コンペ説明会資料・戦略・進捗・環境構築メモ
├── notebooks/
│   └── parc2026_pre_setup.ipynb       # Colab起動用（配布リポジトリのセットアップ・疎通確認）
├── src/parc2026/                      # 共通コード（スコアリング等の自作ユーティリティ）
├── tests/                             # ローカルで軽量に回せるユニットテスト
└── pyproject.toml                     # ローカル用依存関係（uv管理）
```

## 開発体制

- ローカル実装: Claude Code
- Codex CLI（`codex exec` / `codex review`）も連携可能。実装のセカンドオピニオンや並行作業に使う場合はその都度指示する
