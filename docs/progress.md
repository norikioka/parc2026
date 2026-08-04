# 進捗チェックリスト

`docs/strategy.md` のロードマップに対応。日付は更新のたびに追記する。

## ステップ0: プロジェクトセットアップ
- [x] ローカル環境(Python 3.12/uv) — 2026-07-28
- [x] GitHubリポジトリ作成・push（public, https://github.com/norikioka/parc2026） — 2026-07-28
- [x] 戦略・環境構築ドキュメント作成 — 2026-07-28

## 準備作業（ローカルで完結）
- [x] スコアリング4指標(成功率/滑らかさ[jerk・SPARC・EEF回転]/実行効率[ステップ数・軌道距離]/安全性[衝突判定])の
      自己評価用ユーティリティ`src/parc2026/scoring.py`実装、ユニットテスト7件・lint通過 — 2026-07-28
      （公式`pipeline/total_score.py`のSPARC実装とロジック一致を確認済み — 2026-08-03）

## 【探索終了】Pi0.5(LeRobot直接eval)ルートの経緯 — 2026-07-29〜08-03
方針転換前の記録。詳細経緯は`docs/env_setup.md`参照。
- [x] Colab無料枠でLeRobot(pi/libero extras)導入・LIBERO-Plus環境構築完走 — 2026-07-29
- [x] Colab無料枠(RAM12GB)でPi0.5ロード時に毎回OOM Kill発生 → Codex CLIの2段階調査で根本原因特定
      （モデル生成自体がCPU上でfloat32実体を構築してから転送する実装のため）→ RunPodへ移行決定
- [x] RunPod(RTX3090, RAM125GB)でGPU Pod起動・SSH直接接続確立・環境再構築完了 — 2026-07-29
- [x] **2026-08-03: 公式配布物公開により方針転換。Pi0.5でのlerobot-eval直接実行ルートは中止**
      （提出形式がHTTPポリシーサーバーであり、lerobot-eval実行結果は提出物にできないと判明したため）

## 【現行】公式配布リポジトリベースの計画（2026-08-03〜）

### 計算資源の状態（2026-08-04時点）
- [x] Colab Proへアップグレード（月額約1,100円、セッション切断リスク低減のため）
- [x] Colab ProでランタイムGPUをL4に選択 — **本番採点環境と同一GPU**を確保
- [x] RunPod: APIキー設定済み・専用SSH鍵登録済み・以前のPod(`then_white_fly`)はStop状態でデータ保持中（保険として温存）

### ステップ1: 配布リポジトリでの環境構築（Colab Pro / L4で進行中）
- [x] `matsuolab/PARC2026_pre` をclone — 2026-08-04
- [x] `bash setup.sh` 実行・完走（`textures=583`、`suite登録 OK`まで確認） — 2026-08-04
      （python3.10-venv不足のバグに遭遇・修正済み、詳細は`docs/env_setup.md`）
- [x] ランダムポリシーで`policy_server.py`起動 → バックグラウンド起動方式のバグ(`%%bash --bg`不安定)を
      `subprocess.Popen`方式に修正 — 2026-08-04
- [ ] 修正版での`python -m pipeline --track track1 --n-episodes 2`疎通確認 **← 次のアクション**
- [x] 環境情報メモ: 本番採点環境=Python 3.10.12 / torch 2.11.0+cu130 / CUDA13.0 / GPU NVIDIA L4(24GB) / EGLレンダリング

### ステップ2: SmolVLA LoRA学習（Colab Pro / L4、公式サンプル）
- [ ] `examples/smolvla_libero_spatial_lora.ipynb` をColabで実行
- [ ] マージ済みモデル一式(zip)を取得

### ステップ3: MyPolicyへの組み込み・提出物作成
- [ ] `submission_template/policy_server.py`の`MyPolicy`にSmolVLAモデルを実装
      （観測: 128×128画像+関節角+EEF位置/姿勢+グリッパー / 出力: 7次元float32相対アクション）
- [ ] 推論が10秒/リクエスト以内に収まることを確認（超過で即0点の重要制約）
- [ ] `validate_submission.py`で静的検査＋起動スモークテスト

### ステップ4: ロバスト性対策（2026-08-04調査で優先順位確定、余力があれば1〜2個）
- [ ] **最優先**: カメラ視点・ロボット初期姿勢へのaugmentation（LIBERO-Plus論文が最弱点と明言している軸）
- [ ] 次点: StableVLA型の軽量アダプタ（[arXiv:2605.18287](https://arxiv.org/abs/2605.18287)、追加10M未満パラメータ、データ拡張不要）
- [ ] 見送り: 言語指示パラフレーズ対策（LIBERO-Plusで言語はほぼ無視されると判明、優先度低）

### ステップ5: 提出前の最終チェック・提出
- [ ] `evaluate.py`でzipをエンドツーエンドローカル検証
- [ ] リーダーボードに1回提出して動作確認（1日1回制限に注意）
- [ ] 本提出（8/14 23:59締切、ギリギリ厳禁）
- [ ] レポート提出（8/17 23:59締切、PDF2ページ以内、提出漏れ=即失格なので特に注意）

### ステップ6（時間があれば）: TurboVLA並行実験
- [ ] `docs/strategy.md`参照。SmolVLA提出ライン確保後、別ディレクトリで並行実験

---

**現在のブロッカー**: なし。次のアクションはColab上で修正版ノートブックの疎通確認セルを再実行すること。
