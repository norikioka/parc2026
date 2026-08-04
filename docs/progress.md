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
- [x] 修正版での`python -m pipeline --track track1 --n-episodes 2`疎通確認 — 2026-08-04完走
- [x] 環境情報メモ: 本番採点環境=Python 3.10.12 / torch 2.11.0+cu130 / CUDA13.0 / GPU NVIDIA L4(24GB) / EGLレンダリング

**ステップ1完了（2026-08-04）**

### ステップ2: SmolVLA LoRA学習（Colab Pro / L4、公式サンプル）
- [x] `examples/smolvla_libero_spatial_lora.ipynb` をColabで実行・完走 — 2026-08-04
- [x] マージ済みモデル一式(zip)を取得 — `~/projects/PARC/1st/smolvla_libero_plus_spatial_lora_merged/`
      （Base 66.67%→LoRA後73.33%、+6.67pt。ただし評価は1タスク3エピソードのみでノイズ大、参考値）

**ステップ2完了（初回、2026-08-04）。計算資源温存のため再学習は保留、詳細は`docs/strategy.md`参照**

### ステップ3: MyPolicyへの組み込み・提出物作成
- [x] ローカルで実装（`src/parc2026/my_policy_smolvla.py` + `libero_obs_processing.py`） — 2026-08-04
- [x] Colabにデプロイし`policy_server.py`起動・`/health`確認 — 2026-08-04
- [x] `python -m pipeline --track track1 --n-episodes 2`完走（クラッシュなし） — 2026-08-04
      **ただし全4タスク成功率0%**（1442秒、全エピソード600/600ステップまで到達し未成功）
- [x] **criticによる厳格レビュー実施**（Fable/Opus相当） — 2026-08-04
      CRITICAL 0件、MAJOR 8件、MINOR 5件。クォータニオン変換・状態ベクトル構築・LeRobot API呼び出しは
      LeRobot v0.6.0実ソースと1行ずつ突き合わせ、完全に正確と確認された（ACCEPT-WITH-RESERVATIONS判定）
- [x] MAJOR是正（ローカルで完結する分） — 2026-08-04
      - `get_action`に例外処理追加（予選N=1でクラッシュ即0点を回避、ゼロアクションにフォールバック）
      - `assert`を`raise ValueError`に変更（`python -O`で無効化される問題を回避）
      - デプロイ版(`policy_server_smolvla_full.py`)とテスト対象(`libero_obs_processing.py`)の
        整合性を検証するクロスチェックテストを追加(`test_policy_server_consistency.py`)
      - `requirements_smolvla.txt`作成、`lerobot[smolvla]`extra明記（忘れるとImportErrorで即死）
      - 重複していた`my_policy_smolvla_standalone.py`を削除（`policy_server_smolvla_full.py`に一本化）
      - dtype検証・非ゼロ回転角のテストケースを追加（テスト20件、lint通過）
- [x] front/wristカメラ対応の実データ確認 — 2026-08-04完了、**カメラ対応は正しいと確認**
      （front=俯瞰視点、wrist=手先アップの画像を実際に目視確認。MAJOR#3の懸念は解消）
      → 0%の原因はコードバグではなく、**LoRA学習タスク(Spatial 10種)とTrack1 exampleタスク
      (Object/Goal含む、L2〜L5難易度)の分布の違い**である可能性が高いという仮説に更新
- [x] 推論レイテンシ確認 — 2026-08-04、`validate_submission.py`のスモークテストで
      **平均0.29秒・最大0.86秒**（10秒制約に大きく余裕あり）
- [x] `validate_submission.py`で静的検査＋起動スモークテスト — 2026-08-04 **PASS（errors=0, warnings=1）**
      警告は`nondeterministic`（Flow Matchingによる意図的な確率的挙動、対応不要）のみ
- [x] **【訂正】ネットワーク遮断下でのモデルロード確認 → 優先度を最上位に戻す**（後述のplanner設計で判明）

**【重要インシデント】2026-08-04 誤った提出物での初回提出が失敗**:
`1st/smolvla_libero_plus_spatial_lora_merged.zip`（モデル重みのみ、`policy_server.py`/
`requirements.txt`なし）を誤って提出 → Omnicampus実採点で
`struct.no_policy_server`エラーによりFAIL（public_score=0.0, private_score=0.0）。
今日の提出回数を消費した可能性あり（要Omnicampus画面での確認）。

**さらに`requirements.txt`自体にも別の欠陥が発覚**: `lerobot[smolvla] @ git+https://...`という
git参照形式が、提出物バリデーションで「外部ソース（git+https:）の指定は禁止」として拒否されると判明
（ローカルの`validate_submission.py`実行で検出）。PyPIに公式リリース済みの`lerobot==0.6.0`
（gitタグv0.6.0と同一、smolvla extra含む）が存在することを確認し、`lerobot[smolvla]==0.6.0`に修正。
修正後、正しい構造（`policy_server.py`+`requirements.txt`+`model_weights/`）でzip化し、
ローカルバリデーションでPASSを確認済み。**次は正しいzipでの再提出**。

**【2026-08-04 planner設計・重大発見】急いで再提出せず、以下を先に修正することにした**
（ユーザー方針：今日は既に1回提出済みで急ぐ必要がないため、次の貴重な1回の質を上げる）:

1. **【最重要・実ソースで検証済み】VLMトークナイザがHub参照のままだと採点環境で起動失敗する**:
   `SmolVLAPolicy`は`config.vlm_model_name`(既定"HuggingFaceTB/SmolVLM2-500M-Video-Instruct"という
   HF Hub文字列)から`AutoConfig.from_pretrained`/`AutoProcessor.from_pretrained`を必ず呼ぶ
   （`lerobot/policies/smolvla/smolvlm_with_expert.py:99,101`で確認、`load_vlm_weights=False`でも
   `AutoProcessor`側は無条件実行）。評価中は外部ネットワーク禁止のため、ローカルにキャッシュがない
   採点環境では**サーバー起動そのものが失敗する**。ローカルColabで動いていたのは学習時にキャッシュが
   温まっていたためで、この構成のテストでは原理的に検出不可能だった
   → 対応: `model.safetensors`にVLM本体の重みは既に含まれている（500個中490個のテンソルが
   `vlm_with_expert.*`と確認済み）ため、Hubから必要なのは軽量なconfig/tokenizerファイルのみ
   （計約4.8MB）。`src/parc2026/vlm_assets/`にダウンロード・git管理下に置き、`policy_server_smolvla_full.py`
   で`config.vlm_model_name`をこのローカルパスに向けるよう修正済み。Colabノートブックにも
   ダウンロード・配置セルとオフライン起動確認セルを追加済み
2. `n_action_steps`を50→10に変更（`chunk_size`=50とは独立、単純なキュー切り出しのため安全と実ソースで確認済み）。
   600ステップのエピソード中12回しか画像を見ない開ループ状態を解消し、再計画頻度を5倍に
3. アクション統計の診断ログを追加（100ステップごとにグリッパー値等を出力、次回GPU実行の情報量を増やす）
4. torch依存の衝突リスクを確認 → `lerobot==0.6.0`の制約は`torch<2.12.0,>=2.7`で、
   本番環境のtorch2.11.0+cu130と既に互換、問題なし

- [ ] **← 次のアクション**: Colabで上記1〜3を反映したzipを作成し、
      **オフライン起動確認**（新規追加セクション7.5）→ ローカル`validate_submission.py`再PASS →
      `python -m pipeline`で回帰確認（性能measureではなくクラッシュ有無・グリッパー動作の確認）→
      問題なければ次回提出時に公式参考スコア0.0633と比較する（`docs/strategy.md`の判断ゲート参照）

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

**現在のブロッカー**: なし。次のアクションは`examples/smolvla_libero_spatial_lora.ipynb`でのLoRA学習を進めること。
