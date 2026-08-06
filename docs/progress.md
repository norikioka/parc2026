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
- [x] マージ済みモデル一式(zip)を取得 — `~/projects/academic/parc2026/1st/smolvla_libero_plus_spatial_lora_merged/`
      （Base 66.67%→LoRA後73.33%、+6.67pt。ただし評価は1タスク3エピソードのみでノイズ大、参考値）

**ステップ2完了（初回、2026-08-04）。計算資源温存のため再学習は保留、詳細は`docs/strategy.md`参照**

### ステップ3: MyPolicyへの組み込み・提出物作成
- [x] ローカルで実装（`src/parc2026/my_policy_smolvla.py` + `libero_obs_processing.py`） — 2026-08-04
- [x] Colabにデプロイし`policy_server.py`起動・`/health`確認 — 2026-08-04
- [x] `python -m pipeline --track track1 --n-episodes 2`完走（クラッシュなし） — 2026-08-04
      **ただし全4タスク成功率0%**（1442秒、全エピソード600/600ステップまで到達し未成功）
      **【2026-08-05 重要な注記】この時点のColab venvは`python3.10 -m venv`で作られており、
      後日`lerobot[smolvla]==0.6.0`はPython 3.10に`pip install`できないと判明した(下記参照)。
      つまりこの「完走」は、a) 実際は別のPython(学習ノートブック側でlerobotが入ったシステムPython3.12等)を
      見ていた、b) 何らかの理由で当時は解決できていた、のいずれかであり、**本番構成(python3.10 venv)での
      実績として鵜呑みにできない**。0%という結果の原因究明(タスク分布の違い説等)も、その前提ごと
      再検証が必要**
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

- [x] **オフライン起動確認、Colabを待たずローカルMac(CPU)で先行実施・PASS** — 2026-08-05
      `uv run --with "lerobot[smolvla]==0.6.0"`＋`HF_HUB_OFFLINE=1`で、モデルロード→前処理構築→推論まで
      実際にネットワーク遮断下で完走することを確認。この過程で**新たな重大バグを発見**：
      `config.vlm_model_name`の修正だけでは不十分で、`make_pre_post_processors()`が読む
      `policy_preprocessor.json`の`tokenizer_processor`ステップにも独立してHub参照
      (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`)がハードコードされており、
      これが直っていないと採点環境で起動失敗していた。`preprocessor_overrides`で
      同じローカルパスに向ける修正を`policy_server_smolvla_full.py`に反映し、
      修正後のローカル再検証でPASS（詳細は`docs/env_setup.md`）。
      もし気づかず提出していたら8/4と同種の失敗で提出1回を無駄にしていた可能性が高い。
- [x] **【最重大バグ・対応完了】`lerobot[smolvla]==0.6.0`はPython 3.10に`pip install`不可と判明・
      vendoring方式で解決** — 2026-08-05
      公式README改訂で本番Python 3.10.12が確定 → PyPIメタデータで`lerobot>=0.5.0`はすべて
      `Requires-Python>=3.12`と確認（3.10対応は0.4.4が最後だが、0.4.4は0.6.0形式の保存モデルを読めない
      API構造）。lerobot 0.6.0ソースのうちPython 3.12専用構文・API(PEP695ジェネリクス4箇所、
      `typing.Self`/`typing.Unpack`8箇所)をPython 3.10互換に書き換え、`src/parc2026/vendor/lerobot/`に
      同梱（Apache-2.0、`vendor/NOTICE_PARC2026_MODIFICATIONS.md`に変更内容明記）。
      `planner`によるロードマップ再設計・`critic`(Opus)による2段階レビュー(GO→REVISE、
      指摘4件対応済み)を経て実装。`requirements_smolvla.txt`も実測`pip freeze`ベースに全面書き換え
      （torch/torchvisionの扱い、networkxの罠など詳細は`docs/env_setup.md`）。
      **クリーンルーム検証(新規Python 3.10 venv・HF_HUB_OFFLINE=1・実際のpolicy_server.py)で
      /health→/reset→/actまでPASS**。古い`my_policy_smolvla.py`(未修正の参照用ファイル)は削除。
- [x] **【計算基盤の変更】Colab→Modalに切り替え、GPU実機での検証完了** — 2026-08-06
      Colabのランタイム切断が繰り返し発生し、かつ「どのみちフルリビルドが必要」な状況になったため、
      CLIデプロイ完結型のModal(https://modal.com)に切り替えた（ノートブック実行の往復を排除）。
      公式Dockerfileを本番相当(CUDA13.0/GPU)に改造して`modal_deploy/`にベンダリング。
      1. **オフライン起動確認(GPU実機・PASS)**: CUDA 13.0.3・torch 2.11.0+cu130
         (`cuda.is_available()=True`、CPU版に巻き戻っていないことを確認)・`HF_HUB_OFFLINE=1`下で
         `/health`→`/reset`→`/act`まで完走。act latency 1.05秒（10秒制約に大きく余裕）
      2. **LIBERO-Plus実環境での疎通確認(GPU実機・PASS)**: `python -m pipeline --track track1
         --n-episodes 2`で**track1総合成功率62.5%**(4タスク中: 50%/50%/100%/50%)を記録。
         タイムアウト無し。**8/4の「全4タスク成功率0%」は環境側の不具合(VLM Hub参照・Python版非互換)が
         原因で、モデル自体は機能していたことがこれで裏付けられた**（参考スコア0.0633を大幅に上回る）
      （n=2エピソードのため統計的にはまだノイズが大きい参考値、詳細は`docs/env_setup.md`・`modal_deploy/`）
- [x] **提出用zip作成・Omnicampusへ提出** — 2026-08-06
      Modal(GPU実機)で`submission_template/`一式(policy_server.py+requirements.txt+model_weights/+
      vendor/lerobot/)をzip化し、公式`validate_submission.py`でPASS確認
      (errors=0, warnings=1[非決定的出力の警告のみ、想定内]、act latency mean=0.383s max=1.134s)。
      `submission.zip`(688MB)をOmnicampusにアップロード。
- [x] **【本番採点結果】track1 総合スコア 0.09397** — 2026-08-06
      参考ベースライン0.0633を同一単位で上回る(+48%)。Omnicampus側のログでクラッシュ・タイムアウトは
      無く、正常に完走したことも確認済み。**ゲートA(`docs/strategy.md`)判定=GREEN確定**
      （非ゼロかつ0.0633以上のため、ステップ4への着手条件を満たす）。
      **【訂正】ローカルModal検証の62.5%(成功率のみ)と本番0.09397(成功率×衝突ペナルティ×滑らかさの
      合成スコア)は単位が異なり、直接比較は無効だった**とcritic Opusレビューで判明(2026-08-06)。
      「壊滅的な汎化失敗」という当初の懸念は誤りで、実際は地味だが確かな改善が出ている状態。
      あわせて「LIBERO-Spatial限定学習が原因」という8/4来の仮説も、公式説明会資料3点を全文検索した
      結果**出典が存在しないと判明**（Track1は公式には「同一タスク・同一ドメインへの摂動評価」と
      説明されている）。詳細は`docs/strategy.md`のcriticレビュー記録参照
- [x] **提出採用方式を確認: 最新の提出が採用される方式**（ユーザー確認、2026-08-06）。
      以後、現在のスコア(0.09397)を上書きするリスクがあるため、実験的な再提出は慎重に行う。
      現在のコード状態はgitタグ`submission-0.09397`で固定済み（悪化した場合はここに戻せる）
- [x] **`n_action_steps` 50 vs 10のA/B検証完了** — 2026-08-06、Modal上でtrack1 example4タスクを
      同一条件で実行（提出不要）。**n=50: 総合成功率0%(4タスク全て0/2)、n=10: 総合成功率62.5%
      （再現確認済み）**。8/4の設計変更(50→10)は成功率を大幅に改善しており、criticが懸念した
      滑らかさスコアの悪化を考慮しても総合的に10の方が優れていると判断。コード変更は不要、
      現状維持で確定（`policy_server_smolvla_full.py`に環境変数`PARC_N_ACTION_STEPS`での
      A/B切り替え機構を追加、既定値10は変更なし）
- [x] ステップ4着手。planner設計（2026-08-06）を経て、Modal上で改善ループ基盤を構築
      （`modal_deploy/train_app.py::train` → `app.py::run_evaluate_arm` → `docs/step4_experiments.csv`
      に自動記録、という一連の流れ。`policy_server_smolvla_full.py`に`PARC_MODEL_DIR`環境変数で
      モデル切り替え機構を追加、既定値は変更なし）

### ステップ4: ロバスト性対策 — 実験結果（2026-08-06、Modal上で4アーム学習・評価完了）

全アーム同一条件（3000ステップ、track1 example4タスク×n=10エピソード=40エピソード、Modal L4実機）:

| アーム | 設定 | 成功率 |
|---|---|---|
| control | 現行と同一（episodes_per_task=5） | 82.5% |
| **treatment_b** | **image_transforms有効化** | **87.5%（最良、+5.0pt）** |
| treatment_c | episodes_per_task 5→20 | 85.0%（+2.5pt） |
| treatment_d | image_transforms＋episodes_per_task=20（組み合わせ） | 75.0%（controlより悪化） |

**考察**: image_transforms単体が最良。エピソード数増との組み合わせ(treatment_d)はむしろ悪化しており、
3000ステップという短い学習では過剰なaugmentationが収束を妨げた可能性がある。事前登録した基準
（+10pt以上でないと「明確に優位」と言えない、`docs/strategy.md`参照）には届いていないが、
treatment_b・treatment_cの両方がcontrolを上回る方向で一致しており、弱いながらも一貫したシグナル。

**【重要・2日間ユーザー不在に備えた運用】** 8/7・8/8はユーザーが指示を出せないため、
判断せずに使える提出候補を複数用意しておく方針（8/6ユーザー指示）。

- [x] **提出候補1（安全策）**: `submission_0.09397_KEEP.zip`（688MB）——現行の本番提出物そのもの。
      本番実績あり、track1総合スコア0.09397確定済み
- [x] **提出候補2（実験・本日分）**: `submission_treatment_b_0.875local.zip`（688MB）——
      image_transforms版。ローカルn=10で87.5%、`validate_submission.py`でPASS済み
      （errors=0, warnings=1[非決定的出力、想定内]、act latency mean=0.374s max=1.106s）。
      **本番スコアは未検証**（ローカルと本番は単位が異なる、`docs/strategy.md`参照）
- [x] **`submission_treatment_b_0.875local.zip`をOmnicampusへ提出、本番スコア0.0836** — 2026-08-06
      **現行(0.09397)より悪化**。ローカル成功率(87.5%)は本番の隠しタスクでの性能をほとんど
      予測できていなかったことが判明。ローカル4例タスクのうち3つ(トマトソース/牛乳/コンロ)は
      SPATIAL_TASK_NAMES(黒いボウルのみ10種)の学習セットに一度も含まれていないと発覚。
      Spatial限定学習の汎化不足が疑われる（ユーザー指摘、2026-08-06）
- [x] **提出候補3**: `submission_treatment_c_0.85local.zip`（688MB）——episodes_per_task=20版。
      ローカルn=10で85.0%、validate_submission.py PASS済み。**Omnicampusへ提出済み、結果待ち**
- [x] **【抜本的な変更】全40タスクでの学習に着手** — 2026-08-06
      `train_app.py`に`all_tasks`フラグを追加。SPATIAL_TASK_NAMES(黒いボウル10種)限定ではなく、
      `lerobot/libero_plus`の全40タスクを対象に学習(`arm=broad`、episodes_per_task=5、
      steps=6000[epoch数をcontrolと揃えるため増加]、image_transformsは無し単体で検証)。
      汎用性を重視する方向への転換（ユーザー指摘、2026-08-06: 「黒いボウルだけの学習は
      良くないのでは」）
- [ ] **← 次のアクション**: treatment_cの本番スコアを確認 → broadアームの学習・評価完了を待つ →
      `docs/step4_experiments.csv`とあわせて複数候補を比較し、軸にするモデルを決める

### ステップ5: 提出前の最終チェック・提出
- [ ] `evaluate.py`でzipをエンドツーエンドローカル検証
- [ ] リーダーボードに1回提出して動作確認（1日1回制限に注意）
- [ ] 本提出（8/14 23:59締切、ギリギリ厳禁）
- [ ] レポート提出（8/17 23:59締切、PDF2ページ以内、提出漏れ=即失格なので特に注意）

### ステップ6（時間があれば）: TurboVLA並行実験
- [ ] `docs/strategy.md`参照。SmolVLA提出ライン確保後、別ディレクトリで並行実験

---

**現在のブロッカー**: なし。次のアクションは`examples/smolvla_libero_spatial_lora.ipynb`でのLoRA学習を進めること。
