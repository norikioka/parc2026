# PARC2026 攻略戦略（予選突破確実優先）

前提: 目標=予選突破を確実にする（優勝・上位進出は狙わない）／ロボット学習の実装経験はほぼゼロ／
週5〜10時間／GPU環境はRunPod中心（Colab無料枠はメモリ不足で断念、経緯は`docs/env_setup.md`参照）。

**2026-08-03: 公式予選配布物（PDF「PARC2026開発コンペティション_予選」+ GitHub
`matsuolab/PARC2026_pre`）が公開され、以下は全てそちらの一次情報に基づき全面更新した。**
それ以前の記述（LeRobotの`lerobot-eval`直接実行・Pi0.5前提の戦略）は運営の実際の提出仕様と
異なっていたため置き換えている。

## 予選の正確なルール（公式配布物より）

### スケジュール
- 予選開始: 2026/7/31(木) 18:00〜
- **モデル提出締切: 2026/8/14(金) 23:59**
- **レポート提出締切: 2026/8/17(月) 23:59**（PDF2ページ以内・日本語or英語、締切までに複数回提出可）
- 予選最終評価ランキング＋レポート内容を踏まえた上位200人が本選へ選出

### 提出形式（最重要・旧想定と異なる）
LeRobotの`lerobot-eval`をそのまま提出するのではなく、**HTTPポリシーサーバー一式のzip**を提出する。
`submission_template/policy_server.py`の`MyPolicy`クラスだけを編集し、それ以外（FastAPIサーバー部分・
シリアライゼーション）は変更不可。

```
submission.zip
├── policy_server.py   # MyPolicyクラスを編集(必須)
├── requirements.txt   # 追加依存があれば記載(必須)
└── model_weights/     # チェックポイント等を配置(任意)
```

**観測(observation)の形式**:
- `agentview_image`: (128,128,3) uint8
- `robot0_eye_in_hand_image`: (128,128,3) uint8
- `robot0_joint_pos`: (7,) float
- `robot0_eef_pos`: (3,) float
- `robot0_eef_quat`: (4,) float
- `robot0_gripper_qpos`: (2,) float

**アクション出力**: (7,) float32 相対値 `[dx, dy, dz, droll, dpitch, dyaw, gripper]`

**エンドポイント**: `GET /health`（起動確認）、`POST /reset`（instruction, seedを受け取る）、
`POST /act`（msgpack観測→msgpackアクション）

### タイムアウト制約（超重要）
**`/act`・`/reset` の1リクエストが10秒を超えると、そのTrackはerror扱いで即0点**
（平均でも累積でもなく1回でも超過でアウト）。サーバー起動（モデルロード含む）は既定120秒。
提出前に必ず`validate_submission.py`のスモークテストで確認する。

### 評価環境・採点環境
- 本番の採点環境: **単一のNVIDIA L4 GPU（VRAM 24GB）**、Track評価に1時間以上かかるとタイムアウト
- 開発・検証は配布リポジトリの`setup.sh`で本番と同一環境を再現できる（Python 3.10, git, unzip必要、
  初回10〜20分。Dockerでの再現も可能で既存環境への影響を避けたい場合はこちらを推奨）
- リーダーボード提出は1日1回まで（採点完了時点でカウント消費、10〜20分程度かかる）

### 評価指標・スコア式（`pipeline/total_score.py`より）
成功/衝突判定をゲートとして、重み付け正規化した滑らかさ指標の積でスコアを計算：
```
smooth_metrics = w1*time + w2*jerk + w3*SPARC + w4*trajectory + w5*rotation
Total Score = (1/N) * Σ success_i * (1 - collision_penalty_i) * smooth_metrics_i
```
重み(w1〜w5、合計1)と各指標の正規化式は非公開。予選ではN=1（本選ではTrack毎の総Trial数）。

**成功判定**: タスクのゴール条件を満たし、かつ衝突なし。衝突は「操作対象(BDDLの`:obj_of_interest`)
以外の物体の変位が1mmを超えたら失敗」という明確な基準（対象物を掴んで動かすのは当然OK）。

### タスク
配布されているのはexampleタスク5件のみ（`compe/t1/T1_TASKS.csv`）。
**本番のTrack1採点は非公開の別タスクセットで実施される**（配布キットの試行回数既定は1タスクあたり20）。

### 禁止事項（要注意）
タスク固有のハードコード全般（外部プランナー、有限状態機械、行動テーブル、成功条件/報酬の直接参照、
評価環境専用のif分岐、非公開タスク識別用fingerprinting、モデルを実質使わないfallback policy）、
zip提出物側の禁止（入れ子圧縮・zip bomb・シンボリックリンク・パストラバーサル・難読化ファイル）、
評価中の外部ネットワークアクセス（ポリシーサーバー⇔評価クライアント間の通信のみ許可）。

### 独自学習の要件
公開モデル構造・事前学習済み重み・tokenizerの利用は可だが、**最終的なAction生成に実質的に寄与する
独自学習要素**が必須。認められる例: LoRA/adapter等の学習済みパラメータ、独自学習したAction
head/decoder。認められない例: ファイル形式変換のみ、モデル名変更のみ、プロンプト/推論パラメータ変更のみ。

## 採用モデル: SmolVLA（公式サンプルに準拠、方針転換）

配布リポジトリの`examples/`に用意されている唯一の公式学習例は
**`lerobot/smolvla_libero_plus`をベースにしたSmolVLAのLoRA追加学習**（`examples/smolvla_libero_spatial_lora.ipynb`）。
**Colab無料T4で数時間で完走すると明記**されている軽量モデル。

これまで検討していたPi0.5（LeRobot経由）はモデルロード時にColab無料枠のRAM(12GB)を超過し、
根本原因の特定にCodex CLIの2段階調査を要した上でRunPodへの環境移行が必要だった
（経緯は`docs/env_setup.md`参照）。公式が最初からSmolVLAという軽量モデルを提示している以上、
**Pi0.5にこだわる理由はなく、保守的戦略（予選突破確実優先）としてもSmolVLAへ切り替える**のが妥当。

- ノートブックの学習条件: LIBERO-plus Spatial 10タスク×各5エピソード(計50)、3,000 steps、
  バッチサイズ1（Colab完走優先の最小構成、伸ばす場合はここが出発点）
- 出力はLeRobot形式の重み。これ単体では提出できないため、`MyPolicy`に組み込んでHTTPサーバー化する作業が別途必要
- ノートブック内評価とTrack1本番採点は条件が異なる（観測解像度256×256 vs 128×128、試行数3 vs 非公開20など）ため、
  ノートブックの成功率は本番スコアの目安にはならない

## 改訂ロードマップ

### 【2026-08-04更新】計算資源の方針: Colab先行、RunPodは保険

Pi0.5(約25億パラメータ)からSmolVLA(約4.5億パラメータ、1/5以下)への変更に伴い、旧戦略にあった
「最初からRunPod」を撤回し、**Colab無料枠を先行させる**方針に変更した。

- Pi0.5でのOOM原因は「モデル生成時にCPU上でfloat32実体を構築する非効率な実装」×「25億パラメータの
  大きさ」の掛け算で顕在化した。SmolVLAなら同じ非効率な実装であってもCPU一時消費は約1.8GB程度で
  Colab無料枠(RAM12GB)に余裕で収まる計算になる
- SmolVLA LoRA学習は公式が「Colab無料T4で数時間の完走」を明言している（`examples/README.md`）
- ステップ1(環境構築・疎通確認)自体も学習を伴わない軽量な作業のため、まずColabで試す
- RunPod(RTX3090等)は、Colab無料枠で解決しないエラーが出た場合・評価パイプラインがT4で重すぎる場合・
  本番環境(L4 GPU)により近い環境で最終検証したい場合の**保険**として温存する
- トレードオフ: RunPodでは直接SSHで実行できたが、Colabでは「ユーザーがセルを実行→出力を共有」という
  往復作業に戻る。SmolVLAは大きなトラブルが起きにくいと見込んでいるが、詰まった場合はこの往復が発生しうる

### ステップ1: 配布リポジトリの環境構築（Colab優先）
`matsuolab/PARC2026_pre`をclone→`bash setup.sh`→`source env.sh`。**まずColab（GPUランタイム）で試す**。
ランダムポリシーのまま`policy_server.py`を起動し、
`python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 2`で疎通確認する
（ここまでは学習不要、配布のまま動く）。うまくいかない場合のみRunPodに切り替える。

### ステップ2: SmolVLA LoRAノートブックをColabで完走
`examples/smolvla_libero_spatial_lora.ipynb`をColab（T4）で実行し、マージ済みモデル一式(zip)を得る。
公式が「Colabで完走する最小構成」と明言しているため、Pi0.5のような環境問題は起きにくい想定。

### ステップ3: `MyPolicy`への組み込み・提出物作成
ステップ2のモデルを`submission_template/policy_server.py`の`MyPolicy.__init__`/`get_action`/`reset`に実装。
観測画像の前処理（128×128想定）・10秒タイムアウト内に収まる推論速度を確認。
`validate_submission.py`で静的検査＋起動スモークテストを実施してからzip化。
この工程もColabでまず試し、うまくいかなければRunPodへ。

### ステップ4: ロバスト性対策（余力があれば）
LIBERO-plusの摂動（カメラ視点・背景・照明等）に強くするaugmentationを1〜2個追加。
ただし本番タスクは非公開のため、過度なチューニングより「まず確実に動いて提出できる」ことを優先する。

### ステップ5: 提出前の最終チェック
`evaluate.py`でzipをエンドツーエンドローカル検証→リーダーボードに1回提出して動作確認→
本提出は余裕を持って（締切ギリギリ厳禁、大容量zipのアップロードにはラグが生じる）。
レポート（学習・推論の工夫点、モデル情報、学習情報、権利関係）も忘れずに提出（提出漏れは即失格）。

## スケジュール感（週5〜10時間換算）

| 時期 | やること |
|---|---|
| 8月上旬(今週) | 配布リポジトリ環境構築（ステップ1）、SmolVLA LoRA学習（ステップ2） |
| 8月中旬(〜8/14) | MyPolicy組み込み・提出物作成・ロバスト性対策（ステップ3〜4） |
| 8/14締切前 | 最終検証・提出（ステップ5） |
| 8/17締切前 | レポート提出 |

## ローカル⇔Colab(Pro)⇔RunPod⇔Codex CLIの役割分担

**【2026-08-04更新】Colab Proへのアップグレードを決定**（月額約1,100円）。
理由: セッション切断（アイドル90分程度で切れる無料枠の制約）が今日の作業でも支障になっており、
Pro化でバックグラウンド実行・長時間セッション・優先GPU割当・ノートブック内蔵Geminiの強化が受けられる。
ただしProでも**本番採点環境と同じL4 GPUが確実に割り当てられる保証はない**ため、RunPodの役割は残す。

- **ローカル(Mac)**: コード編集・構造化・git管理。MuJoCoのGPU描画非対応のため実行環境としては使わない
- **Colab Pro（メイン）**: 配布リポジトリの`setup.sh`実行・`pipeline`での評価・ポリシーサーバー動作確認・
  SmolVLA LoRA学習ノートブック実行。日常的な開発はここで完結させる
- **RunPod（保険・精密検証用）**: Colab(Pro含む)で解決しないエラーが出た場合、または10秒/リクエストの
  タイムアウト制約ギリギリを検証するなど本番環境(L4 GPU確実に指定可能)により近い環境が必要な場合に使う
- **Codex CLI**: 実装のセカンドオピニオンや、Claude Codeと並行させたい作業がある場合に使う

## 参考

- [PARC2026予選配布リポジトリ](https://github.com/matsuolab/PARC2026_pre)（一次情報・最優先で参照）
- `docs/env_setup.md`（本リポジトリ内、Pi0.5/RunPod移行の経緯・検証済み手順）
- [LIBERO論文](https://arxiv.org/abs/2306.03310) / [LIBERO-Plus論文](https://arxiv.org/abs/2510.13626)
- [LeRobot](https://github.com/huggingface/lerobot) / [SmolVLA](https://huggingface.co/lerobot/smolvla_libero_plus)
