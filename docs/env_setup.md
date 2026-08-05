# 環境構築手順

**【2026-08-04】現行の環境構築手順**: 公式配布リポジトリ`matsuolab/PARC2026_pre`の`setup.sh`を使う。
Colabでの手順は`notebooks/parc2026_pre_setup.ipynb`、詳細は`README.md`・`docs/strategy.md`を参照。
以下（本ファイルの残り全体）は**Pi0.5(LeRobot直接eval)ルート時代の記録**であり、現在は不採用の方針だが、
同種のトラブル（OOM・Colab利用規約違反等）の参考として残してある。

## 現行ルートのつまずきポイント一覧

| 症状 | 原因 | 対策 |
|---|---|---|
| `venv`作成失敗（`python3.10-venv`関連エラー） | Colabの基盤イメージに`python3.10`本体は入っているが`python3.10-venv`パッケージが欠けているケースがあり、ノートブックの旧セルは`which python3.10`の成否だけで分岐していたためインストール自体がスキップされていた（2026-08-04実際に発生・修正済み） | `notebooks/parc2026_pre_setup.ipynb`のセルを修正し、`python3.10 -m venv`の実動作を検査してから常にapt installを実行する形に変更。既に壊れた`venv/`ディレクトリが残っている場合は`setup.sh`実行前に削除する（`setup.sh`は既存venvディレクトリがあれば再作成しない仕様のため） |
| `setup.sh`の最終検証ステップ(`5/5 libero 設定`→`動作確認`)で`ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend`が出て失敗 | Colab/JupyterのMPLBACKEND環境変数(ノートブック内で図を表示するための特殊なバックエンド)が、`setup.sh`が起動するサブプロセスにそのまま引き継がれ、venv内のmatplotlibがこれを認識できない（2026-08-04実際に発生・複数回再発） | `bash setup.sh`の直前に`export MPLBACKEND=Agg`を必ず実行する。`notebooks/parc2026_pre_setup.ipynb`のsetup.shセルには反映済みだが、**チャット上でアドホックにコマンドを再現すると、この修正を含め忘れるリスクがある**。再開手順を案内する際はチャットで手打ちせず、必ずメンテナンス済みのノートブック(`notebooks/parc2026_pre_setup.ipynb`)の該当セルをそのままコピーする |
| 同じMPLBACKENDエラーが、setup.sh以外の複数箇所（疎通確認・SmolVLA疎通確認・提出物検証・カメラ診断の各`%%bash`セル）でも個別に再発した | 上記の対策を`bash setup.sh`セルにだけ`export`していたため、それ以外の`%%bash`セル(それぞれ独立したサブプロセス起動なので`export`は引き継がれない)には効いていなかった。1箇所ずつ対症療法的に直していたため、直していない箇所で毎回同じ症状が再発していた（2026-08-05実際に発生） | 個別セルへの`export`の書き足しをやめ、**セクション0.5でPythonカーネル自体の`os.environ['MPLBACKEND']='Agg'`を一度だけ設定**する方式に変更。Colabの`%%bash`・`!command`・`subprocess.Popen`はいずれもこのPythonカーネルのos.environを継承するため、以降の全セルに自動的に効く。**教訓**: 同じ症状が複数箇所で再発したら、個別に潰すのではなく「継承元を1箇所直せば済まないか」を先に疑う |
| `policy_server.py`をバックグラウンド起動しても`curl http://localhost:8000/health`が「接続拒否」、ログファイルも「存在しない」 | Colabの`%%bash --bg`マジックコマンドが不安定で、バックグラウンドプロセスが実際には起動処理に入れていないケースがあった（2026-08-04実際に発生・修正済み） | `%%bash --bg`をやめ、Pythonの`subprocess.Popen(..., stdout=log_file, stderr=subprocess.STDOUT)`で直接起動する方式に変更。`proc.poll()`でプロセスの生死を確認しながら待つことで、ログファイル未作成の問題自体を回避できる |
| （未発生・予防的メモ）ヘッドレスGPUレンダリング(EGL)関連のエラー | ドライバとGLライブラリのバージョンが完全一致していないと発生しうる（[Zenn記事(inrjin氏)](https://zenn.dev/inrjin/articles/437a359e3ffcd7)より、2026-08-04調査で確認） | ドライバ・GLライブラリのバージョンを揃える。現行の`setup.sh`は`~/.libero/config.yaml`を自動生成するため、同記事にあった「config.yaml未生成→import時に対話式質問でEOFエラー」自体は基本的に解消済みのはず |
| （未発生・予防的メモ）推論サーバ起動後にOOM | 推論サーバ(JAX等)のGPUメモリ先取りと、LIBERO-Plusのレンダリングのメモリ競合が原因になりうる（同Zenn記事より） | `MEM_FRACTION`等でメモリ確保量を調整する、または推論とレンダリングでGPUを分離する |
| Omnicampus実採点で`struct.no_policy_server`エラー、public/private score=0.0 | `policy_server.py`・`requirements.txt`を含まない、モデル重みのみのzip（学習ノートブックが自動生成する`*_merged.zip`）を誤って提出した（2026-08-04実際に発生） | 提出物は必ず`policy_server.py`＋`requirements.txt`＋`model_weights/`の3点を正しい構造でzip化したものにする。**提出前に必ずローカルの`validate_submission.py`でPASSを確認してから**Omnicampusにアップロードする（1日1回の提出制限を無駄にしないため） |
| `validate_submission.py`で「外部ソース（スキーム 'git+https:'）の指定は禁止です」エラー | `requirements.txt`に`lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git@v0.6.0`のようなgit直接参照を書いていた（2026-08-04実際に発生） | PyPIに公開されている同一バージョンを使う。今回は`lerobot==0.6.0`がgitタグv0.6.0と同一内容でPyPI公開されており`smolvla`extraも含まれていたため、`lerobot[smolvla]==0.6.0`に変更して解決 |
| ランタイムが生きたまま同一セッション内でセルを再実行しただけでも、venv構築・pip install・LIBERO-Plusアセット583枚のDLをフルにやり直し、10〜20分かかる | セル9(`bash setup.sh`実行セル)が`rm -rf venv`を無条件実行していた。元々は「壊れている(activateが無い)場合だけ削除」という安全な条件分岐だったが、無関係な別修正(バックグラウンド起動方式の修正)のコミットに紛れる形で無条件削除に退行していた(2026-08-05、git考古学で原因コミット特定・critic Opusレビューで確認)。公式`setup.sh`自体はvenv/LIBERO-plus/LIBERO/アセットが既にあれば再構築しない設計(idempotent)なので、本来2回目以降は数秒〜数十秒で終わるはずだった | `if [ -d venv ] && [ ! -f venv/bin/activate ]; then rm -rf venv; fi`という条件分岐に戻す(修正済み)。**教訓**: エラー対応で安易に`rm -rf`を書き足さない。「壊れている」の判定は`activate`の有無など機械的な基準にする |
| （不採用として却下・実装前に判明）venv・リポジトリをGoogle Drive上に置いて永続化する案 | Google DriveのFUSEマウントは実行可能ファイルへのexec権限を付与しない既知の制限があり(`googlecolab/colabtools` issue #3162, #4495)、`venv/bin/python`を直接execする今の書き方ではDrive配下に置くと`Permission denied`で確実に壊れる(2026-08-05、critic Opusレビューで実装前に発見) | **venv・LIBERO-plus・LIBEROは常に`/content`ローカルに置く**。Driveを使う場合は「zipとしてDriveに保存→新規ランタイムで`/content`に展開」というモデル重み復元セル(セクション6-7)と同じパターンに限定する。上記のセル9修正だけで、最も頻発する「同一セッション内の再実行」はほぼ解消するため、Drive移行自体は現時点では見送り |
| （未発生・予防的メモ）GPU版torchを手動導入した後、`setup.sh`が再実行されると気づかぬうちにCPU版へ巻き戻る | `setup.sh`は無条件で`torch==2.11.0+cpu`をインストールする仕様。`lerobot==0.6.0`の制約(`torch<2.12.0,>=2.7`)をCPU版でも満たすため、pipはエラーを出さずに静かにGPU版をCPU版へ書き換えてしまう(2026-08-05、critic Opusレビューで発見)。10秒/ステップのタイムアウト制約に対し、CPU推論への劣化は致命傷になりうる | セクション9の`torch.cuda.is_available()`確認を、`print`(見逃し可能)から`assert`(即失敗)に格上げ済み。Falseなら`AssertionError`で即座に気づける |
| 【重大・提出前に発見】`config.vlm_model_name`をローカルに向けても、`make_pre_post_processors()`が構築する前処理パイプラインの`tokenizer_processor`ステップが別途Hubの`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`を参照しており、ネットワーク遮断下では`OSError: We couldn't connect to 'https://huggingface.co'...`で起動失敗する | `model_dir/policy_preprocessor.json`内の`tokenizer_processor`ステップの`tokenizer_name`フィールドが、`config.vlm_model_name`とは独立してHub文字列のままハードコードされていた。GPU実機(Colab)を待たず、**ローカルMac(CPU)で`HF_HUB_OFFLINE=1`を設定して`uv run --with "lerobot[smolvla]==0.6.0"`で実際にロード・推論まで再現し、Colabの疎通確認を待たずに発見**(2026-08-05)。もし気づかず提出していたら、8/4と同種の失敗(0.0点)で提出1回を無駄にしていた | `make_pre_post_processors(config, pretrained_path=model_dir, preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": <vlm_local_dirの絶対パス>}})`で上書き。修正後、ローカルでHF_HUB_OFFLINE=1下でのロード→前処理構築→推論まで完走を確認済み(`policy_server_smolvla_full.py`に反映済み)。**教訓**: 同名の設定(モデル名)が複数箇所に独立して重複している場合、1箇所直しただけで「直った」と判断しない |
| 【最重大・本番ブロッカー】`lerobot[smolvla]==0.6.0`はPyPI上`Requires-Python>=3.12`であり、本番採点環境(公式README確定: Python 3.10.12)には`pip install`自体ができない | 学習に使った公式サンプルノートブックはColabの素のPython(3.12)上でlerobotを動かしており、この非互換は今まで表面化していなかった(2026-08-05発覚)。lerobot 0.5.0以降が3.12必須になっているのに対し、Python 3.10対応は0.4.4が最後で、かつ0.4.4は0.6.0で保存された`PolicyProcessorPipeline`形式のモデルを読めない(API構造自体が違う)ため、バージョンダウンでは解決しない | **lerobot 0.6.0のソースをPython 3.10互換にパッチしてリポジトリに同梱(vendoring)**する方針にした。詳細は`src/parc2026/vendor/NOTICE_PARC2026_MODIFICATIONS.md`参照。critic Opusレビュー(2件、GO→REVISE)を経て実装。**教訓**: 学習環境と評価環境でPythonバージョンが違う場合、学習側が動いても評価側で同じライブラリが動く保証はない。今後新しいライブラリを使う前に`pip index versions <pkg>`でrequires-pythonを機械的に確認する |
| PEP695構文(`class Foo[T]:`・`type X = ...`、Python 3.12専用)が`ast.parse`では検出できるが、`typing.Self`/`typing.Unpack`(Python 3.11で追加)はimportしてみるまで検出できない | grepやast.parseによる静的チェックだけでは「3.12専用構文4ファイル」しか見つからず安全と誤認しかけた。critic Opusレビューで「実際にimportして確認していない残り7ファイル」を指摘され、`lerobot.policies.pretrained/factory/smolvla/pi0/pi05/evo1/pi0_fast`・`lerobot.configs.video`が同種の問題を持つと判明(2026-08-05) | 該当7ファイル+video.pyの計8ファイルで`from typing import ... Self/Unpack`を`from typing_extensions import Self, Unpack`に直接書き換え(実行順序に依存する動的shimではなく、importの参照先自体を変える方式に変更)。**教訓**: 「構文チェックで問題なし」は「実行して問題なし」の代わりにならない |
| vendoringしたlerobotのrequirements.txtを、動作確認した環境の`pip freeze`そのままで作ったところ、`networkx==3.6.1`がPython 3.11以上必須でinstall不可 | `pip freeze`を取得した検証環境ではtorchも同時にpipインストールしており、その組み合わせで解決されたnetworkxが3.6.1だった。今回はtorchを意図的にrequirements.txtから外す(本番プリインストール版を使うため)設計のため、その検証環境の値をそのまま持ち込むと矛盾が生じた(2026-08-05、クリーンルームvenvでの`pip install`実行時に発覚) | `networkx`はバージョン固定を外し、pipに解決を委ねる形に変更。**教訓**: 「実際に動いた組み合わせのpip freeze」は原則正しいが、torch関連のようにrequirements.txtから意図的に除外するパッケージがある場合、その除外パッケージに引きずられて決まったバージョンまで持ち込まないよう個別に確認する |

## lerobotベンダリングの検証状況(2026-08-05時点)

- クリーンルームテスト(新規Python 3.10 venv、`pip install -r requirements.txt`→実際の`policy_server.py`を`HF_HUB_OFFLINE=1`で起動→`/health`→`/reset`→`/act`まで全て実施)で**PASS**を確認済み(Macのため`device_processor`のみ検証時にcpuへ一時的に上書き、本番コード自体は変更していない)
- **未検証**: 本番相当のGPU実機(Colab/RunPod)での動作。特に「採点環境のプリインストールtorch(2.11.0+cu130)が、requirements.txtの`torchvision==0.26.0`インストール時に巻き込まれて変わらないか」はMac上では原理的に検証できない(Macにはそもそも比較対象のtorchがシステムに無いため)。次にColabで反映した際、**`torch.cuda.is_available()`のassertチェック(セクション9)で必ず確認する**
- critic Opusレビューが指摘した「本番専用の隔離venv作成ロジック(`sandbox.py`)は非公開」という点は未解消。手元のクリーンルーム検証は近似でしかない

---

# 【過去の記録】Pi0.5(LeRobot直接eval)ルートの環境構築手順（検証済みレシピベース）

出典: [「初めてのフィジカルAI〜PARC入門編 π0.5 × LIBERO / LIBERO-Plus〜」(note, taku_sid氏)](https://note.com/taku_sid/n/n49a0008b29a6)
PARC2026参加者による実践記事。**Macでは動かない（NVIDIA GPU + Linuxが必須）**と明記されているため、
実行はColab（GPUランタイム）または他のLinux GPUマシンで行う。ローカルMacはコード編集専用。

## 前提

- Python **3.12以上必須**（3.10でエラーになったと報告あり）
- NVIDIA GPU + Linux（Mac不可）→ Colabならこの条件を満たす
- HuggingFaceアカウント（PaliGemmaはgated repoのため利用申請が必要）

## 0. アカウント設定（Colab実行前に1回だけ）

1. **HuggingFaceアカウント**: https://huggingface.co/join （未作成なら）
2. **PaliGemmaの利用申請**: https://huggingface.co/google/paligemma-3b-pt-224 を開き、
   利用規約に同意して簡単なフォームに記入。Googleのgated modelは即時承認されることが多い
3. **HFトークン発行**: https://huggingface.co/settings/tokens → Create new token →
   Token type = Read → 名前は `colab-parc2026` 等 → `hf_...` をコピー（再表示不可）
4. **GitHub PAT発行**: https://github.com/settings/personal-access-tokens → Generate new token →
   Repository access = `parc2026` のみ選択 → Permissions = Contents: Read-only
   （Colabはclone/pullのみで書き込みしないため）→ `github_pat_...` をコピー
5. **Colab Secretsに登録**: Colabの左サイドバー鍵アイコン →
   `HF_TOKEN`（手順3の値）と `GH_TOKEN`（手順4の値）を追加し、両方「ノートブックからのアクセス」をON
6. **ノートブックを開く**: リポジトリがprivateなのでGitHub連携より簡単な方法として、
   ローカルの `notebooks/colab_bootstrap.ipynb` をColabの「ファイル」→「ノートブックをアップロード」で開く。
   「ランタイム」→「ランタイムのタイプを変更」でGPU(T4等)を選択してから上から順に実行する

## 手順

### 1. venv作成 + LeRobotのclone

```bash
uv venv .venv --python 3.12
source .venv/bin/activate

git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

### 2. PyTorch（CUDA 12.4向け）

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

`nvcc --version` と `torch.version.cuda` が食い違うとビルド失敗するので要確認。

### 3. LeRobot本体（pi・libero extras込み）

```bash
uv pip install -e ".[pi,libero]"
```

### 4. HuggingFace認証（PaliGemmaのgated repo申請後）

```bash
hf auth login
```

事前にHuggingFace公式サイトでPaliGemmaの利用規約に同意し、トークンを発行しておく。

### 5. LIBERO（無印）で先に評価を完走させる

素のLIBEROとLIBERO-Plusは**同時運用不可**（LIBERO-Plusをインストールすると素のLIBEROが置き換わる）。
必ず**先に無印LIBEROの評価を終わらせてから**LIBERO-Plusに進むこと。

### 6. LIBERO-Plusへの切り替え

```bash
git clone https://github.com/sylvestf/LIBERO-plus.git
cd LIBERO-plus
uv pip install --no-deps -e .
uv pip install robosuite bddl easydict mujoco wand scikit-image gym

# 追加アセット(オブジェクト・テクスチャ)は別配布。Sylvest/LIBERO-plusはHF上ではdatasetリポジトリなので
# --repo-type dataset が必須(付けないとmodelとして探されて「Repository not found」になる)
hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset --local-dir ./tmp_assets
unzip -q ./tmp_assets/assets.zip -d ./libero/libero   # 公式指定の配置先
```

### 7. レンダリング設定（評価実行前に毎回必須）

```bash
export MUJOCO_GL=egl
```

GUIなしのサーバー（Colab含む）でGPU描画を行うための設定。これがないと評価が動かない。

### 8. キャッシュの永続化（Colab特有の注意）

Colabはセッションが切れるとローカルディスクが揮発するため、モデルの再ダウンロードを防ぐには
HuggingFaceのキャッシュをGoogle Drive上に向ける。

```bash
export HF_HOME=/content/drive/MyDrive/PARC2026/hf_cache
```

## つまずきポイント一覧

| 症状 | 原因 | 対策 |
|---|---|---|
| CUDAビルド失敗 | PyTorchとCUDAのバージョン不一致 | `nvcc --version` と `torch.version.cuda` を揃える |
| モデル再DL地獄 | Colabのキャッシュ揮発 | `HF_HOME` をDrive配下に設定 |
| LIBERO-Plusでアセット不足エラー | assets.zip未展開 | `hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset` の後 `./libero/libero` に展開 |
| `hf download`で"Repository not found" | `Sylvest/LIBERO-plus`はdatasetリポジトリなのに`--repo-type dataset`を付けずmodelとして探しに行っていた（2026-07-29実際に発生） | `--repo-type dataset` を必ず付ける |
| flash-attentionのビルドが長い | 依存ビルドが重い | `uv pip install -e ".[pi,libero]"` 実行中に自動処理されるので待つ |
| LIBEROの評価結果がおかしい | LIBERO-Plus導入で無印LIBEROが上書きされた | 無印LIBEROの評価を先に完了させてからLIBERO-Plusへ |

## 【重要・禁止事項】ColabでSSHサーバーを起動しない

`colab_ssh`等でColab上にSSHサーバー(cloudflaredトンネル等)を立てて外部から直接操作する方法は、
**Googleの無料枠の利用規約で明示的に禁止されている**（SSH/RDP/VNC等のリモートデスクトップ
プロトコルの実行は不可）。2026-07-29に実際にこれが原因と思われる「許可されていないコードを実行した」
というランタイム強制切断が発生した。直接操作したい場合はColabではなく、最初からSSH前提で
作られているRunPod等のオンデマンドGPUレンタルを使うこと（`docs/strategy.md`参照）。

## Colabでの実行について

上記記事の著者はLinux GPUマシンで検証しており、Colab実績の明記はない（未確認）。
ただしColabもLinux + NVIDIA GPUなので要件自体は満たすはず。
`notebooks/colab_bootstrap.ipynb` 実行時に一つでも上記の手順で詰まったら、その内容と
エラーメッセージを控えておくこと（対策の記事化・PARC運営への質問にも使える）。
