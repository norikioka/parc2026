# 環境構築手順（検証済みレシピベース）

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
