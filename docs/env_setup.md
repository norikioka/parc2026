# 環境構築手順（検証済みレシピベース）

出典: [「初めてのフィジカルAI〜PARC入門編 π0.5 × LIBERO / LIBERO-Plus〜」(note, taku_sid氏)](https://note.com/taku_sid/n/n49a0008b29a6)
PARC2026参加者による実践記事。**Macでは動かない（NVIDIA GPU + Linuxが必須）**と明記されているため、
実行はColab（GPUランタイム）または他のLinux GPUマシンで行う。ローカルMacはコード編集専用。

## 前提

- Python **3.12以上必須**（3.10でエラーになったと報告あり）
- NVIDIA GPU + Linux（Mac不可）→ Colabならこの条件を満たす
- HuggingFaceアカウント（PaliGemmaはgated repoのため利用申請が必要）

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

# 追加アセット(オブジェクト・テクスチャ)は別配布
hf download Sylvest/LIBERO-plus assets.zip
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
| LIBERO-Plusでアセット不足エラー | assets.zip未展開 | `hf download Sylvest/LIBERO-plus assets.zip` |
| flash-attentionのビルドが長い | 依存ビルドが重い | `uv pip install -e ".[pi,libero]"` 実行中に自動処理されるので待つ |
| LIBEROの評価結果がおかしい | LIBERO-Plus導入で無印LIBEROが上書きされた | 無印LIBEROの評価を先に完了させてからLIBERO-Plusへ |

## Colabでの実行について

上記記事の著者はLinux GPUマシンで検証しており、Colab実績の明記はない（未確認）。
ただしColabもLinux + NVIDIA GPUなので要件自体は満たすはず。
`notebooks/colab_bootstrap.ipynb` 実行時に一つでも上記の手順で詰まったら、その内容と
エラーメッセージを控えておくこと（対策の記事化・PARC運営への質問にも使える）。
