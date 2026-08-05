# このディレクトリについて

`lerobot/` は [huggingface/lerobot](https://github.com/huggingface/lerobot) の
v0.6.0 タグ相当のソースコードを、Apache License 2.0 に基づき改変・同梱したものである。

## 同梱した理由

PARC2026予選の本番採点環境は Python 3.10.12（公式README確定）だが、
`lerobot==0.6.0` は PyPI 上 `Requires-Python >= 3.12` であり、Python 3.10 には
`pip install` できない（2026-08-05判明、PyPI releasesメタデータで確認）。
一方、学習済みモデル(`1st/smolvla_libero_plus_spatial_lora_merged/`)は
`lerobot 0.6.0` の `PolicyProcessorPipeline` 形式で保存されており、
Python 3.10 でも動く `lerobot<=0.4.4` はこの形式に対応していない
(モデルの再学習が必要になる可能性が高い)。

そのため、`lerobot 0.6.0` のソースのうち Python 3.12 専用構文・API のみを
Python 3.10 互換に書き換えたものを同梱し、`sys.path` 経由でこちらを
優先的にimportする（`policy_server_smolvla_full.py`参照）。

## 改変内容（Apache License 2.0 第4条(b)に基づく変更告知）

以下12ファイルを変更した。各ファイルの変更箇所には `py3.10互換パッチ` という
コメントを付与している。

### PEP695構文(Python 3.12)を旧来のTypeVar/Generic構文に書き換え(4ファイル)
- `utils/io_utils.py` — `def deserialize_json_into_object[T: JsonLike](...)` → `TypeVar`使用に変更
- `processor/pipeline.py` — `class DataProcessorPipeline[TInput, TOutput](...)` → `Generic[TInput, TOutput]`使用に変更
- `motors/motors_bus.py` — `type NameOrID = ...` / `type Value = ...` → `Union[...]`使用に変更
- `datasets/streaming_dataset.py` — `class Backtrackable[T]:` → `Generic[T]`使用に変更

### typing モジュールの新機能(Python 3.11以降)を typing_extensions からのimportに変更(8ファイル)
Python 3.10の標準`typing`モジュールには存在しない`Self`/`Unpack`を、
`typing_extensions`(本番採点環境にプリインストール済み、公式README付録で確認)から
importするよう変更した。

- `policies/pretrained.py`（`Unpack`）
- `policies/factory.py`（`Unpack`）
- `policies/smolvla/modeling_smolvla.py`（`Unpack`）
- `policies/pi0/modeling_pi0.py`（`Unpack`）
- `policies/pi05/modeling_pi05.py`（`Unpack`）
- `policies/evo1/modeling_evo1.py`（`Unpack`）
- `policies/pi0_fast/modeling_pi0_fast.py`（`Unpack`）
- `configs/video.py`（`Self`）

## 検証状況

- パッケージ全487ファイルを`ast.parse`で構文チェックし、上記以外にPython 3.12専用構文が
  無いことを確認済み(2026-08-05)
- Python 3.10.20・`HF_HUB_OFFLINE=1`(ネットワーク遮断)環境下で、実際の学習済みモデルを
  使い、モデルロード→前処理構築(tokenizer_processor含む)→推論まで完走することを確認済み
  (critic Opusレビューによる独立検証込み、2026-08-05)
- 未検証: 本番相当のGPU実機・複数エピソードでの動作（詳細は`docs/env_setup.md`参照）

## ライセンス

`lerobot/LICENSE` に原文（Apache License 2.0、Copyright 2026 The Hugging Face team）を
同梱している。改変・再配布は同ライセンスの下で許可されている。
