# 進捗チェックリスト

`docs/strategy.md` のロードマップに対応。日付は更新のたびに追記する。

## ステップ0: プロジェクトセットアップ
- [x] ローカル環境(Python 3.12/uv) — 2026-07-28
- [x] GitHubリポジトリ作成・push（private, https://github.com/norikioka/parc2026） — 2026-07-28
- [x] Colab連携ノートブック(`notebooks/colab_bootstrap.ipynb`)作成 — 2026-07-28
- [x] 戦略・環境構築ドキュメント作成 — 2026-07-28

## 準備作業（ローカルで完結、Colab不要）
- [x] スコアリング4指標(成功率/滑らかさ[jerk・SPARC・EEF回転]/実行効率[ステップ数・軌道距離]/安全性[衝突判定])の
      自己評価用ユーティリティ`src/parc2026/scoring.py`実装、ユニットテスト7件・lint通過 — 2026-07-28
      （LIBERO評価ログの実データ形式が判明次第、`summarize_episode`の入出力を調整する）

## ステップ1: 環境構築を完走させる（最優先）
- [ ] HuggingFaceでPaliGemmaの利用規約に同意・トークン発行 **（ユーザー手作業）**
- [ ] GitHub Personal Access Token(repo scope)発行 **（ユーザー手作業）**
- [ ] Colab SecretsにHF_TOKEN・GH_TOKENを登録 **（ユーザー手作業）**
- [ ] `colab_bootstrap.ipynb` をColabで最後まで実行し、無印LIBEROの公式サンプルタスクが動くことを確認 **（ユーザー実行、詰まったら相談）**
- [ ] LIBERO-Plus導入・アセットダウンロード確認

## ステップ2: 公開チェックポイントで推論再現
- [ ] `moojink/openvla-7b-oft-finetuned-libero-*` 等をロードして評価スクリプトを実行
- [ ] 結果をベースライン(OpenVLA-OFT: 元97.1 / LIBERO-Plus総合69.6〜79.6程度)と比較

## ステップ3: 自力LoRAファインチューニング
- [ ] OpenVLA-OFT公式finetune設定をLoRA+小バッチ+QLoRAで1回完走
- [ ] VRAM不足時はNoraへの切り替えを検討

## ステップ4: ロバスト性対策（1〜2個に絞る）
- [ ] カメラ視点・色調のaugmentation実装
- [ ] 言語指示パラフレーズでの自己点検
- [ ] （余力があれば）domain randomization

## 提出
- [ ] LIBERO-Plus評価スクリプトで自己採点、ベースライン比で大きく劣らないことを確認してから予選提出

---

**現在のブロッカー**: ステップ1の最初の3項目はユーザーによるHuggingFace/GitHubでの
手続き（アカウント操作・トークン発行）が必要で、Claude Codeからは代行できない。
ここが終わり次第、Colab上での実行結果（成功/エラーメッセージ）を共有してもらえれば、
デバッグや`docs/env_setup.md`の更新を継続する。
