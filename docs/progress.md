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
- [x] HuggingFaceでPaliGemmaの利用規約に同意・トークン発行 — 2026-07-29
- [x] GitHub Personal Access Token(repo scope)発行 — 2026-07-29
- [x] Colab SecretsにHF_TOKEN・GH_TOKENを登録 — 2026-07-29
- [x] `colab_bootstrap.ipynb` をColabで実行、LeRobot(pi/libero extras)インストールまで完了 — 2026-07-29
      （無印LIBEROの単独検証はスキップしてLIBERO-Plus導入まで進めた。純粋な無印LIBEROの数値が
      欲しくなったら別ランタイムで再構築が必要）
- [x] LIBERO-Plus導入・アセットダウンロード — 2026-07-29完了
      （途中`colab_ssh`でのSSHサーバー起動がGoogle Colab利用規約違反でランタイム強制切断される
      トラブルがあり、環境を再構築。以後Colab上でSSH系ツールは使用禁止とした。詳細は`docs/env_setup.md`）
- [x] `colab_bootstrap.ipynb` を最後まで実行、`parc2026` importまで到達 — 2026-07-29

**ステップ1完了**

## ステップ2: 公開チェックポイントで推論再現
- [x] **Colab無料枠(RAM 12GB)では動作不可と判明・RunPodへ移行決定** — 2026-07-29
      Pi0.5(lerobot/pi05_libero_finetuned)のロード時に毎回OOM Kill(exit 137)。
      Codex CLIによる2段階のソースコード調査で確定した原因:
      1. `modeling_pi05.py`の`from_pretrained`は、checkpointをまず`safetensors.load_file()`で
         CPU RAMに全ロードしてから`load_state_dict`する実装（`device="cuda"`指定で軽減可能、パッチ済み）
      2. **より根本的な原因**: モデル自体の生成(`model = cls(config)`)が、PaliGemma(2B)+action
         expert(300M)を通常の`nn.Module`としてCPU上にfloat32の実体で構築してから`.to(device)`する
         実装になっており、ここだけで約10GB前後のCPU RAMを消費する。`accelerate.init_empty_weights()`
         によるmeta device初期化等の改修がLeRobot側に必要で、CLIフラグでの回避策は存在しない
      → 結論: コード改修は保守的戦略(予選突破確実)の範囲外。RAM十分な環境(RunPod等)へ移行
- [ ] RunPodでGPU Pod起動・SSH接続確立（進行中）
- [ ] 以下を(RunPod上で)実行し、公式再現値(平均97.5%)に近い結果が出るか確認:
  ```bash
  lerobot-eval \
    --output_dir=./eval_logs/ \
    --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --policy.path=lerobot/pi05_libero_finetuned \
    --policy.n_action_steps=10 \
    --env.max_parallel_tasks=1
  ```
  （出典: https://huggingface.co/docs/lerobot/en/libero 、チェックポイント: lerobot/pi05_libero_finetuned）
- [ ] これは**無印LIBERO**での検証。LIBERO-Plus(ロバスト性評価)側の評価スクリプトは
  `LIBERO-plus`リポジトリ側にある可能性が高く、まだ確認できていない → 次回調査

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
