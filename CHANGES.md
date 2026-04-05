# 変更履歴

元リポジトリ（[seonghyeonye/Knowledge-Entropy](https://github.com/seonghyeonye/Knowledge-Entropy)）からフォークし、現在の環境で動作するように加えた変更の一覧。

---

## 1. 環境構築の変更

### 1.1 conda → venv への移行

| ファイル | 変更内容 |
|---|---|
| `README.md` | インストール手順を `conda` から `python3 -m venv` に書き換え |

**理由**: 実行環境に conda が未インストールだったため、標準ライブラリの `venv` に統一。

### 1.2 依存パッケージのバージョン更新

| ファイル | 変更内容 |
|---|---|
| `pyproject.toml` | `torch>=2.1,<2.3` → `torch>=2.4,<2.5`、`numpy` → `numpy<2` |

**理由**: CUDA 12.4 + A100/H100環境に対応するため PyTorch 2.4 に更新。numpy 2.x は OLMo コードとの後方互換性に問題があるため `<2` で制約。

### 1.3 flash-attn のインストール

**理由**: 元コードは `flash_attention: false` で AMD GPU 向けに無効化されていた。A100/H100/B200 では flash attention が利用可能なので有効化。プリビルトwheel（v2.8.3）を使用し、複数 GPU アーキテクチャ（sm_80, sm_90, sm_100）に対応。

---

## 2. チェックポイントの取得方法の変更

### 2.1 モデルチェックポイント (`scripts/get_model.sh`)

| 変更前 | 変更後 |
|---|---|
| `olmo-checkpoints.org` から `model.pt` + `config.yaml` + `train.pt` をダウンロード | HuggingFace Hub (`allenai/OLMo-1B`) から `model.safetensors` をダウンロード |

**理由**: `olmo-checkpoints.org` が閉鎖されており 404 を返す。HuggingFace Hub にはHF標準形式の `model.safetensors` のみが公開されている。OLMo 独自形式（base64 エンコードキー、3ファイル構成）のチェックポイントはもう公開されていない。

### 2.2 訓練データ順序 (`scripts/get_dataorder.sh`)

| 変更前 | 変更後 |
|---|---|
| `olmo-checkpoints.org` から `global_indices.npy` をダウンロード | ローカルで `numpy.arange` を使い生成 |

**理由**: 同じく URL が 404。OLMo-1B は `data_shuffling: false` で訓練されているため、インデックスは単純な連番で再現可能。

---

## 3. チェックポイント読み込みコードの修正

### 3.1 HuggingFace 標準 safetensors 形式への対応 (`olmo/checkpoint.py`)

`load_state_dict()` 関数に以下の変更を加えた：

1. **safetensors の二方言対応**: OLMo 独自形式（base64 エンコードキー）のデコードに失敗した場合、HF 標準形式（平文文字列キー）にフォールバック
2. **`model.` プレフィックスの除去**: HF 形式のキーは `model.transformer.blocks.0...` だが、OLMo コードは `transformer.blocks.0...` を期待するため
3. **戻り型 `-> Dict[str, Any]` の明示**: 型推論の曖昧さを解消し、下流の checkpointer クラスで発生していた linter エラーを修正

### 3.2 trainer state の不在への対応 (`olmo/checkpoint.py`, `olmo/train.py`)

`FullCheckpointer.restore_checkpoint()` に `load_trainer_state` 引数を追加：

- `reset_trainer_state: true`（= 新規学習を開始する）場合に `train.pt` / `other.pt` を読まず空 dict を返す
- ファイルが存在しない場合も空 dict にフォールバック（HF Hub のチェックポイントには `train.pt` が含まれないため）

`olmo/train.py` の `restore_unsharded_checkpoint()` を修正し、`load_trainer_state` を `FullCheckpointer` に伝播するようにした。

**理由**: HF Hub のチェックポイントにはモデル重みのみが含まれ、optimizer state (`optim.pt`) や trainer state (`train.pt`) は存在しない。config で `reset_optimizer_state: true` / `reset_trainer_state: true` が設定されている場合、これらのファイルは不要だが、元コードは無条件に読み込もうとしていた。

### 3.3 Resuscitation パラメータ変更スクリプトの修正 (`analysis/change_parameters.py`)

| 変更前 | 変更後 |
|---|---|
| `torch.load(f"{olmo_model_path}/model.pt")` | `olmo.checkpoint.load_state_dict(olmo_model_path, "model.pt", map_location="cpu")` |

**理由**: 元コードは `torch.load` で OLMo 独自形式の `model.pt` を直接読み込んでいたが、HF Hub からダウンロードしたチェックポイントは `model.safetensors`（HF 標準形式）であるため、セクション 3.1 で修正した `load_state_dict()` を経由して読み込むように変更。

---

## 4. 評価データセットの互換性修正

### 4.1 piqa データセット (`olmo/eval/downstream.py`)

`PIQA` クラスの `__init__` を修正し、`ybisk/piqa` の Parquet 変換ブランチ (`refs/convert/parquet`) を使用するようにした。

**理由**: `datasets` ライブラリ v3.0 以降でカスタムスクリプトベースのデータセットが非サポートになった。`ybisk/piqa` は HF Hub 上でまだスクリプト形式のままだが、自動変換された Parquet ブランチが利用可能。他の評価データセット（hellaswag, winogrande 等）は既に Parquet 対応済みで問題なし。

### 4.2 `trust_remote_code` 警告の除去 (`olmo/eval/downstream.py`)

`load_dataset()` の呼び出しから `trust_remote_code=True` を削除。

**理由**: `datasets` v4.x では `trust_remote_code` 引数自体が非サポートとなり、渡すと stderr に大量の警告が出力される。実際のデータ読み込みには影響しないため削除。

---

## 5. flash-attn API の互換性修正 (`olmo/train.py`)

`cross_entropy_loss` の `ignored_index` 引数が flash-attn v2.5.8 で `ignore_index` にリネームされた問題に対応。バージョンを検出して適切なキーワード引数を使い分けるようにした。

---

## 6. 学習スクリプトの SLURM 対応

### 6.1 `scripts/train.sh`

| 変更前 | 変更後 |
|---|---|
| `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ...` | SLURM ジョブスクリプト（`ybatch` / `sbatch` 対応） |

主な内容：
- `#YBATCH -r a100_8` でリソース指定
- `SLURM_GPUS_ON_NODE` から GPU 数を自動検出
- config パスを引数で指定可能（デフォルトは baseline config）
- `module load cuda/12.4` で CUDA モジュールをロード

### 6.2 `scripts/prepare_pubmed.sh`（新規作成）

PubMed データ準備用の SLURM ジョブスクリプト。CPU ノード（`epyc-7502`）で実行。

---

## 7. 学習設定の変更

### 7.1 `configs/1B/1B_bs128_lr4e4_pubmed_1ep_738k.yaml`

| 項目 | 変更前 | 変更後 | 理由 |
|---|---|---|---|
| `flash_attention` | `false` | `true` | A100/H100 で flash attention を使用 |
| `device_train_microbatch_size` | `16` | `16` | 8GPU × 16 = 128 = global_train_batch_size |
| `dataset_path` | `path_to_dataset` | `data/pubmed_tokenized` | 実際のデータパスを設定 |

### 7.2 `configs/resuscitation/...yaml`

baseline config と同様の変更（`flash_attention: true`、`dataset_path: data/pubmed_tokenized`）。`device_train_microbatch_size` は A100 4台に合わせて `16` に設定（セクション 10.3 参照）。

---

## 8. データ準備スクリプト（新規作成）

### 8.1 `scripts/prepare_pubmed.py`

NCBI FTP から PubMed Baseline の XML ファイルをダウンロードし、abstract を抽出・トークナイズ・1024トークンのチャンクに分割して HuggingFace Dataset 形式で保存する。

**背景**: 論文の実験では PubMed で約210M トークン（204,800 チャンク × 1024 トークン）の continual learning を行う。`datasets.load_dataset('ncbi/pubmed')` はスクリプト非サポートにより動作しないため、直接 FTP からダウンロードする方式で実装。

### 8.2 `scripts/prepare_dolma_sample.py`

Dolma v1.6-sample から最初の 2048 ドキュメントを取得し、`data/dolma_1B/first_1k.json` として保存する。Knowledge Entropy 計算で使用。

---

## 9. Knowledge Entropy 計算の修正

### 9.1 `analysis/entropy.py` — `ExpOlmoForCausalLM` から標準モデル + hook への移行

| 変更前 | 変更後 |
|---|---|
| カスタム `ExpOlmoForCausalLM`（`analysis/modeling_olmo_hf.py`）を使用し、`outputs.activations` で MLP 中間活性を取得 | 標準 `AutoModelForCausalLM`（transformers 内蔵 `OlmoForCausalLM`）を使用し、`down_proj` への forward hook で MLP 中間活性を取得 |

**理由**: `modeling_olmo_hf.py` は旧バージョンの transformers（v4.3x）から LLaMA のコードをコピーして作られたカスタムモデルで、transformers v5.5.0 との間に複数の非互換が発生していた：

1. **`config.rope_theta` 属性の不在** — OLMo-1B-hf の config ではトップレベルに `rope_theta` が存在せず `AttributeError`（`getattr` でデフォルト値 10000.0 に修正）
2. **`_tied_weights_keys` の型変更** — transformers v5 で `list` → `dict` に変更され `AttributeError`（`{"lm_head.weight": "model.embed_tokens.weight"}` に修正）
3. **`DynamicCache.from_legacy_cache()` の削除** — transformers v5 でメソッドが廃止され `AttributeError`（`DynamicCache()` に修正）
4. **`DynamicCache.to_legacy_cache()` の削除** — 同上（キャッシュをそのまま返すように修正）
5. **`rope_scaling` 辞書形式の不一致** — `_init_rope()` が LLaMA の `{"type": ..., "factor": ...}` 形式を前提としているが、OLMo-1B-hf の config は `{"rope_theta": ..., "rope_type": "default"}` 形式で `KeyError`（`rope_type: "default"` をハンドリングするよう修正）
6. **レイヤー 1 以降で `nan` が伝播** — 上記修正をすべて適用してもカスタム attention 実装に起因する数値不安定が残り、MLP activations が `nan` になる問題が解消されなかった

根本原因は `modeling_olmo_hf.py` が LLaMA の実装をコピーして `Llama` → `Olmo` にリネームした構成であり、OLMo 固有の config 形式や transformers v5 の内部 API 変更に追従できていない点にある。個別パッチではなく、標準 `OlmoForCausalLM`（transformers 組み込み実装で `nan` なし）に切り替え、MLP 中間活性は `down_proj` の入力をフックで捕捉する方式とした。フックで取得される tensor は `act_fn(gate_proj(x)) * up_proj(x)` に相当し、元の `ExpOlmoForCausalLM` の `activations` と同一の値。

### 9.2 `analysis/modeling_olmo_hf.py` — 互換性修正（参考）

`entropy.py` では使用しなくなったが、他の用途に備えて以下の互換性修正を適用済み（ただし LLaMA コピー由来の構造的問題は残る）：

| 行 | 修正内容 |
|---|---|
| L289 | `config.rope_theta` → `getattr(config, "rope_theta", 10000.0)` |
| L304-315 | `_init_rope()` で `rope_scaling` の OLMo 固有形式を処理 |
| L706-710 | `DynamicCache.from_legacy_cache()` → `DynamicCache()` |
| L784-786 | `to_legacy_cache()` 呼び出しを削除 |
| L880 | `_tied_weights_keys` を `list` → `dict` に変更 |

---

## 10. Resuscitation 実験パイプラインの追加

### 10.1 `scripts/entropy.sh`（新規作成）

Knowledge Entropy 計算（`python -m analysis.entropy`）の SLURM ジョブスクリプト。A100 1台で実行。引数 `step`、`data_size`、`batch_size` を受け付ける。

### 10.2 `scripts/resuscitation.sh`（新規作成）

Resuscitation 実験の全パイプラインを一括実行する SLURM ジョブスクリプト。A100 4台で実行。3ステップを順に実行：

1. **Knowledge Entropy 計算** — `mlp_average_coefficients.pt` が存在しなければ `analysis.entropy` を実行
2. **パラメータ変更** — `resuscitation_ratio*.pt` が存在しなければ `analysis.change_parameters` を実行
3. **学習** — resuscitation config で `torchrun` を実行

`set -e` でエラー時に即停止するようにした（前回は Step 1 失敗後に Step 2/3 が空ファイルで進行してしまった）。

### 10.3 `configs/resuscitation/...yaml` — microbatch size の修正

| 項目 | 変更前 | 変更後 | 理由 |
|---|---|---|---|
| `device_train_microbatch_size` | `32` | `16` | A100 4台で OOM 発生（cross_entropy 計算で 6.13 GiB の確保に失敗）。gradient accumulation 2 ステップに分割 |
