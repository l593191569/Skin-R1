# Model Training & Evaluation

Fine-tune **Qwen2.5-VL-7B-Instruct** with LoRA:

1. **SFT** — SkinRationale ([`scripts/run_sft.sh`](scripts/run_sft.sh))
2. **GRPO** — hierarchy-aware RL on sparse public datasets ([`scripts/run_rl.sh`](scripts/run_rl.sh))
3. **Eval** — dermatology benchmarks ([`scripts/run_eval.sh`](scripts/run_eval.sh))

> [!NOTE]
> **Prerequisites:** SkinRationale from [`../data_construction/`](../data_construction/README.md). Public datasets and checkpoints are not included.

## Setup

```bash
cd model_training
pip install -r requirements.txt   # separate venv from data_construction/
```

Data defaults to [`data/`](data/README.md) (`SKIN_R1_DATA_ROOT`). Override:

```bash
export SKIN_R1_DATA_ROOT=/path/to/training/data
export SKIN_R1_CACHE_DIR=/path/to/hf_cache    # optional
```

## Quick start

```bash
# 1. SkinRationale (after Stage 0)
bash scripts/prepare_sft_from_construction.sh

# 2. SFT
bash scripts/run_sft.sh

# 3. RL data + GRPO (resumes the SFT LoRA adapter — see root README)
cp ../data_construction/synonym_and_subtype2.json data/
# download public datasets → data/RL/  (see below)
bash scripts/build_rl_dataset.sh
SFT_CHECKPOINT=output/SFT_trajectory_<TS>/module_checkpoint/module_epoch_4 \
  bash scripts/run_rl.sh

# 4. Eval
huggingface-cli download foreverbeliever/OmniMedVQA \
  --repo-type dataset --local-dir data/OmniMedVQA
export SKIN_R1_DDX_GRAPH=../data/outputs/<run>/ddx_graph_merged.json
bash scripts/build_eval_datasets.sh
CHECKPOINT_PATH=output/RL_openr1_<TS>/lora_checkpoint/lora_step_1500 \
  bash scripts/run_eval.sh
```

## Data sources

Training scripts **load** prepared files; they do not build them.

| Stage | How to obtain | Location under `SKIN_R1_DATA_ROOT` |
| --- | --- | --- |
| SFT | Stage 0 SkinRationale → `prepare_sft_from_construction.sh` | `trajectory_v2/` + `images/` |
| RL | Download public datasets → `build_rl_dataset.sh` | `RL/` → `RL_dataset_prompt_format_4/` |
| Eval | `build_eval_datasets.sh` | `standardized_datasets/` |
| Taxonomy | copied automatically | `synonym_and_subtype2.json` |

Full directory tree: [`data/README.md`](data/README.md).

### SFT bridge (SkinRationale)

After Stage 0 (with `RUN_DIR` set):

```bash
bash scripts/prepare_sft_from_construction.sh
```

Copies SkinRationale (`../data/sft_dataset/train_type*.jsonl`), links images from `$RUN_DIR/pdf_outputs.matched_image_paths_dir/`, and copies `synonym_and_subtype2.json`.

### RL datasets

Download six public dermatology datasets into `data/RL/`:

```
RL/
├── BCN20000/bcn_20k_train/ + bcn_20k_train.csv
├── derm7pt/images/ + meta/meta.csv + meta/{train,valid,test}_indexes.csv
├── derm12345/derm12345_train_part_1/ + derm12345_train_part_2/
│           + derm12345_metadata_{train,test}.csv
├── dermnet/train/ + dermnet/test/
├── HAM10000/HAM10000_images_part_{1,2}/ + HAM10000_metadata/ + ISIC2018 test
└── PAD-UFES-20/images/ + metadata.csv
```

Then:

```bash
bash scripts/build_rl_dataset.sh
```

Folder/column conventions: `src/data.py` → `preprocess_RL_dataset()`.

### Evaluation benchmarks

```bash
export SKIN_R1_DDX_GRAPH=../data/outputs/<run>/ddx_graph_merged.json
bash scripts/build_eval_datasets.sh
```

| Benchmark | Source | Extra input |
| --- | --- | --- |
| `omnimedvqa` | [OmniMedVQA](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA) | `data/OmniMedVQA/` |
| `indomain` | RL test split | after `build_rl_dataset.sh` |
| `indomain_b_or_m` | RL test split | after `build_rl_dataset.sh` |
| `hierarchical` | RL test split | after `build_rl_dataset.sh` |
| `ddx` | RL test split + DDx graph | `SKIN_R1_DDX_GRAPH` |

Subset: `bash scripts/build_eval_datasets.sh --datasets indomain hierarchical`  
Skip missing: `DATASETS="indomain ddx" bash scripts/run_eval.sh`

## Training details

**SFT** — LoRA r=64, α=32 on SkinRationale (`trajectory_v2/`); output `output/SFT_trajectory_<TS>/module_checkpoint/module_epoch_{N}`. `run_sft.sh` defaults to a smoke run (`--sft_test --max_train_samples 3000`); remove those flags for full training.

**GRPO** — resumes the **same LoRA adapter** from SFT; hierarchy-aware reward (format + option score + benign/malignant). Trainer: `open_r1/trainer/grpo_trainer.py`. Output `output/RL_openr1_<TS>/lora_checkpoint/lora_step_{N}`.

## Important notes

See the [root README](../README.md#important-notes) for PDF/layout sensitivity (Stage 0), LoRA resume SFT→GRPO, **`transformers==4.52.3`** pin, and eval without LoRA merge.

## Scripts

| Script | Entry point |
| --- | --- |
| `scripts/prepare_sft_from_construction.sh` | Bridge SkinRationale → `trajectory_v2/` |
| `scripts/run_sft.sh` | `src/train_sft_trajectory.py` |
| `scripts/build_rl_dataset.sh` | `load_RL_data_raw.py` → `RL_data_construct.py` |
| `scripts/run_rl.sh` | `src/train_rl_grpo.py` |
| `scripts/build_eval_datasets.sh` | `src/organize_eval_datasets.py` |
| `scripts/run_eval.sh` | `src/test_performance.py` |

## Key source files

| File | Role |
| --- | --- |
| `src/data_v2_old.py` | SkinRationale loader (SFT) |
| `src/data_v2.py` | RL verl-format loader |
| `src/data.py` | Shared collator + RL raw dataset preprocessing |
| `src/report_RL_diagnosis_hits.py` | Taxonomy / synonym / b_or_m resolution |
| `src/organize_eval_datasets.py` | Build standardized eval JSON |
| `src/paths.py` | Default `SKIN_R1_DATA_ROOT` |

Base model: `Qwen/Qwen2.5-VL-7B-Instruct`. Paper hardware: 1× A100 40GB (SFT), 2× A100 40GB (GRPO).

<p align="right"><sub><a href="../README.md">← Back to main README</a></sub></p>
