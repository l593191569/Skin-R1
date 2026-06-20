# model_training/data/

Default **`SKIN_R1_DATA_ROOT`** — all inputs for SFT, RL, and evaluation. Git-ignored except this README and `.gitkeep` placeholders.

> [!NOTE]
> Stage 0 artifacts and SkinRationale live under [`../../data/`](../../data/README.md) (`SKIN_R1_DATA_DIR`). Use `prepare_sft_from_construction.sh` to bridge SkinRationale jsonl and images here.

```bash
export SKIN_R1_DATA_ROOT=/path/to/your/data   # default: model_training/data
```

## Layout

```
data/
├── synonym_and_subtype2.json          # from data_construction/ (taxonomy)
├── trajectory_v2/                     # SkinRationale (SFT)
│   ├── images/                        # linked from Stage 0 RUN_DIR
│   └── train_type*.jsonl              # SkinRationale splits
├── RL/                                # raw public datasets (user download)
│   ├── BCN20000/  derm7pt/  derm12345/
│   ├── dermnet/   HAM10000/  PAD-UFES-20/
├── RL_dataset_prompt_format_4/        # built by build_rl_dataset.sh
│   ├── RL_dataset_verl_train.json
│   ├── RL_dataset_verl_valid.json
│   └── RL_dataset_verl_test.json
├── OmniMedVQA/                        # for omnimedvqa eval (HF download)
│   └── OmniMedVQA/Images/ …
└── standardized_datasets/             # built by build_eval_datasets.sh
    ├── omnimedvqa_standardized.json
    ├── indomain_standardized.json
    ├── indomain_b_or_m_standardized.json
    ├── hierarchical_standardized.json
    └── ddx_standardized.json
```

RL construction also writes `RL_dataset_all.json`, `RL_dataset_filtered.json`, and pickle caches under this tree.

## Setup checklist

```bash
# After Stage 0 — bridge SkinRationale (recommended):
cd model_training
source ../data/outputs/.env_run          # sets RUN_DIR
bash scripts/prepare_sft_from_construction.sh

# For RL:
cp ../data_construction/synonym_and_subtype2.json data/
# download datasets → data/RL/
bash scripts/build_rl_dataset.sh

# For eval:
huggingface-cli download foreverbeliever/OmniMedVQA \
  --repo-type dataset --local-dir data/OmniMedVQA
export SKIN_R1_DDX_GRAPH=../data/outputs/<run>/ddx_graph_merged.json
bash scripts/build_eval_datasets.sh
```

Then follow [`../README.md`](../README.md) for SFT / GRPO / eval.

<p align="right"><sub><a href="../../README.md">← Back to main README</a></sub></p>
