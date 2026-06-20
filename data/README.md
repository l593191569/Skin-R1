# data/

Stage 0 (**data_construction**) storage. Contents are git-ignored except this README and `.gitkeep` placeholders — you populate locally.

> [!NOTE]
> **Not the same as** [`model_training/data/`](../model_training/data/README.md) (`SKIN_R1_DATA_ROOT`), which holds bridged SkinRationale (`trajectory_v2/`), RL, and evaluation inputs.

## Layout

```
data/
├── pdfs/         # Input textbook PDF(s) — you provide (copyright not included)
├── outputs/      # One sub-directory per run:
│                 #   bbc_<timestamp>/            — after pre-cluster script
│                 #   bbc_continue_<timestamp>/     — after post-cluster re-filter
│                 #   clustering, matched CSV, ddx_graph_merged.json,
│                 #   refined_data.csv, matched images, …
│                 #   After each run: source outputs/.env_run → sets RUN_DIR
└── sft_dataset/  # SkinRationale: train_type{1..5}.jsonl
                  # Images remain under $RUN_DIR/pdf_outputs.matched_image_paths_dir/
```

## Usage

Default root for all `data_construction/` scripts:

```bash
export SKIN_R1_DATA_DIR=/path/to/data   # default: <repo>/data
```

After a PDF or continue run:

```bash
source data/outputs/.env_run            # export RUN_DIR=...
```

When Stage 0 finishes, bridge SkinRationale into the training layout:

```bash
bash model_training/scripts/prepare_sft_from_construction.sh
# requires RUN_DIR + data/sft_dataset/train_type*.jsonl (SkinRationale)
```

See [`data_construction/README.md`](../data_construction/README.md) for the full Stage 0 pipeline.

<p align="right"><sub><a href="../README.md">← Back to main README</a></sub></p>
