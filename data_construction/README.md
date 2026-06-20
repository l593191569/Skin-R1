# Data Construction (Stage 0)

Turn dermatology textbook **PDFs** into **SkinRationale** — hierarchy-aware and DDx **reasoning trajectories** for SFT. Image–text pairing uses vendored [MONET](https://github.com/suinleelab/MONET) (`MONET-main/`).

| | |
| --- | --- |
| **Outputs** | `data/sft_dataset/train_type{1..5}.jsonl` + images under `$RUN_DIR` |
| **Next step** | Bridge to [`model_training/data/`](../model_training/data/README.md) via `prepare_sft_from_construction.sh` |

## Setup

```bash
cd data_construction
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
```

Place your PDF at `../data/pdfs/<PDF_NAME>.pdf` (default name set by `PDF_NAME` env var; see root README).

Storage: [`../data/`](../data/README.md) via `SKIN_R1_DATA_DIR`.

## Scripts

| Script | Role |
| --- | --- |
| **`run_stage0_pre_cluster.sh`** | PDF → extract → featurize → cluster → **stop** for manual cluster review |
| **`run_stage0_post_cluster.sh`** | After cluster choice: re-filter → match → LLM → DDx graph |
| `run_stage0_post_cluster.sh --finish` | Taxonomy → SkinRationale → `model_training/data/trajectory_v2/` |
| `process_bbc_pdf.sh` | One-shot full MONET run (fixed default cluster labels) |
| `process_bbc_pdf_continue.sh` | Re-filter + match only |
| `run_llm_diagnosis.sh` | LLM diagnosis-rule rephrase |
| `run_llm_ddx.sh` | LLM DDx extraction → graph → synonym merge |
| `run_filter.sh` | **Interactive** image curation → `refined_data.csv` |
| `run_taxonomy_generate.sh` | Disease taxonomy tree |
| `generate_sft_samples.sh` | Synthesize SkinRationale (`train_type*.jsonl`) |

`RUN_DIR` is written to `../data/outputs/.env_run` after PDF / continue runs:

```bash
source ../data/outputs/.env_run
```

## Recommended workflow

```bash
cd data_construction

# 1. Automated through clustering
bash scripts/run_stage0_pre_cluster.sh

# 2. Manual: inspect cluster labels
less $RUN_DIR/pdf_outputs.clustering/kmeans_label_lower.csv

# 3. Re-filter with your chosen exclude labels
bash scripts/run_stage0_post_cluster.sh bbc_<timestamp> "02_01 02_03 01 00 03"

# 4. Manual: interactive image curation
bash scripts/run_filter.sh          # menu: 1 → curate in browser → 2

# 5. Finish Stage 0 + bridge SkinRationale to model_training
bash scripts/run_stage0_post_cluster.sh --finish
```

Two manual steps: **cluster label selection** and **image curation**.

## Config files (included)

| File | Purpose |
| --- | --- |
| `pdf_files.config.json` | MONET image–text matching parameters |
| `table_of_contents_part20.json` | Chapter / page ranges for taxonomy |
| `synonym_and_subtype2.json` | **Single taxonomy source** — synonyms, subtypes, benign/malignant buckets |
| `excluded_images.json` | Default excluded image IDs for the reference textbook run |

Edit `synonym_and_subtype2.json` when adapting to a new textbook; a copy is placed under `model_training/data/` by `prepare_sft_from_construction.sh`.

## Cross-module dependencies

- `training_sample_generator.py` → `model_training/src/report_RL_diagnosis_hits.py` (override with `SKIN_R1_MODEL_TRAINING_SRC` if needed).
- `model_training/src/organize_eval_datasets.py` → imports `TrainingSampleGenerator` when building the `ddx` eval set.

## Requirements

- GPU recommended for MONET featurization (downloads CLIP + EfficientNet).
- `OPENAI_API_KEY` for LLM rephrase, DDx extraction, and SkinRationale types 2–5.
- Use a **separate venv** from `model_training/` (`transformers==4.39.1` here vs `4.52.3` for Qwen training).
- PDF layout / MONET config strongly affects extraction quality — see [root README Important notes](../README.md#important-notes).
- `taxonomy_generate.py` default page offset is tuned for the reference textbook; use `--page-offset` for other sources.

<p align="right"><sub><a href="../README.md">← Back to main README</a></sub></p>
