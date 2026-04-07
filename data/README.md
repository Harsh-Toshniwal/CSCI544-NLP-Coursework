# Paraphrase datasets: preparation and preprocessing

This repository contains a small pipeline to **download standard paraphrase benchmarks from Hugging Face**, **convert a raw TwitterPPDB (Language-Net) text dump into CSV**, and **clean / split** data for training and evaluation.

---

## Requirements

```bash
pip install pandas datasets scikit-learn
```

- **`prepare_datasets.py`** needs network access to pull [Hugging Face Datasets](https://huggingface.co/datasets).
- **`preprocess_datasets.py`** needs **`scikit-learn`** for stratified random splits.

---

## Pipeline overview

| Step | Script | Role |
|------|--------|------|
| 1 | `prepare_datasets.py` | Download **QQP**, **PAWS-X (en)**, and **MRPC** from Hugging Face; write unified CSVs (`source`, `target`, `label`). |
| 2 | `TwitterQQDP_preprocess.py` | **TwitterPPDB only:** turn a raw **tab-separated `.txt`** file into `twitterppdb.csv` (see [TwitterPPDB / Language-Net](#twitterppdb--language-net-manual-download)). |
| 3 | `preprocess_datasets.py` | Load the CSVs from `raw_data/`, apply cleaning, build **train/val/test** for modeling, and **eval** files (MRPC, QQP held-out, optional TwitterPPDB). |

---

## 1. `prepare_datasets.py`

Downloads GLUE **QQP**, **PAWS-X English**, and GLUE **MRPC**, standardizes columns, and concatenates all available labeled splits per dataset into one table each.

### Output (default: `data/`)

| File | Contents |
|------|----------|
| `qqp.csv` | QQP pairs (`question1` → `source`, `question2` → `target`) |
| `paws.csv` | PAWS-X en pairs |
| `mrpc.csv` | MRPC pairs |

### Usage

```bash
python prepare_datasets.py --output_dir data
```

Skip individual datasets if needed:

```bash
python prepare_datasets.py --output_dir data --skip_qqp
python prepare_datasets.py --output_dir data --skip_paws
python prepare_datasets.py --output_dir data --skip_mrpc
```

**Note:** TwitterPPDB is **not** fetched here. It is handled separately (see below).

---

## 2. TwitterPPDB / Language-Net (manual download)

**TwitterPPDB** (also referred to as **Language-Net** in the literature) is a Twitter paraphrase corpus. In this project:

- There is **no Hugging Face dataset loader** wired into `prepare_datasets.py` for this corpus.
- The **official** distribution page for the corpus is the [Language-Net / Twitter URL Corpus](https://lanwuwei.github.io/Twitter-URL-Corpus/) project site. The **official download / request workflow has been unreliable**: for example, the historical Heroku request form and similar links are often **broken or inactive**, so automated or one-click downloads may fail.
- **Practical workaround:** obtain the raw corpus as a **plain text file** (typically **tab-separated**, two columns: source sentence, target sentence) using the **Dropbox link shared by a project contributor** (or your course staff). Place that file on disk (e.g. `raw_data/twitterppdb.txt`) and convert it with `TwitterQQDP_preprocess.py`.

Respect the original **license** (e.g. non-commercial use where applicable) and **Twitter / X terms** when using the data.

---

## 3. `TwitterQQDP_preprocess.py`

Converts the **raw TwitterPPDB `.txt`** (no header; tab-separated `source` and `target`) into a CSV with columns `source`, `target`, `label` (all labels set to `1` for paraphrase compatibility).

### Defaults

| Argument | Default |
|----------|---------|
| `--input_path` | `raw_data/twitterppdb.txt` |
| `--output_path` | `processed/eval/twitterppdb.csv` |

For use with **`preprocess_datasets.py`**, point the output to **`raw_data/twitterppdb.csv`** so the main preprocessor can read it together with QQP / PAWS / MRPC:

```bash
python TwitterQQDP_preprocess.py ^
  --input_path raw_data/twitterppdb.txt ^
  --output_path raw_data/twitterppdb.csv
```

Optional flags (see `--help`): `--lowercase`, `--remove_mentions`, `--remove_hashtags`, `--max_length_ratio`, `--max_per_source`, `--seed`.

---

## 4. `preprocess_datasets.py`

Expects **input CSVs** under `--input_dir` (default: `raw_data/`):

| File | Required |
|------|----------|
| `qqp.csv` | Yes |
| `paws.csv` | Yes |
| `mrpc.csv` | Yes |
| `twitterppdb.csv` | No; if present, an additional eval file is written |

**Prepare `raw_data/`** by copying or symlinking the outputs of `prepare_datasets.py`, e.g.:

```bash
mkdir raw_data
copy data\qqp.csv raw_data\
copy data\paws.csv raw_data\
copy data\mrpc.csv raw_data\
```

If you generated TwitterPPDB with the step above, ensure `raw_data/twitterppdb.csv` exists before running.

### Processing behavior (summary)

- **Training pool:** QQP rows with `label == 1` and PAWS rows with `label == 1`, concatenated, cleaned, then split **80% / 10% / 10%** (`train` / `val` / `test`) via two-stage `train_test_split` with `--seed`.
- **QQP held-out:** cleaned QQP positives only, same 3-way split; the **test** portion is written as in-domain held-out evaluation.
- **MRPC:** full cleaned file written to eval (no split inside this script).
- **TwitterPPDB:** optional; cleaned and written to eval if `twitterppdb.csv` is present.

### Usage

```bash
python preprocess_datasets.py --input_dir raw_data --output_dir processed --seed 42
python preprocess_datasets.py --input_dir raw_data --output_dir processed --lowercase --max_length_ratio 3.0
```

### Output (default: `processed/`)

| Path | Description |
|------|-------------|
| `processed/train/train.csv` | Training split |
| `processed/train/val.csv` | Validation split |
| `processed/train/test.csv` | Test split (from QQP+PAWS positives) |
| `processed/eval/qqp_heldout.csv` | In-domain held-out (QQP positives, test fraction of 3-way split) |
| `processed/eval/mrpc.csv` | Cleaned MRPC |
| `processed/eval/twitterppdb.csv` | Only if `raw_data/twitterppdb.csv` was provided |

---

## End-to-end example

```bash
# 1) HF datasets → CSV bundle
python prepare_datasets.py --output_dir data

# 2) Lay out inputs for preprocessing (Windows example)
mkdir raw_data
copy data\qqp.csv raw_data\
copy data\paws.csv raw_data\
copy data\mrpc.csv raw_data\

# 3) TwitterPPDB: place contributor-provided twitterppdb.txt, then:
python TwitterQQDP_preprocess.py --input_path raw_data/twitterppdb.txt --output_path raw_data/twitterppdb.csv

# 4) Unified preprocessing
python preprocess_datasets.py --input_dir raw_data --output_dir processed
```

If you skip TwitterPPDB, omit steps 3 and any copy of `twitterppdb.csv`; `preprocess_datasets.py` will still run and only skip the Twitter eval output.

---

## File reference

| File | Purpose |
|------|---------|
| `prepare_datasets.py` | HF → `qqp.csv`, `paws.csv`, `mrpc.csv` |
| `TwitterQQDP_preprocess.py` | Raw TwitterPPDB `.txt` → `twitterppdb.csv` |
| `preprocess_datasets.py` | `raw_data/*.csv` → cleaned `processed/train/*` and `processed/eval/*` |
