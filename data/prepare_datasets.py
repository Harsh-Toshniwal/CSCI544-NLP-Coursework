#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def ensure_dir(root: Path):
    root.mkdir(parents=True, exist_ok=True)


def standardize_rows(df, source_col, target_col, label_col):
    out = pd.DataFrame({
        "source": df[source_col].astype(str),
        "target": df[target_col].astype(str),
        "label": df[label_col].astype(int),
    })

    # optional basic cleanup: remove rows with missing text after astype(str)
    out["source"] = out["source"].str.strip()
    out["target"] = out["target"].str.strip()

    return out


def save_dataset(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def load_qqp():
    from datasets import load_dataset

    ds = load_dataset("glue", "qqp")

    frames = []
    for split in ["train", "validation", "test"]:
        df = ds[split].to_pandas()

        # glue/qqp test split may not have labels in some setups
        if "label" not in df.columns:
            continue

        frames.append(df)

    if not frames:
        raise ValueError("QQP loaded successfully, but no labeled splits were found.")

    return pd.concat(frames, ignore_index=True)


def load_paws():
    from datasets import load_dataset

    ds = load_dataset("paws-x", "en")

    frames = []
    for split in ["train", "validation", "test"]:
        df = ds[split].to_pandas()
        if "label" not in df.columns:
            continue
        frames.append(df)

    if not frames:
        raise ValueError("PAWS-X loaded successfully, but no labeled splits were found.")

    return pd.concat(frames, ignore_index=True)


def load_mrpc():
    from datasets import load_dataset

    ds = load_dataset("glue", "mrpc")

    frames = []
    for split in ["train", "validation", "test"]:
        df = ds[split].to_pandas()
        if "label" not in df.columns:
            continue
        frames.append(df)

    if not frames:
        raise ValueError("MRPC loaded successfully, but no labeled splits were found.")

    return pd.concat(frames, ignore_index=True)



def prepare_qqp(output_dir: Path, counts: dict):
    qqp = load_qqp()
    qqp_std = standardize_rows(qqp, "question1", "question2", "label")
    out_path = output_dir / "qqp.csv"
    save_dataset(qqp_std, out_path)
    counts["qqp.csv"] = len(qqp_std)


def prepare_paws(output_dir: Path, counts: dict):
    paws = load_paws()
    paws_std = standardize_rows(paws, "sentence1", "sentence2", "label")
    out_path = output_dir / "paws.csv"
    save_dataset(paws_std, out_path)
    counts["paws.csv"] = len(paws_std)


def prepare_mrpc(output_dir: Path, counts: dict):
    mrpc = load_mrpc()
    mrpc_std = standardize_rows(mrpc, "sentence1", "sentence2", "label")
    out_path = output_dir / "mrpc.csv"
    save_dataset(mrpc_std, out_path)
    counts["mrpc.csv"] = len(mrpc_std)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare paraphrase datasets and save standardized CSV files."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data")
    )

    parser.add_argument(
        "--skip_qqp",
        action="store_true",
        help="Skip preparing QQP."
    )
    parser.add_argument(
        "--skip_paws",
        action="store_true",
        help="Skip preparing PAWS-X."
    )
    parser.add_argument(
        "--skip_mrpc",
        action="store_true",
        help="Skip preparing MRPC."
    )

    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    counts = {}

    print(f"Output root: {output_dir}")

    if not args.skip_qqp:
        prepare_qqp(output_dir, counts)

    if not args.skip_paws:
        prepare_paws(output_dir, counts)

    if not args.skip_mrpc:
        prepare_mrpc(output_dir, counts)


    print("\nSaved files:")
    for name, n in counts.items():
        print(f"{name}: {n} rows")


if __name__ == "__main__":
    main()