#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLS = ["source", "target", "label"]


# =========================
# TEXT NORMALIZATION
# =========================
def normalize_text(text, lowercase=False):
    if pd.isna(text):
        return ""

    s = str(text).strip()
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s)

    if lowercase:
        s = s.lower()

    return s.strip()


def canonicalize(text):
    if pd.isna(text):
        return ""

    s = str(text).strip().lower()
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def token_len(s):
    return len(str(s).split())


# =========================
# CLEANING
# =========================
def clean_df(df, args):
    df = df.copy()

    # drop empty
    df = df[
        (df["source"].astype(str).str.strip() != "") &
        (df["target"].astype(str).str.strip() != "")
    ]

    # exact dup
    df = df.drop_duplicates(["source", "target"])

    # canonical dup
    key = df["source"].map(canonicalize) + "\t" + df["target"].map(canonicalize)
    df = df.loc[~key.duplicated()]

    # identity
    df = df[
        df["source"].map(canonicalize) != df["target"].map(canonicalize)
    ]

    # length ratio
    def ok(row):
        a = token_len(row["source"])
        b = token_len(row["target"])
        if a == 0 or b == 0:
            return False
        return max(a, b) / min(a, b) <= args.max_length_ratio

    df = df[df.apply(ok, axis=1)]

    # normalize
    df["source"] = df["source"].map(lambda x: normalize_text(x, args.lowercase))
    df["target"] = df["target"].map(lambda x: normalize_text(x, args.lowercase))

    return df.reset_index(drop=True)


# =========================
# SPLIT
# =========================
def split_3way(df, seed):
    train, temp = train_test_split(df, test_size=0.2, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)
    return train, val, test


# =========================
# MAIN PIPELINE
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--output_dir", type=Path, default=Path("processed"))
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--max_length_ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "eval").mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # =====================
    # LOAD
    # =====================
    qqp = pd.read_csv(input_dir / "qqp.csv")
    paws = pd.read_csv(input_dir / "paws.csv")
    mrpc = pd.read_csv(input_dir / "mrpc.csv")

    twitter_path = input_dir / "twitterppdb.csv"
    twitter = pd.read_csv(twitter_path) if twitter_path.exists() else None

    # =====================
    # TRAIN DATA
    # =====================
    qqp_pos = qqp[qqp["label"] == 1]
    paws_pos = paws[paws["label"] == 1]

    train_df = pd.concat([qqp_pos, paws_pos], ignore_index=True)

    train_df = clean_df(train_df, args)

    train, val, test = split_3way(train_df, args.seed)

    train.to_csv(output_dir / "train/train.csv", index=False)
    val.to_csv(output_dir / "train/val.csv", index=False)
    test.to_csv(output_dir / "train/test.csv", index=False)

    print(f"[train] total={len(train_df)}")

    # =====================
    # QQP HELDOUT (in-domain test)
    # =====================
    qqp_clean = clean_df(qqp_pos, args)
    _, _, qqp_test = split_3way(qqp_clean, args.seed)

    qqp_test.to_csv(output_dir / "eval/qqp_heldout.csv", index=False)

    # =====================
    # EVAL SETS
    # =====================
    mrpc_clean = clean_df(mrpc, args)
    mrpc_clean.to_csv(output_dir / "eval/mrpc.csv", index=False)

    if twitter is not None:
        twitter_clean = clean_df(twitter, args)
        twitter_clean.to_csv(output_dir / "eval/twitterppdb.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()