#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import html
import re
from pathlib import Path

import pandas as pd


def normalize_text(text, lowercase=False, remove_mentions=False, remove_hashtags=False):
    if pd.isna(text):
        return ""

    s = str(text).strip()

    # Basic HTML unescape
    s = html.unescape(s)

    # Normalize common quote characters
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")

    # Fix some common malformed HTML fragments often seen in Twitter dumps
    s = s.replace("& amp ;", "&")
    s = s.replace("& quot ;", '"')
    s = s.replace("& apos ;", "'")
    s = s.replace("& lt ;", "<")
    s = s.replace("& gt ;", ">")

    # Optional Twitter-specific cleanup
    if remove_mentions:
        s = re.sub(r"@\w+", "", s)

    if remove_hashtags:
        # remove whole hashtag token, keep normal words untouched
        s = re.sub(r"#\w+", "", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    if lowercase:
        s = s.lower()

    return s


def canonicalize(text):
    if pd.isna(text):
        return ""

    s = str(text).strip()
    s = html.unescape(s)

    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")

    s = s.replace("& amp ;", "&")
    s = s.replace("& quot ;", '"')
    s = s.replace("& apos ;", "'")
    s = s.replace("& lt ;", "<")
    s = s.replace("& gt ;", ">")

    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_len(text):
    if not text:
        return 0
    return len(str(text).split())


def read_twitterppdb(path: Path):
    # Typical TwitterPPDB sample is tab-separated, no header, 2 columns
    # source \t target
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["source", "target"],
        encoding="utf-8",
        on_bad_lines="skip",
        quoting=3,   # safer for noisy raw text
    )
    return df


def drop_null_and_empty(df):
    before = len(df)

    source = df["source"].fillna("").astype(str).str.strip()
    target = df["target"].fillna("").astype(str).str.strip()

    keep = (source != "") & (target != "")
    out = df.loc[keep].copy()
    out["source"] = source[keep].values
    out["target"] = target[keep].values

    return out.reset_index(drop=True), before - len(out)


def drop_exact_duplicates(df):
    before = len(df)
    out = df.drop_duplicates(subset=["source", "target"], keep="first").reset_index(drop=True)
    return out, before - len(out)


def drop_canonical_duplicates(df):
    before = len(df)

    key = (
        df["source"].map(canonicalize)
        + "\t"
        + df["target"].map(canonicalize)
    )

    out = df.loc[~key.duplicated(keep="first")].reset_index(drop=True)
    return out, before - len(out)


def drop_identity_pairs(df):
    before = len(df)

    same = df["source"].map(canonicalize) == df["target"].map(canonicalize)
    out = df.loc[~same].reset_index(drop=True)

    return out, before - len(out)


def filter_length_ratio(df, max_ratio):
    before = len(df)

    def keep_row(row):
        a = token_len(row["source"])
        b = token_len(row["target"])

        if a == 0 or b == 0:
            return False

        return max(a, b) / min(a, b) <= max_ratio

    mask = df.apply(keep_row, axis=1)
    out = df.loc[mask].reset_index(drop=True)

    return out, before - len(out)


def normalize_frame(df, lowercase=False, remove_mentions=False, remove_hashtags=False):
    df = df.copy()
    df["source"] = df["source"].map(
        lambda x: normalize_text(
            x,
            lowercase=lowercase,
            remove_mentions=remove_mentions,
            remove_hashtags=remove_hashtags,
        )
    )
    df["target"] = df["target"].map(
        lambda x: normalize_text(
            x,
            lowercase=lowercase,
            remove_mentions=remove_mentions,
            remove_hashtags=remove_hashtags,
        )
    )
    return df


def limit_per_source(df, max_per_source, seed):
    if max_per_source is None or max_per_source <= 0:
        return df.reset_index(drop=True)

    # Use canonical source for grouping so tiny formatting differences
    # do not create separate buckets.
    tmp = df.copy()
    tmp["_source_key"] = tmp["source"].map(canonicalize)

    out = (
        tmp.groupby("_source_key", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), max_per_source), random_state=seed))
        .reset_index(drop=True)
        .drop(columns=["_source_key"], errors="ignore")
    )

    return out.reset_index(drop=True)


def print_stats(title, n_before, n_after, removed_dict):
    print(f"[{title}] input={n_before}, output={n_after}")
    print(
        "  removed: "
        f"null_empty={removed_dict['null_empty']}, "
        f"exact_dup={removed_dict['exact_dup']}, "
        f"canonical_dup={removed_dict['canonical_dup']}, "
        f"identity={removed_dict['identity']}, "
        f"length_ratio={removed_dict['length_ratio']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess TwitterPPDB raw text file into standardized CSV."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("raw_data/twitterppdb.txt"),
        help="Path to raw TwitterPPDB .txt file",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("processed/eval/twitterppdb.csv"),
        help="Path to output CSV",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase source and target text",
    )
    parser.add_argument(
        "--remove_mentions",
        action="store_true",
        help="Remove @mentions",
    )
    parser.add_argument(
        "--remove_hashtags",
        action="store_true",
        help="Remove #hashtags",
    )
    parser.add_argument(
        "--max_length_ratio",
        type=float,
        default=3.0,
        help="Maximum allowed token length ratio",
    )
    parser.add_argument(
        "--max_per_source",
        type=int,
        default=5,
        help="Maximum number of target paraphrases kept per source; <=0 means keep all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    input_path = args.input_path.resolve()
    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_path}")
    df = read_twitterppdb(input_path)
    n_input = len(df)

    removed = {
        "null_empty": 0,
        "exact_dup": 0,
        "canonical_dup": 0,
        "identity": 0,
        "length_ratio": 0,
    }

    df, removed["null_empty"] = drop_null_and_empty(df)
    df, removed["exact_dup"] = drop_exact_duplicates(df)
    df, removed["canonical_dup"] = drop_canonical_duplicates(df)
    df, removed["identity"] = drop_identity_pairs(df)
    df, removed["length_ratio"] = filter_length_ratio(df, args.max_length_ratio)

    df = normalize_frame(
        df,
        lowercase=args.lowercase,
        remove_mentions=args.remove_mentions,
        remove_hashtags=args.remove_hashtags,
    )

    # Normalize may create new empties
    df, extra_removed = drop_null_and_empty(df)
    removed["null_empty"] += extra_removed

    # Optional source balancing
    before_limit = len(df)
    df = limit_per_source(df, args.max_per_source, args.seed)
    source_limit_removed = before_limit - len(df)

    # Add label for compatibility with your unified schema
    df["label"] = 1

    # Final schema
    df = df[["source", "target", "label"]].reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8")

    print_stats("twitterppdb", n_input, len(df), removed)
    print(f"  removed_by_source_cap={source_limit_removed}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()