"""Create and save separate train/val/test datasets with stratified sampling"""

import argparse
import logging
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from paragen.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_stratified_splits(
    all_pairs,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    Create stratified train/val/test splits
    
    Args:
        all_pairs: List of (source, target) tuples
        train_ratio: Proportion for training (default 0.8 = 80%)
        val_ratio: Proportion for validation (default 0.2 = 20%)
        test_ratio: Proportion for testing (default 0.0 = no test set)
        seed: Random seed for reproducibility
    
    Returns:
        train_pairs, val_pairs, test_pairs (test_pairs is empty if test_ratio=0)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    # Compute length bins for stratification
    length_bins = []
    for source, target in all_pairs:
        source_len = len(source.split())
        # Bin by length quartiles
        if source_len < 10:
            length_bins.append(0)
        elif source_len < 20:
            length_bins.append(1)
        elif source_len < 30:
            length_bins.append(2)
        else:
            length_bins.append(3)
    
    # If no test set needed, skip first split
    if test_ratio == 0.0:
        # Adjust val ratio relative to train+val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_pairs, val_pairs = train_test_split(
            all_pairs,
            test_size=val_ratio_adjusted,
            random_state=seed,
            stratify=length_bins,
        )
        test_pairs = []
    else:
        # Split 1: Separate test set
        temp_pairs, test_pairs = train_test_split(
            all_pairs,
            test_size=test_ratio,
            random_state=seed,
            stratify=length_bins,
        )
        
        # Update length bins for remaining data
        temp_length_bins = [
            length_bins[all_pairs.index(p)] for p in temp_pairs
        ]
        
        # Split 2: Separate val from train
        # Adjust val_ratio relative to remaining data
        val_ratio_adjusted = val_ratio / (1 - test_ratio)
        train_pairs, val_pairs = train_test_split(
            temp_pairs,
            test_size=val_ratio_adjusted,
            random_state=seed,
            stratify=temp_length_bins,
        )
    
    return train_pairs, val_pairs, test_pairs


def save_pairs_to_file(pairs, file_path):
    """Save (source, target) pairs to JSON file"""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    
    data = [
        {"source": source, "target": target}
        for source, target in pairs
    ]
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(pairs)} pairs to {file_path}")


def save_pairs_to_txt(pairs, source_file, target_file):
    """Save sources and targets to separate text files (one per line)"""
    os.makedirs(os.path.dirname(source_file) or ".", exist_ok=True)
    
    with open(source_file, "w", encoding="utf-8") as f:
        for source, _ in pairs:
            f.write(source + "\n")
    
    with open(target_file, "w", encoding="utf-8") as f:
        for _, target in pairs:
            f.write(target + "\n")
    
    logger.info(f"Saved {len(pairs)} sources to {source_file}")
    logger.info(f"Saved {len(pairs)} targets to {target_file}")


def create_datasets(
    csv_file=None,
    output_dir="./data/splits",
    train_ratio=0.8,
    val_ratio=0.2,
    test_ratio=0.0,
):
    """
    Create and save train/val/test datasets from local CSV file
    
    Args:
        csv_file: Path to CSV file (QQP format with question1, question2, is_duplicate columns)
        output_dir: Directory to save splits
        train_ratio: Training proportion
        val_ratio: Validation proportion
        test_ratio: Testing proportion
    """
    config = get_config()
    
    # Use provided CSV or default to QQP
    if csv_file is None:
        csv_file = "./quora-question-pairs/train.csv/train.csv"
    
    # Load data from CSV
    logger.info(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} rows from CSV")
    
    # Filter for semantic duplicates only (is_duplicate == 1)
    # Handle different column name possibilities
    if 'is_duplicate' in df.columns:
        duplicate_col = 'is_duplicate'
    elif 'label' in df.columns:
        duplicate_col = 'label'
    else:
        duplicate_col = None
    
    # Handle different question column names
    if 'question1' in df.columns and 'question2' in df.columns:
        q1_col, q2_col = 'question1', 'question2'
    elif 'sent1' in df.columns and 'sent2' in df.columns:
        q1_col, q2_col = 'sent1', 'sent2'
    elif 'sentence1' in df.columns and 'sentence2' in df.columns:
        q1_col, q2_col = 'sentence1', 'sentence2'
    else:
        # Use first two columns as source and target
        q1_col, q2_col = df.columns[0], df.columns[1]
    
    logger.info(f"Using columns: '{q1_col}' and '{q2_col}'")
    
    # Filter for duplicates if label column exists
    if duplicate_col and duplicate_col in df.columns:
        df = df[df[duplicate_col] == 1]
        logger.info(f"Filtered to {len(df)} duplicate pairs")
    
    # Create pairs list
    all_pairs = []
    for idx, row in df.iterrows():
        q1 = row[q1_col]
        q2 = row[q2_col]
        
        # Skip if either is NaN
        if pd.isna(q1) or pd.isna(q2):
            continue
        
        q1 = str(q1).strip()
        q2 = str(q2).strip()
        
        # Skip empty pairs
        if q1 and q2:
            all_pairs.append((q1, q2))
    
    logger.info(f"Total valid pairs: {len(all_pairs)}")
    
    # Create stratified splits
    logger.info(f"\nCreating stratified splits:")
    logger.info(f"  Train: {train_ratio*100:.0f}%")
    logger.info(f"  Val: {val_ratio*100:.0f}%")
    if test_ratio > 0:
        logger.info(f"  Test: {test_ratio*100:.0f}%")
    
    train_pairs, val_pairs, test_pairs = create_stratified_splits(
        all_pairs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=42,
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    logger.info("\nSaving as JSON files...")
    save_pairs_to_file(train_pairs, os.path.join(output_dir, "train.json"))
    save_pairs_to_file(val_pairs, os.path.join(output_dir, "val.json"))
    if test_pairs:
        save_pairs_to_file(test_pairs, os.path.join(output_dir, "test.json"))
    
    # Save as separate text files (sources and targets)
    logger.info("\nSaving as text files...")
    save_pairs_to_txt(
        train_pairs,
        os.path.join(output_dir, "train_sources.txt"),
        os.path.join(output_dir, "train_targets.txt"),
    )
    save_pairs_to_txt(
        val_pairs,
        os.path.join(output_dir, "val_sources.txt"),
        os.path.join(output_dir, "val_targets.txt"),
    )
    if test_pairs:
        save_pairs_to_txt(
            test_pairs,
            os.path.join(output_dir, "test_sources.txt"),
            os.path.join(output_dir, "test_targets.txt"),
        )
    
    # Print summary
    print("\n" + "=" * 80)
    print("DATASET CREATION SUMMARY")
    print("=" * 80)
    print(f"Input file: {csv_file}")
    print(f"Output directory: {output_dir}")
    print(f"\nDataset splits (stratified by source length):")
    print(f"  Train: {len(train_pairs):,} pairs ({len(train_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"  Val:   {len(val_pairs):,} pairs ({len(val_pairs)/len(all_pairs)*100:.1f}%)")
    if test_pairs:
        print(f"  Test:  {len(test_pairs):,} pairs ({len(test_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"  Total: {len(all_pairs):,} pairs")
    print("\nFiles created:")
    print(f"  - {output_dir}/train.json")
    print(f"  - {output_dir}/train_sources.txt")
    print(f"  - {output_dir}/train_targets.txt")
    print(f"  - {output_dir}/val.json")
    print(f"  - {output_dir}/val_sources.txt")
    print(f"  - {output_dir}/val_targets.txt")
    if test_pairs:
        print(f"  - {output_dir}/test.json")
        print(f"  - {output_dir}/test_sources.txt")
        print(f"  - {output_dir}/test_targets.txt")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and save stratified train/val/test datasets from CSV"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="./quora-question-pairs/train.csv/train.csv",
        help="Path to CSV file (default: ./quora-question-pairs/train.csv/train.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/splits",
        help="Directory to save dataset splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training proportion (default 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation proportion (default 0.2)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Testing proportion (default 0.0)",
    )

    args = parser.parse_args()

    create_datasets(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
