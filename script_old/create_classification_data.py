"""
Extract classification data from the Quora Dataset for Encoder-only training.

Example Usage via CLI:

# Extract the default 20% of the dataset:
python scripts/create_classification_data.py

# Extract a specific number of records (e.g., 10,000) for faster testing:
python scripts/create_classification_data.py --limit-data 10000

# Extract and save to a custom directory:
python scripts/create_classification_data.py --output-dir data/custom_splits
"""

import os
import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dataset(limit_data=None, output_dir="data/classification_splits"):
    csv_path = "quora-question-pairs/train.csv/train.csv"
    logger.info(f"Loading {csv_path}...")
    
    # Read the data and drop null values
    df = pd.read_csv(csv_path).dropna(subset=['question1', 'question2', 'is_duplicate'])
    
    # Calculate weights to enforce stratified sampling directly from pandas DataFrame
    if limit_data:
        # Use sklearn train_test_split to strictly stratify sample a subset safely
        df, _ = train_test_split(df, train_size=limit_data, stratify=df['is_duplicate'], random_state=42)
        logger.info(f"Stratified sample of specifically {limit_data} records.")
    else:
        # Use sklearn train_test_split to strictly stratify sample 20%
        df, _ = train_test_split(df, train_size=0.20, stratify=df['is_duplicate'], random_state=42)
        logger.info("No limit_data specified. Defaulting to 20% stratified sample of the dataset.")
    
    # Format texts natively inside the CSV to make it easier to load directly later
    logger.info("Formatting labels and text pairings...")
    texts = df.apply(lambda row: f"question1: {row['question1']} question2: {row['question2']}", axis=1).tolist()
    labels = df['is_duplicate'].astype(int).tolist()
    
    # 80/20 train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)
        
    # Save the splits to disk explicitly
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    
    pd.DataFrame({"text": train_texts, "label": train_labels}).to_csv(train_path, index=False)
    pd.DataFrame({"text": val_texts, "label": val_labels}).to_csv(val_path, index=False)
    
    logger.info(f"Saved {len(train_texts)} records to {train_path}")
    logger.info(f"Saved {len(val_texts)} records to {val_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract classification data from Quora Dataset")
    parser.add_argument("--limit-data", type=int, default=None, help="Use smaller subset. Defaults to 20%.")
    parser.add_argument("--output-dir", type=str, default="data/classification_splits", help="Where to save the splits.")
    args = parser.parse_args()
    
    create_dataset(limit_data=args.limit_data, output_dir=args.output_dir)
