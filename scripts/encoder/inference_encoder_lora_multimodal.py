"""
Inference script for LoRA-fine-tuned T5 Encoder for paraphrase/duplicate detection.

This script:
1. Loads a trained encoder model from checkpoint
2. Reads test data from CSV
3. Combines sentence_a and sentence_b exactly as done during training
4. Extracts categorical features (style_label, length_label)
5. Makes predictions and computes confidence scores
6. Computes classification metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
7. Saves predictions and metrics to output files

Example Usage:

# Run inference on test set
python scripts/encoder/inference_encoder_lora_multimodal.py \
    --checkpoint checkpoints/encoder_lora_multimodal_combined_AdamW_1e-04_ep30 \
    --test-file data/classification_splits/test.csv \
    --output predictions.csv

# Run with custom batch size and device
python scripts/encoder/inference_encoder_lora_multimodal.py \
    --checkpoint checkpoints/encoder_lora_multimodal_combined_AdamW_1e-04_ep30 \
    --test-file data/classification_splits/test.csv \
    --output predictions.csv \
    --batch-size 32 \
    --device cuda

Output files:
- predictions.csv: Predictions with all features and confidence scores
- predictions_metrics.txt: Classification metrics report (if ground truth available)
"""

import os
import argparse
import logging
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import numpy as np

from transformers import T5EncoderModel, AutoTokenizer
from peft import PeftModel

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceMultiModalDataset(Dataset):
    """
    Dataset for inference on sentence pairs with categorical modalities.
    
    Expects CSV with columns:
    - sentence_a or source: First sentence
    - sentence_b or target: Second sentence
    - style_label: Categorical feature (e.g., CONSERVATIVE, CREATIVE)
    - length_label: Categorical feature (e.g., SAME, LONG, SHORT)
    """
    def __init__(self, sentence_a_list, sentence_b_list, style_labels, length_labels,
                 tokenizer, style_label_vocab, length_label_vocab, max_length=256):
        self.sentence_a_list = sentence_a_list
        self.sentence_b_list = sentence_b_list
        self.style_labels = style_labels
        self.length_labels = length_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.style_label_vocab = style_label_vocab
        self.length_label_vocab = length_label_vocab

    def __len__(self):
        return len(self.sentence_a_list)

    def __getitem__(self, idx):
        sentence_a = self.sentence_a_list[idx]
        sentence_b = self.sentence_b_list[idx]
        style_label = self.style_labels[idx]
        length_label = self.length_labels[idx]

        # Combine sentences EXACTLY as in training
        combined_text = f"paraphrase detection: sentence1: {sentence_a} </s> sentence2: {sentence_b}"
        
        # Encode combined text
        encoding = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )

        # Encode categorical labels as indices
        style_idx = self.style_label_vocab.get(style_label, 0)  # Default to 0 if unknown
        length_idx = self.length_label_vocab.get(length_label, 0)  # Default to 0 if unknown

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "style_label_idx": torch.tensor(style_idx, dtype=torch.long),
            "length_label_idx": torch.tensor(length_idx, dtype=torch.long)
        }


class LoRAMultiModalEncoderInference(nn.Module):
    """
    T5 Encoder with LoRA for inference.
    Loads pre-trained LoRA weights and freezes them.
    """
    def __init__(self, base_model_path, lora_weights_path, 
                 num_style_labels=2, num_length_labels=3, categorical_embed_dim=32):
        super().__init__()
        
        # Load base encoder
        self.encoder = T5EncoderModel.from_pretrained(base_model_path)
        
        # Load LoRA weights
        logger.info(f"Loading LoRA weights from {lora_weights_path}...")
        self.encoder = PeftModel.from_pretrained(self.encoder, lora_weights_path)
        
        # Set to eval mode and freeze parameters
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        hidden_size = self.encoder.config.hidden_size
        
        # Categorical embeddings (load from checkpoint if available)
        self.style_embedding = nn.Embedding(num_style_labels, categorical_embed_dim)
        self.length_embedding = nn.Embedding(num_length_labels, categorical_embed_dim)
        
        # Classification head
        classifier_input_size = hidden_size + categorical_embed_dim * 2
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(classifier_input_size, 2)  # Binary classification
        )
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def mean_pool(self, hidden_states, attention_mask):
        """Apply mean pooling with attention mask"""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask, style_label_idx, length_label_idx):
        """Forward pass for inference"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embedding = self.mean_pool(outputs.last_hidden_state, attention_mask)
        
        style_embedding = self.style_embedding(style_label_idx)
        length_embedding = self.length_embedding(length_label_idx)
        
        combined = torch.cat([
            embedding,
            style_embedding,
            length_embedding
        ], dim=1)
        
        logits = self.classifier(combined)
        return logits


def load_inference_data(file_path):
    """
    Load inference data from CSV.
    Supports both sentence_a/sentence_b and source/target naming conventions.
    Also extracts relevant features.
    """
    df = pd.read_csv(file_path).dropna()
    
    # Check for sentence_a/sentence_b or source/target columns
    if 'sentence_a' in df.columns and 'sentence_b' in df.columns:
        sentence_a = df['sentence_a'].tolist()
        sentence_b = df['sentence_b'].tolist()
    elif 'source' in df.columns and 'target' in df.columns:
        sentence_a = df['source'].tolist()
        sentence_b = df['target'].tolist()
    else:
        available = list(df.columns)
        logger.warning(f"Available columns: {available}")
        raise ValueError("CSV must have either [sentence_a, sentence_b] or [source, target] columns")
    
    # Extract categorical labels
    style_labels = df['style_label'].tolist() if 'style_label' in df.columns else ['UNKNOWN'] * len(sentence_a)
    length_labels = df['length_label'].tolist() if 'length_label' in df.columns else ['UNKNOWN'] * len(sentence_a)
    
    # Extract optional features
    features = {}
    feature_cols = ['lexical_overlap', 'seq_similarity', 'length_ratio', 'src_tokens', 'tgt_tokens']
    for col in feature_cols:
        if col in df.columns:
            features[col] = df[col].tolist()
    
    # Extract ground truth labels if available
    labels = None
    if 'label' in df.columns:
        labels = df['label'].tolist()
    
    return sentence_a, sentence_b, style_labels, length_labels, features, labels


def compute_classification_metrics(y_true, y_pred, y_probs=None):
    """
    Compute classic classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (for ROC-AUC)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, 
                                                       target_names=['not_paraphrase', 'paraphrase'],
                                                       zero_division=0)
    }
    
    # Add ROC-AUC if probabilities are available
    if y_probs is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs)
        except:
            metrics['roc_auc'] = None
    
    return metrics


def save_metrics_report(metrics, output_path):
    """Save metrics report to file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLASSIFICATION METRICS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
        if metrics.get('roc_auc'):
            f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 80 + "\n")
        f.write("                 Predicted\n")
        f.write("                 Not Para  Para\n")
        cm = metrics['confusion_matrix']
        f.write(f"Actual Not Para  {cm[0][0]:6d}    {cm[0][1]:6d}\n")
        f.write(f"Actual Para      {cm[1][0]:6d}    {cm[1][1]:6d}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(metrics['classification_report'] + "\n")
    
    logger.info(f"Metrics report saved to {output_path}")



def run_inference(checkpoint_dir, test_file, output_file, batch_size=16, device=DEFAULT_DEVICE):
    """
    Run inference on test data.
    
    Args:
        checkpoint_dir: Directory containing checkpoint with best_model.pt and model_config.json
        test_file: Path to test CSV file
        output_file: Path to save predictions
        batch_size: Batch size for inference
        device: Device to run on (cuda or cpu)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load model config
    config_path = os.path.join(checkpoint_dir, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    logger.info(f"Model config: {model_config}")
    
    base_model_path = model_config['model_path']
    style_vocab = model_config['style_vocab']
    length_vocab = model_config['length_vocab']
    categorical_embed_dim = model_config.get('categorical_embed_dim', 32)
    
    # Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "best_encoder_lora")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        logger.info(f"Loaded tokenizer from base model {base_model_path}")
    
    # Ensure special tokens
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load inference data
    logger.info(f"Loading test data from {test_file}...")
    sentence_a, sentence_b, style_labels, length_labels, features, ground_truth = load_inference_data(test_file)
    logger.info(f"Loaded {len(sentence_a)} test samples")
    
    # Create inference dataset
    inference_dataset = InferenceMultiModalDataset(
        sentence_a, sentence_b, style_labels, length_labels,
        tokenizer, style_vocab, length_vocab
    )
    
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    logger.info("Loading model...")
    model = LoRAMultiModalEncoderInference(
        base_model_path=base_model_path,
        lora_weights_path=os.path.join(checkpoint_dir, "best_encoder_lora"),
        num_style_labels=len(style_vocab),
        num_length_labels=len(length_vocab),
        categorical_embed_dim=categorical_embed_dim
    ).to(device)
    
    # Load best model weights (classifier and embeddings)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        logger.info(f"Loading model weights from {best_model_path}...")
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    
    # Run inference
    logger.info("Running inference...")
    all_predictions = []
    all_confidences = []
    all_logits = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Inferencing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            style_idx = batch["style_label_idx"].to(device)
            length_idx = batch["length_label_idx"].to(device)
            
            logits = model(input_ids, attention_mask, style_idx, length_idx)
            
            # Get predictions and confidence scores
            probs = softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_logits.extend(logits.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_confidences.extend(probs.max(dim=-1).values.cpu().numpy())
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'source': sentence_a,
        'target': sentence_b,
        'style_label': style_labels,
        'length_label': length_labels,
        'prediction': all_predictions,
        'prediction_label': ['paraphrase' if p == 1 else 'not_paraphrase' for p in all_predictions],
        'confidence': all_confidences
    })
    
    # Add features if available
    for feature_name, feature_values in features.items():
        output_df[feature_name] = feature_values
    
    # Add ground truth if available
    if ground_truth is not None:
        output_df['ground_truth_label'] = ground_truth
        output_df['ground_truth_label_str'] = ['paraphrase' if l == 1 else 'not_paraphrase' for l in ground_truth]
        
        # Calculate accuracy
        correct = sum(output_df['prediction'] == output_df['ground_truth_label'])
        accuracy = correct / len(output_df)
        logger.info(f"\nAccuracy on test set: {accuracy:.4f} ({correct}/{len(output_df)})")
        
        # Compute comprehensive metrics
        logger.info("\n" + "=" * 80)
        logger.info("COMPUTING CLASSIFICATION METRICS")
        logger.info("=" * 80)
        
        metrics = compute_classification_metrics(
            output_df['ground_truth_label'].values,
            output_df['prediction'].values,
            output_df['confidence'].values
        )
        
        # Display metrics
        logger.info(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        if metrics.get('roc_auc'):
            logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        logger.info("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        logger.info("                 Predicted")
        logger.info("                 Not Para  Para")
        logger.info(f"Actual Not Para  {cm[0][0]:6d}    {cm[0][1]:6d}")
        logger.info(f"Actual Para      {cm[1][0]:6d}    {cm[1][1]:6d}")
        
        logger.info("\nDetailed Classification Report:")
        logger.info(metrics['classification_report'])
        
        # Save metrics report
        metrics_report_path = output_file.replace('.csv', '_metrics.txt')
        save_metrics_report(metrics, metrics_report_path)
    
    # Save predictions
    output_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")
    
    # Print sample predictions
    logger.info("\nSample predictions:")
    logger.info(output_df.head(10).to_string())
    
    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference for LoRA-tuned T5 Encoder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--test-file", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV file for predictions")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    run_inference(
        checkpoint_dir=args.checkpoint,
        test_file=args.test_file,
        output_file=args.output,
        batch_size=args.batch_size,
        device=args.device
    )
