"""
Train the T5 Encoder on the duplicate classification dataset.

Example Usage via CLI:

# Train with default settings (10 epochs, small flan-t5):
python scripts/encoder/train_encoder_classifier.py

# Train with custom batch size and learning rate:
python scripts/encoder/train_encoder_classifier.py --batch-size 8 --learning-rate 5e-5 --epochs 20

# Point to custom pre-extracted data files (if you changed the output directory):
python scripts/encoder/train_encoder_classifier.py --train-file path/to/train.csv --val-file path/to/val.csv

# Freeze the entire T5 model and only train the new classification head:
python scripts/encoder/train_encoder_classifier.py --head-only

# Example combining head-only with specific batch sizes and epochs:
python scripts/encoder/train_encoder_classifier.py --head-only --batch-size 32 --learning-rate 0.2 --epochs 30
"""

import os
import argparse
import logging
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

from transformers import T5EncoderModel, AutoTokenizer

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuoraDuplicateDataset(Dataset):
    """Dataset for training the encoder on Quora pairs predicting is_duplicate"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

class T5EncoderClassifier(nn.Module):
    """Extracts only the T5 encoder and adds a classification head for representations"""
    def __init__(self, model_path="checkpoints/flan-t5-small", num_classes=2, use_lora=False, head_only=False):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_path)
        
        self.use_lora = use_lora
        
        if head_only:
            logger.info("Freezing base encoder. Training classification head ONLY.")
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.use_lora = False # Cannot use LoRA if we are strictly skipping encoder training
            
        elif self.use_lora:
            from peft import get_peft_model, LoraConfig
            logger.info("Applying LoRA to Encoder!")
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            
        hidden_size = self.encoder.config.hidden_size if not self.use_lora else self.encoder.base_model.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling based on attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # Ensure pooled tensor explicitly tracks gradients if head_only is True
        # but the base model was frozen. (Normally taken care of by the linear layer, but safe to force)
        if not pooled.requires_grad:
            pooled = pooled.clone().detach().requires_grad_(True)
            
        logits = self.classifier(pooled)
        return logits

def train_encoder(batch_size=16, epochs=10, learning_rate=1e-4, model_path="checkpoints/flan-t5-small", train_file="data/classification_splits/train.csv", val_file="data/classification_splits/val.csv", use_lora=True, head_only=False):
    device = DEFAULT_DEVICE
    
    logger.info(f"Loading pre-computed training data from {train_file}...")
    train_df = pd.read_csv(train_file).dropna()
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].astype(int).tolist()
    
    logger.info(f"Loading pre-computed validation data from {val_file}...")
    val_df = pd.read_csv(val_file).dropna()
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label'].astype(int).tolist()
    
    logger.info(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dataset = QuoraDuplicateDataset(train_texts, train_labels, tokenizer)
    val_dataset = QuoraDuplicateDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = T5EncoderClassifier(model_path=model_path, use_lora=use_lora, head_only=head_only).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Only pass trainable parameters to the optimizer (vital if freezing the whole encoder)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=learning_rate)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, eps=1e-6)
    
    best_val_loss = float("inf")
    
    # Adjust checkpoint naming based on what we are training
    mode_str = "head_only_" if head_only else ("lora_" if use_lora else "")
    checkpoint_dir = f"checkpoints/encoder_classifier_{mode_str}AdamW_{learning_rate:.0e}_ep{epochs}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info(f"Starting Encoder-Only classification training. saving to {checkpoint_dir}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()
            
            loss.backward()
            
            # Clip gradients ONLY for parameters that require gradients
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            
            progress_bar.set_postfix({"loss": loss.item(), "acc": correct/total})
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels = batch["labels"].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, batch_labels)
                
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_labels.size(0)
                
        avg_val_loss = val_loss/len(val_loader)
        val_acc = val_correct/val_total
        logger.info(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info("Saving best model...")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_encoder.pt"))
            model.encoder.save_pretrained(os.path.join(checkpoint_dir, "best_t5_encoder"))
            tokenizer.save_pretrained(os.path.join(checkpoint_dir, "best_t5_encoder"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train T5 Encoder on Quora Duplicate Classification")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="checkpoints/flan-t5-small")
    parser.add_argument("--train-file", type=str, default="data/classification_splits/train.csv")
    parser.add_argument("--val-file", type=str, default="data/classification_splits/val.csv")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA and train the full encoder")
    parser.add_argument("--head-only", action="store_true", help="Freeze the entire T5 encoder and ONLY train the newly initialized classification head")
    args = parser.parse_args()
    
    train_encoder(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_path=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        use_lora=not args.no_lora,
        head_only=args.head_only
    )
