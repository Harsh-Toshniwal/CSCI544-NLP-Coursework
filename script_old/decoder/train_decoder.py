"""
Script to train the Decoder of a T5 model using LoRA for Paraphrasing.
The Encoder weights and non-LoRA Decoder weights are kept frozen.

Example Usage:
python scripts/decoder/train_decoder.py --epochs 10 --batch-size 16 --learning-rate 1e-4
"""

import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParaphraseSeq2SeqDataset(Dataset):
    def __init__(self, sources, targets, tokenizer, max_length=256):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = str(self.sources[idx])
        target = str(self.targets[idx])
        
        # Add a task prefix (optional, but standard for T5)
        source = f"paraphrase: {source}"

        model_inputs = self.tokenizer(
            source, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = self.tokenizer(
            target, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)
        
        # Replace padding token id in labels with -100 so CrossEntropy loss ignores them
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids
        }

def train_decoder_lora(batch_size=16, epochs=10, learning_rate=1e-4, model_path="checkpoints/flan-t5-small"):
    device = DEFAULT_DEVICE
    
    # 1. Load Data
    csv_path = "quora-question-pairs/train.csv/train.csv"
    logger.info(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path).dropna(subset=['question1', 'question2', 'is_duplicate'])
    
    # Only train on actual duplicate pairs (is_duplicate == 1) to learn paraphrasing
    df_dupes = df[df['is_duplicate'] == 1].sample(frac=0.20, random_state=42) # Taking 20% to test quickly
    logger.info(f"Filtered down to {len(df_dupes)} actual duplicate pairs for training.")
    
    sources = df_dupes['question1'].tolist()
    targets = df_dupes['question2'].tolist()
    
    train_src, val_src, train_tgt, val_tgt = train_test_split(sources, targets, test_size=0.2, random_state=42)
    logger.info(f"Train size: {len(train_src)}, Val size: {len(val_src)}")

    # 2. Setup Model & LoRA
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Configure LoRA to strictly target the decoder's attention modules only
    # T5 module names look like : "decoder.block.0.layer.0.SelfAttention.q"
    logger.info("Applying LoRA strictly to Decoder Q and V layers...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        # Targeting only decoder modules using regex
        target_modules=[r".*decoder.*\.q", r".*decoder.*\.v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # 3. Datasets and Loaders
    train_dataset = ParaphraseSeq2SeqDataset(train_src, train_tgt, tokenizer)
    val_dataset = ParaphraseSeq2SeqDataset(val_src, val_tgt, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 4. Optimizer and Scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, eps=1e-6)

    best_val_loss = float("inf")
    checkpoint_dir = f"checkpoints/decoder_lora_AdamW_{learning_rate:.0e}_ep{epochs}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 5. Training Loop
    logger.info(f"Starting Decoder LoRA training. Saving to {checkpoint_dir}")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        # Tunable weight for the token overlap penalty
        overlap_alpha = 0.5  

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ce_loss = outputs.loss
            
            # --- Auxiliary Token Overlap Penalty ---
            # 1. Create a multi-hot mask of the source sentence tokens
            vocab_size = outputs.logits.size(-1)
            src_multi_hot = torch.zeros(input_ids.size(0), vocab_size, device=device)
            src_multi_hot.scatter_(1, input_ids, 1.0)
            
            # Ignore special tokens (pad=0, eos=1, unk=2) so we don't penalize grammar/structure too heavily
            src_multi_hot[:, :3] = 0.0 
            
            # 2. Get model's predicted token probabilities
            probs = torch.softmax(outputs.logits, dim=-1)
            
            # 3. Calculate probability mass assigned to source tokens
            overlap_prob = (probs * src_multi_hot.unsqueeze(1)).sum(dim=-1)
            
            # 4. Average this penalty only over valid target words (ignoring padding)
            valid_tgt_mask = (labels != -100).float()
            overlap_penalty = (overlap_prob * valid_tgt_mask).sum() / (valid_tgt_mask.sum() + 1e-8)
            
            # Final combined loss
            loss = ce_loss + (overlap_alpha * overlap_penalty)
            # ---------------------------------------
            
            total_train_loss += loss.item()
            loss.backward()
            
            # Clip only trainable parameters
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            
            # Show individual losses in the progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "ce": ce_loss.item(),
                "overlap": overlap_penalty.item()
            })
            
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info("Saving best model...")
            model.save_pretrained(os.path.join(checkpoint_dir, "best_decoder_lora"))
            tokenizer.save_pretrained(os.path.join(checkpoint_dir, "best_decoder_lora"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train T5 Decoder using LoRA for Paraphrasing")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="checkpoints/flan-t5-small")
    args = parser.parse_args()
    
    train_decoder_lora(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_path=args.model
    )
