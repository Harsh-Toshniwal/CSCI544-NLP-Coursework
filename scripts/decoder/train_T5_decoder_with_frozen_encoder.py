"""
Decoder LoRA Fine-tuning with Pre-trained Frozen Encoder

This script:
1. Loads a pre-trained encoder (with LoRA weights) from checkpoint
2. Freezes the encoder completely
3. Applies LoRA only to the decoder
4. Fine-tunes the decoder for paraphrase generation

The encoder remains frozen with its original LoRA weights intact.
Only the decoder receives new LoRA layers for fine-tuning.

Example Usage:
    python scripts/train_decoder_with_frozen_encoder.py \
        --encoder-checkpoint checkpoints/encoder_lora_multimodal_combined_16_epc_50_lr_5e_5 \
        --data-path sample_data/train.tsv \
        --output-dir checkpoints/decoder_lora_with_frozen_encoder \
        --num-epochs 5 \
        --batch-size 16 \
        --learning-rate 5e-4
"""

import argparse
import logging
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from typing import Dict, Tuple, List
from pathlib import Path

from transformers import (
    T5ForConditionalGeneration,
    T5EncoderModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from paragen.data_loader import ParaphraseDataset
from paragen.evaluation import ParaphraseEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DecoderLoRAWithFrozenEncoderTrainer:
    """Trainer class for decoder LoRA with frozen pre-trained encoder"""

    def __init__(
        self,
        encoder_checkpoint: str,
        base_model_name: str = None,
        device: str = DEFAULT_DEVICE,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
    ):
        """
        Initialize trainer with frozen encoder and LoRA decoder.

        Args:
            encoder_checkpoint: Path to encoder checkpoint with LoRA weights
            base_model_name: Base model name (auto-detected if None)
            device: Device to train on
            lora_r: LoRA rank for decoder
            lora_alpha: LoRA alpha for decoder
            lora_dropout: LoRA dropout
            lora_target_modules: Decoder modules to apply LoRA to
        """
        self.device = device
        self.encoder_checkpoint = encoder_checkpoint
        self.evaluator = ParaphraseEvaluator()

        # Default target modules for T5 decoder
        if lora_target_modules is None:
            lora_target_modules = ["q", "v"]

        # Load encoder config to get base model name
        config_path = os.path.join(encoder_checkpoint, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                encoder_config = json.load(f)
            base_model_name = encoder_config.get("model_path", "flan-t5-small")
        else:
            base_model_name = base_model_name or "flan-t5-small"

        logger.info(f"Base model: {base_model_name}")
        logger.info(f"Encoder checkpoint: {encoder_checkpoint}")

        # Load tokenizer from encoder checkpoint
        logger.info("Loading tokenizer...")
        encoder_lora_dir = os.path.join(encoder_checkpoint, "best_encoder_lora")
        if os.path.exists(encoder_lora_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_lora_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load full T5 model
        logger.info(f"Loading base T5 model: {base_model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(base_model_name)

        # Load pre-trained encoder with LoRA weights
        logger.info(f"Loading pre-trained encoder from {encoder_checkpoint}")
        self._load_frozen_encoder(encoder_checkpoint, base_model_name)

        # Freeze encoder completely
        logger.info("Freezing encoder...")
        self._freeze_encoder()

        # Apply LoRA to decoder only
        logger.info(f"Applying LoRA to decoder (r={lora_r}, alpha={lora_alpha})")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.to(device)

        # Print trainable parameters
        self._print_parameter_stats()

    def _load_frozen_encoder(self, encoder_checkpoint: str, base_model_name: str):
        """Load pre-trained encoder with LoRA weights"""
        encoder_lora_dir = os.path.join(encoder_checkpoint, "best_encoder_lora")

        if os.path.exists(encoder_lora_dir):
            # Load encoder model with LoRA
            logger.info(f"Loading encoder LoRA from {encoder_lora_dir}")
            temp_encoder = T5EncoderModel.from_pretrained(base_model_name)
            temp_encoder = PeftModel.from_pretrained(temp_encoder, encoder_lora_dir)
            
            # Replace encoder in full model
            self.model.encoder = temp_encoder
            logger.info("Encoder loaded successfully with LoRA weights")
        else:
            logger.warning(
                f"LoRA directory not found at {encoder_lora_dir}. "
                "Using base encoder without fine-tuning."
            )

    def _freeze_encoder(self):
        """Freeze all encoder parameters"""
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def _print_parameter_stats(self):
        """Print parameter statistics"""
        encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
        encoder_trainable = sum(
            p.numel() for p in self.model.encoder.parameters() if p.requires_grad
        )
        decoder_trainable = sum(
            p.numel() for p in self.model.decoder.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info("\n" + "=" * 60)
        logger.info("Parameter Statistics:")
        logger.info("=" * 60)
        logger.info(f"Encoder total params:      {encoder_params:>15,}")
        logger.info(f"Encoder trainable params:  {encoder_trainable:>15,} (frozen)")
        logger.info(f"Decoder trainable params:  {decoder_trainable:>15,}")
        logger.info(f"Total params:              {total_params:>15,}")
        logger.info(f"Total trainable params:    {total_trainable:>15,}")
        logger.info(
            f"Trainable percentage:      {100 * total_trainable / total_params:>14.2f}%"
        )
        logger.info("=" * 60 + "\n")

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training")

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
            )

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} - Average Training Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int = -1) -> Tuple[float, Dict]:
        """Validate the model and compute metrics"""
        self.model.eval()
        total_loss = 0
        all_similarities = []
        all_inverse_bleu = []

        progress_bar = tqdm(val_loader, desc="Validating")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()

            # Generate paraphrases
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5,
                early_stopping=True,
            )

            generated_texts = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            target_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            source_texts = batch["source"]

            for source, generated, target in zip(source_texts, generated_texts, target_texts):
                similarity = self.evaluator.compute_bert_score_similarity(generated, target)
                all_similarities.append(similarity)

                inverse_bleu = self.evaluator.compute_inverse_bleu(source, generated)
                all_inverse_bleu.append(inverse_bleu)

        avg_loss = total_loss / len(val_loader)
        avg_similarity = np.mean(all_similarities) if all_similarities else 0.0
        avg_diversity = np.mean(all_inverse_bleu) if all_inverse_bleu else 0.0

        metrics = {
            "val_loss": avg_loss,
            "similarity": avg_similarity,
            "diversity": avg_diversity,
        }

        if epoch >= 0:
            logger.info(
                f"Epoch {epoch + 1} - Val Loss: {avg_loss:.4f}, "
                f"Similarity: {avg_similarity:.4f}, Diversity: {avg_diversity:.4f}"
            )

        return avg_loss, metrics

    def save_checkpoint(self, output_dir: str, epoch: int = -1):
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving checkpoint to {output_dir}")

        # Save decoder LoRA weights
        self.model.save_pretrained(output_dir)

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        # Save config with encoder checkpoint path
        config = {
            "encoder_checkpoint": self.encoder_checkpoint,
            "device": self.device,
            "epoch": epoch,
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Checkpoint saved successfully")

    @torch.no_grad()
    def generate_paraphrases(
        self,
        source_texts: List[str],
        max_length: int = 128,
        num_beams: int = 5,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Generate paraphrases"""
        self.model.eval()

        inputs = self.tokenizer(
            source_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
        )

        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return generated_texts


def create_data_loaders(
    pairs: List[Tuple[str, str]],
    tokenizer,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    max_source_length: int = 128,
    max_target_length: int = 128,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    indices = np.random.permutation(len(pairs))
    pairs = [pairs[i] for i in indices]

    train_size = int(len(pairs) * train_ratio)
    val_size = int(len(pairs) * val_ratio)

    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size : train_size + val_size]
    test_pairs = pairs[train_size + val_size :]

    logger.info(
        f"Data split - Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}"
    )

    train_dataset = ParaphraseDataset(
        train_pairs,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        add_attributes=False,
    )

    val_dataset = ParaphraseDataset(
        val_pairs,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        add_attributes=False,
    )

    test_dataset = ParaphraseDataset(
        test_pairs,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        add_attributes=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def load_paraphrase_pairs(
    data_path: str = None,
    num_samples: int = None,
) -> List[Tuple[str, str]]:
    """Load paraphrase pairs from file"""
    import pandas as pd

    if data_path is None:
        data_path = "sample_data/train.tsv"

    logger.info(f"Loading data from {data_path}")

    try:
        df = pd.read_csv(data_path, sep="\t")
    except Exception:
        df = pd.read_csv(data_path, sep=",")

    pairs = []
    for idx, row in df.iterrows():
        if "sentence1" in df.columns and "sentence2" in df.columns:
            source = row["sentence1"]
            target = row["sentence2"]
        elif "source" in df.columns and "target" in df.columns:
            source = row["source"]
            target = row["target"]
        else:
            cols = df.columns.tolist()
            source = row[cols[0]]
            target = row[cols[1]]

        if pd.notna(source) and pd.notna(target):
            pairs.append((str(source), str(target)))

        if num_samples and len(pairs) >= num_samples:
            break

    logger.info(f"Loaded {len(pairs)} paraphrase pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune T5 decoder with LoRA, keeping encoder frozen"
    )

    # Encoder arguments
    parser.add_argument(
        "--encoder-checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained encoder checkpoint",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (auto-detected from encoder config if None)",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
    )

    # Training arguments
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps (default: 500)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="sample_data/train.tsv",
        help="Path to training data file",
    )
    parser.add_argument(
        "--max-source-length",
        type=int,
        default=128,
        help="Maximum source sequence length (default: 128)",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=128,
        help="Maximum target sequence length (default: 128)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Proportion of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Proportion of data for validation (default: 0.1)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/decoder_lora_with_frozen_encoder",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device to use (default: {DEFAULT_DEVICE})",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DECODER LoRA WITH FROZEN ENCODER FINE-TUNING")
    logger.info("=" * 70)

    # Load data
    pairs = load_paraphrase_pairs(args.data_path)

    if not pairs:
        logger.error(f"No data loaded from {args.data_path}")
        return

    # Initialize trainer
    trainer = DecoderLoRAWithFrozenEncoderTrainer(
        encoder_checkpoint=args.encoder_checkpoint,
        base_model_name=args.base_model,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        pairs,
        trainer.tokenizer,
        batch_size=args.batch_size,
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    # Setup optimizer
    optimizer = AdamW(
        trainer.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Setup scheduler
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)

    best_val_loss = float("inf")
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "similarity": [],
        "diversity": [],
    }

    for epoch in range(args.num_epochs):
        train_loss = trainer.train_epoch(train_loader, optimizer, scheduler, epoch)
        training_history["train_loss"].append(train_loss)

        val_loss, metrics = trainer.validate(val_loader, epoch)
        training_history["val_loss"].append(val_loss)
        training_history["similarity"].append(metrics["similarity"])
        training_history["diversity"].append(metrics["diversity"])

        checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
        trainer.save_checkpoint(checkpoint_dir, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = os.path.join(args.output_dir, "best_model")
            trainer.save_checkpoint(best_dir, epoch)
            logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")

    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    # Final evaluation on test set
    logger.info("=" * 70)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 70)

    test_loss, test_metrics = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Similarity: {test_metrics['similarity']:.4f}")
    logger.info(f"Test Diversity: {test_metrics['diversity']:.4f}")

    results_path = os.path.join(args.output_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "test_loss": test_loss,
                "test_similarity": test_metrics["similarity"],
                "test_diversity": test_metrics["diversity"],
            },
            f,
            indent=2,
        )

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
