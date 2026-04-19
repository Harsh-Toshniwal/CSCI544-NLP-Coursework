"""Training script for ParaGen"""

import argparse
import logging
import os
import torch
from torch.optim import Adam
from tqdm import tqdm
import json

from paragen.config import Config, get_config
from paragen.data_loader import load_qqp, load_paws, create_data_loaders
from paragen.model import ParaphraseModel
from transformers import get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    model.model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch in progress_bar:
        optimizer.zero_grad()

        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, val_loader, device):
    """Validation"""
    model.model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(val_loader)
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def train(config: Config = None, resume_from: str = None):
    """Main training function"""
    if config is None:
        config = get_config()

    device = config.training.device
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    # Load data
    logger.info("Loading datasets...")
    all_pairs = []

    if config.data.train_dataset in ["qqp", "both"]:
        qqp_pairs = load_qqp(config.data.data_dir)
        all_pairs.extend(qqp_pairs)

    if config.data.train_dataset in ["paws", "both"]:
        paws_pairs = load_paws(config.data.data_dir)
        all_pairs.extend(paws_pairs)

    logger.info(f"Total pairs: {len(all_pairs)}")

    # Create model
    logger.info(f"Loading model: {config.model.model_name}")
    model = ParaphraseModel(
        model_name=config.model.model_name,
        device=device,
        length_tokens=config.attributes.length_tokens,
        diversity_tokens=config.attributes.diversity_tokens,
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        all_pairs,
        model.tokenizer,
        batch_size=config.data.batch_size,
        train_ratio=config.data.train_size,
        val_ratio=config.data.val_size,
        max_source_length=config.data.max_source_length,
        max_target_length=config.data.max_target_length,
        num_workers=config.data.num_workers,
        seed=config.data.seed,
    )

    # Setup optimizer
    optimizer = Adam(model.model.parameters(), lr=config.training.learning_rate)

    # Setup scheduler
    total_steps = len(train_loader) * config.training.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    best_val_loss = float("inf")
    train_history = {"loss": [], "val_loss": []}

    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from {resume_from}")
        checkpoint = torch.load(os.path.join(resume_from, "training_state.pt"))
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        train_history = checkpoint["history"]

    for epoch in range(start_epoch, config.training.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss = validate(model, val_loader, device)

        train_history["loss"].append(train_loss)
        train_history["val_loss"].append(val_loss)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                config.training.checkpoint_dir, f"checkpoint_epoch_{epoch}"
            )
            os.makedirs(checkpoint_path, exist_ok=True)

            model.save(checkpoint_path)

            # Save training state
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "history": train_history,
                },
                os.path.join(checkpoint_path, "training_state.pt"),
            )

            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final training history
    with open(os.path.join(config.training.checkpoint_dir, "training_history.json"), "w") as f:
        json.dump(train_history, f, indent=2)

    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ParaGen model")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on",
    )

    args = parser.parse_args()

    config = get_config()
    if args.device:
        config.training.device = args.device

    train(config=config, resume_from=args.resume_from)
