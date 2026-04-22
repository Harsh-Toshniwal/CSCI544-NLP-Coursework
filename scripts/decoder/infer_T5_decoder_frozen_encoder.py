"""
Inference script for Decoder LoRA with Frozen Encoder.

This script loads:
1. Pre-trained frozen encoder from checkpoint
2. Fine-tuned decoder LoRA from another checkpoint
3. Generates paraphrases and compares with reference

Example Usage:
    python scripts/infer_decoder_frozen_encoder.py \
        --encoder-checkpoint checkpoints/encoder_lora_multimodal_combined_16_epc_50_lr_5e_5 \
        --decoder-checkpoint checkpoints/decoder_lora_with_frozen_encoder/best_model \
        --input-texts "The quick brown fox" "A fast brown fox"
"""

import argparse
import logging
import os
import json
import torch
from typing import List

from transformers import T5ForConditionalGeneration, T5EncoderModel, AutoTokenizer
from peft import PeftModel

from paragen.evaluation import ParaphraseEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FrozenEncoderDecoderLoRAInferencer:
    """Inferencer for frozen encoder with fine-tuned decoder LoRA"""

    def __init__(
        self,
        encoder_checkpoint: str,
        decoder_checkpoint: str,
        device: str = DEFAULT_DEVICE,
    ):
        """
        Initialize inferencer.

        Args:
            encoder_checkpoint: Path to frozen encoder checkpoint
            decoder_checkpoint: Path to decoder LoRA checkpoint
            device: Device to use
        """
        self.device = device
        self.evaluator = ParaphraseEvaluator()

        # Load decoder config to get base model
        decoder_config_path = os.path.join(decoder_checkpoint, "config.json")
        if os.path.exists(decoder_config_path):
            with open(decoder_config_path, "r") as f:
                decoder_config = json.load(f)
            encoder_checkpoint = decoder_config.get("encoder_checkpoint", encoder_checkpoint)

        # Load encoder config to get base model
        encoder_config_path = os.path.join(encoder_checkpoint, "model_config.json")
        if os.path.exists(encoder_config_path):
            with open(encoder_config_path, "r") as f:
                encoder_config = json.load(f)
            base_model_name = encoder_config.get("model_path", "flan-t5-small")
        else:
            base_model_name = "flan-t5-small"

        logger.info(f"Base model: {base_model_name}")
        logger.info(f"Encoder checkpoint: {encoder_checkpoint}")
        logger.info(f"Decoder checkpoint: {decoder_checkpoint}")

        # Load tokenizer from decoder checkpoint
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)

        # Load base model
        logger.info(f"Loading base T5 model: {base_model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(base_model_name)

        # Load frozen encoder with LoRA
        logger.info("Loading frozen encoder with LoRA...")
        encoder_lora_dir = os.path.join(encoder_checkpoint, "best_encoder_lora")
        if os.path.exists(encoder_lora_dir):
            temp_encoder = T5EncoderModel.from_pretrained(base_model_name)
            temp_encoder = PeftModel.from_pretrained(temp_encoder, encoder_lora_dir)
            self.model.encoder = temp_encoder
            logger.info("Encoder loaded successfully")

        # Load decoder LoRA
        logger.info("Loading decoder LoRA...")
        self.model = PeftModel.from_pretrained(self.model, decoder_checkpoint)

        self.model.to(device)
        self.model.eval()

        logger.info("Model loaded successfully")

    @torch.no_grad()
    def generate_paraphrase(
        self,
        source_text: str,
        max_length: int = 128,
        num_beams: int = 5,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Generate paraphrases"""
        inputs = self.tokenizer(
            f"paraphrase: {source_text}",
            return_tensors="pt",
            padding=True,
            max_length=512,
            truncation=True,
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

        paraphrases = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return paraphrases

    def compare_with_reference(
        self,
        source_text: str,
        reference_text: str,
        num_paraphrases: int = 1,
        max_length: int = 128,
        num_beams: int = 5,
    ) -> dict:
        """Generate paraphrases and compare with reference"""
        logger.info(f"Source: {source_text}")
        logger.info(f"Reference: {reference_text}")

        paraphrases = self.generate_paraphrase(
            source_text,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_paraphrases,
        )

        results = {
            "source": source_text,
            "reference": reference_text,
            "paraphrases": paraphrases,
            "similarities": [],
            "diversity_scores": [],
        }

        logger.info(f"\nGenerated {len(paraphrases)} paraphrase(s):")

        for i, paraphrase in enumerate(paraphrases):
            logger.info(f"  [{i + 1}] {paraphrase}")

            similarity = self.evaluator.compute_bert_score_similarity(
                paraphrase, reference_text
            )
            results["similarities"].append(similarity)

            diversity = self.evaluator.compute_inverse_bleu(source_text, paraphrase)
            results["diversity_scores"].append(diversity)

            logger.info(f"      Similarity: {similarity:.4f} | Diversity: {diversity:.4f}")

        avg_similarity = sum(results["similarities"]) / len(results["similarities"])
        avg_diversity = sum(results["diversity_scores"]) / len(results["diversity_scores"])

        results["avg_similarity"] = avg_similarity
        results["avg_diversity"] = avg_diversity

        logger.info(f"\nAverage Similarity: {avg_similarity:.4f}")
        logger.info(f"Average Diversity: {avg_diversity:.4f}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Inference with frozen encoder and decoder LoRA"
    )

    parser.add_argument(
        "--encoder-checkpoint",
        type=str,
        required=True,
        help="Path to encoder checkpoint",
    )

    parser.add_argument(
        "--decoder-checkpoint",
        type=str,
        required=True,
        help="Path to decoder LoRA checkpoint",
    )

    parser.add_argument(
        "--input-texts",
        type=str,
        nargs="+",
        required=True,
        help="Input texts",
    )

    parser.add_argument(
        "--num-paraphrases",
        type=int,
        default=3,
        help="Number of paraphrases to generate",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max output length",
    )

    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Number of beams",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device to use (default: {DEFAULT_DEVICE})",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("INFERENCE: FROZEN ENCODER + DECODER LoRA")
    logger.info("=" * 70)

    inferencer = FrozenEncoderDecoderLoRAInferencer(
        args.encoder_checkpoint,
        args.decoder_checkpoint,
        device=args.device,
    )

    if len(args.input_texts) == 2:
        logger.info("\nMode: Comparison (source vs reference)")
        logger.info("-" * 70)
        result = inferencer.compare_with_reference(
            args.input_texts[0],
            args.input_texts[1],
            num_paraphrases=args.num_paraphrases,
            max_length=args.max_length,
            num_beams=args.num_beams,
        )
    else:
        logger.info(f"\nMode: Generation ({len(args.input_texts)} text(s))")
        logger.info("-" * 70)
        for text in args.input_texts:
            logger.info(f"\nSource: {text}")
            paraphrases = inferencer.generate_paraphrase(
                text,
                max_length=args.max_length,
                num_beams=args.num_beams,
                num_return_sequences=args.num_paraphrases,
            )
            for i, p in enumerate(paraphrases):
                logger.info(f"  [{i + 1}] {p}")

    logger.info("\n" + "=" * 70)
    logger.info("INFERENCE COMPLETED")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
