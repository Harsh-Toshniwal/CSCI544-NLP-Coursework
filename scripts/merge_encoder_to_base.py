"""
Merges a trained Encoder LoRA back into a full T5 Sequence-to-Sequence model.
This allows you to train a Decoder LoRA on top of your newly trained custom Encoder!

Example Usage:
python scripts/merge_encoder_to_base.py \
    --lora-dir "checkpoints/encoder_classifier_lora_AdamW_5e-05_ep10/best_t5_encoder" \
    --output-dir "checkpoints/flan-t5-custom-encoder"
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5EncoderModel
from peft import PeftModel

def merge_models(lora_dir, base_model, output_dir):
    print(f"1. Loading base full Seq2Seq model: {base_model}")
    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"2. Loading base Encoder and attaching LoRA adapters from {lora_dir}...")
    # We load just the encoder because that's what we trained the LoRA on
    base_encoder = T5EncoderModel.from_pretrained(base_model)
    peft_encoder = PeftModel.from_pretrained(base_encoder, lora_dir)

    print("3. Merging LoRA weights permanently into the Encoder...")
    merged_encoder = peft_encoder.merge_and_unload()

    print("4. Stitching the custom merged Encoder back into the full Seq2Seq model...")
    # Overwrite the vanilla encoder with our newly merged smart encoder!
    seq2seq_model.encoder = merged_encoder

    print(f"5. Saving new custom Seq2Seq base model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    seq2seq_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\nSuccess! You can now use this as the --model argument for training your Decoder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Merge Encoder LoRA into a Base Seq2Seq Model")
    parser.add_argument("--lora-dir", type=str, required=True, help="Path to the trained best_t5_encoder LoRA folder")
    parser.add_argument("--base-model", type=str, default="checkpoints/flan-t5-small")
    parser.add_argument("--output-dir", type=str, default="checkpoints/flan-t5-custom-encoder")
    args = parser.parse_args()
    
    merge_models(args.lora_dir, args.base_model, args.output_dir)
