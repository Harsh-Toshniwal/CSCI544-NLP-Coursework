"""
Inference script to test the trained Decoder LoRA on paraphrasing.

Example Usage:
python scripts/decoder/inference_decoder.py \
    --lora-dir "checkpoints/decoder_lora_AdamW_5e-05_ep10/best_decoder_lora" \
    --input "How do I invest in the stock market?"
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_paraphrase(lora_dir, text, base_model="checkpoints/flan-t5-small"):
    if not os.path.exists(lora_dir):
        print(f"Error: Could not find LoRA weights at {lora_dir}")
        return

    print(f"Loading Base Model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(lora_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    
    print(f"Loading Decoder LoRA weights from: {lora_dir}")
    model = PeftModel.from_pretrained(model, lora_dir)
    model.to(DEFAULT_DEVICE)
    model.eval()
    
    # Prefix input text
    input_text = f"paraphrase: {text}"
    print(f"\nOriginal Input : {text}")
    print("-" * 50)
    
    encoding = tokenizer(
        input_text, return_tensors="pt", max_length=256, truncation=True
    )
    input_ids = encoding["input_ids"].to(DEFAULT_DEVICE)
    attention_mask = encoding["attention_mask"].to(DEFAULT_DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
    paraphrase = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Paraphrase Cnd : {paraphrase}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test T5 Decoder Paraphrase Inference")
    parser.add_argument("--lora-dir", type=str, required=True, help="Path to best_decoder_lora")
    parser.add_argument("--input", type=str, required=True, help="Sentence to paraphrase")
    parser.add_argument("--base-model", type=str, default="checkpoints/flan-t5-small")
    
    args = parser.parse_args()
    
    generate_paraphrase(
        lora_dir=args.lora_dir,
        text=args.input,
        base_model=args.base_model
    )
