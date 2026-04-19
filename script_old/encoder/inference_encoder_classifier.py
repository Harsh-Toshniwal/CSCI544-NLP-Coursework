"""
Inference script to test the trained T5 Encoder on classifying if two questions are duplicates.

Example Usage via CLI:

# Test a specific checkpoint on two questions:
python scripts/inference_encoder_classifier.py \
    --checkpoint-dir "checkpoints/encoder_classifier_lora_AdamW_5e-05_ep10" \
    --q1 "How can I invest in the share market?" \
    --q2 "What is the best way to buy stocks?"

# If you trained the model without LoRA, you must pass the --no-lora flag:
python scripts/inference_encoder_classifier.py \
    --checkpoint-dir "checkpoints/encoder_classifier_AdamW_5e-05_ep10" \
    --no-lora \
    --q1 "How can I invest in the share market?" \
    --q2 "What is the best way to buy stocks?"
# Run inference on 50 random questions from test.csv:
python scripts/encoder/inference_encoder_classifier.py \
    --checkpoint-dir "checkpoints/encoder_classifier_lora_AdamW_5e-05_ep10" \
    --batch-test "quora-question-pairs/test.csv" \
    --num-samples 50
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Ensure we can import the model class from the training script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_encoder_classifier import T5EncoderClassifier

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def predict(checkpoint_dir, q1=None, q2=None, batch_test_file=None, num_samples=50, base_model="checkpoints/flan-t5-small", use_lora=True):
    # Verify checkpoint exists
    state_dict_path = os.path.join(checkpoint_dir, "best_encoder.pt")
    if not os.path.exists(state_dict_path):
        print(f"Error: Could not find model weights at {state_dict_path}")
        print("Did you provide the correct --checkpoint-dir?")
        sys.exit(1)

    print(f"Loading model from {state_dict_path} (LoRA: {use_lora})...")
    
    # Try to load the tokenizer saved with the run, otherwise fallback to the base model's tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "best_t5_encoder")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = base_model
        
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Reconstruct the exact model architecture
    model = T5EncoderClassifier(model_path=base_model, use_lora=use_lora)
    
    # Load the trained weights
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    model.to(DEFAULT_DEVICE)
    model.eval()  # Set to evaluation mode (turns off dropout)

    # -----------------------------
    # BATCH TESTING LOGIC
    # -----------------------------
    if batch_test_file:
        print(f"\nProcessing {num_samples} random pairs from {batch_test_file}...")
        df = pd.read_csv(batch_test_file).dropna(subset=['question1', 'question2'])
        samples = df.sample(n=num_samples, random_state=42)
        
        results = []
        with torch.no_grad():
            for idx, row in samples.iterrows():
                text = f"question1: {row['question1']} question2: {row['question2']}"
                encoding = tokenizer(
                    text, max_length=256, padding="max_length", truncation=True, return_tensors="pt"
                )
                input_ids = encoding["input_ids"].to(DEFAULT_DEVICE)
                attention_mask = encoding["attention_mask"].to(DEFAULT_DEVICE)

                logits = model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=1).squeeze()
                pred = torch.argmax(probs).item()
                
                results.append({
                    "Q1": row['question1'],
                    "Q2": row['question2'],
                    "Prediction": "DUPLICATE" if pred == 1 else "NOT DUPLICATE",
                    "Confidence": f"{probs[pred].item() * 100:.1f}%",
                    "Prob_Dup": f"{probs[1].item() * 100:.1f}%"
                })
        
        # Display the results neatly
        for i, res in enumerate(results):
            print(f"\n--- Sample {i+1} ---")
            print(f"Q1: {res['Q1']}")
            print(f"Q2: {res['Q2']}")
            print(f"--> {res['Prediction']} (Confidence: {res['Confidence']})")
        
        # Save to CSV for easy reviewing
        out_name = f"inference_results_{num_samples}_samples.csv"
        pd.DataFrame(results).to_csv(out_name, index=False)
        print(f"\nAll {num_samples} results saved to {out_name}!")
        return

    # -----------------------------
    # SINGLE PAIR LOGIC
    # -----------------------------
    if not q1 or not q2:
        print("Error: You must provide either --q1 and --q2 OR --batch-test")
        sys.exit(1)

    # Format the input exactly as we did during training
    text = f"question1: {q1} question2: {q2}"
    
    encoding = tokenizer(
        text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(DEFAULT_DEVICE)
    attention_mask = encoding["attention_mask"].to(DEFAULT_DEVICE)

    print("\nRunning Inference...")
    print("-" * 50)
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()

    # Present results clearly
    class_label = "DUPLICATE" if pred == 1 else "NOT DUPLICATE"
    conf = probs[pred].item() * 100
    
    print(f"Question 1: {q1}")
    print(f"Question 2: {q2}")
    print("-" * 50)
    print(f"Prediction : {class_label}")
    print(f"Confidence : {conf:.2f}%")
    print(f"Raw Probs  : [0: Not Duplicate = {probs[0].item()*100:.1f}%]  [1: Duplicate = {probs[1].item()*100:.1f}%]")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test T5 Encoder Classifier Inference")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to the directory containing best_encoder.pt")
    parser.add_argument("--q1", type=str, required=False, help="First question (Single test)")
    parser.add_argument("--q2", type=str, required=False, help="Second question (Single test)")
    parser.add_argument("--batch-test", type=str, required=False, help="Path to CSV file to pull random samples from")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of random samples to test if batch-test is provided")
    parser.add_argument("--base-model", type=str, default="checkpoints/flan-t5-small", help="Path to base T5 model")
    parser.add_argument("--no-lora", action="store_true", help="Set this if the model was trained WITHOUT LoRA")
    
    args = parser.parse_args()
    
    predict(
        checkpoint_dir=args.checkpoint_dir,
        q1=args.q1,
        q2=args.q2,
        batch_test_file=args.batch_test,
        num_samples=args.num_samples,
        base_model=args.base_model,
        use_lora=not args.no_lora
    )
