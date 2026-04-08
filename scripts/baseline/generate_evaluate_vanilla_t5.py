"""Run vanilla T5-base on validation set and evaluate"""

import argparse
import logging
import json
import torch
from tqdm import tqdm
from baseline.train_baseline import VanillaT5Model
from paragen.evaluation import ParaphraseEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Auto-detect CUDA
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sentences(file_path):
    """Load sentences from text file (one per line)"""
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences


def run_vanilla_t5_evaluation(
    val_sources_file="./data/splits_no_test/val_sources.txt",
    val_targets_file="./data/splits_no_test/val_targets.txt",
    output_dir="./results/vanilla_t5_baseline",
    model_name="google/flan-t5-base",
    num_beams=5,
    device=None,
):
    if device is None:
        device = DEFAULT_DEVICE
    """
    Run vanilla T5-base on validation set and evaluate
    
    Args:
        val_sources_file: Path to validation source sentences
        val_targets_file: Path to validation target (reference) sentences
        output_dir: Directory to save results
        model_name: T5 model to use (default: t5-base, untrained)
        num_beams: Number of beams for generation
        device: Device to use (cuda or cpu)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load validation data
    logger.info("Loading validation data...")
    val_sources = load_sentences(val_sources_file)
    val_targets = load_sentences(val_targets_file)
    
    assert len(val_sources) == len(val_targets), "Mismatch in source and target counts"
    logger.info(f"Loaded {len(val_sources)} validation pairs")
    
    # Initialize vanilla T5 model
    logger.info(f"Loading {model_name} (no fine-tuning on your data)...")
    model = VanillaT5Model(model_name=model_name, device=device)
    
    # Generate paraphrases
    logger.info("Generating paraphrases...")
    predictions = []
    
    # Add instruction prefix for FLAN-T5 (instruction-tuned model)
    instruction_prefix = "Generate Paraphrase, if it is a question, don't answer it. Rephrase it/rewrite it using other words, : " if "flan" in model_name.lower() else ""
    
    for i, source in enumerate(tqdm(val_sources)):
        if (i + 1) % 500 == 0:
            logger.info(f"Generated {i + 1}/{len(val_sources)} paraphrases")
        
        # Add instruction prefix for better results with FLAN-T5
        input_text = instruction_prefix + source
        paraphrase = model.generate(input_text, num_beams=num_beams, max_length=128)
        predictions.append({
            "source": source,
            "reference": val_targets[i],
            "generated": paraphrase,
        })
    
    # Save predictions
    pred_file = os.path.join(output_dir, "predictions.json")
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions to {pred_file}")
    
    # Evaluate
    logger.info("Evaluating predictions")
    evaluator = ParaphraseEvaluator()
    
    sources = [p["source"] for p in predictions]
    generated = [p["generated"] for p in predictions]
    
    all_scores, summary = evaluator.evaluate_batch(
        sources, generated, compute_all=True
    )
    
    # Save evaluation results
    eval_file = os.path.join(output_dir, "evaluation.json")
    results = {
        "model": model_name,
        "num_samples": len(predictions),
        "summary": summary,
        "detailed_scores": {metric: scores for metric, scores in all_scores.items()},
    }
    
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved evaluation results to {eval_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"VANILLA T5 EVALUATION SUMMARY ({model_name})")
    print("=" * 80)
    print(f"Validation samples: {len(predictions)}")
    print("\nMetrics:")
    for metric, value in summary.items():
        print(f"  {metric}: {value:.4f}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run T5-base on validation set and evaluate"
    )
    parser.add_argument(
        "--val-sources",
        type=str,
        default="./data/splits_no_test/val_sources.txt",
        help="Path to validation source sentences",
    )
    parser.add_argument(
        "--val-targets",
        type=str,
        default="./data/splits_no_test/val_targets.txt",
        help="Path to validation target sentences",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/vanilla_t5_baseline",
        help="Directory to save results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-base",
        help="T5 model name (default: google/flan-t5-base instruction-tuned)",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Number of beams for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device to use (default: cuda if available, else cpu)",
    )
    
    args = parser.parse_args()
    
    run_vanilla_t5_evaluation(
        val_sources_file=args.val_sources,
        val_targets_file=args.val_targets,
        output_dir=args.output_dir,
        model_name=args.model,
        num_beams=args.num_beams,
        device=args.device,
    )
