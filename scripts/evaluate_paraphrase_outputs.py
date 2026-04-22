"""
Evaluation script for paraphrase generation outputs.

This script:
1. Loads generated paraphrases (from CSV or JSON)
2. Loads reference paraphrases if available (from same CSV)
3. Evaluates them using the ParaphraseEvaluator metrics
4. Computes semantic similarity, diversity, fluency scores
5. Saves evaluation results in standard format
6. Generates predictions JSON with all metrics

Evaluation metrics computed:
- Semantic similarity (Sentence-BERT)
- Inverse BLEU (diversity measure)
- Lexical diversity ratio
- METEOR score
- Perplexity (fluency)
- ParaScore (composite metric)

Example Usage:

# Evaluate CSV with source, reference, and generated outputs
python scripts/evaluate_paraphrase_outputs.py \
    --predictions results/generated_outputs.csv \
    --output-dir results/evaluation_outputs \
    --output-name encoder_lora_eval \
    --source-col source \
    --generated-col generated_output \
    --reference-col target

# Auto-detect columns (will look for source, generated_output, target)
python scripts/evaluate_paraphrase_outputs.py \
    --predictions results/generated_outputs.csv \
    --output-dir results/evaluation_outputs \
    --output-name encoder_lora_eval

# With custom model name
python scripts/evaluate_paraphrase_outputs.py \
    --predictions results/generated_outputs.csv \
    --output-dir results/evaluation_outputs \
    --output-name encoder_lora_eval \
    --model-name "LoRA Encoder + Vanilla Decoder"
"""

import argparse
import json
import logging
import os
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

from paragen.evaluation import ParaphraseEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_predictions(predictions_file: str, source_col: str = None, 
                     generated_col: str = None, reference_col: str = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Load predictions from CSV or JSON file.
    Supports loading source, generated output, and optional reference paraphrases.
    
    Args:
        predictions_file: Path to predictions file (CSV or JSON)
        source_col: Column name for source text (auto-detected if None)
        generated_col: Column name for generated text (auto-detected if None)
        reference_col: Column name for reference paraphrases (auto-detected if None)
    
    Returns:
        Tuple of (sources, generated_outputs, references)
    """
    file_ext = os.path.splitext(predictions_file)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(predictions_file)
        
        # Auto-detect source column
        if source_col is None:
            for col in ['source', 'sentence_a', 'input_text', 'original']:
                if col in df.columns:
                    source_col = col
                    break
        
        # Auto-detect generated column
        if generated_col is None:
            for col in ['generated_output', 'generated', 'prediction']:
                if col in df.columns:
                    generated_col = col
                    break
        
        # Auto-detect reference column
        if reference_col is None:
            for col in ['target', 'sentence_b', 'reference', 'paraphrase']:
                if col in df.columns:
                    reference_col = col
                    break
        
        if source_col is None or generated_col is None:
            available_cols = list(df.columns)
            raise ValueError(f"Could not auto-detect source/generated columns. Available: {available_cols}")
        
        sources = df[source_col].tolist()
        generated = df[generated_col].tolist()
        references = df[reference_col].tolist() if reference_col and reference_col in df.columns else None
        
        logger.info(f"Loaded {len(sources)} predictions from CSV")
        if references:
            logger.info(f"Loaded {len([r for r in references if r])} reference paraphrases")
        
    elif file_ext == '.json':
        with open(predictions_file, 'r') as f:
            data = json.load(f)
        
        # Handle list of dicts format
        if isinstance(data, list):
            sources = [item.get('source', item.get('input', '')) for item in data]
            generated = [item.get('generated', item.get('prediction', '')) for item in data]
            references = [item.get('target', item.get('reference', '')) for item in data]
        else:
            raise ValueError("JSON format not recognized")
        
        logger.info(f"Loaded {len(sources)} predictions from JSON")
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    return sources, generated, references


def load_references(references_file: str, source_col: str = None,
                   reference_col: str = None) -> Dict[str, str]:
    """
    Load reference paraphrases.
    
    Args:
        references_file: Path to references file
        source_col: Column with source text
        reference_col: Column with reference paraphrases
    
    Returns:
        Dictionary mapping source -> reference
    """
    df = pd.read_csv(references_file)
    
    if source_col is None:
        for col in ['source', 'sentence_a', 'input_text']:
            if col in df.columns:
                source_col = col
                break
    
    if reference_col is None:
        for col in ['reference', 'paraphrase', 'target', 'sentence_b']:
            if col in df.columns:
                reference_col = col
                break
    
    if source_col is None or reference_col is None:
        raise ValueError("Could not find source/reference columns")
    
    reference_dict = dict(zip(df[source_col], df[reference_col]))
    logger.info(f"Loaded {len(reference_dict)} references")
    
    return reference_dict


def evaluate_outputs(sources: List[str], generated: List[str], 
                    references: List[str] = None,
                    model_name: str = "Unknown Model") -> Tuple[Dict, Dict]:
    """
    Evaluate generated outputs.
    
    Args:
        sources: List of source sentences
        generated: List of generated paraphrases
        references: Optional list of reference paraphrases
        model_name: Name of the model for reporting
    
    Returns:
        Tuple of (all_scores, summary)
    """
    logger.info("Initializing evaluator...")
    evaluator = ParaphraseEvaluator()
    
    # Use references for evaluation if available, otherwise use source
    comparison_texts = references if references else sources
    
    eval_mode = "against references" if references else "against source (diversity)"
    logger.info(f"Evaluating {len(generated)} outputs {eval_mode}...")
    all_scores, summary = evaluator.evaluate_batch(comparison_texts, generated, compute_all=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info(f"Mode: Comparing generated outputs {eval_mode}")
    logger.info("=" * 80)
    for metric, value in summary.items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")
    logger.info("=" * 80 + "\n")
    
    return all_scores, summary


def create_predictions_json(sources: List[str], generated: List[str], 
                           all_scores: Dict[str, List[float]],
                           references: List[str] = None,
                           model_name: str = "Unknown Model") -> List[Dict]:
    """
    Create predictions JSON in format compatible with predictions_t5_base.json.
    
    Args:
        sources: List of source sentences
        generated: List of generated paraphrases
        all_scores: Dictionary of metric scores
        references: Optional list of reference paraphrases
        model_name: Model name for reporting
    
    Returns:
        List of prediction dictionaries
    """
    predictions = []
    
    for i, (source, gen_text) in enumerate(zip(sources, generated)):
        pred_dict = {
            "source": source,
            "generated": gen_text,
        }
        
        # Add reference if available
        if references and i < len(references):
            pred_dict["reference"] = references[i]
        
        # Add individual scores
        for metric, scores in all_scores.items():
            if i < len(scores):
                pred_dict[metric] = scores[i]
        
        predictions.append(pred_dict)
    
    return predictions


def save_evaluation_results(all_scores: Dict[str, List[float]], 
                           summary: Dict[str, float],
                           output_dir: str, output_name: str,
                           model_name: str = "Unknown Model",
                           num_samples: int = 0):
    """
    Save evaluation results in standard format.
    
    Args:
        all_scores: Dictionary of per-sample scores
        summary: Summary statistics
        output_dir: Output directory
        output_name: Base name for output files
        model_name: Model name
        num_samples: Number of samples evaluated
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results dictionary
    results = {
        "model": model_name,
        "num_samples": num_samples,
        "summary": summary,
        "detailed_scores": all_scores
    }
    
    # Save evaluation results
    eval_output = os.path.join(output_dir, f"evaluation_{output_name}.json")
    with open(eval_output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved to {eval_output}")
    
    return eval_output


def save_predictions_json(predictions: List[Dict], output_dir: str, 
                         output_name: str):
    """
    Save predictions with scores.
    
    Args:
        predictions: List of prediction dictionaries
        output_dir: Output directory
        output_name: Base name for output file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pred_output = os.path.join(output_dir, f"predictions_{output_name}.json")
    with open(pred_output, 'w') as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Predictions saved to {pred_output}")
    
    return pred_output


def generate_text_report(sources: List[str], generated: List[str],
                        all_scores: Dict[str, List[float]],
                        summary: Dict[str, float],
                        output_dir: str, output_name: str,
                        references: List[str] = None,
                        model_name: str = "Unknown Model"):
    """
    Generate a human-readable text report.
    
    Args:
        sources: List of source sentences
        generated: List of generated paraphrases
        all_scores: Dictionary of scores
        summary: Summary statistics
        output_dir: Output directory
        output_name: Base name for output file
        references: Optional list of reference paraphrases
        model_name: Model name
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, f"report_{output_name}.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"PARAPHRASE EVALUATION REPORT\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Number of samples: {len(sources)}\n")
        if references:
            f.write(f"Evaluation mode: Against reference paraphrases\n")
        else:
            f.write(f"Evaluation mode: Diversity (against source)\n")
        f.write("=" * 100 + "\n\n")
        
        # Summary metrics
        f.write("SUMMARY METRICS\n")
        f.write("-" * 100 + "\n")
        for metric, value in summary.items():
            if isinstance(value, float):
                f.write(f"{metric:.<50} {value:>10.4f}\n")
            else:
                f.write(f"{metric:.<50} {str(value):>10}\n")
        f.write("\n")
        
        # Per-metric statistics
        f.write("DETAILED METRIC STATISTICS\n")
        f.write("-" * 100 + "\n")
        for metric, scores in all_scores.items():
            if scores:
                f.write(f"\n{metric}:\n")
                f.write(f"  Mean:   {np.mean(scores):.4f}\n")
                f.write(f"  Std:    {np.std(scores):.4f}\n")
                f.write(f"  Min:    {np.min(scores):.4f}\n")
                f.write(f"  Max:    {np.max(scores):.4f}\n")
                f.write(f"  Median: {np.median(scores):.4f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("SAMPLE PREDICTIONS (First 10)\n")
        f.write("=" * 100 + "\n\n")
        
        for i in range(min(10, len(sources))):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  Source:    {sources[i][:80]}...\n" if len(sources[i]) > 80 else f"  Source:    {sources[i]}\n")
            if references and i < len(references):
                f.write(f"  Reference: {references[i][:80]}...\n" if len(references[i]) > 80 else f"  Reference: {references[i]}\n")
            f.write(f"  Generated: {generated[i][:80]}...\n" if len(generated[i]) > 80 else f"  Generated: {generated[i]}\n")
            
            # Add scores for this sample
            for metric, scores in all_scores.items():
                if i < len(scores):
                    f.write(f"  {metric}: {scores[i]:.4f}\n")
            f.write("\n")
    
    logger.info(f"Report saved to {report_path}")
    return report_path


def main(predictions_file: str, output_dir: str = "results/evaluation_outputs",
         output_name: str = "predictions", model_name: str = None,
         source_col: str = None, generated_col: str = None,
         reference_col: str = None, generate_report: bool = True):
    """
    Main evaluation pipeline.
    
    Args:
        predictions_file: Path to predictions file
        output_dir: Output directory for results
        output_name: Base name for output files
        model_name: Model name for reporting
        source_col: Source column name
        generated_col: Generated column name
        reference_col: Reference column name
        generate_report: Whether to generate text report
    """
    logger.info(f"Loading predictions from {predictions_file}...")
    sources, generated, references = load_predictions(
        predictions_file, source_col, generated_col, reference_col
    )
    
    if model_name is None:
        model_name = os.path.basename(os.path.dirname(predictions_file))
    
    # Evaluate
    all_scores, summary = evaluate_outputs(sources, generated, references, model_name)
    
    # Save evaluation results
    save_evaluation_results(all_scores, summary, output_dir, output_name,
                           model_name, len(sources))
    
    # Create and save predictions JSON
    predictions = create_predictions_json(sources, generated, all_scores, references, model_name)
    save_predictions_json(predictions, output_dir, output_name)
    
    # Generate text report
    if generate_report:
        generate_text_report(sources, generated, all_scores, summary,
                            output_dir, output_name, references, model_name)
    
    logger.info(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate paraphrase generation outputs")
    parser.add_argument("--predictions", type=str, required=True,
                       help="Path to predictions CSV or JSON file")
    parser.add_argument("--output-dir", type=str, default="results/evaluation_outputs",
                       help="Output directory for results")
    parser.add_argument("--output-name", type=str, default="predictions",
                       help="Base name for output files")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Model name for reporting")
    parser.add_argument("--source-col", type=str, default=None,
                       help="Source column name")
    parser.add_argument("--generated-col", type=str, default=None,
                       help="Generated output column name")
    parser.add_argument("--reference-col", type=str, default=None,
                       help="Reference paraphrase column name (optional)")
    parser.add_argument("--no-report", action="store_true",
                       help="Skip generating text report")
    
    args = parser.parse_args()
    
    main(
        predictions_file=args.predictions,
        output_dir=args.output_dir,
        output_name=args.output_name,
        model_name=args.model_name,
        source_col=args.source_col,
        generated_col=args.generated_col,
        reference_col=args.reference_col,
        generate_report=not args.no_report
    )
