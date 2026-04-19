"""Evaluation script for ParaGen"""

import argparse
import json
import logging
from typing import List
from paragen.evaluation import ParaphraseEvaluator
from paragen.semantic_reranker import SemanticReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_predictions(
    source_file: str,
    prediction_file: str,
    reference_file: str = None,
    output_file: str = "evaluation_results.json",
):
    """
    Evaluate generated paraphrases against references

    Args:
        source_file: File with source sentences
        prediction_file: File with generated paraphrases (JSON or plain text)
        reference_file: File with reference paraphrases
        output_file: Output file for results
    """
    logger.info("Loading files...")

    # Load sources
    with open(source_file, "r") as f:
        sources = [line.strip() for line in f if line.strip()]

    # Load predictions
    try:
        with open(prediction_file, "r") as f:
            pred_data = json.load(f)
            if isinstance(pred_data, list) and isinstance(pred_data[0], dict):
                predictions = [item.get("best_paraphrase", "") for item in pred_data]
            else:
                predictions = pred_data
    except json.JSONDecodeError:
        with open(prediction_file, "r") as f:
            predictions = [line.strip() for line in f if line.strip()]

    # Load references if provided
    references = None
    if reference_file:
        with open(reference_file, "r") as f:
            references = [line.strip() for line in f if line.strip()]

    assert len(sources) == len(predictions), "Mismatch in number of sources and predictions"

    logger.info(f"Evaluating {len(sources)} paraphrase pairs...")

    # Initialize evaluator
    evaluator = ParaphraseEvaluator()

    # Evaluate pairs
    all_scores, summary = evaluator.evaluate_batch(sources, predictions, compute_all=True)

    # Compile results
    results = {
        "num_samples": len(sources),
        "summary": summary,
        "detailed_scores": {
            metric: scores for metric, scores in all_scores.items()
        },
    }

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 80)

    return results


def evaluate_with_human_scores(
    prediction_file: str,
    human_scores_file: str,
    output_file: str = "human_eval_analysis.json",
):
    """
    Analyze correlation between automatic metrics and human scores

    Args:
        prediction_file: File with predictions and automatic scores
        human_scores_file: File with human annotation scores (JSON)
        output_file: Output file for analysis
    """
    logger.info("Loading files...")

    # Load predictions with scores
    with open(prediction_file, "r") as f:
        pred_data = json.load(f)

    # Load human scores
    with open(human_scores_file, "r") as f:
        human_data = json.load(f)

    logger.info("Computing correlations with human judgments...")

    # This requires computing correlations - placeholder for now
    results = {
        "num_samples": len(pred_data),
        "correlations": {},
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Analysis saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ParaGen outputs")
    parser.add_argument(
        "--source-file",
        type=str,
        required=True,
        help="File with source sentences",
    )
    parser.add_argument(
        "--prediction-file",
        type=str,
        required=True,
        help="File with generated paraphrases",
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        help="File with reference paraphrases",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation_results.json",
        help="Output file",
    )

    args = parser.parse_args()

    evaluate_predictions(
        source_file=args.source_file,
        prediction_file=args.prediction_file,
        reference_file=args.reference_file,
        output_file=args.output_file,
    )
