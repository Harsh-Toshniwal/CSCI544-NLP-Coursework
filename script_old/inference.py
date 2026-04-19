"""Inference script for ParaGen"""

import argparse
import logging
from paragen.model import ParaphraseModel
from paragen.semantic_reranker import SemanticReranker
from paragen.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_paraphrases_with_reranking(
    model: ParaphraseModel,
    reranker: SemanticReranker,
    source_text: str,
    length_control: str = None,
    diversity_control: str = None,
    num_candidates: int = 5,
    use_reranking: bool = True,
):
    """
    Generate paraphrases with optional semantic reranking

    Args:
        model: Paraphrase generation model
        reranker: Semantic reranker
        source_text: Source sentence
        length_control: Length control token
        diversity_control: Diversity control token
        num_candidates: Number of candidates to generate
        use_reranking: Whether to apply semantic reranking

    Returns:
        Best paraphrase and optionally all ranked candidates
    """
    # Generate candidates
    candidates = model.generate(
        source_text,
        length_control=length_control,
        diversity_control=diversity_control,
        num_beams=num_candidates,
        num_return_sequences=num_candidates,
        diversity_penalty=0.5,
    )

    # Rerank if requested
    if use_reranking and len(candidates) > 1:
        best, ranked = reranker.rerank(source_text, candidates, return_scores=True)
        return best, ranked
    else:
        return candidates[0], [(c, None) for c in candidates]


def interactive_inference(model_checkpoint: str = None, use_reranking: bool = True):
    """Interactive inference mode"""
    config = get_config()

    logger.info("Loading model...")
    model_name = model_checkpoint or config.model.model_name
    model = ParaphraseModel(
        model_name=model_name,
        device=config.training.device,
    )

    if use_reranking:
        logger.info("Loading semantic reranker...")
        reranker = SemanticReranker(
            model_name=config.reranker.reranker_model,
            metric=config.reranker.rerank_metric,
            lambda_weight=config.reranker.lambda_weight,
        )
    else:
        reranker = None

    print("\n" + "=" * 80)
    print("ParaGen Interactive Inference")
    print("=" * 80)
    print("Commands:")
    print("  - Enter a sentence to generate paraphrases")
    print("  - Use flags: --length [SHORT|SAME|LONG] --diversity [CONSERVATIVE|CREATIVE]")
    print("  - Type 'quit' to exit")
    print("=" * 80 + "\n")

    while True:
        try:
            user_input = input("Enter source sentence (or 'quit' to exit): ").strip()

            if user_input.lower() == "quit":
                break

            # Parse flags
            length_control = None
            diversity_control = None

            if "--length" in user_input:
                parts = user_input.split("--length")
                user_input = parts[0].strip()
                flags = parts[1].strip().split()
                if flags:
                    length_control = flags[0]

            if "--diversity" in user_input:
                parts = user_input.split("--diversity")
                user_input = parts[0].strip()
                flags = parts[1].strip().split()
                if flags:
                    diversity_control = flags[0]

            if not user_input:
                continue

            print("\nGenerating paraphrases...")

            best, ranked = generate_paraphrases_with_reranking(
                model,
                reranker if use_reranking else None,
                user_input,
                length_control=length_control,
                diversity_control=diversity_control,
                num_candidates=5,
                use_reranking=use_reranking,
            )

            print("\n" + "-" * 80)
            print(f"Source: {user_input}")
            print(f"\nBest Paraphrase: {best}")

            if use_reranking and ranked:
                print("\nAll Candidates (ranked):")
                for i, (candidate, score) in enumerate(ranked, 1):
                    if score:
                        print(f"  {i}. {candidate} (score: {score:.4f})")
                    else:
                        print(f"  {i}. {candidate}")

            print("-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            continue


def batch_inference(
    model_checkpoint: str,
    input_file: str,
    output_file: str,
    use_reranking: bool = True,
):
    """Batch inference from file"""
    import json

    config = get_config()

    logger.info("Loading model...")
    model = ParaphraseModel(
        model_name=model_checkpoint,
        device=config.training.device,
    )

    if use_reranking:
        logger.info("Loading semantic reranker...")
        reranker = SemanticReranker(
            model_name=config.reranker.reranker_model,
            metric=config.reranker.rerank_metric,
        )
    else:
        reranker = None

    # Read input
    with open(input_file, "r") as f:
        sentences = [line.strip() for line in f if line.strip()]

    logger.info(f"Processing {len(sentences)} sentences...")

    results = []
    for i, source in enumerate(sentences):
        logger.info(f"Processing {i + 1}/{len(sentences)}...")

        best, ranked = generate_paraphrases_with_reranking(
            model,
            reranker if use_reranking else None,
            source,
            num_candidates=5,
            use_reranking=use_reranking,
        )

        results.append(
            {
                "source": source,
                "best_paraphrase": best,
                "candidates": [c for c, _ in ranked] if ranked else [best],
            }
        )

    # Write output
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ParaGen Inference")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch"],
        default="interactive",
        help="Inference mode",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file for batch inference (one sentence per line)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output.json",
        help="Output file for batch inference",
    )
    parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Disable semantic reranking",
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        interactive_inference(
            model_checkpoint=args.checkpoint,
            use_reranking=not args.no_reranking,
        )
    else:  # batch
        if not args.input_file:
            raise ValueError("--input-file required for batch mode")
        batch_inference(
            model_checkpoint=args.checkpoint or "t5-base",
            input_file=args.input_file,
            output_file=args.output_file,
            use_reranking=not args.no_reranking,
        )
