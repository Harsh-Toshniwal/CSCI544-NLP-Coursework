# ParaGen: Controllable Neural Paraphrase Generation with Semantic Preservation

A neural paraphrase generation system that extends fine-tuned T5 with controllable generation techniques focused on length and lexical diversity control, augmented with semantic reranking.

## Project Structure

```
paragen/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration dataclasses
├── data_loader.py              # Data loading and preprocessing
├── model.py                    # T5-based paraphrase model with attribute tokens
├── semantic_reranker.py        # Semantic reranking module
└── evaluation.py               # Evaluation metrics

scripts/
├── train.py                    # Training script
├── inference.py                # Inference and interactive generation
└── evaluate.py                 # Evaluation script

configs/                        # Configuration files (if needed)

data/                          # Data directory (will be populated with datasets)

checkpoints/                   # Model checkpoints during training

results/                       # Evaluation results
```

## Installation

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Training

```bash
# Train with default configuration
python scripts/train.py

# Train using specific device
python scripts/train.py --device cuda

# Resume from checkpoint
python scripts/train.py --resume-from checkpoints/checkpoint_epoch_5
```

### 2. Inference

#### Interactive Mode (default)
```bash
python scripts/inference.py --checkpoint checkpoints/checkpoint_epoch_best
```

Then you can enter sentences interactively:
```
Enter source sentence: The quick brown fox jumps over the lazy dog
--length LONG --diversity CREATIVE
```

#### Batch Mode
```bash
python scripts/inference.py \
  --mode batch \
  --checkpoint checkpoints/checkpoint_epoch_best \
  --input-file input_sentences.txt \
  --output-file output_paraphrases.json
```

### 3. Evaluation

```bash
python scripts/evaluate.py \
  --source-file sources.txt \
  --prediction-file predictions.txt \
  --output-file results.json
```

## Configuration

Edit `paragen/config.py` to customize:

- **DataConfig**: Dataset selection, data paths, batch sizes
- **ModelConfig**: Model architecture, generation parameters
- **TrainingConfig**: Learning rate, epochs, optimization settings
- **RerankerConfig**: Semantic reranking settings
- **AttributeConfig**: Control tokens for length and diversity
- **EvaluationConfig**: Evaluation metrics and settings

## Key Features

### 1. Controllable Generation
- **Length Control**: `[SHORT]`, `[SAME]`, `[LONG]`
- **Diversity Control**: `[CONSERVATIVE]`, `[CREATIVE]`
- Attributes are automatically inferred during training from source-target pair statistics

### 2. Semantic Reranking
- Generates k diverse candidates using diverse beam search
- Reranks using combination of:
  - Sentence-BERT cosine similarity
  - BERTScore F1 score
  - Tunable lambda parameter (default: 0.6)

### 3. Comprehensive Evaluation
- **Semantic Preservation**: Sentence-BERT similarity, BERTScore, LLM-as-a-judge
- **Diversity Metrics**: Inverse BLEU, Self-BLEU, Lexical Diversity Ratio
- **Fluency Metrics**: Perplexity, METEOR
- **Composite Score**: ParaScore balancing semantic and diversity

## Datasets

The system is trained on:
- **QQP (Quora Question Pairs)**: 400,000+ question pairs
- **PAWS (Paraphrase Adversaries)**: 108,463 well-formed paraphrase pairs

Evaluation on:
- **MRPC**: 5,801 high-quality sentence pairs
- **TwitterPPDB**: 51,524 pairs from Twitter
- **Held-out QQP**: In-domain test set

## Module Details

### model.py - ParaphraseModel
```python
from paragen.model import ParaphraseModel

model = ParaphraseModel(model_name="t5-base", device="cuda")

# Generate with controls
paraphrases = model.generate(
    "The quick brown fox jumps over the lazy dog",
    length_control="[LONG]",
    diversity_control="[CREATIVE]"
)
```

### semantic_reranker.py - SemanticReranker
```python
from paragen.semantic_reranker import SemanticReranker

reranker = SemanticReranker(metric="combined", lambda_weight=0.6)

# Rerank candidates
best, ranked_list = reranker.rerank(
    source="The quick brown fox jumps over the lazy dog",
    candidates=[...],
    return_scores=True
)
```

### evaluation.py - ParaphraseEvaluator
```python
from paragen.evaluation import ParaphraseEvaluator

evaluator = ParaphraseEvaluator()

scores = evaluator.evaluate_pair(
    source="The quick brown fox jumps over the lazy dog",
    target="A fast auburn canine leaps across the sleeping dog"
)
```

## Timeline (Week 11-13)

**Week 11 (3/24-3/30)**: 
- ✅ Implement controllable generation with attribute tokens
- ✅ Fine-tune T5 on QQP+PAWS
- □ Conduct validation experiments

**Week 12 (3/31-4/6)**:
- □ Implement diverse beam search
- □ Implement semantic reranking
- □ Hyperparameter tuning
- □ Comprehensive evaluation

**Week 13 (4/7-4/13)**:
- □ Human evaluation (200 samples)
- □ Final analysis and documentation
- □ Presentation preparation

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- torch
- transformers
- datasets
- sentence-transformers
- nltk
- scipy
- scikit-learn

## References

[1] Elgaar, M., & Amiri, H. (2025). Linguistically-Controlled Paraphrase Generation. In Findings of the Association for Computational Linguistics: EMNLP 2025.

[2] Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer (T5).

[3-9] See proposal document for additional references
