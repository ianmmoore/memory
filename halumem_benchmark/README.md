# HaluMem Benchmark

Implementation of the HaluMem benchmark (arXiv:2511.03506) for evaluating memory systems.

## Quick Start

```bash
# Dry run to validate setup
python run.py --dry-run

# Run full benchmark (requires OPENAI_API_KEY)
export OPENAI_API_KEY=your-key
python run.py

# Quick test with 50 QA pairs
python run.py --sample-size 50
```

## Dataset

Downloaded from [IAAR-Shanghai/HaluMem](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem).

| Variant | Dialogues | Memories | QA Pairs | Tokens |
|---------|-----------|----------|----------|--------|
| Long | 2,417 | 14,948 | 3,467 | 20M+ |
| Medium | 1,387 | 14,948 | 3,467 | 6.7M |

### Question Types
- Basic Fact Recall (746)
- Dynamic Update (180)
- Generalization & Application (746)
- Memory Boundary (828)
- Memory Conflict (769)
- Multi-hop Inference (198)

## Cost Estimates

| Phase | With Prefiltering | Without Prefiltering |
|-------|-------------------|---------------------|
| Extraction | ~$2 | ~$2 |
| Updating | ~$1 | ~$1 |
| QA | ~$62 | ~$3,000 |
| **Total** | **~$65** | **~$3,003** |

Prefiltering uses embeddings to select top-100 candidates before LLM scoring, reducing QA costs by ~98%.

## Evaluation Phases

### 1. Extraction
Evaluate memory extraction from conversations:
- **Recall**: % of ground-truth memories extracted
- **Weighted Recall**: Importance-weighted recall
- **Precision**: % of extracted memories that are correct

### 2. Updating
Evaluate conflict resolution when new information arrives:
- **Accuracy**: Correct update decisions
- **Omission Rate**: Missing critical updates
- **Hallucination Rate**: Incorrect updates

### 3. Question Answering
Evaluate answering questions using stored memories:
- **Correctness**: Accurate answers (LLM-as-judge)
- **Hallucination Rate**: Fabricated information
- **Omission Rate**: Missing information

## Architecture

```
halumem_benchmark/
├── config.py      # HaluMemConfig, ModelConfig
├── dataset.py     # HaluMemDataset, data classes
├── metrics.py     # Metric computation
├── extraction.py  # Extraction evaluation
├── updating.py    # Update evaluation
├── qa.py          # QA evaluation
├── runner.py      # HaluMemBenchmark orchestrator
├── run.py         # CLI entry point
├── convert_dataset.py    # Dataset conversion
└── download_dataset.py   # Dataset download
```

## API Usage

```python
import asyncio
from halumem_benchmark import run_benchmark

async def main():
    results = await run_benchmark(
        variant="long",
        prefilter=True,
        sample_size=100  # Optional limit
    )
    results.print_summary()
    results.save("results/my_run.json")

asyncio.run(main())
```

## Custom Model Integration

```python
from halumem_benchmark import HaluMemConfig, HaluMemBenchmark

async def my_model(prompt: str) -> str:
    # Your model implementation
    return response

async def my_embeddings(texts: list[str]) -> list[list[float]]:
    # Your embedding implementation
    return embeddings

config = HaluMemConfig.for_long()
benchmark = HaluMemBenchmark(config)
await benchmark.setup()

benchmark.configure_models(
    extraction_model_fn=my_model,
    scoring_model_fn=my_model,
    primary_model_fn=my_model,
    embedding_fn=my_embeddings
)

results = await benchmark.run_full()
```

## Files

- `data/halumem/long/` - HaluMem-Long dataset
- `data/halumem/medium/` - HaluMem-Medium dataset
- `results/` - Benchmark results (created on run)
