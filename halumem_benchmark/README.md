# HaluMem Benchmark Suite

Implementation of the HaluMem benchmark (arXiv:2511.03506) for evaluating memory systems on extraction, updating, and question answering tasks.

## Overview

HaluMem is the first benchmark specifically designed to evaluate hallucinations in memory systems at the **operation level**. Unlike end-to-end benchmarks, HaluMem tests three fundamental capabilities independently:

1. **Memory Extraction**: How well does the system extract memories from conversations?
2. **Memory Updating**: How accurately does the system handle conflicting information?
3. **Question Answering**: How correctly does the system answer questions using stored memories?

This granular approach enables identification of specific failure modes in memory systems.

## Dataset

Downloaded from [IAAR-Shanghai/HaluMem](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem).

| Variant | Dialogues | Memories | QA Pairs | Tokens | Use Case |
|---------|-----------|----------|----------|--------|----------|
| Long | 2,417 | 14,948 | 3,467 | 20M+ | Production evaluation |
| Medium | 1,387 | 14,948 | 3,467 | 6.7M | Quick testing |

### Question Types

| Type | Count | Description |
|------|-------|-------------|
| Basic Fact Recall | 746 | Direct retrieval of stored facts |
| Dynamic Update | 180 | Questions about updated information |
| Generalization & Application | 746 | Applying knowledge to new situations |
| Memory Boundary | 828 | Testing what the system doesn't know |
| Memory Conflict | 769 | Resolving contradictory information |
| Multi-hop Inference | 198 | Reasoning across multiple memories |

## Installation

```bash
# From the memory project root
pip install -r requirements.txt

# Verify installation
python -c "from halumem_benchmark import run_benchmark; print('OK')"
```

## Quick Start

### Dry Run (Validate Setup)

```bash
python run.py --dry-run
```

This validates your setup without making API calls:
- Loads and parses the dataset
- Displays statistics and question type distribution
- Shows cost estimates

### Run Benchmark

```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"

# Full HaluMem-Long with prefiltering (recommended)
python run.py

# HaluMem-Medium (faster, cheaper)
python run.py --variant medium

# Quick test with limited samples
python run.py --sample-size 100

# Without prefiltering (WARNING: ~$3,000!)
python run.py --no-prefilter
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--variant` | Dataset variant (`medium` or `long`) | `long` |
| `--no-prefilter` | Disable embedding prefiltering | Enabled |
| `--sample-size N` | Limit QA pairs for testing | All (3,467) |
| `--dry-run` | Validate setup only | False |
| `-v, --verbose` | Enable debug logging | False |

## Cost Estimates

| Phase | With Prefiltering | Without Prefiltering |
|-------|-------------------|---------------------|
| Extraction | ~$2 | ~$2 |
| Updating | ~$1 | ~$1 |
| QA | ~$62 | ~$3,000 |
| **Total** | **~$65** | **~$3,003** |

**Prefiltering** uses embeddings to select top-100 candidates before LLM scoring, reducing QA costs by ~98%.

## Evaluation Phases

### Phase 1: Memory Extraction

Evaluates how well the system extracts memories from dialogues.

**Metrics:**
- **Recall**: Proportion of ground-truth memories that were extracted
- **Weighted Recall**: Recall weighted by memory importance (critical facts weighted higher)
- **Precision**: Proportion of extracted memories that are correct (not hallucinated)

### Phase 2: Memory Updating

Evaluates conflict resolution when new information contradicts existing memories.

**Metrics:**
- **Accuracy**: Proportion of correct update decisions (ADD/UPDATE/DELETE/NOOP)
- **Omission Rate**: How often the system fails to update when it should
- **Hallucination Rate**: How often the system creates false updates

### Phase 3: Question Answering

Evaluates end-to-end question answering using the memory system.

**Metrics:**
- **Correctness**: Proportion of correctly answered questions (LLM-as-judge)
- **Hallucination Rate**: How often answers contain fabricated information
- **Omission Rate**: How often correct information is missing from answers

## Architecture

```
halumem_benchmark/
├── __init__.py          # Package exports
├── config.py            # HaluMemConfig, ModelConfig dataclasses
├── dataset.py           # HaluMemDataset, DialogueTurn, MemoryPoint, QAPair
├── metrics.py           # ExtractionMetrics, UpdateMetrics, QAMetrics
├── extraction.py        # ExtractionEvaluator, create_llm_extraction_fn
├── updating.py          # UpdateEvaluator, UpdateDecision, create_llm_update_fn
├── qa.py                # QAEvaluator, QAResult, create_llm_judge_fn
├── runner.py            # HaluMemBenchmark orchestrator, BenchmarkResults
├── run.py               # CLI entry point
├── batch_api.py         # OpenAI Batch API utilities
├── convert_dataset.py   # Dataset format conversion
└── download_dataset.py  # Dataset download from HuggingFace
```

## API Usage

### Quick Start

```python
import asyncio
from halumem_benchmark import run_benchmark

async def main():
    results = await run_benchmark(
        variant="long",
        prefilter=True,
        sample_size=100  # Optional limit for testing
    )
    results.print_summary()
    results.save("results/my_run.json")

asyncio.run(main())
```

### Custom Configuration

```python
from halumem_benchmark import HaluMemConfig, HaluMemBenchmark

async def custom_benchmark():
    # Create custom configuration
    config = HaluMemConfig(
        variant="long",
        relevance_threshold=0.75,
        max_memories=50,
        batch_size=20,
        prefilter_enabled=True,
        prefilter_top_k=100,
        sample_size=500
    )

    # Initialize benchmark
    benchmark = HaluMemBenchmark(config)
    await benchmark.setup()

    # Configure models
    benchmark.configure_models(
        extraction_model_fn=your_extraction_model,
        scoring_model_fn=your_small_model,
        primary_model_fn=your_primary_model,
        judge_model_fn=your_judge_model,     # Optional
        embedding_fn=your_embedding_fn        # Optional, for prefiltering
    )

    # Run full benchmark
    results = await benchmark.run_full()

    # Access individual metrics
    print(f"Extraction Recall: {results.extraction['recall']:.2%}")
    print(f"Update Accuracy: {results.updating['accuracy']:.2%}")
    print(f"QA Correctness: {results.qa['correctness']:.2%}")

asyncio.run(custom_benchmark())
```

### Custom Model Integration

```python
async def my_model(prompt: str) -> str:
    """Your LLM implementation."""
    # Call your API
    return response

async def my_embeddings(texts: list[str]) -> list[list[float]]:
    """Your embedding implementation."""
    # Generate embeddings
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

## Configuration Reference

### HaluMemConfig

```python
@dataclass
class HaluMemConfig:
    variant: str = "long"                    # "medium" or "long"
    dataset_path: Path = "./data/halumem"    # Path to dataset files
    output_path: Path = "./results/halumem"  # Path for results

    # Memory system settings
    relevance_threshold: float = 0.7         # Minimum relevance score (0-1)
    max_memories: int = 50                   # Max memories per query
    batch_size: int = 20                     # Concurrent API calls

    # Prefiltering (cost optimization)
    prefilter_enabled: bool = True           # Use embedding prefilter
    prefilter_top_k: int = 100               # Candidates before LLM scoring

    # Evaluation settings
    num_runs: int = 3                        # Runs for mean/std calculation
    sample_size: Optional[int] = None        # Limit QA pairs (None = all)

    # Factory methods
    @classmethod
    def for_medium(cls) -> "HaluMemConfig": ...
    @classmethod
    def for_long(cls) -> "HaluMemConfig": ...
    @classmethod
    def for_quick_test(cls) -> "HaluMemConfig": ...
```

### ModelConfig

```python
@dataclass
class ModelConfig:
    extraction_model: str = "gpt-5.1"              # Memory extraction
    scoring_model: str = "gpt-5-nano"              # Relevance scoring (fast/cheap)
    primary_model: str = "gpt-5.1"                 # QA responses
    judge_model: str = "gpt-5.1"                   # Answer evaluation
    embedding_model: str = "text-embedding-3-small"

    # API settings
    api_key: Optional[str] = None
    max_retries: int = 3
    retry_delay: float = 1.0
```

## Output Format

Results are saved as JSON and printed to console:

```
============================================================
HaluMem Benchmark Results
============================================================
Run ID: halumem_20240115_143022
Variant: long
Prefilter: enabled

EXTRACTION METRICS:
  Recall:          53.02%
  Weighted Recall: 70.73%
  Precision:       85.82%

UPDATE METRICS:
  Accuracy:        17.01%
  Omission Rate:   82.42%
  Hallucination:   0.58%

QA METRICS:
  Correctness:     53.77%
  Hallucination:   22.21%
  Omission:        24.02%

TIMING:
  extraction: 1234.5s
  updating: 89.2s
  qa: 4567.8s

ESTIMATED COST:
  extraction: $2.00
  updating: $1.00
  qa: $62.00
  TOTAL: $65.00
============================================================
```

## API Reference

### Core Classes

#### HaluMemBenchmark

Main orchestrator class for running the benchmark.

```python
class HaluMemBenchmark:
    def __init__(self, config: HaluMemConfig): ...
    async def setup(self) -> None: ...
    def configure_models(
        self,
        extraction_model_fn: Callable[[str], Awaitable[str]],
        scoring_model_fn: Callable[[str], Awaitable[str]],
        primary_model_fn: Callable[[str], Awaitable[str]],
        judge_model_fn: Optional[Callable] = None,
        embedding_fn: Optional[Callable] = None
    ) -> None: ...
    async def run_extraction(self) -> ExtractionMetrics: ...
    async def run_updating(self) -> UpdateMetrics: ...
    async def run_qa(self) -> QAMetrics: ...
    async def run_full(self) -> BenchmarkResults: ...
```

#### BenchmarkResults

Contains all benchmark results with serialization support.

```python
@dataclass
class BenchmarkResults:
    config: Dict[str, Any]
    extraction: Dict[str, float]
    updating: Dict[str, Any]
    qa: Dict[str, Any]
    timing: Dict[str, float]
    cost_estimate: Dict[str, float]
    timestamp: str
    run_id: str

    def to_dict(self) -> Dict[str, Any]: ...
    def save(self, path: Path) -> None: ...
    def print_summary(self) -> None: ...
```

### Convenience Function

```python
async def run_benchmark(
    variant: str = "long",
    prefilter: bool = True,
    sample_size: Optional[int] = None,
    api_key: Optional[str] = None
) -> BenchmarkResults: ...
```

## Data Files

- `data/halumem/long/` - HaluMem-Long dataset
  - `dialogues.jsonl` - Conversation data
  - `ground_truth_memories.jsonl` - Expected memories
  - `qa_pairs.jsonl` - Question-answer pairs
  - `update_scenarios.jsonl` - Update test cases
  - `importance_weights.json` - Memory importance weights
- `data/halumem/medium/` - HaluMem-Medium dataset (same structure)
- `results/` - Benchmark results (created on run)

## Comparison with Other Benchmarks

| Benchmark | Focus | Token Scale | Unique Features |
|-----------|-------|-------------|-----------------|
| LoCoMo | Conversational memory | 26K | Multi-hop reasoning |
| LongMemEval | Long-context memory | 1.5M | Abstention testing |
| **HaluMem** | Hallucination detection | 1M+ | **Operation-level evaluation** |

HaluMem is unique in testing memory operations independently, enabling targeted improvement.

## References

- HaluMem Paper: [arXiv:2511.03506](https://arxiv.org/abs/2511.03506)
- Dataset: [IAAR-Shanghai/HaluMem](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem)
- Memory System Documentation: See `docs/` folder
- LoCoMo: ACL 2024
- LongMemEval: [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)

## License

See parent project LICENSE.
