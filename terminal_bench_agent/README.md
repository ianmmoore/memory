# Terminal-Bench Agent with Memory Integration

A memory-guided coding agent for Terminal-Bench that leverages the code memory system for intelligent task solving.

## Features

- ✅ **Terminal-Bench Compatible**: Implements BaseAgent interface
- ✅ **Memory Integration**: Uses code memory system for context
- ✅ **Adaptive Planning**: Creates and adjusts plans based on feedback
- ✅ **Error Recovery**: Handles errors with memory-guided fixes
- ✅ **Fully Async**: Efficient parallel execution

## Installation

### Prerequisites

1. **Python 3.9+**
2. **Terminal-Bench** (from GitHub)
3. **Memory System** (from parent directory)
4. **LLM API Key** (OpenAI or Anthropic)

### Step-by-Step Installation

```bash
# 1. Clone this repository (if not already done)
cd /path/to/memory

# 2. Install Terminal-Bench
pip install terminal-bench
# Or from source:
# git clone https://github.com/laude-institute/terminal-bench.git
# cd terminal-bench
# pip install -e .

# 3. Install memory system (if not already installed)
cd memory_lib
pip install -e .
cd ..

# 4. Install agent
cd terminal_bench_agent
pip install -e .

# 5. Set up environment variables
cp .env.example .env
# Edit .env and add your API key:
#   OPENAI_API_KEY=your_key_here
#   or
#   ANTHROPIC_API_KEY=your_key_here
```

## Quick Start

### Running with Terminal-Bench CLI

The agent integrates directly with Terminal-Bench:

```bash
# Run on a single task
tb run --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent --task-id example_task_001

# Run on full benchmark
tb run --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent --dataset-name terminal-bench-core --dataset-version 0.1.1

# Run with specific model
tb run \
  --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent \
  --dataset-name terminal-bench-core \
  --model-name gpt-4
```

### Using the Provided Scripts

We provide convenience scripts for common operations:

```bash
# Run benchmark with default settings
python scripts/run_benchmark.py

# Run with custom configuration
python scripts/run_benchmark.py --config config/agent_config.yaml

# Run on specific task IDs
python scripts/run_benchmark.py --tasks task_001 task_002 task_003

# Bootstrap memory with existing solutions
python scripts/bootstrap_memory.py --solutions-dir /path/to/solutions
```

## Configuration

Edit `config/agent_config.yaml` to customize:

```yaml
agent:
  max_steps: 50  # Maximum execution steps
  max_time_seconds: 600  # Timeout per task

llm:
  primary_model: "gpt-4"  # Or claude-3-opus-20240229
  small_model: "gpt-3.5-turbo"  # For memory scoring

memory:
  enabled: true
  relevance_threshold: 0.7
  max_memories: 10
```

## How It Works

### Architecture

```
Task → Planner → Executor → Observation
         ↑                      ↓
         └──── Memory System ───┘
```

### Execution Flow

1. **Task Loading**: Receives task from Terminal-Bench
2. **Memory Retrieval**: Queries memory for similar tasks/patterns
3. **Planning**: Creates step-by-step plan using LLM + memories
4. **Execution**: Executes plan steps in terminal environment
5. **Adaptation**: Adjusts plan based on observations/errors
6. **Learning**: Stores successful solutions in memory

### Memory Integration

The agent uses the code memory system in several ways:

- **Pre-planning**: Retrieve similar past task solutions
- **Error handling**: Find and apply fixes for common errors
- **Code patterns**: Use known patterns for task types
- **Post-completion**: Store successful solutions for future use

## Benchmarking Instructions

### Running a Full Benchmark

```bash
# 1. Ensure Terminal-Bench is installed
tb --version

# 2. Run benchmark (this will take several hours)
tb run \
  --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent \
  --dataset-name terminal-bench-core \
  --dataset-version 0.1.1 \
  --output-dir ./results \
  --parallel 4

# 3. View results
tb results ./results
```

### Benchmark Output

Results are saved to the output directory:

```
results/
├── summary.json          # Overall statistics
├── task_results/         # Individual task results
│   ├── task_001.json
│   ├── task_002.json
│   └── ...
└── logs/                 # Execution logs
    ├── task_001.log
    └── ...
```

### Analyzing Results

```bash
# View summary statistics
tb results ./results --summary

# View results by task type
tb results ./results --group-by task_type

# View failed tasks
tb results ./results --filter failed

# Compare with baseline
tb compare ./results ./baseline_results
```

## Advanced Usage

### Custom LLM Functions

You can provide custom LLM functions:

```python
import asyncio
from terminal_bench_agent.core import MemoryGuidedAgent
from memory_lib import CodeMemorySystem

async def my_llm_function(prompt: str) -> str:
    # Your custom LLM integration
    response = await your_llm_api.complete(prompt)
    return response

async def my_small_model(prompt: str) -> str:
    # For memory scoring
    response = await your_fast_llm.complete(prompt)
    return response

# Initialize memory system
memory = CodeMemorySystem(
    small_model_fn=my_small_model,
    db_path="my_memories.db"
)

# Create agent
agent = MemoryGuidedAgent(
    llm_function=my_llm_function,
    memory_system=memory
)

# Use with Terminal-Bench
# (Register agent class and run via tb CLI)
```

### Memory Bootstrap

Pre-populate memory with existing solutions:

```python
from scripts.bootstrap_memory import bootstrap_from_solutions

# Load solutions from directory
bootstrap_from_solutions(
    solutions_dir="./past_solutions",
    memory_db="terminal_bench_memories.db"
)
```

### Debugging

Enable verbose logging:

```bash
# Set log level in config
# Edit config/agent_config.yaml:
#   logging:
#     level: "DEBUG"

# Or via environment variable
export AGENT_LOG_LEVEL=DEBUG

# Run with logging
tb run \
  --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent \
  --task-id example_task_001 \
  --log-dir ./debug_logs
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Test agent standalone (without Terminal-Bench)
python -m terminal_bench_agent.core
```

### Project Structure

```
terminal_bench_agent/
├── __init__.py           # Package init
├── core.py               # Main agent (BaseAgent implementation)
├── agent/
│   ├── actions.py        # Action definitions
│   ├── executor.py       # Command execution
│   └── planner.py        # Planning logic
├── config/
│   └── agent_config.yaml # Configuration
├── scripts/
│   ├── run_benchmark.py  # Benchmark runner
│   └── bootstrap_memory.py # Memory initialization
└── tests/
    ├── test_agent.py
    ├── test_executor.py
    └── test_planner.py
```

## Performance Tips

### Optimizing for Speed

1. **Reduce max_steps**: Lower for simple tasks
2. **Use smaller model**: GPT-3.5 instead of GPT-4 for simple planning
3. **Disable memory**: For tasks where memory doesn't help
4. **Parallel execution**: Use `--parallel` flag in tb run

### Optimizing for Accuracy

1. **Increase max_steps**: Allow more exploration
2. **Use better model**: GPT-4 or Claude Opus
3. **Enable memory**: Learn from past solutions
4. **Lower temperature**: More deterministic outputs

### Cost Optimization

1. **Cache memory queries**: Enabled by default
2. **Use small model for memory**: GPT-3.5 for scoring
3. **Batch tasks**: Run multiple tasks in same session
4. **Limit retries**: Reduce max_retries in config

## Troubleshooting

### Common Issues

**"terminal-bench not found"**
```bash
# Install Terminal-Bench
pip install terminal-bench
```

**"Memory system not found"**
```bash
# Install memory system
cd ../memory_lib
pip install -e .
```

**"API key not set"**
```bash
# Set environment variable
export OPENAI_API_KEY=your_key_here
```

**"Agent timeout"**
```bash
# Increase timeout in config
# Edit config/agent_config.yaml:
#   agent:
#     max_time_seconds: 1200
```

### Getting Help

- Check [Terminal-Bench docs](https://www.tbench.ai/docs)
- Review example logs in `examples/`
- Open an issue on GitHub

## Roadmap

- [ ] Task-specific strategies
- [ ] Multi-agent collaboration
- [ ] Better error pattern detection
- [ ] Automatic prompt optimization
- [ ] Integration with more LLM providers

## License

MIT

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Citation

If you use this agent in your research:

```bibtex
@software{terminal_bench_agent,
  title={Terminal-Bench Agent with Memory Integration},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/terminal-bench-agent}
}
```
