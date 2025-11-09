# Quick Start Guide - Terminal-Bench Agent

Get the agent running on Terminal-Bench in 5 minutes!

## Step 1: Install Dependencies

```bash
# Install Terminal-Bench
pip install terminal-bench

# Install memory system
cd ../memory_lib
pip install -e .
cd ../terminal_bench_agent

# Install agent
pip install -e .

# Install LLM client (choose one)
pip install openai  # For GPT models
# OR
# pip install anthropic  # For Claude models
```

## Step 2: Set API Key

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
# For OpenAI:
echo "OPENAI_API_KEY=your_key_here" >> .env

# For Anthropic:
# echo "ANTHROPIC_API_KEY=your_key_here" >> .env

# Load environment
source .env  # or: export $(cat .env | xargs)
```

## Step 3: Test Installation

```bash
# Test agent standalone (without Terminal-Bench)
python -m terminal_bench_agent.core

# Expected output: "Testing agent in standalone mode..."
```

## Step 4: Run a Single Task

```bash
# Run on a specific task
tb run \
  --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent \
  --task-id example_task_001

# This will:
# 1. Load the task
# 2. Run the agent
# 3. Execute tests
# 4. Show results
```

## Step 5: Run Full Benchmark

```bash
# Run on full Terminal-Bench dataset
tb run \
  --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent \
  --dataset-name terminal-bench-core \
  --dataset-version 0.1.1 \
  --output-dir ./results

# This will take several hours!
# Results saved to: ./results
```

## Step 6: View Results

```bash
# View summary
tb results ./results --summary

# View details
tb results ./results

# View failed tasks
tb results ./results --filter failed
```

## Using the Convenience Scripts

Instead of the `tb` CLI directly, you can use our scripts:

```bash
# Run benchmark with defaults
python scripts/run_benchmark.py

# Run specific tasks
python scripts/run_benchmark.py --tasks task_001 task_002 task_003

# Run with parallel workers
python scripts/run_benchmark.py --parallel 4

# Use custom config
python scripts/run_benchmark.py --config config/my_config.yaml
```

## Bootstrap Memory (Optional)

Pre-populate memory with common patterns:

```bash
# Add common patterns
python scripts/bootstrap_memory.py --add-patterns

# Import solutions from file
python scripts/bootstrap_memory.py --import-file solutions.json

# Import from directory
python scripts/bootstrap_memory.py --solutions-dir ./past_solutions
```

## Troubleshooting

### "tb: command not found"

```bash
pip install terminal-bench
# OR install from source:
# git clone https://github.com/laude-institute/terminal-bench.git
# cd terminal-bench && pip install -e .
```

### "ModuleNotFoundError: No module named 'memory_lib'"

```bash
cd ../memory_lib
pip install -e .
```

### "API key not set"

```bash
# For OpenAI
export OPENAI_API_KEY=your_key_here

# For Anthropic
export ANTHROPIC_API_KEY=your_key_here
```

### Agent times out

Edit `config/agent_config.yaml`:
```yaml
agent:
  max_time_seconds: 1200  # Increase from 600
```

## Configuration

Edit `config/agent_config.yaml` to customize:

- **Model**: Change `primary_model` to "gpt-4", "gpt-3.5-turbo", or "claude-3-opus-20240229"
- **Steps**: Adjust `max_steps` (default: 50)
- **Timeout**: Adjust `max_time_seconds` (default: 600)
- **Memory**: Enable/disable with `memory.enabled` (default: true)

## Performance Expectations

Based on current SOTA:

- **Simple tasks**: 70-80% success rate
- **Medium tasks**: 40-50% success rate
- **Hard tasks**: 20-30% success rate
- **Overall**: Target 50-60% (current SOTA: ~50-58%)

## Next Steps

1. **Run on subset**: Test on 10 tasks first
2. **Analyze failures**: Use `tb results` to understand what failed
3. **Tune config**: Adjust parameters based on results
4. **Bootstrap memory**: Add successful solutions to memory
5. **Full benchmark**: Run on entire dataset
6. **Compare**: Compare against baseline agents

## Getting Help

- **Terminal-Bench docs**: https://www.tbench.ai/docs
- **Issues**: Check README.md troubleshooting section
- **Examples**: See `examples/` directory

## What's Next?

After running the benchmark:

1. Analyze which task types work best
2. Add task-specific strategies
3. Improve error recovery
4. Optimize prompts
5. Contribute improvements!

Good luck! 🚀
