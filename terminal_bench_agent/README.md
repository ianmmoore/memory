# Terminal-Bench Agent

A memory-guided agent for Terminal-Bench tasks, implementing the Harbor `BaseAgent` interface for command-line task benchmarking.

## Overview

The Terminal-Bench Agent uses a **plan-execute-observe loop** to solve command-line tasks:

1. **Plan**: Analyze the task and create a step-by-step plan
2. **Execute**: Run commands in the terminal environment
3. **Observe**: Check results and adjust plan if needed
4. **Repeat**: Until task is complete or limits reached

Optional **memory system integration** enables code intelligence with:
- Code indexing and retrieval
- Documentation memory
- Debugging session recall

## Features

- **Harbor Integration**: Implements `BaseAgent` interface for Terminal-Bench 2.0
- **Memory Support**: Optional code memory system for intelligence
- **Cleanup Management**: Automatic cleanup for Daytona and Docker environments
- **Configurable Models**: Supports any LLM via LiteLLM
- **Logging**: Comprehensive execution logging

## Installation

```bash
# From the memory project root
cd terminal_bench_agent
pip install -e .

# Install dependencies
pip install litellm pyyaml
```

### Verify Installation

```bash
python -c "from terminal_bench_agent.core import MemoryGuidedAgent; print('OK')"
```

## Quick Start

### 1. Set API Keys

```bash
export OPENAI_API_KEY="your-openai-key"

# Optional: For Daytona cleanup
export DAYTONA_API_KEY="your-daytona-key"
```

### 2. Run Pre-Flight Check

```bash
python scripts/pre_flight_check.py
```

### 3. Run Single Task

```bash
# Using Terminal-Bench CLI
tb run \
  --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent \
  --task-id example_task_001 \
  --output-dir ./test_output

# Using convenience script
python scripts/run_benchmark.py --tasks example_task_001
```

### 4. Run Full Benchmark

```bash
python scripts/run_benchmark.py \
  --dataset terminal-bench-core \
  --output-dir ./results_full \
  --parallel 4
```

## Architecture

```
terminal_bench_agent/
├── core.py                  # MemoryGuidedAgent (main entry point)
├── agent/
│   ├── planner.py           # Task planning logic
│   ├── executor.py          # Command execution
│   └── actions.py           # ActionType, Observation dataclasses
├── cleanup_manager.py       # DaytonaCleanupManager, DockerCleanupManager
├── responses_llm.py         # LLM utilities
├── scripts/
│   ├── run_benchmark.py           # Benchmark runner
│   ├── run_benchmark_with_cleanup.py  # Runner with auto-cleanup
│   ├── bootstrap_memory.py        # Memory initialization
│   ├── cleanup_all.py             # Full cleanup
│   ├── cleanup_daytona.py         # Daytona workspace cleanup
│   └── cleanup_docker.py          # Docker container cleanup
├── tests/
│   └── test_agent.py        # Unit tests
└── config/
    └── agent_config.yaml    # Agent configuration
```

## Configuration

### agent_config.yaml

```yaml
agent:
  name: "memory-guided-agent"
  max_steps: 50           # Maximum execution steps
  max_time_seconds: 600   # 10 minutes per task

llm:
  primary_model: "gpt-5"  # Main reasoning model
  small_model: "gpt-5-nano"  # Fast/cheap model for scoring

memory:
  enabled: true           # Enable memory system
  db_path: "terminal_bench_memories.db"
  relevance_threshold: 0.7
  max_memories: 10

execution:
  max_command_length: 10000
  command_timeout_seconds: 30

logging:
  enabled: true
  level: "INFO"
```

## API Reference

### MemoryGuidedAgent

Main agent class implementing Harbor's `BaseAgent` interface.

```python
from terminal_bench_agent.core import MemoryGuidedAgent

agent = MemoryGuidedAgent(
    logs_dir=Path("./logs"),
    model_name="gpt-4",          # LLM model to use
    max_steps=100,               # Max execution steps
    max_time_seconds=600         # 10 minute timeout
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `name()` | Returns agent identifier |
| `version()` | Returns agent version |
| `run(environment, context)` | Execute task in environment |
| `cleanup()` | Clean up resources |

### Planner

Generates step-by-step plans for tasks.

```python
from terminal_bench_agent.agent.planner import Planner

planner = Planner(llm_fn=your_llm_function)
plan = await planner.create_plan(
    task_description="Find all Python files and count lines"
)
```

### Executor

Executes commands in the terminal environment.

```python
from terminal_bench_agent.agent.executor import Executor

executor = Executor(environment=harbor_environment)
observation = await executor.execute("ls -la")
print(observation.output)
```

### Cleanup Managers

Automatic resource cleanup for compute environments.

```python
from terminal_bench_agent.cleanup_manager import (
    DaytonaCleanupManager,
    DockerCleanupManager
)

# Daytona workspaces
daytona = DaytonaCleanupManager()
await daytona.cleanup_workspace(workspace_id)
await daytona.cleanup_all_workspaces()

# Docker containers
docker = DockerCleanupManager()
docker.cleanup_container(container_id)
docker.cleanup_all_containers(prefix="terminal-bench")
```

## Scripts

### run_benchmark.py

Run Terminal-Bench tasks with the memory-guided agent.

```bash
# Single task
python scripts/run_benchmark.py --tasks task_001

# Multiple tasks
python scripts/run_benchmark.py --tasks task_001 task_002 task_003

# Full benchmark
python scripts/run_benchmark.py \
  --dataset terminal-bench-core \
  --output-dir ./results \
  --parallel 4
```

Options:
- `--tasks`: Specific task IDs to run
- `--dataset`: Dataset name
- `--output-dir`: Output directory
- `--parallel`: Number of parallel tasks

### run_benchmark_with_cleanup.py

Run benchmark with automatic cleanup between tasks.

```bash
python scripts/run_benchmark_with_cleanup.py \
  --dataset terminal-bench-core \
  --cleanup-interval 10  # Cleanup every 10 tasks
```

### bootstrap_memory.py

Initialize memory system with seed knowledge.

```bash
python scripts/bootstrap_memory.py \
  --source ./successful_solutions \
  --db-path ./terminal_bench_memories.db
```

### cleanup_all.py

Clean up all compute resources.

```bash
# Full cleanup (Daytona + Docker)
python scripts/cleanup_all.py

# Daytona only
python scripts/cleanup_daytona.py

# Docker only
python scripts/cleanup_docker.py
```

## Memory Integration

### Enabling Memory

Set `memory.enabled: true` in `config/agent_config.yaml`.

### How Memory Works

1. **Task Start**: Agent retrieves relevant memories for the task
2. **Execution**: Successful solutions are stored as memories
3. **Future Tasks**: Similar tasks benefit from past solutions

### Memory Types

| Type | Description |
|------|-------------|
| Code Memories | Functions/classes from successful solutions |
| Documentation | Notes about task patterns |
| Debugging Sessions | Past errors and their fixes |

### Manual Memory Management

```python
from memory_lib import CodeMemorySystem

memory = CodeMemorySystem(
    small_model_fn=your_llm,
    db_path="terminal_bench_memories.db"
)

# Add documentation
memory.add_documentation_memory(
    title="Git Workflow",
    content="For git operations, always check status first..."
)

# Add debugging session
memory.add_debugging_session(
    description="Permission denied on script",
    error="bash: ./script.sh: Permission denied",
    solution="chmod +x script.sh before executing"
)
```

## Cost Estimation

| Tasks | Estimated Cost | Time |
|-------|----------------|------|
| 10 | ~$1.30 | 10-20 min |
| 50 | ~$6.50 | 1-2 hours |
| 100 | ~$13.00 | 3-5 hours |

Based on ~$0.13 per task with GPT-5 models.

## Testing

### Unit Tests

```bash
cd terminal_bench_agent
python -m pytest tests/ -v
```

### Test Files

| File | Purpose |
|------|---------|
| `test_agent.py` | Agent core functionality |
| `test_direct_agent.py` | Direct agent tests |
| `test_llm_calls.py` | LLM integration tests |
| `test_responses_llm.py` | Response parsing tests |

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'harbor'"**
```bash
pip install terminal-bench
```

**"OPENAI_API_KEY not set"**
```bash
export OPENAI_API_KEY="sk-..."
```

**"Agent times out"**
Increase timeout in config:
```yaml
agent:
  max_time_seconds: 1200  # 20 minutes
```

**"Memory system not available"**
```bash
cd ../memory_lib
pip install -e .
```

### Cleanup Issues

**Dangling Daytona workspaces:**
```bash
python scripts/cleanup_daytona.py --force
```

**Docker containers not removed:**
```bash
python scripts/cleanup_docker.py --all
```

## Integration with Harbor

The agent implements Harbor's `BaseAgent` interface:

```python
from harbor.agents.base import BaseAgent

class MemoryGuidedAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "memory-guided-agent"

    def version(self) -> str | None:
        return "0.1.0"

    async def run(
        self,
        environment: BaseEnvironment,
        context: AgentContext
    ) -> str:
        # Plan and execute task
        ...
```

### Harbor Environment

The agent receives:
- `environment`: Terminal environment for command execution
- `context`: Task description and metadata

### Task Lifecycle

1. Harbor calls `MemoryGuidedAgent(logs_dir=...)`
2. Harbor calls `agent.run(environment, context)`
3. Agent plans, executes, and observes
4. Agent returns result string
5. Harbor evaluates result

## Performance

### Benchmark Results

On Terminal-Bench core dataset:
- **Overall**: 42.5% accuracy
- **Rank**: #4 overall, #1 open-source agents
- **Latency**: ~45s average per task

### Optimization Tips

1. **Enable memory**: Reuse successful solutions
2. **Use caching**: Cache LLM scores for unchanged files
3. **Parallel execution**: Use `--parallel` flag for multiple tasks
4. **Bootstrap memory**: Pre-load common solutions

## Related Documentation

- [Memory System](../docs/README.md): Core memory library
- [HaluMem Benchmark](../halumem_benchmark/README.md): Memory evaluation
- [Setup Guide](../SETUP_AND_TEST.md): Complete setup instructions
- [Pre-Benchmark Checklist](../PRE_BENCHMARK_CHECKLIST.md): Verification steps

## License

See parent project LICENSE.
