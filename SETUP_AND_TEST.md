# Setup and Test Guide - Before Terminal-Bench

Complete this guide to ensure everything is ready for Terminal-Bench testing.

## Prerequisites

- Python 3.9+
- pip
- Git
- API key (OpenAI or Anthropic)

## Step-by-Step Setup

### 1. Install Memory System

```bash
cd memory_lib
pip install -e .
cd ..
```

**Verify:**
```bash
python -c "from memory_lib import MemorySystem, CodeMemorySystem; print('✓ Memory system installed')"
```

### 2. Install Terminal-Bench Agent

```bash
cd terminal_bench_agent
pip install -e .
cd ..
```

**Verify:**
```bash
python -c "from terminal_bench_agent.core import MemoryGuidedAgent; print('✓ Agent installed')"
```

### 3. Install Terminal-Bench

```bash
pip install terminal-bench
```

**Verify:**
```bash
tb --version
```

If `terminal-bench` isn't available via pip yet, install from source:
```bash
git clone https://github.com/laude-institute/terminal-bench.git
cd terminal-bench
pip install -e .
cd ..
```

### 4. Set Up API Keys

Choose your LLM provider:

**For OpenAI (GPT-5):**
```bash
export OPENAI_API_KEY="your-key-here"
```

**For Anthropic (Claude):**
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Make it permanent:
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 5. Run Pre-Flight Check

This will verify everything is working:

```bash
python scripts/pre_flight_check.py
```

**Expected output:**
```
======================================================================
Terminal-Bench Agent Pre-Flight Check
======================================================================

1. Checking imports...
  ✓ Memory system imports
  ✓ CodeContext import
  ✓ Agent imports

2. Checking API keys...
  ✓ OPENAI_API_KEY found

3. Testing memory system...
  ✓ MemorySystem created
  ✓ Memory added (ID: abc12345)
  ✓ Memory retrieved correctly
  ✓ Memory count: 1
  ✓ Cleanup complete

4. Testing code memory system...
  ✓ CodeMemorySystem created
  ✓ Code memory added (ID: def67890)
  ✓ Documentation added (ID: ghi12345)
  ✓ Code memories: 1
  ✓ Non-code memories: 1
  ✓ Cleanup complete

5. Testing Terminal-Bench agent...
  ✓ Agent created
  ✓ Plan created with 1 steps
  ✓ Agent components working

6. Checking Terminal-Bench installation...
  ✓ Terminal-Bench installed: 0.1.1

7. Checking dependencies...
  ✓ openai installed
  ✓ yaml installed
  ✓ asyncio installed

======================================================================
Summary
======================================================================

  Imports.................................... PASS
  API Keys................................... PASS
  Memory System.............................. PASS
  Code Memory System......................... PASS
  Agent...................................... PASS
  Terminal-Bench............................. PASS
  Dependencies............................... PASS

Total: 7/7 checks passed

✓ All checks passed! Ready to run Terminal-Bench.
```

---

## Quick Test on Single Task

Before running the full benchmark, test on a single task:

### Method 1: Using Terminal-Bench CLI

```bash
tb run \
  --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent \
  --task-id example_task_001 \
  --output-dir ./test_output
```

### Method 2: Using Convenience Script

```bash
python scripts/run_benchmark.py --tasks example_task_001
```

**Expected output:**
```
Loading configuration from: config/agent_config.yaml
Setting up LLM...
Setting up memory system...
  ✓ Memory system enabled

Starting Terminal-Bench...

Running command: tb run --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent ...

Output will be saved to: ./results/run_20240115_143022

[Terminal-Bench executes task...]

✓ Benchmark completed successfully!

View results with: tb results ./results/run_20240115_143022
```

---

## Verify Results

Check that the agent actually ran:

```bash
# View results
tb results ./test_output

# Expected output shows:
# - Task ID
# - Success/Failure
# - Execution time
# - Test results
```

Check logs:

```bash
# View agent execution log
cat ./test_output/logs/example_task_001.log

# Should show:
# - Task description
# - Plan created
# - Commands executed
# - Observations
# - Final result
```

---

## Common Issues & Fixes

### Issue: "ModuleNotFoundError: No module named 'memory_lib'"

**Fix:**
```bash
cd memory_lib
pip install -e .
```

### Issue: "ModuleNotFoundError: No module named 'terminal_bench_agent'"

**Fix:**
```bash
cd terminal_bench_agent
pip install -e .
```

### Issue: "tb: command not found"

**Fix:**
```bash
pip install terminal-bench
# Or from source:
# git clone https://github.com/laude-institute/terminal-bench.git
# cd terminal-bench && pip install -e .
```

### Issue: "OPENAI_API_KEY not set"

**Fix:**
```bash
export OPENAI_API_KEY="sk-..."
# Or add to ~/.bashrc for persistence
```

### Issue: "Agent times out"

**Fix:** Increase timeout in config:
```yaml
# terminal_bench_agent/config/agent_config.yaml
agent:
  max_time_seconds: 1200  # Increase from 600
```

### Issue: ImportError for terminal_bench.agents

This means Terminal-Bench isn't installed properly.

**Fix:**
```bash
# Check if installed
pip list | grep terminal-bench

# If not, install
pip install terminal-bench

# If still issues, install from source
git clone https://github.com/laude-institute/terminal-bench.git
cd terminal-bench
pip install -e .
```

### Issue: Agent doesn't use memory

Check that memory is enabled:
```yaml
# config/agent_config.yaml
memory:
  enabled: true  # Must be true
```

---

## Configuration Checklist

Before running the full benchmark, verify your configuration:

**File: `terminal_bench_agent/config/agent_config.yaml`**

```yaml
agent:
  name: "memory-guided-agent"
  max_steps: 50           # ✓ Reasonable limit
  max_time_seconds: 600   # ✓ 10 minutes per task

llm:
  primary_model: "gpt-5"  # ✓ Or your chosen model
  small_model: "gpt-5-nano"

memory:
  enabled: true           # ✓ MUST be true
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

---

## Estimating Cost Before Running

Calculate expected cost:

```python
# For 10 tasks (testing):
tasks = 10
cost_per_task = 0.13
total = tasks * cost_per_task
print(f"10 tasks: ${total:.2f}")  # $1.30

# For 100 tasks (full benchmark):
tasks = 100
total = tasks * cost_per_task
print(f"100 tasks: ${total:.2f}")  # $13.00
```

---

## Ready to Run Full Benchmark

Once all checks pass and single task works:

### Small Test (10 tasks)

```bash
python scripts/run_benchmark.py \
  --tasks task_001 task_002 task_003 task_004 task_005 \
         task_006 task_007 task_008 task_009 task_010
```

Expected time: 10-20 minutes
Expected cost: ~$1.30

### Medium Test (50 tasks)

```bash
python scripts/run_benchmark.py \
  --output-dir ./results_50tasks \
  --parallel 2
```

Run first 50 tasks from dataset
Expected time: 1-2 hours
Expected cost: ~$6.50

### Full Benchmark (100+ tasks)

```bash
python scripts/run_benchmark.py \
  --dataset terminal-bench-core \
  --output-dir ./results_full \
  --parallel 4
```

Expected time: 3-5 hours
Expected cost: ~$13.00

---

## Post-Run Analysis

After running:

### 1. View Summary

```bash
tb results ./results_full --summary
```

Shows:
- Total tasks
- Success rate
- Average execution time
- Failures by category

### 2. View Failed Tasks

```bash
tb results ./results_full --filter failed
```

### 3. Compare with Baseline

```bash
tb compare ./results_full ./baseline_results
```

### 4. View Detailed Stats

```bash
python scripts/analyze_results.py ./results_full
```

### 5. Check Memory Usage

See what was learned:

```bash
# Count memories stored
python -c "
from memory_lib import CodeMemorySystem
memory = CodeMemorySystem(lambda x: None, db_path='terminal_bench_memories.db')
stats = memory.get_stats()
print(f'Total memories: {stats[\"total_memories\"]}')
print(f'Code memories: {stats[\"code_memories\"]}')
print(f'Non-code memories: {stats[\"non_code_memories\"]}')
"
```

---

## Next Steps After Testing

1. **Analyze failures**: Understand what task types are hardest
2. **Tune configuration**: Adjust based on results
3. **Bootstrap memory**: Add successful solutions to memory
4. **Iterate**: Re-run with improvements
5. **Compare**: Compare against baseline/other agents

---

## Getting Help

If you encounter issues:

1. **Check logs**: `./results/*/logs/*.log`
2. **Run pre-flight**: `python scripts/pre_flight_check.py`
3. **Check Terminal-Bench docs**: https://www.tbench.ai/docs
4. **Review agent README**: `terminal_bench_agent/README.md`
5. **Check GitHub issues**: Look for similar problems

---

## Success Criteria

You're ready when:

- ✓ Pre-flight check passes 7/7
- ✓ Single task runs successfully
- ✓ Agent logs show proper execution
- ✓ Memory system stores and retrieves
- ✓ Cost is as expected (~$0.13/task)

**Then proceed to full benchmark!** 🚀
