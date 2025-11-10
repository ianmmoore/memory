# Pre-Benchmark Checklist

Complete this checklist before running Terminal-Bench to ensure everything is ready.

## ☐ Installation

### Memory System
```bash
cd memory_lib
pip install -e .
```
- [ ] Memory system installed
- [ ] Can import: `from memory_lib import MemorySystem, CodeMemorySystem`

### Terminal-Bench Agent
```bash
cd terminal_bench_agent
pip install -e .
```
- [ ] Agent installed
- [ ] Can import: `from terminal_bench_agent.core import MemoryGuidedAgent`

### Terminal-Bench
```bash
pip install terminal-bench
# Or from source if not available on PyPI
```
- [ ] Terminal-Bench installed
- [ ] `tb --version` works

### Dependencies
```bash
pip install openai pyyaml python-dotenv
# Or: pip install anthropic pyyaml python-dotenv
```
- [ ] LLM client installed (openai or anthropic)
- [ ] Other dependencies installed

---

## ☐ Configuration

### API Keys
```bash
export OPENAI_API_KEY="your-key-here"
# Or: export ANTHROPIC_API_KEY="your-key-here"
```
- [ ] API key set in environment
- [ ] Key is valid and has credit

### Agent Config
File: `terminal_bench_agent/config/agent_config.yaml`

```yaml
memory:
  enabled: true  # Must be true!
```
- [ ] Memory enabled in config
- [ ] Model names are correct
- [ ] Paths are valid

---

## ☐ Testing

### Pre-Flight Check
```bash
python scripts/pre_flight_check.py
```
- [ ] All 7 checks pass
- [ ] No errors in output

### Integration Test (Optional but Recommended)
```bash
python scripts/integration_test.py
```
- [ ] Memory system test passes
- [ ] Agent planning test passes
- [ ] Cost estimate shown

### Single Task Test
```bash
tb run \
  --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent \
  --task-id example_task_001
```
- [ ] Task runs without errors
- [ ] Logs show execution
- [ ] Results are generated

---

## ☐ Verification

### Check Logs
```bash
cat ./results/*/logs/*.log
```
- [ ] Logs show task description
- [ ] Logs show plan creation
- [ ] Logs show command execution
- [ ] Logs show observations

### Check Results
```bash
tb results ./results
```
- [ ] Results command works
- [ ] Shows task status
- [ ] Shows execution time

### Check Memory
```python
from memory_lib import CodeMemorySystem
memory = CodeMemorySystem(lambda x: None, db_path='terminal_bench_memories.db')
print(memory.get_stats())
```
- [ ] Memory database exists
- [ ] Can query stats
- [ ] No errors

---

## ☐ Cost Planning

### Estimate Budget
For 100 tasks with GPT-5:
- Per task: ~$0.13
- Total: ~$13.00

- [ ] Budget allocated
- [ ] Cost estimate reviewed
- [ ] API rate limits checked

### Set Cost Alerts (Optional)
- [ ] OpenAI usage alerts configured
- [ ] Monitoring dashboard ready

---

## ☐ Benchmark Strategy

### Start Small
```bash
# Test on 10 tasks first
python scripts/run_benchmark.py --tasks task_001 task_002 ... task_010
```
- [ ] 10-task test planned
- [ ] Expected cost: ~$1.30
- [ ] Expected time: 10-20 minutes

### Scale Up
```bash
# Then 50 tasks
python scripts/run_benchmark.py --output-dir ./results_50
```
- [ ] 50-task test planned
- [ ] Expected cost: ~$6.50
- [ ] Expected time: 1-2 hours

### Full Benchmark
```bash
# Finally 100+ tasks
python scripts/run_benchmark.py --dataset terminal-bench-core
```
- [ ] Full benchmark planned
- [ ] Expected cost: ~$13.00
- [ ] Expected time: 3-5 hours

---

## ☐ Monitoring

### During Run
- [ ] Monitor API usage dashboard
- [ ] Check agent logs periodically
- [ ] Watch for errors in output

### After Run
- [ ] Review results summary
- [ ] Analyze failed tasks
- [ ] Check memory growth
- [ ] Calculate actual cost

---

## ☐ Backup Plan

### If Something Goes Wrong
- [ ] Can stop benchmark mid-run (Ctrl+C)
- [ ] Can resume from checkpoint
- [ ] Have backup API key if rate limited
- [ ] Know how to clear memory database if corrupted

### Support Resources
- [ ] Terminal-Bench docs: https://www.tbench.ai/docs
- [ ] Agent README reviewed
- [ ] Setup guide reviewed (SETUP_AND_TEST.md)

---

## Quick Command Reference

### Pre-Flight
```bash
python scripts/pre_flight_check.py
```

### Integration Test
```bash
python scripts/integration_test.py
```

### Single Task
```bash
tb run --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent --task-id task_001
```

### Small Benchmark (10 tasks)
```bash
python scripts/run_benchmark.py --tasks task_001 task_002 task_003 task_004 task_005 task_006 task_007 task_008 task_009 task_010
```

### Full Benchmark
```bash
python scripts/run_benchmark.py --dataset terminal-bench-core --output-dir ./results
```

### View Results
```bash
tb results ./results --summary
```

---

## Final Check

Before running the full benchmark, confirm:

- ✅ All installation items checked
- ✅ All configuration items checked
- ✅ All testing items passed
- ✅ All verification items confirmed
- ✅ Budget and strategy planned
- ✅ Monitoring plan ready
- ✅ Backup plan understood

**If all checked, you're ready to benchmark!** 🚀

---

## Expected Results

After running full benchmark (100 tasks):

### Success Metrics
- Success rate: 50-60% (target)
- Average execution time: 2-5 minutes per task
- Total cost: $10-15
- Total time: 3-5 hours

### Outputs
- Results in `./results/`
- Logs in `./results/logs/`
- Memory database: `terminal_bench_memories.db`
- Cost report from API provider

### Next Steps After Benchmark
1. Analyze results by task type
2. Identify failure patterns
3. Bootstrap memory with successful solutions
4. Iterate and improve
5. Re-run benchmark to measure improvement

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Import errors | `cd memory_lib && pip install -e .` |
| tb not found | `pip install terminal-bench` |
| No API key | `export OPENAI_API_KEY=...` |
| Agent timeout | Increase `max_time_seconds` in config |
| Out of memory | Clear old databases, reduce `max_memories` |
| Rate limited | Reduce `batch_size`, add delays |
| High costs | Check config, verify models used |

---

## Contact & Support

- **Terminal-Bench Docs**: https://www.tbench.ai/docs
- **Agent README**: `terminal_bench_agent/README.md`
- **Setup Guide**: `SETUP_AND_TEST.md`
- **This Checklist**: `PRE_BENCHMARK_CHECKLIST.md`

Good luck! 🎯
