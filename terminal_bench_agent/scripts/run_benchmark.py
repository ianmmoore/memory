#!/usr/bin/env python3
"""Script to run Terminal-Bench benchmark with the memory-guided agent.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --config config/custom_config.yaml
    python scripts/run_benchmark.py --tasks task_001 task_002 task_003
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
import yaml
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from terminal_bench_agent.core import MemoryGuidedAgent


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_llm_function(config: dict):
    """Set up LLM function based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Async LLM function
    """
    model_name = config['llm']['primary_model']

    if 'gpt' in model_name.lower():
        # OpenAI
        import openai

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        client = openai.AsyncOpenAI(api_key=api_key)

        async def llm_function(prompt: str) -> str:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=config['llm'].get('planning_temperature', 0.1),
                max_tokens=config['llm'].get('max_tokens', 2000)
            )
            return response.choices[0].message.content

        return llm_function

    elif 'claude' in model_name.lower():
        # Anthropic
        import anthropic

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        client = anthropic.AsyncAnthropic(api_key=api_key)

        async def llm_function(prompt: str) -> str:
            response = await client.messages.create(
                model=model_name,
                max_tokens=config['llm'].get('max_tokens', 2000),
                messages=[{"role": "user", "content": prompt}],
                temperature=config['llm'].get('planning_temperature', 0.1)
            )
            return response.content[0].text

        return llm_function

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def setup_memory_system(config: dict):
    """Set up memory system if enabled.

    Args:
        config: Configuration dictionary

    Returns:
        CodeMemorySystem or None
    """
    if not config['memory'].get('enabled', False):
        return None

    try:
        from memory_lib import CodeMemorySystem

        # Set up small model for memory scoring
        small_model_name = config['llm']['small_model']

        if 'gpt' in small_model_name.lower():
            import openai
            client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            async def small_model_fn(prompt: str) -> str:
                response = await client.chat.completions.create(
                    model=small_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=200
                )
                return response.choices[0].message.content

        elif 'claude' in small_model_name.lower():
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

            async def small_model_fn(prompt: str) -> str:
                response = await client.messages.create(
                    model=small_model_name,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
        else:
            raise ValueError(f"Unsupported small model: {small_model_name}")

        # Create memory system
        memory = CodeMemorySystem(
            small_model_fn=small_model_fn,
            db_path=config['memory']['db_path'],
            relevance_threshold=config['memory']['relevance_threshold'],
            max_memories=config['memory']['max_memories'],
            enable_caching=config['memory'].get('enable_caching', True),
            enable_dependency_boost=config['memory'].get('enable_dependency_boost', True),
            enable_recency_boost=config['memory'].get('enable_recency_boost', True)
        )

        return memory

    except Exception as e:
        print(f"Warning: Could not initialize memory system: {e}")
        return None


def run_terminal_bench(
    agent_class,
    dataset_name: str = "terminal-bench-core",
    dataset_version: str = "0.1.1",
    task_ids: list = None,
    output_dir: str = None,
    parallel: int = 1
):
    """Run Terminal-Bench using subprocess (since tb is a CLI tool).

    Args:
        agent_class: Agent class to use
        dataset_name: Dataset name
        dataset_version: Dataset version
        task_ids: Specific task IDs to run
        output_dir: Output directory
        parallel: Number of parallel workers
    """
    import subprocess

    # Build command
    cmd = [
        "tb", "run",
        "--agent-import-path", "terminal_bench_agent.core:MemoryGuidedAgent",
        "--dataset-name", dataset_name,
        "--dataset-version", dataset_version,
    ]

    if task_ids:
        for task_id in task_ids:
            cmd.extend(["--task-id", task_id])

    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    else:
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./results/run_{timestamp}"
        cmd.extend(["--output-dir", output_dir])

    if parallel > 1:
        cmd.extend(["--parallel", str(parallel)])

    print(f"\nRunning command: {' '.join(cmd)}\n")
    print(f"Output will be saved to: {output_dir}\n")

    # Run Terminal-Bench
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Benchmark completed successfully!")
        print(f"\nView results with: tb results {output_dir}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Benchmark failed with error code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("\n✗ Error: 'tb' command not found.")
        print("Please install Terminal-Bench: pip install terminal-bench")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Terminal-Bench with memory-guided agent"
    )
    parser.add_argument(
        "--config",
        default="config/agent_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--dataset",
        default="terminal-bench-core",
        help="Dataset name"
    )
    parser.add_argument(
        "--version",
        default="0.1.1",
        help="Dataset version"
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="Specific task IDs to run"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Set up LLM
    print("Setting up LLM...")
    llm_function = setup_llm_function(config)

    # Set up memory system
    print("Setting up memory system...")
    memory_system = setup_memory_system(config)
    if memory_system:
        print("  ✓ Memory system enabled")
    else:
        print("  ✗ Memory system disabled")

    # Note: We've set up the agent, but Terminal-Bench will instantiate it
    # via the import path when running. The config will be read by the agent
    # when it's instantiated by Terminal-Bench.

    print("\nStarting Terminal-Bench...\n")

    # Run benchmark
    exit_code = run_terminal_bench(
        agent_class=MemoryGuidedAgent,
        dataset_name=args.dataset,
        dataset_version=args.version,
        task_ids=args.tasks,
        output_dir=args.output_dir,
        parallel=args.parallel
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
