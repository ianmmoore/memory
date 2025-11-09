#!/usr/bin/env python3
"""Bootstrap memory system with existing Terminal-Bench solutions.

This script helps pre-populate the memory database with known solutions,
patterns, and best practices for Terminal-Bench tasks.

Usage:
    python scripts/bootstrap_memory.py --solutions-dir ./solutions
    python scripts/bootstrap_memory.py --import-file solutions.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_lib import CodeMemorySystem


async def bootstrap_from_solutions_dir(solutions_dir: str, memory_db: str):
    """Bootstrap memory from a directory of solution files.

    Args:
        solutions_dir: Directory containing solution files
        memory_db: Path to memory database
    """
    print(f"Bootstrapping memory from: {solutions_dir}")

    # We'll need a small model function for the memory system
    # For bootstrapping, we'll use a dummy function since we're just storing
    async def dummy_small_model(prompt: str) -> str:
        return "Score: 0.5\nReason: Dummy"

    # Initialize memory system
    memory = CodeMemorySystem(
        small_model_fn=dummy_small_model,
        db_path=memory_db
    )

    solutions_path = Path(solutions_dir)
    if not solutions_path.exists():
        print(f"Error: Directory not found: {solutions_dir}")
        return

    count = 0
    # Look for solution files
    for solution_file in solutions_path.rglob("*.json"):
        try:
            with open(solution_file, 'r') as f:
                solution_data = json.load(f)

            # Extract task info
            task_id = solution_data.get('task_id', solution_file.stem)
            task_type = solution_data.get('task_type', 'unknown')
            description = solution_data.get('description', '')
            solution = solution_data.get('solution', '')
            commands = solution_data.get('commands', [])

            # Store as documentation memory
            memory.add_documentation_memory(
                title=f"Solution: {task_id}",
                content=f"""Task Type: {task_type}

Description: {description}

Solution Steps:
{solution}

Commands Used:
{chr(10).join(f"- {cmd}" for cmd in commands)}
""",
                category="terminal_bench_solution",
                metadata={
                    "task_id": task_id,
                    "task_type": task_type,
                    "success": True
                }
            )

            count += 1
            print(f"  ✓ Loaded solution: {task_id}")

        except Exception as e:
            print(f"  ✗ Failed to load {solution_file}: {e}")

    print(f"\nBootstrapped {count} solutions into memory database")


async def bootstrap_from_json(json_file: str, memory_db: str):
    """Bootstrap memory from a JSON file containing solutions.

    Args:
        json_file: JSON file with solutions
        memory_db: Path to memory database
    """
    print(f"Bootstrapping memory from: {json_file}")

    async def dummy_small_model(prompt: str) -> str:
        return "Score: 0.5\nReason: Dummy"

    memory = CodeMemorySystem(
        small_model_fn=dummy_small_model,
        db_path=memory_db
    )

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        solutions = data if isinstance(data, list) else [data]

        for solution in solutions:
            task_id = solution.get('task_id', 'unknown')
            task_type = solution.get('task_type', 'unknown')
            description = solution.get('description', '')
            solution_text = solution.get('solution', '')

            memory.add_documentation_memory(
                title=f"Solution: {task_id}",
                content=f"""Task: {description}

Type: {task_type}

Solution:
{solution_text}
""",
                category="terminal_bench_solution",
                metadata={"task_id": task_id, "task_type": task_type}
            )

            print(f"  ✓ Loaded: {task_id}")

        print(f"\nBootstrapped {len(solutions)} solutions")

    except Exception as e:
        print(f"Error: {e}")


async def add_common_patterns(memory_db: str):
    """Add common Terminal-Bench patterns to memory.

    Args:
        memory_db: Path to memory database
    """
    print("Adding common patterns to memory...")

    async def dummy_small_model(prompt: str) -> str:
        return "Score: 0.5\nReason: Dummy"

    memory = CodeMemorySystem(
        small_model_fn=dummy_small_model,
        db_path=memory_db
    )

    patterns = [
        {
            "title": "File Exploration Pattern",
            "content": """Common pattern for exploring unknown environments:

1. Check current directory: pwd
2. List files: ls -la
3. Check for README or documentation: cat README.md
4. Explore subdirectories: find . -type f -name "*.txt"
5. Search for specific patterns: grep -r "pattern" .
""",
            "category": "pattern"
        },
        {
            "title": "Python Script Pattern",
            "content": """Common pattern for Python tasks:

1. Check Python version: python --version or python3 --version
2. Check for requirements: cat requirements.txt
3. Install dependencies: pip install -r requirements.txt
4. Run script: python script.py or python3 script.py
5. Check output: cat output.txt or ls -la
""",
            "category": "pattern"
        },
        {
            "title": "Network Task Pattern",
            "content": """Common pattern for network tasks:

1. Check network interfaces: ip addr or ifconfig
2. Test connectivity: ping hostname
3. Check open ports: netstat -tuln or ss -tuln
4. Make HTTP request: curl URL or wget URL
5. Parse response: jq for JSON, grep for text
""",
            "category": "pattern"
        },
        {
            "title": "Data Processing Pattern",
            "content": """Common pattern for data processing:

1. Check data format: head file.csv or cat file.json
2. Count lines/records: wc -l file.txt
3. Filter data: grep pattern file.txt or awk '{print $1}' file.txt
4. Transform data: sed 's/old/new/g' file.txt
5. Output results: > output.txt or | tee output.txt
""",
            "category": "pattern"
        },
    ]

    for pattern in patterns:
        memory.add_documentation_memory(
            title=pattern["title"],
            content=pattern["content"],
            category=pattern["category"]
        )
        print(f"  ✓ Added: {pattern['title']}")

    print(f"\nAdded {len(patterns)} common patterns")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bootstrap memory with Terminal-Bench solutions"
    )
    parser.add_argument(
        "--solutions-dir",
        help="Directory containing solution files"
    )
    parser.add_argument(
        "--import-file",
        help="JSON file containing solutions"
    )
    parser.add_argument(
        "--add-patterns",
        action="store_true",
        help="Add common patterns to memory"
    )
    parser.add_argument(
        "--memory-db",
        default="terminal_bench_memories.db",
        help="Path to memory database"
    )

    args = parser.parse_args()

    if not any([args.solutions_dir, args.import_file, args.add_patterns]):
        parser.print_help()
        print("\nError: Specify at least one of: --solutions-dir, --import-file, or --add-patterns")
        sys.exit(1)

    # Bootstrap from solutions directory
    if args.solutions_dir:
        asyncio.run(bootstrap_from_solutions_dir(args.solutions_dir, args.memory_db))

    # Bootstrap from JSON file
    if args.import_file:
        asyncio.run(bootstrap_from_json(args.import_file, args.memory_db))

    # Add common patterns
    if args.add_patterns:
        asyncio.run(add_common_patterns(args.memory_db))

    print(f"\n✓ Memory database ready at: {args.memory_db}")


if __name__ == "__main__":
    main()
