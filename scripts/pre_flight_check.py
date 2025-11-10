"""Pre-flight test script for memory system and Terminal-Bench agent.

Run this before attempting full Terminal-Bench benchmark to verify everything works.

Usage:
    python scripts/pre_flight_check.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_step(step: str, status: str = ""):
    """Print a test step."""
    if status == "pass":
        print(f"  {GREEN}✓{RESET} {step}")
    elif status == "fail":
        print(f"  {RED}✗{RESET} {step}")
    elif status == "warning":
        print(f"  {YELLOW}⚠{RESET} {step}")
    else:
        print(f"  {BLUE}→{RESET} {step}")


def check_imports():
    """Check if all required imports work."""
    print(f"\n{BLUE}1. Checking imports...{RESET}")

    try:
        from memory_lib import MemorySystem, CodeMemorySystem
        print_step("Memory system imports", "pass")
    except ImportError as e:
        print_step(f"Memory system imports: {e}", "fail")
        return False

    try:
        from memory_lib.codebase import CodeContext
        print_step("CodeContext import", "pass")
    except ImportError as e:
        print_step(f"CodeContext import: {e}", "fail")
        return False

    try:
        from terminal_bench_agent.core import MemoryGuidedAgent
        from terminal_bench_agent.agent import Planner, Executor
        print_step("Agent imports", "pass")
    except ImportError as e:
        print_step(f"Agent imports: {e}", "fail")
        return False

    return True


def check_api_keys():
    """Check if API keys are configured."""
    print(f"\n{BLUE}2. Checking API keys...{RESET}")

    has_openai = os.getenv("OPENAI_API_KEY")
    has_anthropic = os.getenv("ANTHROPIC_API_KEY")

    if has_openai:
        print_step("OPENAI_API_KEY found", "pass")
    else:
        print_step("OPENAI_API_KEY not found", "warning")

    if has_anthropic:
        print_step("ANTHROPIC_API_KEY found", "pass")
    else:
        print_step("ANTHROPIC_API_KEY not found", "warning")

    if not has_openai and not has_anthropic:
        print_step("No API keys found! Set OPENAI_API_KEY or ANTHROPIC_API_KEY", "fail")
        return False

    return True


async def test_memory_system():
    """Test basic memory system operations."""
    print(f"\n{BLUE}3. Testing memory system...{RESET}")

    try:
        from memory_lib import MemorySystem

        # Mock LLM
        async def mock_llm(prompt: str) -> str:
            return "Score: 0.8\nReason: Test memory is relevant"

        # Create system
        print_step("Creating MemorySystem...")
        memory = MemorySystem(
            small_model_fn=mock_llm,
            db_path="test_memory.db"
        )
        print_step("MemorySystem created", "pass")

        # Add memory
        print_step("Adding test memory...")
        mem_id = memory.add_memory(
            "Test memory about Python",
            metadata={"category": "test"}
        )
        print_step(f"Memory added (ID: {mem_id[:8]})", "pass")

        # Retrieve memory
        print_step("Retrieving memory...")
        retrieved = memory.get_memory(mem_id)
        if retrieved and retrieved["text"] == "Test memory about Python":
            print_step("Memory retrieved correctly", "pass")
        else:
            print_step("Memory retrieval failed", "fail")
            return False

        # Count memories
        count = memory.count_memories()
        print_step(f"Memory count: {count}", "pass")

        # Clean up
        memory.clear_all_memories()
        import os
        if os.path.exists("test_memory.db"):
            os.remove("test_memory.db")
        print_step("Cleanup complete", "pass")

        return True

    except Exception as e:
        print_step(f"Memory system test failed: {e}", "fail")
        import traceback
        traceback.print_exc()
        return False


async def test_code_memory_system():
    """Test code memory system operations."""
    print(f"\n{BLUE}4. Testing code memory system...{RESET}")

    try:
        from memory_lib import CodeMemorySystem
        from memory_lib.codebase import CodeContext

        # Mock LLM
        async def mock_llm(prompt: str) -> str:
            return "Score: 0.8\nReason: Code is relevant"

        # Create system
        print_step("Creating CodeMemorySystem...")
        code_memory = CodeMemorySystem(
            small_model_fn=mock_llm,
            db_path="test_code_memory.db"
        )
        print_step("CodeMemorySystem created", "pass")

        # Add code memory
        print_step("Adding code memory...")
        mem_id = code_memory.add_code_memory(
            file_path="test.py",
            entity_name="test_function",
            code_snippet="def test_function(): pass",
            language="python"
        )
        print_step(f"Code memory added (ID: {mem_id[:8]})", "pass")

        # Add documentation
        print_step("Adding documentation...")
        doc_id = code_memory.add_documentation_memory(
            title="Test Doc",
            content="Test documentation",
            category="test"
        )
        print_step(f"Documentation added (ID: {doc_id[:8]})", "pass")

        # Count memories
        stats = code_memory.get_stats()
        print_step(f"Code memories: {stats['code_memories']}", "pass")
        print_step(f"Non-code memories: {stats['non_code_memories']}", "pass")

        # Clean up
        code_memory.clear_all_memories()
        import os
        if os.path.exists("test_code_memory.db"):
            os.remove("test_code_memory.db")
        print_step("Cleanup complete", "pass")

        return True

    except Exception as e:
        print_step(f"Code memory system test failed: {e}", "fail")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_standalone():
    """Test agent in standalone mode."""
    print(f"\n{BLUE}5. Testing Terminal-Bench agent...{RESET}")

    try:
        from terminal_bench_agent.core import MemoryGuidedAgent
        from terminal_bench_agent.agent import Planner, Executor

        # Mock LLM
        async def mock_llm(prompt: str) -> str:
            if "Create a detailed plan" in prompt:
                return '''[
                    {"description": "Test step", "action_type": "bash",
                     "command": "echo test", "expected_outcome": "Success"}
                ]'''
            return '{"action_type": "bash", "command": "echo test", "reasoning": "Test"}'

        print_step("Creating agent...")
        agent = MemoryGuidedAgent(
            llm_function=mock_llm,
            memory_system=None,
            max_steps=5
        )
        print_step("Agent created", "pass")

        # Test planner
        print_step("Testing planner...")
        planner = Planner(mock_llm)
        plan = await planner.create_plan("Test task")
        if len(plan) > 0:
            print_step(f"Plan created with {len(plan)} steps", "pass")
        else:
            print_step("Plan creation failed", "fail")
            return False

        print_step("Agent components working", "pass")
        return True

    except Exception as e:
        print_step(f"Agent test failed: {e}", "fail")
        import traceback
        traceback.print_exc()
        return False


def check_terminal_bench():
    """Check if Terminal-Bench is installed."""
    print(f"\n{BLUE}6. Checking Terminal-Bench installation...{RESET}")

    import subprocess

    try:
        result = subprocess.run(
            ["tb", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print_step(f"Terminal-Bench installed: {result.stdout.strip()}", "pass")
            return True
        else:
            print_step("Terminal-Bench command failed", "fail")
            return False
    except FileNotFoundError:
        print_step("Terminal-Bench not installed", "fail")
        print_step("Install with: pip install terminal-bench", "warning")
        return False
    except subprocess.TimeoutExpired:
        print_step("Terminal-Bench command timed out", "fail")
        return False


def check_dependencies():
    """Check if all dependencies are installed."""
    print(f"\n{BLUE}7. Checking dependencies...{RESET}")

    required = [
        ("openai", "openai"),
        ("yaml", "pyyaml"),
        ("asyncio", None),  # Built-in
    ]

    all_good = True
    for module_name, package_name in required:
        try:
            __import__(module_name)
            print_step(f"{module_name} installed", "pass")
        except ImportError:
            print_step(f"{module_name} not installed", "fail")
            if package_name:
                print_step(f"  Install with: pip install {package_name}", "warning")
            all_good = False

    return all_good


async def main():
    """Run all pre-flight checks."""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Terminal-Bench Agent Pre-Flight Check{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

    results = []

    # Run checks
    results.append(("Imports", check_imports()))
    results.append(("API Keys", check_api_keys()))
    results.append(("Memory System", await test_memory_system()))
    results.append(("Code Memory System", await test_code_memory_system()))
    results.append(("Agent", await test_agent_standalone()))
    results.append(("Terminal-Bench", check_terminal_bench()))
    results.append(("Dependencies", check_dependencies()))

    # Summary
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Summary{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {check_name:.<40} {status}")

    print(f"\n{BLUE}Total: {passed}/{total} checks passed{RESET}")

    if passed == total:
        print(f"\n{GREEN}✓ All checks passed! Ready to run Terminal-Bench.{RESET}")
        print(f"\n{BLUE}Next steps:{RESET}")
        print(f"  1. Run a single task:")
        print(f"     tb run --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent --task-id example_task_001")
        print(f"  2. Or use the convenience script:")
        print(f"     python scripts/run_benchmark.py --tasks task_001")
        return 0
    else:
        print(f"\n{RED}✗ Some checks failed. Fix issues before running benchmark.{RESET}")
        print(f"\n{BLUE}Common fixes:{RESET}")
        print(f"  - Install dependencies: pip install -r requirements.txt")
        print(f"  - Install memory lib: cd memory_lib && pip install -e .")
        print(f"  - Install agent: cd terminal_bench_agent && pip install -e .")
        print(f"  - Set API key: export OPENAI_API_KEY=your_key_here")
        print(f"  - Install Terminal-Bench: pip install terminal-bench")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
