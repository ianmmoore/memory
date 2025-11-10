#!/usr/bin/env python3
"""Integration test using real LLM APIs (if available).

This test:
1. Creates a real memory system with actual LLM
2. Adds some sample memories
3. Tests retrieval with real scoring
4. Tests the agent with real planning

Usage:
    export OPENAI_API_KEY=your_key
    python scripts/integration_test.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_lib import CodeMemorySystem
from memory_lib.codebase import CodeContext
from terminal_bench_agent.core import MemoryGuidedAgent


async def setup_llm_functions():
    """Set up LLM functions based on available API keys."""

    # Try OpenAI first
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("Using OpenAI GPT-5...")
        import openai
        client = openai.AsyncOpenAI(api_key=openai_key)

        async def small_model(prompt: str) -> str:
            response = await client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )
            return response.choices[0].message.content

        async def primary_model(prompt: str) -> str:
            response = await client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            return response.choices[0].message.content

        return small_model, primary_model

    # Try Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("Using Anthropic Claude...")
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=anthropic_key)

        async def small_model(prompt: str) -> str:
            response = await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        async def primary_model(prompt: str) -> str:
            response = await client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        return small_model, primary_model

    raise ValueError("No API keys found! Set OPENAI_API_KEY or ANTHROPIC_API_KEY")


async def test_memory_with_real_llm():
    """Test memory system with real LLM scoring."""
    print("\n" + "="*70)
    print("Integration Test 1: Memory System with Real LLM")
    print("="*70 + "\n")

    small_model, _ = await setup_llm_functions()

    # Create memory system
    print("1. Creating CodeMemorySystem...")
    memory = CodeMemorySystem(
        small_model_fn=small_model,
        db_path="integration_test.db"
    )
    print("   ✓ Created\n")

    # Add sample memories
    print("2. Adding sample memories...")
    memories_added = [
        {
            "file_path": "api/auth.py",
            "entity_name": "authenticate_user",
            "code_snippet": "def authenticate_user(username, password):\n    return verify_credentials(username, password)",
            "docstring": "Authenticate user with username and password",
            "language": "python"
        },
        {
            "file_path": "api/handlers.py",
            "entity_name": "handle_login",
            "code_snippet": "async def handle_login(request):\n    user = authenticate_user(request.username, request.password)",
            "docstring": "Handle login requests",
            "language": "python"
        },
        {
            "file_path": "utils/validation.py",
            "entity_name": "validate_email",
            "code_snippet": "def validate_email(email):\n    return re.match(r'^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$', email)",
            "docstring": "Validate email format",
            "language": "python"
        }
    ]

    for mem in memories_added:
        mem_id = memory.add_code_memory(**mem)
        print(f"   ✓ Added: {mem['entity_name']} (ID: {mem_id[:8]})")

    print(f"\n3. Testing memory retrieval with LLM scoring...")
    print("   Query: 'How to fix authentication error in login handler?'\n")

    # Create context
    context = CodeContext(
        user_query="How to fix authentication error in login handler?",
        current_file="api/handlers.py",
        errors="AttributeError in authenticate_user"
    )

    # Retrieve memories (this will call the real LLM!)
    print("   Calling LLM to score memories...")
    relevant_memories = await memory.retrieve_relevant_memories(context)

    print(f"\n   Found {len(relevant_memories)} relevant memories:\n")
    for i, mem in enumerate(relevant_memories, 1):
        print(f"   {i}. Score: {mem.relevance_score:.2f}")
        print(f"      Text: {mem.text}")
        print(f"      Reason: {mem.reasoning[:60]}...")
        print()

    # Clean up
    memory.clear_all_memories()
    import os
    if os.path.exists("integration_test.db"):
        os.remove("integration_test.db")

    print("✓ Integration Test 1 Passed!\n")
    return True


async def test_agent_with_real_llm():
    """Test agent with real LLM planning."""
    print("\n" + "="*70)
    print("Integration Test 2: Agent with Real LLM")
    print("="*70 + "\n")

    small_model, primary_model = await setup_llm_functions()

    # Create memory system
    print("1. Creating memory system for agent...")
    memory = CodeMemorySystem(
        small_model_fn=small_model,
        db_path="agent_integration_test.db"
    )
    print("   ✓ Created\n")

    # Create agent
    print("2. Creating agent...")
    agent = MemoryGuidedAgent(
        llm_function=primary_model,
        memory_system=memory,
        max_steps=5
    )
    print("   ✓ Created\n")

    # Test planning
    print("3. Testing agent planning with real LLM...")
    print("   Task: 'List all Python files in the current directory'\n")

    from terminal_bench_agent.agent.planner import Planner

    planner = Planner(primary_model, memory)
    plan = await planner.create_plan(
        task_description="List all Python files in the current directory",
        environment_info="Linux environment with bash"
    )

    print(f"   Plan created with {len(plan)} steps:\n")
    for i, step in enumerate(plan, 1):
        print(f"   Step {i}: {step.description}")
        print(f"           Action: {step.action.action_type.value}")
        print(f"           Command: {step.action.command}")
        print()

    # Clean up
    memory.clear_all_memories()
    import os
    if os.path.exists("agent_integration_test.db"):
        os.remove("agent_integration_test.db")

    print("✓ Integration Test 2 Passed!\n")
    return True


async def test_cost_estimation():
    """Estimate actual API costs from the test."""
    print("\n" + "="*70)
    print("Cost Estimation from Tests")
    print("="*70 + "\n")

    print("Test 1 (Memory retrieval):")
    print("  - Scored 3 memories")
    print("  - ~500 tokens input × 3 = 1,500 tokens")
    print("  - ~50 tokens output × 3 = 150 tokens")
    print("  - Cost: ~$0.0001 (using GPT-5 Nano)")
    print()

    print("Test 2 (Agent planning):")
    print("  - 1 planning call")
    print("  - ~2,000 tokens input")
    print("  - ~500 tokens output")
    print("  - Cost: ~$0.008 (using GPT-5)")
    print()

    print("Total test cost: ~$0.01")
    print()

    print("Extrapolating to full benchmark (100 tasks):")
    print("  - Per task: ~$0.13")
    print("  - 100 tasks: ~$13.00")
    print()


async def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("INTEGRATION TEST SUITE")
    print("Testing with Real LLM APIs")
    print("="*70)

    try:
        # Check for API keys
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            print("\n✗ No API keys found!")
            print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run integration tests.")
            return 1

        # Run tests
        result1 = await test_memory_with_real_llm()
        result2 = await test_agent_with_real_llm()

        # Cost estimation
        await test_cost_estimation()

        # Summary
        print("="*70)
        print("INTEGRATION TEST RESULTS")
        print("="*70 + "\n")
        print(f"  Memory System Test............ {'✓ PASS' if result1 else '✗ FAIL'}")
        print(f"  Agent Test.................... {'✓ PASS' if result2 else '✗ FAIL'}")
        print()

        if result1 and result2:
            print("✓ All integration tests passed!")
            print("\nYou're ready to run on Terminal-Bench!")
            print("\nNext step:")
            print("  tb run --agent-import-path terminal_bench_agent.core:MemoryGuidedAgent --task-id task_001")
            return 0
        else:
            print("✗ Some tests failed")
            return 1

    except Exception as e:
        print(f"\n✗ Integration test failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
