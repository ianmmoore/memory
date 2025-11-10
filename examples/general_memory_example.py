"""Example usage of the general memory system.

This script demonstrates how to use the general MemorySystem for storing
and retrieving memories with LLM-based relevance scoring.
"""

import asyncio
import sys
sys.path.insert(0, '..')

from memory_lib import MemorySystem


# Mock LLM functions for demonstration
# In production, replace these with actual API calls to your LLM provider

async def mock_small_model(prompt: str) -> str:
    """Mock small model that scores relevance based on keyword matching.

    In production, replace this with actual API calls to a small model
    like GPT-3.5-turbo, Claude Haiku, or similar.
    """
    # Simple keyword-based scoring for demo
    prompt_lower = prompt.lower()

    if "python" in prompt_lower and "python" in prompt.lower():
        return "Score: 0.9\nReason: Directly mentions Python, highly relevant"
    elif "programming" in prompt_lower:
        return "Score: 0.75\nReason: Related to programming topic"
    elif "web" in prompt_lower and "framework" in prompt_lower:
        return "Score: 0.8\nReason: Related to web development"
    else:
        return "Score: 0.3\nReason: Limited relevance to the query"


async def mock_primary_model(prompt: str) -> str:
    """Mock primary model that generates responses.

    In production, replace this with actual API calls to a primary model
    like GPT-4, Claude Opus, or similar.
    """
    return f"Based on the provided memories, here is my response:\n[This would be the LLM's actual response based on the context]"


async def main():
    """Main example demonstrating the general memory system."""

    print("=" * 70)
    print("General Memory System Example")
    print("=" * 70)
    print()

    # 1. Initialize the memory system
    print("1. Initializing memory system...")
    memory_system = MemorySystem(
        small_model_fn=mock_small_model,
        db_path="example_memories.db",
        relevance_threshold=0.7,
        max_memories=5
    )
    print(f"   ✓ System initialized")
    print()

    # 2. Add some memories
    print("2. Adding memories...")

    memories_to_add = [
        {
            "text": "Python is a high-level, interpreted programming language with dynamic typing.",
            "metadata": {"topic": "python", "category": "language-basics"}
        },
        {
            "text": "FastAPI is a modern, fast web framework for building APIs with Python.",
            "metadata": {"topic": "python", "category": "web-framework"}
        },
        {
            "text": "JavaScript is primarily used for web development and runs in browsers.",
            "metadata": {"topic": "javascript", "category": "language-basics"}
        },
        {
            "text": "React is a JavaScript library for building user interfaces.",
            "metadata": {"topic": "javascript", "category": "frontend"}
        },
        {
            "text": "async/await in Python allows for asynchronous programming.",
            "metadata": {"topic": "python", "category": "async"}
        },
        {
            "text": "Django is a high-level Python web framework that encourages rapid development.",
            "metadata": {"topic": "python", "category": "web-framework"}
        }
    ]

    for mem in memories_to_add:
        mem_id = memory_system.add_memory(mem["text"], mem["metadata"])
        print(f"   ✓ Added: {mem['text'][:50]}... (ID: {mem_id[:8]})")
    print()

    # 3. Get statistics
    print("3. System statistics:")
    stats = memory_system.get_stats()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Relevance threshold: {stats['retrieval_config']['relevance_threshold']}")
    print(f"   Max memories returned: {stats['retrieval_config']['max_memories']}")
    print()

    # 4. Retrieve relevant memories
    print("4. Retrieving relevant memories for query...")
    context = "Tell me about Python web frameworks"
    print(f"   Context: '{context}'")
    print()

    relevant_memories = await memory_system.retrieve_relevant_memories(context)

    print(f"   Found {len(relevant_memories)} relevant memories:")
    for i, mem in enumerate(relevant_memories, 1):
        print(f"   {i}. Score: {mem.relevance_score:.2f}")
        print(f"      Text: {mem.text[:60]}...")
        print(f"      Reason: {mem.reasoning}")
        print()

    # 5. Format memories for a prompt
    print("5. Formatting memories for prompt...")
    formatted = memory_system.format_memories_for_prompt(relevant_memories)
    print(f"   Formatted output:")
    print("   " + "-" * 66)
    for line in formatted.split("\n")[:10]:  # Show first 10 lines
        print(f"   {line}")
    print("   ...")
    print()

    # 6. Complete query with primary model
    print("6. Running complete query with primary model...")
    response = await memory_system.query(
        context="I'm building a web API with Python",
        task="Which framework should I use and why?",
        primary_model_fn=mock_primary_model
    )
    print(f"   Response: {response[:100]}...")
    print()

    # 7. Update retrieval configuration
    print("7. Updating retrieval configuration...")
    memory_system.update_retrieval_config(
        relevance_threshold=0.8,
        max_memories=3
    )
    print(f"   ✓ Updated threshold to 0.8 and max memories to 3")
    print()

    # 8. Query again with new configuration
    print("8. Querying with updated configuration...")
    relevant_memories_v2 = await memory_system.retrieve_relevant_memories(
        "Asynchronous programming in Python"
    )
    print(f"   Found {len(relevant_memories_v2)} memories (with higher threshold)")
    print()

    # 9. Clean up
    print("9. Cleaning up...")
    count = memory_system.clear_all_memories()
    print(f"   ✓ Deleted {count} memories")
    print()

    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
