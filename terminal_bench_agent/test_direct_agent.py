"""Direct test of MemoryGuidedTerminus agent LLM calls."""
import os
import asyncio
from pathlib import Path
import tempfile

from terminal_bench_agent.terminus_agent import MemoryGuidedTerminus


async def test_direct_agent():
    """Test agent LLM calls directly."""

    # Create temp directory for logs
    with tempfile.TemporaryDirectory() as tmpdir:
        logs_dir = Path(tmpdir)

        print("=== Creating MemoryGuidedTerminus agent ===")
        try:
            agent = MemoryGuidedTerminus(
                logs_dir=logs_dir,
                model_name="gpt-5-codex",
                enable_memory=False,  # Disable memory for simplicity
                temperature=1.0,
                parser_name="json",
            )
            print(f"✓ Agent created successfully")
            print(f"  Agent LLM type: {type(agent._llm)}")
            print(f"  Agent LLM model: {agent._llm._model_name}")

            # Try to call the LLM directly
            print("\n=== Testing direct LLM call ===")
            response = await agent._llm.call(
                prompt="Say hello in one word",
                message_history=[]
            )
            print(f"✓ LLM call succeeded!")
            print(f"  Response type: {type(response)}")
            print(f"  Response: {response}")

            # Check usage
            usage = agent._llm.get_last_usage()
            if usage:
                print(f"  Tokens - Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}")
            else:
                print("  ⚠ No usage info available")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    print(f"Using API key: {api_key[:20]}...")

    asyncio.run(test_direct_agent())