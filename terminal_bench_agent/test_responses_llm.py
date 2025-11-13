"""Test ResponsesLLM wrapper implementation."""
import os
import asyncio
from pathlib import Path

from terminal_bench_agent.responses_llm import ResponsesLLM


async def test_responses_llm():
    """Test ResponsesLLM routing and functionality."""

    # Test 1: Verify GPT-5 detection
    print("=== Test 1: Model Detection ===")
    llm_gpt5 = ResponsesLLM(
        model_name="gpt-5-codex",
        temperature=1.0,
    )
    print(f"gpt-5-codex uses Responses API: {llm_gpt5._use_responses_api}")
    assert llm_gpt5._use_responses_api == True, "gpt-5-codex should use Responses API"

    llm_gpt4 = ResponsesLLM(
        model_name="gpt-4o-mini",
        temperature=0.7,
    )
    print(f"gpt-4o-mini uses Responses API: {llm_gpt4._use_responses_api}")
    assert llm_gpt4._use_responses_api == False, "gpt-4o-mini should NOT use Responses API"

    llm_o3 = ResponsesLLM(
        model_name="o3-mini",
        temperature=1.0,
    )
    print(f"o3-mini uses Responses API: {llm_o3._use_responses_api}")
    assert llm_o3._use_responses_api == True, "o3-mini should use Responses API"

    print("✓ Model detection works correctly\n")

    # Test 2: Test actual API call with GPT-5-Codex (requires valid API key)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠ Skipping API call test - no OPENAI_API_KEY found")
        return

    print("=== Test 2: API Call ===")
    print(f"Using API key: {api_key[:20]}...")

    try:
        response = await llm_gpt5.call(
            prompt="Say hello in one word",
            message_history=[],
        )
        print(f"✓ API call succeeded!")
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        print(f"Response length: {len(response)} chars")
        print(f"Is string: {isinstance(response, str)}")

        # Check usage info
        usage = llm_gpt5.get_last_usage()
        if usage:
            print(f"Tokens - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}")
            print(f"Cost: ${usage.cost_usd:.6f}")
        else:
            print("⚠ No usage info available")

    except Exception as e:
        print(f"✗ API call failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_responses_llm())
