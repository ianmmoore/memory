#!/usr/bin/env python3
"""Quick test to verify gpt-5-codex model name with LiteLLM."""

import asyncio
import os
from litellm import acompletion


async def test_model(model_name: str, temperature: float = 1.0):
    """Test if a model name works with LiteLLM."""
    print(f"\nTesting model: {model_name} (temperature={temperature})")
    print("-" * 60)

    try:
        response = await acompletion(
            model=model_name,
            messages=[{"role": "user", "content": "Say 'Hello' and nothing else."}],
            temperature=temperature,
            max_tokens=10,
        )

        print(f"✓ SUCCESS")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")
        return True

    except Exception as e:
        print(f"✗ FAILED")
        print(f"Error: {type(e).__name__}: {str(e)[:200]}")
        return False


async def main():
    """Test both model name formats."""
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return

    # Test both formats
    results = []

    # Test 1: Plain model name with temperature=1.0
    results.append(await test_model("gpt-5-codex", temperature=1.0))

    # Test 2: Responses API format with temperature=1.0
    results.append(await test_model("openai/responses/gpt-5-codex", temperature=1.0))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"gpt-5-codex (temp=1.0):                 {'✓ WORKS' if results[0] else '✗ FAILS'}")
    print(f"openai/responses/gpt-5-codex (temp=1.0): {'✓ WORKS' if results[1] else '✗ FAILS'}")


if __name__ == "__main__":
    asyncio.run(main())
