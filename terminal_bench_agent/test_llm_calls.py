#!/usr/bin/env python3
"""Test script to verify LLM calls are working."""

import os
import asyncio
from litellm import acompletion

async def test_llm():
    """Test if LLM calls work."""
    model = os.environ.get("OPENAI_MODEL", "gpt-4")
    api_key = os.environ.get("OPENAI_API_KEY")

    print(f"Testing LLM: {model}")
    print(f"API Key set: {'Yes' if api_key else 'No'}")

    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    try:
        print("\nSending test prompt...")
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": "Say 'Hello, I am working!' in JSON format: {\"message\": \"...\"}"}],
            temperature=1.0,
            max_tokens=100
        )

        content = response.choices[0].message.content
        print(f"\nSUCCESS! LLM Response:")
        print(content)

    except Exception as e:
        print(f"\nERROR: LLM call failed")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_llm())
