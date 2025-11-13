"""Test GPT-5-Codex with litellm.aresponses() (Responses API)."""
import os
import asyncio
import litellm

async def test_responses_api():
    """Test using litellm.aresponses for GPT-5-Codex."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No OPENAI_API_KEY found")
        return

    print(f"API Key found: {api_key[:20]}...")

    # Test 1: Using aresponses (Responses API)
    try:
        print("\n=== Test 1: Using litellm.aresponses() ===")
        response = await litellm.aresponses(
            model="gpt-5-codex",
            input="Say hello in one word",
            temperature=1.0,
        )
        print(f"✓ aresponses() works!")
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"✗ aresponses() failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Using acompletion (Chat Completions API) - expected to fail
    try:
        print("\n=== Test 2: Using litellm.acompletion() ===")
        response = await litellm.acompletion(
            model="gpt-5-codex",
            messages=[{"role": "user", "content": "Say hello in one word"}],
            temperature=1.0,
        )
        print(f"✓ acompletion() works!")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"✗ acompletion() failed (expected): {e}")

if __name__ == "__main__":
    asyncio.run(test_responses_api())
