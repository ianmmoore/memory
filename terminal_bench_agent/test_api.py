"""Simple test to verify API access."""
import os
import asyncio
from openai import AsyncOpenAI

async def test_openai():
    """Test basic OpenAI API access."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No OPENAI_API_KEY found")
        return

    print(f"API Key found: {api_key[:20]}...")

    client = AsyncOpenAI(api_key=api_key)

    try:
        print("\nTesting with gpt-4o-mini...")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print(f"✓ gpt-4o-mini works! Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"✗ gpt-4o-mini failed: {e}")

    try:
        print("\nTesting with gpt-5-codex...")
        response = await client.chat.completions.create(
            model="gpt-5-codex",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
            temperature=1.0
        )
        print(f"✓ gpt-5-codex works! Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"✗ gpt-5-codex failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_openai())
