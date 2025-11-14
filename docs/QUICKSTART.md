# Memory Systems - Quick Start Guide

Get up and running with Memory Systems in under 10 minutes!

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Tutorial 1: Your First Memory System](#tutorial-1-your-first-memory-system)
4. [Tutorial 2: Building a Chatbot with Memory](#tutorial-2-building-a-chatbot-with-memory)
5. [Tutorial 3: Code Intelligence System](#tutorial-3-code-intelligence-system)
6. [Tutorial 4: Advanced Configuration](#tutorial-4-advanced-configuration)
7. [Next Steps](#next-steps)

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.9 or higher**
  ```bash
  python --version  # Should show 3.9+
  ```

- **An LLM API account** (choose one):
  - OpenAI (recommended for beginners)
  - Anthropic Claude
  - Or any other async LLM API

- **API Key** from your chosen provider

---

## Installation

### Step 1: Clone or Download the Repository

```bash
# If using git
git clone <repository-url>
cd memory

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```python
python -c "from memory_lib import MemorySystem, CodeMemorySystem; print('✅ Installation successful!')"
```

---

## Tutorial 1: Your First Memory System

**Goal**: Create a simple memory system that stores and retrieves facts about programming languages.

**Time**: 5 minutes

### Step 1: Set Up Your LLM Function

Create a file called `tutorial1.py`:

```python
import asyncio
import os
from memory_lib import MemorySystem

# Using OpenAI (install: pip install openai)
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your API key

async def small_llm(prompt: str) -> str:
    """Call GPT-3.5-turbo for memory scoring."""
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )
    return response.choices[0].message.content

async def primary_llm(prompt: str) -> str:
    """Call GPT-4 for main responses."""
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content
```

### Step 2: Initialize Memory System

```python
async def main():
    # Create memory system
    system = MemorySystem(
        small_model_fn=small_llm,
        db_path="tutorial1.db",
        relevance_threshold=0.7,
        max_memories=5
    )

    print("✅ Memory system initialized!")
```

### Step 3: Add Memories

```python
    # Add programming language facts
    memories = [
        ("Python uses dynamic typing and has significant whitespace", {"language": "python"}),
        ("JavaScript is single-threaded with an event loop", {"language": "javascript"}),
        ("Rust provides memory safety without garbage collection", {"language": "rust"}),
        ("Go was designed at Google for system programming", {"language": "go"}),
        ("Java uses a virtual machine (JVM) for platform independence", {"language": "java"}),
    ]

    for text, metadata in memories:
        memory_id = system.add_memory(text, metadata=metadata)
        print(f"Added memory: {memory_id}")

    print(f"\n✅ Added {len(memories)} memories")
```

### Step 4: Retrieve Relevant Memories

```python
    # Query the system
    context = "Tell me about languages with automatic memory management"

    print(f"\n🔍 Retrieving memories for: '{context}'")
    relevant = await system.retrieve_relevant_memories(context)

    print(f"\n📊 Found {len(relevant)} relevant memories:\n")
    for mem in relevant:
        print(f"Score: {mem.relevance_score:.2f}")
        print(f"Text: {mem.text}")
        print(f"Reasoning: {mem.reasoning}")
        print("-" * 60)
```

### Step 5: Complete Query

```python
    # Ask a question with memory context
    response = await system.query(
        context="User asking about Python and Java",
        task="Compare Python and Java's memory management approaches",
        primary_model_fn=primary_llm
    )

    print("\n🤖 AI Response:")
    print(response)

    # Cleanup
    system.close()
    print("\n✅ Tutorial complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 6: Run It!

```bash
export OPENAI_API_KEY="your-api-key-here"
python tutorial1.py
```

**Expected Output**:
```
✅ Memory system initialized!
Added memory: 123e4567-e89b-12d3-a456-426614174000
Added memory: 123e4567-e89b-12d3-a456-426614174001
...

✅ Added 5 memories

🔍 Retrieving memories for: 'Tell me about languages with automatic memory management'

📊 Found 3 relevant memories:

Score: 0.85
Text: Python uses dynamic typing and has significant whitespace
Reasoning: Python has automatic memory management through garbage collection
------------------------------------------------------------
Score: 0.80
Text: Java uses a virtual machine (JVM) for platform independence
Reasoning: JVM includes automatic garbage collection
------------------------------------------------------------
...

🤖 AI Response:
Python and Java both use automatic memory management through garbage collection...

✅ Tutorial complete!
```

---

## Tutorial 2: Building a Chatbot with Memory

**Goal**: Build a chatbot that remembers facts about a user and provides personalized responses.

**Time**: 10 minutes

### Step 1: Create the Chatbot Script

Create `tutorial2_chatbot.py`:

```python
import asyncio
import os
from memory_lib import MemorySystem
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

async def small_llm(prompt: str) -> str:
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )
    return response.choices[0].message.content

async def primary_llm(prompt: str) -> str:
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content

class MemoryChatbot:
    def __init__(self):
        self.system = MemorySystem(
            small_model_fn=small_llm,
            db_path="chatbot_memory.db",
            relevance_threshold=0.6,  # Lower threshold for broader recall
            max_memories=10
        )

    def remember(self, fact: str, category: str = "general"):
        """Store a fact in memory."""
        memory_id = self.system.add_memory(
            fact,
            metadata={"category": category, "source": "user"}
        )
        return memory_id

    async def respond(self, user_message: str) -> str:
        """Generate a response with memory context."""
        response = await self.system.query(
            context=f"User said: {user_message}",
            task=f"Respond to the user's message: '{user_message}'",
            primary_model_fn=primary_llm
        )
        return response

    def close(self):
        self.system.close()
```

### Step 2: Interactive Chat Loop

```python
async def chat_session():
    bot = MemoryChatbot()

    print("🤖 Memory Chatbot Started!")
    print("Commands:")
    print("  - Type normally to chat")
    print("  - Type 'remember: <fact>' to store a memory")
    print("  - Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'quit':
            break

        # Check if user wants to store a memory
        if user_input.lower().startswith('remember:'):
            fact = user_input[9:].strip()
            memory_id = bot.remember(fact)
            print(f"🧠 Remembered: {fact}\n")
            continue

        # Regular conversation
        response = await bot.respond(user_input)
        print(f"Bot: {response}\n")

    bot.close()
    print("👋 Goodbye!")

if __name__ == "__main__":
    asyncio.run(chat_session())
```

### Step 3: Test the Chatbot

```bash
python tutorial2_chatbot.py
```

**Example Conversation**:
```
🤖 Memory Chatbot Started!

You: remember: My name is Alice and I love Python programming
🧠 Remembered: My name is Alice and I love Python programming

You: remember: I'm working on a web scraping project
🧠 Remembered: I'm working on a web scraping project

You: What's my name?
Bot: Your name is Alice!

You: What am I working on?
Bot: You're working on a web scraping project. Since you love Python programming,
you might find libraries like BeautifulSoup and Scrapy very useful for that!

You: quit
👋 Goodbye!
```

---

## Tutorial 3: Code Intelligence System

**Goal**: Index a codebase and query it intelligently.

**Time**: 10 minutes

### Step 1: Create Sample Code

Create a directory `sample_project/`:

```bash
mkdir -p sample_project
```

Create `sample_project/utils.py`:

```python
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b

def calculate_product(a, b):
    """Calculate the product of two numbers."""
    return a * b

def validate_input(value):
    """Validate that input is a number."""
    if not isinstance(value, (int, float)):
        raise ValueError("Input must be a number")
    return True
```

Create `sample_project/auth.py`:

```python
import hashlib

def hash_password(password):
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_token(token):
    """Validate an authentication token."""
    if not token or len(token) < 32:
        return False
    return True

class User:
    """Represents a user in the system."""

    def __init__(self, username, email):
        self.username = username
        self.email = email

    def authenticate(self, password):
        """Authenticate user with password."""
        hashed = hash_password(password)
        # Compare with stored hash
        return True
```

### Step 2: Create Code Intelligence Script

Create `tutorial3_code.py`:

```python
import asyncio
import os
from memory_lib import CodeMemorySystem
from memory_lib.codebase import CodeContext
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

async def small_llm(prompt: str) -> str:
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )
    return response.choices[0].message.content

async def primary_llm(prompt: str) -> str:
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content

async def main():
    # Initialize code memory system
    system = CodeMemorySystem(
        small_model_fn=small_llm,
        db_path="code_memory.db",
        max_memories=10,
        enable_caching=True
    )

    print("🔍 Indexing codebase...")

    # Index the sample project
    memory_ids = system.index_repository(
        "sample_project",
        recursive=True
    )

    print(f"✅ Indexed {len(memory_ids)} code entities\n")

    # Add documentation
    doc_id = system.add_documentation_memory(
        title="Authentication System",
        content="Our authentication system uses SHA256 for password hashing. "
                "Tokens must be at least 32 characters long.",
        category="security"
    )
    print(f"📝 Added documentation: {doc_id}\n")

    # Query 1: How does authentication work?
    print("=" * 60)
    print("Query 1: How does authentication work?")
    print("=" * 60)

    context1 = CodeContext(
        user_query="How does the authentication system work?"
    )

    response1 = await system.query(context1, primary_model_fn=primary_llm)
    print(f"\n{response1}\n")

    # Query 2: Code context-aware query
    print("=" * 60)
    print("Query 2: Debugging authentication issue")
    print("=" * 60)

    context2 = CodeContext(
        user_query="Fix the authentication bug",
        current_file="sample_project/auth.py",
        errors="Token validation always returns False",
        accessed_files=["sample_project/auth.py"]
    )

    response2 = await system.query(context2, primary_model_fn=primary_llm)
    print(f"\n{response2}\n")

    # Get statistics
    stats = system.get_stats()
    print("=" * 60)
    print("System Statistics")
    print("=" * 60)
    print(f"Total code entities: {stats['total_code_memories']}")
    print(f"Total documentation: {stats['total_non_code_memories']}")
    print(f"Cache size: {stats['cache_stats']['size']}")

    system.close()
    print("\n✅ Tutorial complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3: Run It!

```bash
python tutorial3_code.py
```

**Expected Output**:
```
🔍 Indexing codebase...
✅ Indexed 7 code entities

📝 Added documentation: 123e4567-e89b-12d3-a456-426614174002

============================================================
Query 1: How does authentication work?
============================================================

The authentication system uses SHA256 for password hashing. The hash_password()
function in auth.py takes a password and returns its SHA256 hash. For token
validation, the validate_token() function checks that tokens are at least 32
characters long. The User class has an authenticate() method that hashes the
provided password and compares it with the stored hash.

============================================================
Query 2: Debugging authentication issue
============================================================

Looking at the validate_token() function in auth.py, the issue is likely in the
condition. The function checks if token length is less than 32, which would fail
for valid tokens. You might want to review the validation logic. Additionally,
ensure that tokens being passed aren't None or empty strings, which would cause
the length check to fail.

============================================================
System Statistics
============================================================
Total code entities: 7
Total documentation: 1
Cache size: 14

✅ Tutorial complete!
```

---

## Tutorial 4: Advanced Configuration

**Goal**: Learn to fine-tune the memory system for optimal performance.

**Time**: 5 minutes

### Scenario 1: High Precision (Strict Filtering)

```python
from memory_lib import MemorySystem

# For use cases where precision is critical
system = MemorySystem(
    small_model_fn=small_llm,
    relevance_threshold=0.85,  # Very strict
    max_memories=3,            # Only top 3
    batch_size=20              # Faster processing
)
```

**Use cases**:
- Legal document analysis
- Medical information retrieval
- Financial data queries

### Scenario 2: High Recall (Broad Retrieval)

```python
system = MemorySystem(
    small_model_fn=small_llm,
    relevance_threshold=0.5,   # Permissive
    max_memories=30,           # Many results
    batch_size=10
)
```

**Use cases**:
- Research and exploration
- Brainstorming sessions
- Comprehensive context gathering

### Scenario 3: Cost-Optimized

```python
from memory_lib import CodeMemorySystem

system = CodeMemorySystem(
    small_model_fn=small_llm,
    enable_caching=True,            # Maximize cache hits
    relevance_threshold=0.75,       # Balanced
    max_memories=10,                # Moderate context
    batch_size=5,                   # Conservative API usage
    dependency_boost_amount=0.20,   # Strong boost reduces queries
    recency_boost_amount=0.15       # Strong recency preference
)
```

**Use cases**:
- High-volume applications
- Limited API budget
- Development/testing

### Scenario 4: Latency-Optimized

```python
system = CodeMemorySystem(
    small_model_fn=small_llm,
    enable_caching=True,
    batch_size=50,              # Maximum parallelism
    max_memories=5,             # Smaller context = faster
    relevance_threshold=0.9     # Fewer memories to process
)
```

**Use cases**:
- Real-time applications
- Interactive tools
- High-frequency queries

### Dynamic Configuration Updates

```python
# Start with default config
system = MemorySystem(small_model_fn=small_llm)

# Adjust based on results
response = await system.query(context, task, primary_llm)

# If response lacks context, increase recall
system.update_retrieval_config(
    relevance_threshold=0.6,
    max_memories=20
)

# If response is too verbose or off-topic, increase precision
system.update_retrieval_config(
    relevance_threshold=0.85,
    max_memories=5
)
```

### Monitoring Performance

```python
import time

async def monitored_query(system, context, task, primary_llm):
    """Query with performance monitoring."""

    start = time.time()

    # Get stats before
    stats_before = system.get_stats()

    # Execute query
    response = await system.query(context, task, primary_llm)

    # Calculate metrics
    latency = time.time() - start
    stats_after = system.get_stats()

    print(f"📊 Performance Metrics:")
    print(f"  Latency: {latency:.2f}s")
    print(f"  Total memories: {stats_after['total_memories']}")
    print(f"  Threshold: {stats_after['relevance_threshold']}")
    print(f"  Max returned: {stats_after['max_memories']}")

    if hasattr(system, 'retrieval'):
        cache_stats = system.retrieval.get_cache_stats()
        print(f"  Cache size: {cache_stats.get('size', 'N/A')}")

    return response
```

---

## Next Steps

Congratulations! You've completed the Quick Start tutorials. Here's where to go next:

### Learn More

1. **API Reference** (`API_REFERENCE.md`)
   - Complete method documentation
   - All parameters and options
   - Advanced usage patterns

2. **Architecture Guide** (`01_ARCHITECTURE.md`)
   - System design
   - Component interactions
   - Design decisions

3. **Component Details** (`02_COMPONENTS.md`)
   - Deep dive into each component
   - Internal workings
   - Extension points

### Explore Examples

Check the `examples/` directory:
- `general_memory_example.py` - Comprehensive general system demo
- `code_memory_example.py` - Full code intelligence demo

### Common Use Cases

1. **Customer Support Bot**
   - Store company policies, FAQs
   - Retrieve relevant answers
   - Personalize responses

2. **Code Assistant**
   - Index your codebase
   - Answer "how do I..." questions
   - Debug with context

3. **Research Assistant**
   - Store papers, articles, notes
   - Find relevant information
   - Synthesize knowledge

4. **Personal Knowledge Base**
   - Store learnings, ideas
   - Retrieve when needed
   - Build on past knowledge

### Best Practices

1. **Start Simple**: Begin with default configuration
2. **Iterate**: Adjust based on results
3. **Monitor**: Track performance metrics
4. **Cache**: Enable caching for repeated queries
5. **Clean Up**: Close systems when done

### Get Help

- **Documentation**: See docs/ directory
- **Examples**: See examples/ directory
- **Issues**: [Your issue tracker]
- **Community**: [Your community forum]

---

Happy coding! 🚀
