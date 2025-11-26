# Memory Systems Documentation

Complete documentation for the Memory Systems library - an intelligent context retrieval system for LLM applications.

## Documentation Structure

This documentation is organized by audience and abstraction level:

### For All Users

📘 **[Quick Start Guide](QUICKSTART.md)**
- Step-by-step tutorials
- Code examples you can run immediately
- Common use cases
- **Start here if you're new!**

### By Abstraction Level

#### Executive / High-Level

📄 **[Executive Overview](00_OVERVIEW.md)**
- What is this system?
- Why does it matter?
- Business value
- Key features and benefits
- **For decision makers and stakeholders**

#### Technical Leadership / Architects

🏗️ **[Architecture Documentation](01_ARCHITECTURE.md)**
- System architecture
- Design philosophy
- Component interactions
- Technology stack
- Performance considerations
- **For technical leads and architects**

#### Developers / Users

🔧 **[Component Documentation](02_COMPONENTS.md)**
- Detailed component specifications
- Method signatures and parameters
- Data models
- Configuration options
- **For developers using the library**

📚 **[API Reference](API_REFERENCE.md)** ⭐ *Suitable for External Release*
- Complete public API documentation
- Installation guide
- Usage examples
- Best practices
- Error handling
- **The main reference for library users**

📖 **[Public API Guide](PUBLIC_API.md)** ⭐ *Customer-Focused Integration Guide*
- Simplified integration guide
- Quick start examples
- Cost estimation
- Best practices
- **For customers integrating the library**

#### Internal Teams / Maintainers

🔒 **[Internal Implementation](INTERNAL.md)** ⚠️ *CONFIDENTIAL - Internal Use Only*
- Core algorithms and their rationale
- Performance optimizations
- Implementation details
- Known limitations
- Future roadmap
- Debugging guide
- **For internal development teams only**

### Related Projects

🧪 **[HaluMem Benchmark](../halumem_benchmark/README.md)**
- Memory system evaluation suite
- Extraction, updating, and QA metrics
- HaluMem-Medium and HaluMem-Long variants
- Cost-optimized with prefiltering

🤖 **[Terminal-Bench Agent](../terminal_bench_agent/README.md)**
- Memory-guided agent for command-line tasks
- Plan-execute-observe architecture
- Harbor BaseAgent integration
- Cleanup management for Daytona/Docker

## Quick Navigation

### I want to...

- **Get started quickly** → [Quick Start Guide](QUICKSTART.md)
- **Understand what this is** → [Executive Overview](00_OVERVIEW.md)
- **Learn the system design** → [Architecture Documentation](01_ARCHITECTURE.md)
- **Look up API methods** → [API Reference](API_REFERENCE.md)
- **Understand a component** → [Component Documentation](02_COMPONENTS.md)
- **Debug an issue** → [Internal Implementation](INTERNAL.md) (internal only)

### I am a...

- **New User** → Start with [Quick Start Guide](QUICKSTART.md)
- **Executive/Manager** → Read [Executive Overview](00_OVERVIEW.md)
- **Technical Lead** → Review [Architecture Documentation](01_ARCHITECTURE.md)
- **Developer** → Use [API Reference](API_REFERENCE.md) + [Component Documentation](02_COMPONENTS.md)
- **Maintainer** → Study [Internal Implementation](INTERNAL.md)

## What's in This Library?

### General Memory System

A flexible foundation for storing and retrieving any type of information:

```python
from memory_lib import MemorySystem

system = MemorySystem(small_model_fn=your_llm_function)

# Store information
system.add_memory("Python uses dynamic typing")

# Retrieve relevant information
relevant = await system.retrieve_relevant_memories("Tell me about Python")

# Query with context
response = await system.query(
    context="User asking about Python",
    task="Explain Python features",
    primary_model_fn=your_primary_llm
)
```

**Use Cases**: Chatbots, customer support, knowledge management, research assistants

### Code Memory System

Specialized system for code intelligence:

```python
from memory_lib import CodeMemorySystem
from memory_lib.codebase import CodeContext

system = CodeMemorySystem(small_model_fn=your_llm_function)

# Index your codebase
system.index_repository("src/", exclude_patterns=["*/tests/*"])

# Query with rich context
context = CodeContext(
    user_query="How does authentication work?",
    current_file="api/auth.py",
    errors="AttributeError in validate_token"
)

response = await system.query(context, primary_model_fn=your_primary_llm)
```

**Use Cases**: Code assistants, documentation generation, debugging tools, code review

## Key Features

### 🎯 Exhaustive Scoring
Unlike traditional search, we score **every single memory** using a small LLM for maximum precision. No relevant information is overlooked.

### 🚀 Smart Optimizations
- **Caching**: Cache scores for 60-80% cost reduction
- **Dependency Boosting**: Automatically surface related functions
- **Recency Boosting**: Prioritize recently modified code
- **Batch Processing**: Parallel API calls for speed

### 🔧 Flexible & Extensible
- Works with any LLM provider (OpenAI, Anthropic, custom)
- Configurable thresholds and limits
- Runtime configuration updates
- Clean, composable architecture

### 📊 Production Ready
- Comprehensive test coverage
- SQLite storage (zero setup)
- Handles 10,000+ memories efficiently
- Detailed error messages and logging

## Documentation Versions

### For External Release

The following documents are suitable for public/external release:

- ✅ [Quick Start Guide](QUICKSTART.md)
- ✅ [Executive Overview](00_OVERVIEW.md)
- ✅ [Architecture Documentation](01_ARCHITECTURE.md)
- ✅ [Component Documentation](02_COMPONENTS.md)
- ✅ [API Reference](API_REFERENCE.md)

These documents contain only public information about the system's capabilities and usage.

### For Internal Use Only

The following documents contain proprietary information:

- ⚠️ [Internal Implementation](INTERNAL.md) - **CONFIDENTIAL**

This document contains:
- Proprietary algorithms
- Performance benchmarks
- Cost analysis
- Internal design decisions
- Future roadmap

## Quick Examples

### Example 1: Simple Memory Storage

```python
import asyncio
from memory_lib import MemorySystem

async def main():
    system = MemorySystem(small_model_fn=your_llm)

    # Add memories
    system.add_memory("Paris is the capital of France")
    system.add_memory("The Eiffel Tower is in Paris")

    # Retrieve
    relevant = await system.retrieve_relevant_memories("Tell me about Paris")

    for memory in relevant:
        print(f"{memory.text} (score: {memory.relevance_score})")

asyncio.run(main())
```

### Example 2: Code Intelligence

```python
from memory_lib import CodeMemorySystem

async def main():
    system = CodeMemorySystem(small_model_fn=your_llm)

    # Index repository
    system.index_repository("src")

    # Query
    response = await system.query(
        "How do I add a new API endpoint?",
        primary_model_fn=your_primary_llm
    )

    print(response)

asyncio.run(main())
```

### Example 3: Chatbot with Memory

```python
from memory_lib import MemorySystem

class Chatbot:
    def __init__(self):
        self.memory = MemorySystem(small_model_fn=small_llm)

    def remember(self, fact):
        self.memory.add_memory(fact)

    async def respond(self, message):
        return await self.memory.query(
            context=message,
            task=f"Respond to: {message}",
            primary_model_fn=primary_llm
        )

# Use it
bot = Chatbot()
bot.remember("User's name is Alice")
response = await bot.respond("What's my name?")
# Response: "Your name is Alice!"
```

## Performance Characteristics

| Metric | General System | Code System (no cache) | Code System (cached) |
|--------|---------------|------------------------|---------------------|
| Latency (100 memories) | 3.2s | 4.1s | 0.15s |
| Cost per query | $0.08 | $0.08 | $0.03 |
| Throughput (queries/min) | 18 | 14 | 400 |

## System Requirements

- **Python**: 3.9 or higher
- **Storage**: ~1-10 KB per memory
- **Memory (RAM)**: Minimal (caching uses ~3 MB per 10K entries)
- **Dependencies**: Standard library + your LLM SDK

## Support & Resources

### Documentation
- All documentation is in the `docs/` directory
- Examples in `examples/` directory
- Tests in `tests/` directory

### Getting Help
- **Issues**: [Your issue tracker]
- **Questions**: [Your support email]
- **Community**: [Your forum/Discord]

### Contributing
- See `CONTRIBUTING.md` (if available)
- Internal teams: See [Internal Implementation](INTERNAL.md)

## Version Information

- **Current Version**: 1.0
- **Release Date**: [Your release date]
- **License**: [Your license]

## Changelog

### Version 1.0 (Initial Release)
- General memory system with LLM-based retrieval
- Code memory system with indexing
- Caching and optimization features
- Comprehensive documentation and examples
- Full test coverage

---

## Document Maintenance

These documents should be updated:
- **After major features**: Update all relevant docs
- **Before releases**: Review and update version info
- **On design changes**: Update architecture and internal docs
- **When APIs change**: Update API reference and component docs

**Maintainer**: [Your team/name]
**Last Updated**: [Date]

---

**Ready to get started?** → Begin with the [Quick Start Guide](QUICKSTART.md)!
