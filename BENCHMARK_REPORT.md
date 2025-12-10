# HaluMem Benchmark Report: MemorySystem

**Run ID:** memsys_batch_long_20251123_213256
**Variant:** HaluMem-long
**Completed:** November 28, 2025 at 23:37:17 UTC
**Total Runtime:** 74 hours (3.1 days)

---

## Executive Summary

MemorySystem was evaluated on the HaluMem-long benchmark, which tests memory extraction, updating, and retrieval capabilities. The system achieved moderate performance with **57.2% accuracy on memory updates** and **58.1% correctness on QA tasks**.

### Key Findings

| Metric | Score | Assessment |
|--------|-------|------------|
| Update Accuracy | 57.2% | Moderate |
| QA Correctness | 58.1% | Moderate |
| Hallucination Rate (Updates) | 0% | Excellent |
| Hallucination Rate (QA) | 56.9% | Needs Improvement |
| Omission Rate (QA) | 53.4% | Needs Improvement |

---

## Dataset Statistics

| Metric | Count |
|--------|-------|
| Dialogues | 2,417 |
| Total Turns | 107,032 |
| Ground Truth Memories | 14,948 |
| QA Pairs | 3,467 |
| Update Scenarios | 3,788 |

### Memory Categories
- **Persona Memory:** 9,116 (61%)
- **Event Memory:** 4,550 (30%)
- **Relationship Memory:** 1,282 (9%)

### Question Types
- Memory Boundary: 828
- Memory Conflict: 769
- Generalization & Application: 746
- Basic Fact Recall: 746
- Multi-hop Inference: 198
- Dynamic Update: 180

---

## Phase 2: Memory Updating Results

The updating phase tests the system's ability to correctly handle new information that may contradict, refine, or supplement existing memories.

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 57.2% |
| Refinement Accuracy | 57.2% |
| Omission Rate | 25.4% |
| Hallucination Rate | 0% |
| Scenarios Evaluated | 3,788 |

### Analysis

- **Zero hallucinations** in update decisions is excellent - the system never fabricated information during updates
- **25.4% omission rate** indicates the system sometimes fails to incorporate important new information
- Room for improvement in correctly identifying when memories need refinement

---

## Phase 3: Question Answering Results

The QA phase tests retrieval accuracy - whether the system can find and use relevant memories to answer questions correctly.

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Correctness** | 58.1% |
| Hallucination Rate | 56.9% |
| Omission Rate | 53.4% |
| Questions Evaluated | 3,467 |

### Performance by Question Type

| Question Type | Correctness | Assessment |
|---------------|-------------|------------|
| Memory Boundary | **87.8%** | Excellent |
| Memory Conflict | 69.8% | Good |
| Generalization & Application | 45.2% | Needs Work |
| Basic Fact Recall | 41.1% | Needs Work |
| Multi-hop Inference | 34.9% | Poor |
| Dynamic Update | **20.8%** | Critical |

### Analysis

**Strengths:**
- **Memory Boundary (87.8%)**: Excellent at knowing what it does and doesn't know
- **Memory Conflict (69.8%)**: Good at handling contradictory information

**Weaknesses:**
- **Dynamic Update (20.8%)**: Critical failure - system struggles to use recently updated memories
- **Multi-hop Inference (34.9%)**: Difficulty connecting multiple memories for complex reasoning
- **Basic Fact Recall (41.1%)**: Below expectations for straightforward retrieval

**High Hallucination Rate (56.9%)**: The system frequently generates plausible-sounding but incorrect information when it can't find the right memory.

---

## System Configuration

| Setting | Value |
|---------|-------|
| Total Memories Stored | 23,521 |
| Memories with Embeddings | 22,866 |
| Prefilter Enabled | Yes |
| Prefilter Top-K | 2,287 |
| Relevance Threshold | 0.5 |
| Max Memories per Query | 20 |

### Models Used
- **Extraction & Decisions:** gpt-5.1
- **Relevance Scoring:** gpt-5-nano
- **Answer Generation:** gpt-5.1
- **Answer Judging:** gpt-5.1

---

## Recommendations

### Critical Priority
1. **Fix Dynamic Update handling** - 20.8% accuracy indicates the system isn't properly surfacing recently updated memories. Investigate temporal weighting or recency signals.

### High Priority
2. **Reduce hallucinations in QA** - 56.9% rate is too high. Consider:
   - Lower confidence threshold before answering
   - Add "I don't know" as valid response
   - Improve retrieval to find more relevant memories

3. **Improve Multi-hop Inference** - 34.9% suggests memory connections aren't being made. Consider:
   - Graph-based memory relationships
   - Chain-of-thought retrieval

### Medium Priority
4. **Basic Fact Recall** - 41.1% should be higher. Review:
   - Embedding quality for factual statements
   - Retrieval relevance scoring

---

## Appendix: Timing Breakdown

| Phase | Duration |
|-------|----------|
| Extraction | (completed prior) |
| Updating | 39.5 hours |
| QA Scoring | 34.5 hours |
| **Total** | **74 hours** |

---

*Report generated: November 29, 2025*
*Benchmark: HaluMem-long v1.0*
