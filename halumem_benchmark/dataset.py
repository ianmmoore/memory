"""HaluMem dataset loading and parsing.

Handles loading dialogue data, ground truth memories, QA pairs, and update scenarios
from the HaluMem benchmark dataset (arXiv:2511.03506).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
import logging

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    """A single turn in a dialogue.

    Attributes:
        role: Either "user" or "assistant".
        content: The text content of the turn.
        turn_id: Unique identifier for this turn.
        timestamp: Optional timestamp for the turn.
    """
    role: str
    content: str
    turn_id: int
    timestamp: Optional[str] = None


@dataclass
class MemoryPoint:
    """A ground truth memory point.

    Attributes:
        memory_id: Unique identifier.
        text: The memory content.
        source_turns: Turn IDs where this memory was mentioned.
        importance: Importance weight (0-1) for weighted recall.
        category: Category of the memory (e.g., "personal", "preference").
        is_target: Whether this is a target memory for retrieval.
    """
    memory_id: str
    text: str
    source_turns: List[int] = field(default_factory=list)
    importance: float = 1.0
    category: Optional[str] = None
    is_target: bool = False


@dataclass
class QAPair:
    """A question-answer pair for evaluation.

    Attributes:
        qa_id: Unique identifier.
        question: The question to answer.
        ground_truth_answer: The expected answer.
        relevant_memory_ids: IDs of memories needed to answer.
        question_type: Type of question (single-hop, multi-hop, temporal, etc.).
        difficulty: Difficulty level (easy, medium, hard).
    """
    qa_id: str
    question: str
    ground_truth_answer: str
    relevant_memory_ids: List[str] = field(default_factory=list)
    question_type: str = "single-hop"
    difficulty: str = "medium"


@dataclass
class UpdateScenario:
    """A memory update scenario for testing update capabilities.

    Attributes:
        scenario_id: Unique identifier.
        old_memory: The existing memory content.
        new_information: New information that may conflict.
        expected_action: Expected action (ADD, UPDATE, DELETE, NOOP).
        expected_result: Expected memory state after update.
        conflict_type: Type of conflict (contradiction, refinement, etc.).
    """
    scenario_id: str
    old_memory: str
    new_information: str
    expected_action: str
    expected_result: Optional[str] = None
    conflict_type: str = "contradiction"


class HaluMemDataset:
    """Loader for HaluMem benchmark dataset.

    This class handles loading and parsing the HaluMem dataset files,
    providing iterators for dialogues, memories, QA pairs, and update scenarios.

    Example:
        >>> from halumem_benchmark import HaluMemDataset, HaluMemConfig
        >>> config = HaluMemConfig.for_long()
        >>> dataset = HaluMemDataset(config)
        >>> await dataset.load()
        >>> for qa in dataset.get_qa_pairs():
        ...     print(qa.question)
    """

    def __init__(self, config: "HaluMemConfig"):
        """Initialize the dataset loader.

        Args:
            config: HaluMem configuration with dataset paths.
        """
        from .config import HaluMemConfig
        self.config = config
        self.dialogues: List[List[DialogueTurn]] = []
        self.memories: Dict[str, MemoryPoint] = {}
        self.qa_pairs: List[QAPair] = []
        self.update_scenarios: List[UpdateScenario] = []
        self.importance_weights: Dict[str, float] = {}
        self._loaded = False

    async def load(self) -> None:
        """Load the dataset from files.

        This loads all dataset files asynchronously. If files don't exist,
        it will attempt to download them from the HaluMem repository.
        """
        files = self.config.get_dataset_files()

        # Check if dataset exists, if not provide instructions
        if not files["dialogues"].exists():
            await self._setup_dataset()

        # Load each component
        self._load_dialogues(files["dialogues"])
        self._load_memories(files["memories"])
        self._load_qa_pairs(files["qa_pairs"])
        self._load_update_scenarios(files["updates"])
        self._load_importance_weights(files["importance_weights"])

        self._loaded = True
        logger.info(
            f"Loaded HaluMem-{self.config.variant}: "
            f"{len(self.dialogues)} dialogues, "
            f"{len(self.memories)} memories, "
            f"{len(self.qa_pairs)} QA pairs, "
            f"{len(self.update_scenarios)} update scenarios"
        )

    async def _setup_dataset(self) -> None:
        """Set up the dataset directory with instructions or download."""
        dataset_path = self.config.dataset_path / self.config.variant
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Create placeholder files with instructions
        readme_path = dataset_path / "README.md"
        if not readme_path.exists():
            readme_content = f"""# HaluMem-{self.config.variant.capitalize()} Dataset

## Setup Instructions

The HaluMem dataset is from arXiv:2511.03506.

### Option 1: Download from official source
1. Visit the HaluMem paper/repository
2. Download the {self.config.variant} variant
3. Extract files to this directory

### Option 2: Generate synthetic test data
Run the following to generate synthetic test data for development:

```python
from halumem_benchmark.dataset import HaluMemDataset
dataset = HaluMemDataset.generate_synthetic(variant="{self.config.variant}")
dataset.save("{dataset_path}")
```

### Expected Files
- dialogues.jsonl: Conversation turns
- ground_truth_memories.jsonl: Expected extracted memories
- qa_pairs.jsonl: Question-answer pairs for evaluation
- update_scenarios.jsonl: Memory update test cases
- importance_weights.json: Weights for weighted recall
"""
            readme_path.write_text(readme_content)
            logger.warning(f"Dataset not found. Created README at {readme_path}")

        # Generate minimal synthetic data for testing
        self._generate_synthetic_placeholder(dataset_path)

    def _generate_synthetic_placeholder(self, path: Path) -> None:
        """Generate minimal synthetic data for testing."""
        # Dialogues
        dialogues_file = path / "dialogues.jsonl"
        if not dialogues_file.exists():
            sample_dialogue = {
                "dialogue_id": "synth_001",
                "turns": [
                    {"role": "user", "content": "Hi, my name is Alice and I work as a software engineer.", "turn_id": 0},
                    {"role": "assistant", "content": "Nice to meet you, Alice! How long have you been a software engineer?", "turn_id": 1},
                    {"role": "user", "content": "I've been coding for about 5 years. I mainly use Python.", "turn_id": 2},
                    {"role": "assistant", "content": "Python is a great language! What kind of projects do you work on?", "turn_id": 3},
                    {"role": "user", "content": "I build machine learning systems, particularly for NLP.", "turn_id": 4},
                ]
            }
            with open(dialogues_file, "w") as f:
                json.dump(sample_dialogue, f)
                f.write("\n")

        # Ground truth memories
        memories_file = path / "ground_truth_memories.jsonl"
        if not memories_file.exists():
            memories = [
                {"memory_id": "m001", "text": "Alice is a software engineer", "source_turns": [0], "importance": 0.9, "category": "personal"},
                {"memory_id": "m002", "text": "Alice has 5 years of coding experience", "source_turns": [2], "importance": 0.8, "category": "experience"},
                {"memory_id": "m003", "text": "Alice mainly uses Python", "source_turns": [2], "importance": 0.7, "category": "skills"},
                {"memory_id": "m004", "text": "Alice works on machine learning systems for NLP", "source_turns": [4], "importance": 0.85, "category": "work"},
            ]
            with open(memories_file, "w") as f:
                for m in memories:
                    json.dump(m, f)
                    f.write("\n")

        # QA pairs
        qa_file = path / "qa_pairs.jsonl"
        if not qa_file.exists():
            qa_pairs = [
                {"qa_id": "q001", "question": "What is Alice's profession?", "ground_truth_answer": "Software engineer", "relevant_memory_ids": ["m001"], "question_type": "single-hop"},
                {"qa_id": "q002", "question": "How many years has Alice been coding?", "ground_truth_answer": "5 years", "relevant_memory_ids": ["m002"], "question_type": "single-hop"},
                {"qa_id": "q003", "question": "What programming language does Alice use and what does she build with it?", "ground_truth_answer": "Alice uses Python to build machine learning systems for NLP", "relevant_memory_ids": ["m003", "m004"], "question_type": "multi-hop"},
            ]
            with open(qa_file, "w") as f:
                for qa in qa_pairs:
                    json.dump(qa, f)
                    f.write("\n")

        # Update scenarios
        updates_file = path / "update_scenarios.jsonl"
        if not updates_file.exists():
            updates = [
                {"scenario_id": "u001", "old_memory": "Alice has 5 years of coding experience", "new_information": "Alice mentioned she recently completed her 6th year as a developer", "expected_action": "UPDATE", "expected_result": "Alice has 6 years of coding experience", "conflict_type": "refinement"},
                {"scenario_id": "u002", "old_memory": "Alice mainly uses Python", "new_information": "Alice is now learning Rust for systems programming", "expected_action": "ADD", "expected_result": "Alice is learning Rust", "conflict_type": "addition"},
            ]
            with open(updates_file, "w") as f:
                for u in updates:
                    json.dump(u, f)
                    f.write("\n")

        # Importance weights
        weights_file = path / "importance_weights.json"
        if not weights_file.exists():
            weights = {"m001": 0.9, "m002": 0.8, "m003": 0.7, "m004": 0.85}
            with open(weights_file, "w") as f:
                json.dump(weights, f, indent=2)

        logger.info(f"Generated synthetic placeholder data at {path}")

    def _load_dialogues(self, path: Path) -> None:
        """Load dialogue data from JSONL file."""
        if not path.exists():
            logger.warning(f"Dialogues file not found: {path}")
            return

        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    turns = [
                        DialogueTurn(
                            role=t["role"],
                            content=t["content"],
                            turn_id=t.get("turn_id", i),
                            timestamp=t.get("timestamp")
                        )
                        for i, t in enumerate(data.get("turns", []))
                    ]
                    self.dialogues.append(turns)

    def _load_memories(self, path: Path) -> None:
        """Load ground truth memories from JSONL file."""
        if not path.exists():
            logger.warning(f"Memories file not found: {path}")
            return

        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    memory = MemoryPoint(
                        memory_id=data["memory_id"],
                        text=data["text"],
                        source_turns=data.get("source_turns", []),
                        importance=data.get("importance", 1.0),
                        category=data.get("category"),
                        is_target=data.get("is_target", False)
                    )
                    self.memories[memory.memory_id] = memory

    def _load_qa_pairs(self, path: Path) -> None:
        """Load QA pairs from JSONL file."""
        if not path.exists():
            logger.warning(f"QA pairs file not found: {path}")
            return

        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    qa = QAPair(
                        qa_id=data["qa_id"],
                        question=data["question"],
                        ground_truth_answer=data["ground_truth_answer"],
                        relevant_memory_ids=data.get("relevant_memory_ids", []),
                        question_type=data.get("question_type", "single-hop"),
                        difficulty=data.get("difficulty", "medium")
                    )
                    self.qa_pairs.append(qa)

    def _load_update_scenarios(self, path: Path) -> None:
        """Load update scenarios from JSONL file."""
        if not path.exists():
            logger.warning(f"Update scenarios file not found: {path}")
            return

        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    scenario = UpdateScenario(
                        scenario_id=data["scenario_id"],
                        old_memory=data["old_memory"],
                        new_information=data["new_information"],
                        expected_action=data["expected_action"],
                        expected_result=data.get("expected_result"),
                        conflict_type=data.get("conflict_type", "contradiction")
                    )
                    self.update_scenarios.append(scenario)

    def _load_importance_weights(self, path: Path) -> None:
        """Load importance weights from JSON file."""
        if not path.exists():
            logger.warning(f"Importance weights file not found: {path}")
            return

        with open(path) as f:
            self.importance_weights = json.load(f)

    def get_dialogues(self) -> List[List[DialogueTurn]]:
        """Get all dialogues."""
        return self.dialogues

    def get_dialogue_text(self, dialogue_idx: int = 0) -> str:
        """Get full dialogue as formatted text.

        Args:
            dialogue_idx: Index of the dialogue to retrieve.

        Returns:
            Formatted dialogue text.
        """
        if dialogue_idx >= len(self.dialogues):
            return ""

        turns = self.dialogues[dialogue_idx]
        lines = []
        for turn in turns:
            role_label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role_label}: {turn.content}")
        return "\n\n".join(lines)

    def get_memories(self) -> Dict[str, MemoryPoint]:
        """Get all ground truth memories."""
        return self.memories

    def get_qa_pairs(self, limit: Optional[int] = None) -> List[QAPair]:
        """Get QA pairs, optionally limited.

        Args:
            limit: Maximum number of pairs to return. Uses config.sample_size if None.

        Returns:
            List of QA pairs.
        """
        effective_limit = limit or self.config.sample_size
        if effective_limit:
            return self.qa_pairs[:effective_limit]
        return self.qa_pairs

    def get_update_scenarios(self) -> List[UpdateScenario]:
        """Get all update scenarios."""
        return self.update_scenarios

    def get_importance_weight(self, memory_id: str) -> float:
        """Get importance weight for a memory.

        Args:
            memory_id: The memory ID.

        Returns:
            Importance weight (0-1), defaults to 1.0 if not found.
        """
        return self.importance_weights.get(memory_id, 1.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_turns = sum(len(d) for d in self.dialogues)
        return {
            "variant": self.config.variant,
            "num_dialogues": len(self.dialogues),
            "total_turns": total_turns,
            "num_memories": len(self.memories),
            "num_qa_pairs": len(self.qa_pairs),
            "num_update_scenarios": len(self.update_scenarios),
            "question_types": self._count_by_field(self.qa_pairs, "question_type"),
            "memory_categories": self._count_by_field(list(self.memories.values()), "category")
        }

    def _count_by_field(self, items: List[Any], field: str) -> Dict[str, int]:
        """Count items by a field value."""
        counts: Dict[str, int] = {}
        for item in items:
            value = getattr(item, field, None) or "unknown"
            counts[value] = counts.get(value, 0) + 1
        return counts
