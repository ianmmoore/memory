"""Simple Terminus agent without any overrides for testing."""

from pathlib import Path

from harbor.agents.terminus_2 import Terminus2
from harbor.models.agent.name import AgentName

# Import custom ResponsesLLM for GPT-5 models
from terminal_bench_agent.responses_llm import ResponsesLLM


class SimpleTerminus(Terminus2):
    """Plain Terminus2 agent with no modifications - for testing."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *args,
        **kwargs,
    ):
        """Initialize with gpt-5-codex compatibility."""
        # gpt-5-codex requires temperature=1.0
        if model_name and "gpt-5" in model_name.lower():
            kwargs.setdefault("temperature", 1.0)

        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            *args,
            **kwargs,
        )

        # Override LLM with ResponsesLLM to support GPT-5 models
        self._llm = ResponsesLLM(
            model_name=model_name,
            api_base=kwargs.get("api_base"),
            temperature=kwargs.get("temperature", 0.7),
            logprobs=kwargs.get("logprobs", False),
            top_logprobs=kwargs.get("top_logprobs"),
            session_id=kwargs.get("session_id"),
        )

    @staticmethod
    def name() -> AgentName:
        """Return agent name."""
        return AgentName.TERMINUS_2

    @staticmethod
    def version() -> str:
        """Return agent version."""
        return "1.0.0-test"
