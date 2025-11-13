"""Simple Terminus agent with built-in GPT-5 Responses API support."""

import os
import litellm
from pathlib import Path
from harbor.agents.terminus_2 import Terminus2
from harbor.models.agent.name import AgentName


# Configure litellm to use Responses API for GPT-5 models
# This is a global configuration that affects all litellm calls
litellm.set_verbose = False  # Set to True for debugging


class SimpleTerminusGPT5(Terminus2):
    """Simple Terminus2 agent configured for GPT-5 models.

    This agent configures litellm to automatically use the Responses API
    for GPT-5 models without needing a custom LLM wrapper.
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_turns: int | None = None,
        parser_name: str = "json",
        api_base: str | None = None,
        temperature: float = 0.7,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        session_id: str | None = None,
        enable_summarize: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize Terminus agent with GPT-5 support.

        Args:
            logs_dir: Directory for logs
            model_name: Model name (e.g., "gpt-5-codex")
            max_turns: Maximum conversation turns
            parser_name: Response parser ("json" or "xml")
            api_base: Optional API base URL
            temperature: LLM temperature
            logprobs: Whether to request log probabilities
            top_logprobs: Number of top log probs to return
            session_id: Optional session ID
            enable_summarize: Whether to enable summarization
        """
        # Force temperature=1.0 for GPT-5 models
        if model_name and "gpt-5" in model_name.lower():
            temperature = 1.0
            print(f"Setting temperature=1.0 for GPT-5 model: {model_name}")

        # Configure litellm for Responses API
        # This tells litellm to use /v1/responses endpoint for GPT-5 models
        if model_name and ("gpt-5" in model_name.lower() or "o3" in model_name.lower() or "o1" in model_name.lower()):
            # Set up custom model routing for Responses API
            os.environ["LITELLM_USE_RESPONSES_API"] = "true"
            print(f"Configured litellm to use Responses API for {model_name}")

        # Initialize parent Terminus2 with standard configuration
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            max_turns=max_turns,
            parser_name=parser_name,
            api_base=api_base,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            session_id=session_id,
            enable_summarize=enable_summarize,
            *args,
            **kwargs,
        )

    @staticmethod
    def name() -> AgentName:
        """Return agent name."""
        return AgentName.TERMINUS_2

    @staticmethod
    def version() -> str:
        """Return agent version."""
        return "1.0.0"