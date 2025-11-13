"""Custom LiteLLM wrapper that uses Responses API for GPT-5 models."""

from pathlib import Path
from typing import Any
import litellm
from litellm import Message

from harbor.llms.lite_llm import LiteLLM
from harbor.models.metric import UsageInfo


class ResponsesLLM(LiteLLM):
    """LiteLLM wrapper that uses Responses API for GPT-5 models.

    GPT-5-Codex and other GPT-5 models only work with the /v1/responses endpoint,
    not the standard /v1/chat/completions endpoint. This class automatically
    routes GPT-5 models to use litellm.aresponses() instead of litellm.acompletion().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if this is a GPT-5 model that needs Responses API
        self._use_responses_api = self._should_use_responses_api(self._model_name)

    def _should_use_responses_api(self, model_name: str) -> bool:
        """Check if model should use Responses API."""
        # GPT-5 models (gpt-5-codex, gpt-5, gpt-5-nano, etc.) need Responses API
        model_lower = model_name.lower()
        return "gpt-5" in model_lower or "o3" in model_lower or "o1" in model_lower

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any] | Message] = [],
        response_format: dict | None = None,
        logging_path: Path | None = None,
        **kwargs,
    ) -> str:
        """Call LLM using appropriate API (Responses API for GPT-5 models)."""

        if not self._use_responses_api:
            # Use standard completion API for non-GPT-5 models
            return await super().call(
                prompt=prompt,
                message_history=message_history,
                response_format=response_format,
                logging_path=logging_path,
                **kwargs,
            )

        # Use Responses API for GPT-5 models
        return await self._call_responses_api(
            prompt=prompt,
            message_history=message_history,
            logging_path=logging_path,
            **kwargs,
        )

    async def _call_responses_api(
        self,
        prompt: str,
        message_history: list[dict[str, Any] | Message] = [],
        logging_path: Path | None = None,
        **kwargs,
    ) -> str:
        """Call Responses API for GPT-5 models.

        The Responses API uses a different format:
        - Uses 'input' instead of 'messages'
        - Returns response in a different format
        """
        # Convert message history to input string
        # For now, we'll combine all messages into a single input
        full_input = ""
        for msg in message_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                full_input += f"{role}: {content}\n\n"

        # Add current prompt
        full_input += f"user: {prompt}"

        try:
            # Call Responses API
            response = await litellm.aresponses(
                model=self._model_name,
                input=full_input,
                temperature=self._temperature,
                **kwargs,
            )

            # Store response for usage extraction
            self._last_response = response

            # Extract text content from response
            # The Responses API returns ResponsesAPIResponse with an 'output' attribute
            # containing list of items [reasoning, message]
            try:
                # Get the output list from ResponsesAPIResponse or use response directly if it's a list
                if hasattr(response, 'output'):
                    items = response.output
                elif isinstance(response, list):
                    items = response
                elif isinstance(response, dict) and 'output' in response:
                    items = response['output']
                else:
                    # Not a recognized format, return as string
                    return str(response)

                # Find the output message (type='message')
                for item in reversed(items):
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content') and isinstance(item.content, list) and len(item.content) > 0:
                            content_item = item.content[0]
                            if hasattr(content_item, 'text'):
                                return content_item.text

                # Fallback - no message found
                return str(response)

            except (TypeError, AttributeError) as e:
                # Error extracting - return as string
                return str(response)

        except Exception as e:
            # Log the error and re-raise
            import logging
            logging.error(f"Responses API call failed: {e}")
            raise

    def get_last_usage(self) -> UsageInfo | None:
        """Extract token usage from Responses API response."""
        if self._last_response is None:
            return None

        try:
            # Responses API has different usage structure
            if hasattr(self._last_response, 'usage'):
                usage = self._last_response.usage

                # Responses API uses input_tokens/output_tokens instead of prompt_tokens/completion_tokens
                prompt_tokens = getattr(usage, 'input_tokens', 0) or 0
                completion_tokens = getattr(usage, 'output_tokens', 0) or 0

                # Get cache tokens from input_tokens_details
                cache_tokens = 0
                if hasattr(usage, 'input_tokens_details'):
                    input_tokens_details = usage.input_tokens_details
                    if input_tokens_details is not None:
                        cache_tokens = getattr(input_tokens_details, 'cached_tokens', 0) or 0

                # Get cost from _hidden_params
                cost = 0.0
                if hasattr(self._last_response, '_hidden_params'):
                    hidden_params = self._last_response._hidden_params
                    if isinstance(hidden_params, dict):
                        cost = hidden_params.get('response_cost', 0.0) or 0.0

                # Fallback cost calculation
                if cost == 0.0:
                    try:
                        cost = litellm.completion_cost(completion_response=self._last_response) or 0.0
                    except Exception:
                        cost = 0.0

                return UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cache_tokens=cache_tokens,
                    cost_usd=float(cost)
                )
        except (AttributeError, TypeError) as e:
            import logging
            logging.warning(f"Failed to extract usage from Responses API: {e}")
            return None

        return None
