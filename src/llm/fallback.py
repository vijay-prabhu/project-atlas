"""Provider failover chain for LLM calls.

Default order: OpenAI -> Anthropic -> Bedrock. If the primary provider fails
(network error, rate limit, timeout), the chain tries the next one.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.core.config import get_settings
from src.core.exceptions import LLMError
from src.core.observability import get_logger

from .client import LLMClient, LLMResponse
from .router import ModelRouter

logger = get_logger(__name__)

# Maps provider name to a sensible default model for that provider.
_PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "bedrock": "amazon.nova-lite-v1:0",
}

_DEFAULT_CHAIN_ORDER = ["openai", "anthropic", "bedrock"]


class FallbackChain:
    """Tries providers in order until one succeeds.

    Usage::

        chain = FallbackChain()
        response = chain.call_with_fallback(
            prompt="Classify this document.",
            system_prompt="You are a classifier.",
            task_type="classification",
        )
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        router: Optional[ModelRouter] = None,
        provider_order: Optional[list[str]] = None,
    ) -> None:
        self._client = client or LLMClient()
        self._router = router or ModelRouter()
        self._provider_order = provider_order or list(_DEFAULT_CHAIN_ORDER)

    def call_with_fallback(
        self,
        prompt: str,
        system_prompt: str = "",
        task_type: str = "simple_qa",
        complexity: str = "normal",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Try each provider in order until one returns a response.

        The router picks the ideal model for the task type. If that provider
        fails, we fall through to the next provider's default model.

        Args:
            prompt: User prompt.
            system_prompt: System-level instructions.
            task_type: Routing key (classification, extraction, etc.).
            complexity: "normal" or "high".
            temperature: Override the router's default temperature.
            max_tokens: Override the router's default max_tokens.

        Returns:
            LLMResponse from the first provider that succeeds.

        Raises:
            LLMError: If all providers in the chain fail.
        """
        config = self._router.route(task_type, complexity)
        temp = temperature if temperature is not None else config.temperature
        tokens = max_tokens if max_tokens is not None else config.max_tokens

        # Build the ordered list of (provider, model) pairs to try.
        # Start with the router's pick, then fall through the rest.
        attempts: list[tuple[str, str]] = [(config.provider, config.model_name)]
        for provider in self._provider_order:
            if provider == config.provider:
                continue
            attempts.append((provider, _PROVIDER_DEFAULT_MODELS[provider]))

        errors: list[str] = []

        for provider, model in attempts:
            try:
                logger.info(
                    "Fallback chain attempting provider",
                    extra={
                        "provider": provider,
                        "model": model,
                        "task_type": task_type,
                        "attempt": len(errors) + 1,
                    },
                )
                response = self._client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temp,
                    max_tokens=tokens,
                )
                if errors:
                    logger.info(
                        "Fallback chain recovered",
                        extra={
                            "provider": provider,
                            "model": model,
                            "previous_failures": len(errors),
                        },
                    )
                return response

            except (LLMError, Exception) as exc:
                reason = str(exc)
                errors.append(f"{provider}/{model}: {reason}")
                logger.warning(
                    "Provider failed, trying next",
                    extra={
                        "provider": provider,
                        "model": model,
                        "error": reason,
                    },
                )

        raise LLMError(
            provider="fallback_chain",
            message=(
                f"All providers failed for task '{task_type}'. "
                f"Errors: {'; '.join(errors)}"
            ),
        )
