"""Multi-provider LLM client supporting OpenAI, Anthropic, and AWS Bedrock."""

import json
import time
from dataclasses import dataclass
from typing import Optional

import anthropic
import boto3
import openai

from src.core.config import get_settings
from src.core.exceptions import LLMError
from src.core.observability import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient:
    """Unified client that routes to OpenAI, Anthropic, or Bedrock based on model name.

    Provider detection:
        gpt-*       → OpenAI
        claude-*    → Anthropic
        amazon.*    → AWS Bedrock
    """

    _PROVIDER_PREFIXES = {
        "gpt-": "openai",
        "claude-": "anthropic",
        "amazon.": "bedrock",
    }

    def __init__(self) -> None:
        settings = get_settings()
        self._openai_client: Optional[openai.OpenAI] = None
        self._anthropic_client: Optional[anthropic.Anthropic] = None
        self._bedrock_client = None

        if settings.openai_api_key:
            self._openai_client = openai.OpenAI(api_key=settings.openai_api_key)
        if settings.anthropic_api_key:
            self._anthropic_client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key,
            )
        self._aws_region = settings.aws_region

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send a prompt to the right provider and return a normalized response."""
        model = model or get_settings().default_model
        provider = self._detect_provider(model)

        logger.info(
            "LLM call started",
            extra={"model": model, "provider": provider},
        )

        start = time.perf_counter()
        try:
            if provider == "openai":
                response = self._call_openai(
                    prompt, system_prompt, model, temperature, max_tokens,
                )
            elif provider == "anthropic":
                response = self._call_anthropic(
                    prompt, system_prompt, model, temperature, max_tokens,
                )
            elif provider == "bedrock":
                response = self._call_bedrock(
                    prompt, system_prompt, model, temperature, max_tokens,
                )
            else:
                raise LLMError(provider, f"Unsupported provider for model {model}")
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(provider, str(exc)) from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.latency_ms = round(elapsed_ms, 2)

        logger.info(
            "LLM call completed",
            extra={
                "model": model,
                "tokens": response.input_tokens + response.output_tokens,
                "duration_ms": response.latency_ms,
            },
        )
        return response

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _call_openai(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if self._openai_client is None:
            raise LLMError("openai", "OpenAI API key not configured")

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=0.0,  # filled by generate()
        )

    def _call_anthropic(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if self._anthropic_client is None:
            raise LLMError("anthropic", "Anthropic API key not configured")

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._anthropic_client.messages.create(**kwargs)

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return LLMResponse(
            content=content,
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=0.0,
        )

    def _call_bedrock(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=self._aws_region,
            )

        # Bedrock uses the Converse API for Claude-family models hosted on AWS.
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        system_config = [{"text": system_prompt}] if system_prompt else []

        response = self._bedrock_client.converse(
            modelId=model,
            messages=messages,
            system=system_config,
            inferenceConfig={
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        )

        output = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})

        return LLMResponse(
            content=output,
            model=model,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            latency_ms=0.0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_provider(self, model: str) -> str:
        for prefix, provider in self._PROVIDER_PREFIXES.items():
            if model.startswith(prefix):
                return provider
        raise LLMError("unknown", f"Cannot detect provider for model: {model}")
