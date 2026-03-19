"""Task-based model routing that picks the optimal model per task type."""

from dataclasses import dataclass
from typing import Optional

from src.core.observability import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Routing decision for a single LLM call."""

    model_name: str
    provider: str
    max_tokens: int
    temperature: float
    estimated_cost_per_1k_tokens: float


# Routing table: task_type -> (default config, high-complexity config)
_ROUTING_RULES: dict[str, tuple[ModelConfig, ModelConfig]] = {
    "classification": (
        ModelConfig(
            model_name="gpt-4o-mini",
            provider="openai",
            max_tokens=256,
            temperature=0.0,
            estimated_cost_per_1k_tokens=0.00015,
        ),
        ModelConfig(
            model_name="gpt-4o",
            provider="openai",
            max_tokens=512,
            temperature=0.0,
            estimated_cost_per_1k_tokens=0.0025,
        ),
    ),
    "extraction": (
        ModelConfig(
            model_name="gpt-4o",
            provider="openai",
            max_tokens=2048,
            temperature=0.0,
            estimated_cost_per_1k_tokens=0.0025,
        ),
        ModelConfig(
            model_name="gpt-4o",
            provider="openai",
            max_tokens=4096,
            temperature=0.0,
            estimated_cost_per_1k_tokens=0.0025,
        ),
    ),
    "search_synthesis": (
        ModelConfig(
            model_name="claude-sonnet-4-20250514",
            provider="anthropic",
            max_tokens=2048,
            temperature=0.3,
            estimated_cost_per_1k_tokens=0.003,
        ),
        ModelConfig(
            model_name="claude-sonnet-4-20250514",
            provider="anthropic",
            max_tokens=4096,
            temperature=0.3,
            estimated_cost_per_1k_tokens=0.003,
        ),
    ),
    "simple_qa": (
        ModelConfig(
            model_name="gpt-4o-mini",
            provider="openai",
            max_tokens=512,
            temperature=0.2,
            estimated_cost_per_1k_tokens=0.00015,
        ),
        ModelConfig(
            model_name="gpt-4o",
            provider="openai",
            max_tokens=1024,
            temperature=0.2,
            estimated_cost_per_1k_tokens=0.0025,
        ),
    ),
}

_DEFAULT_CONFIG = ModelConfig(
    model_name="gpt-4o-mini",
    provider="openai",
    max_tokens=512,
    temperature=0.0,
    estimated_cost_per_1k_tokens=0.00015,
)

_DEFAULT_HIGH_CONFIG = ModelConfig(
    model_name="gpt-4o",
    provider="openai",
    max_tokens=1024,
    temperature=0.0,
    estimated_cost_per_1k_tokens=0.0025,
)


class ModelRouter:
    """Picks the right model for a given task type and complexity level.

    Low-stakes, fast tasks go to gpt-4o-mini. Complex reasoning tasks go to
    bigger models. The routing table is static for now but can later be backed
    by live cost/latency metrics.
    """

    def route(
        self,
        task_type: str,
        complexity: str = "normal",
    ) -> ModelConfig:
        """Return the best ModelConfig for the given task and complexity.

        Args:
            task_type: One of the known task types (classification, extraction,
                       search_synthesis, simple_qa) or any custom string.
            complexity: "normal" or "high". High complexity escalates to a
                        more capable (and expensive) model.
        """
        rules = _ROUTING_RULES.get(task_type)
        if rules is not None:
            normal_config, high_config = rules
        else:
            normal_config, high_config = _DEFAULT_CONFIG, _DEFAULT_HIGH_CONFIG

        escalated = complexity == "high"
        config = high_config if escalated else normal_config

        logger.info(
            "Model routing decision",
            extra={
                "task_type": task_type,
                "complexity": complexity,
                "model": config.model_name,
                "provider": config.provider,
                "escalated": escalated,
                "cost": config.estimated_cost_per_1k_tokens,
            },
        )

        return config
