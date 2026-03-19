"""Application configuration via pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables."""

    # LLM Providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    aws_region: str = "ca-central-1"

    # Vector DB
    pinecone_api_key: str = ""
    pinecone_index_name: str = "project-atlas"

    # Local Development
    dynamodb_endpoint: str = ""
    log_level: str = "INFO"

    # Agent Configuration
    confidence_auto_threshold: float = 0.85
    confidence_review_threshold: float = 0.5
    max_agent_steps: int = 5

    # Model Defaults
    default_model: str = "gpt-4o-mini"
    complex_model: str = "gpt-4o"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
