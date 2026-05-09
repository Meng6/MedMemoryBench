"""Application configuration."""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    llm_model: str = "gpt-4o"
    llm_model_small: str = "gpt-4o-mini"
    llm_timeout: int = 300
    openai_api_key: str | None = None
    openai_api_base: str | None = None
    azure_api_key: str | None = None
    azure_api_base: str | None = None

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/med_eve.db"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Dialogue settings
    max_dialogue_turns: int = 20
    event_context_limit: int = 3
    default_time_span_days: int = 90

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
