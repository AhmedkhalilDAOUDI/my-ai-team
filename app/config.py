from functools import lru_cache
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY") or None
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or None
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.6-sol")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
    jury_model: str = os.getenv("JURY_MODEL", "gpt-5.6-sol")
    auditor_model: str = os.getenv("AUDITOR_MODEL", "deepseek-v4-pro")
    synthesis_provider: str = os.getenv("SYNTHESIS_PROVIDER", "jury")
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "1600"))
    openai_input_cost_per_million: float = float(os.getenv("OPENAI_INPUT_COST_PER_MILLION", "4.00"))
    openai_output_cost_per_million: float = float(os.getenv("OPENAI_OUTPUT_COST_PER_MILLION", "20.00"))
    deepseek_input_cost_per_million: float = float(os.getenv("DEEPSEEK_INPUT_COST_PER_MILLION", "1.32"))
    deepseek_output_cost_per_million: float = float(os.getenv("DEEPSEEK_OUTPUT_COST_PER_MILLION", "3.96"))
    gemini_input_cost_per_million: float = float(os.getenv("GEMINI_INPUT_COST_PER_MILLION", "0.75"))
    gemini_output_cost_per_million: float = float(os.getenv("GEMINI_OUTPUT_COST_PER_MILLION", "3.75"))
    anthropic_input_cost_per_million: float = float(os.getenv("ANTHROPIC_INPUT_COST_PER_MILLION", "5.00"))
    anthropic_output_cost_per_million: float = float(os.getenv("ANTHROPIC_OUTPUT_COST_PER_MILLION", "25.00"))
    cost_warning_usd: float = float(os.getenv("COST_WARNING_USD", "0.05"))
    daily_budget_usd: float = float(os.getenv("DAILY_BUDGET_USD", "5.00"))
    app_access_token: str | None = os.getenv("APP_ACCESS_TOKEN") or None
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    video_vision_model: str = os.getenv("VIDEO_VISION_MODEL", "gpt-5.6-luna")
    enable_embeddings: bool = os.getenv("ENABLE_EMBEDDINGS", "false").lower() in {"1", "true", "yes", "on"}
    neo4j_uri: str | None = os.getenv("NEO4J_URI") or None
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str | None = os.getenv("NEO4J_PASSWORD") or None
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")


@lru_cache
def get_settings() -> Settings:
    return Settings()
