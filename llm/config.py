import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration shared by all OpenAI-compatible providers."""

    provider: str = field(default_factory=lambda: os.getenv("PROVIDER", "openai_compatible"))
    api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
    )
    base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL")
        or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
    )
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "256")))
    top_p: float = field(default_factory=lambda: float(os.getenv("TOP_P", "1.0")))
    timeout: float = field(default_factory=lambda: float(os.getenv("TIMEOUT", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))

    # Backward-compatible aliases for code using the old Groq-specific names.
    @property
    def GROQ_API_KEY(self) -> str:
        return self.api_key

    @property
    def GROQ_BASE_URL(self) -> str:
        return self.base_url

    @property
    def MODEL_NAME(self) -> str:
        return self.model_name

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("LLM_API_KEY (or GROQ_API_KEY) must be set in the environment.")
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1.")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
