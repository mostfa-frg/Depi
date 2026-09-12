import os 
from dataclasses import dataclass, field # dataclass decorator is used to automatically generate special methods like __init__() and __repr__() for the class
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    """
    Configuration class for the application.
    This class holds all the configuration parameters required for the application to run.
    """
    provider: str = field(default_factory=lambda: os.environ.get("PROVIDER", "groq"))
    GROQ_API_KEY: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    GROQ_BASE_URL: str = field(default_factory=lambda: os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    MODEL_NAME: str = field(default_factory=lambda: os.environ.get("MODEL_NAME", "openai/gpt-oss-20b"))
    temperature: float = field(default_factory=lambda: float(os.environ.get("TEMPERATURE", 0.7)))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("MAX_TOKENS", 256)))
    top_p: float = field(default_factory=lambda: float(os.environ.get("TOP_P", 1.0)))
    timeout: int = field(default_factory=lambda: int(os.environ.get("TIMEOUT", 30)))
    max_retries: int = field(default_factory=lambda: int(os.environ.get("MAX_RETRIES", 3)))

    def __post_init__(self):
        """
        Post-initialization processing to ensure that the configuration parameters are valid.
        """
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY must be set in the environment variables.")
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1.")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be a positive integer.")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("Top P must be between 0 and 1.")
        if self.timeout <= 0:
            raise ValueError("Timeout must be a positive integer.")
        if self.max_retries < 0:
            raise ValueError("Max retries must be a non-negative integer.")
