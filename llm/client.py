from collections.abc import Callable

from llm.config import LLMConfig
from llm.model import MessageRequest
from providers.base import LLMProvider
from providers.openai_compatible import OpenAICompatibleProvider


class LLMClient:
    """Facade that routes requests to the configured provider."""

    def __init__(
        self,
        config: LLMConfig,
        provider: LLMProvider | None = None,
        provider_factory: Callable[[LLMConfig], LLMProvider] | None = None,
    ) -> None:
        self.config = config
        self.provider = provider or self._build_provider(provider_factory)

    def _build_provider(
        self, provider_factory: Callable[[LLMConfig], LLMProvider] | None
    ) -> LLMProvider:
        if provider_factory is not None:
            return provider_factory(self.config)

        provider_name = self.config.provider.lower().replace("-", "_")
        if provider_name in {"openai", "openai_compatible", "groq"}:
            return OpenAICompatibleProvider(self.config)
        raise ValueError(
            f"Unsupported provider '{self.config.provider}'. "
            "Use 'openai_compatible', 'openai', or 'groq', or pass a provider instance."
        )

    def generate(self, request: str | MessageRequest) -> str:
        if isinstance(request, str):
            request = MessageRequest.from_text(
                request,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        return self.provider.generate(request)

    def generate_response(self, request: str | MessageRequest) -> str:
        """Backward-compatible alias for :meth:`generate`."""
        return self.generate(request)
