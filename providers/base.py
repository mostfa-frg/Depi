from abc import ABC, abstractmethod

from llm.model import MessageRequest


class LLMProvider(ABC):
    """Interface implemented by every LLM backend."""

    @abstractmethod
    def generate(self, request: MessageRequest) -> str:
        """Generate text for a structured message request."""
        raise NotImplementedError

    def generate_response(self, request: str | MessageRequest) -> str:
        """Backward-compatible convenience method accepting plain text."""
        if isinstance(request, str):
            request = MessageRequest.from_text(request)
        return self.generate(request)
