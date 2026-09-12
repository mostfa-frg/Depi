from dataclasses import dataclass
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: MessageRole
    content: str


@dataclass(frozen=True)
class MessageRequest:
    messages: list[Message]
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float | None = 1.0
    top_k: int | None = None

    @classmethod
    def from_text(cls, prompt: str, **kwargs: object) -> "MessageRequest":
        return cls(messages=[Message(MessageRole.USER, prompt)], **kwargs)
