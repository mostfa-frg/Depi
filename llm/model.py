from dataclasses import dataclass
from enum import Enum


class MessageRole(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    
    role: MessageRole
    content: str
    content: str



@dataclass(frozen=True)
class MessageRequest:
    messages: list[Message]
    top_p: float | None = None
    max_new_tokens: int
    top_k: int |  None = None
    temperature: float = 0.7


