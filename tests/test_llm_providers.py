from types import SimpleNamespace

import pytest

from llm.client import LLMClient
from llm.config import LLMConfig
from llm.model import Message, MessageRequest, MessageRole
from providers.openai_compatible import OpenAICompatibleProvider


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="generated text"))]
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def config(**overrides):
    values = dict(
        api_key="test-key",
        base_url="https://example.test/v1",
        model_name="test-model",
    )
    values.update(overrides)
    return LLMConfig(**values)


def test_openai_compatible_provider_maps_messages_and_options():
    fake = FakeClient()
    provider = OpenAICompatibleProvider(config(), client=fake)
    request = MessageRequest(
        messages=[
            Message(MessageRole.SYSTEM, "Be concise."),
            Message(MessageRole.USER, "Hello"),
        ],
        max_new_tokens=42,
        temperature=0.2,
        top_p=0.9,
    )

    assert provider.generate(request) == "generated text"
    assert fake.chat.completions.kwargs == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ],
        "temperature": 0.2,
        "max_tokens": 42,
        "top_p": 0.9,
    }


def test_client_accepts_plain_text_and_uses_config_defaults():
    fake = FakeClient()
    client = LLMClient(config(max_tokens=12, temperature=0.1), provider=OpenAICompatibleProvider(config(), client=fake))

    assert client.generate_response("Hello") == "generated text"
    assert fake.chat.completions.kwargs["max_tokens"] == 12
    assert fake.chat.completions.kwargs["temperature"] == 0.1


def test_config_requires_an_api_key():
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        config(api_key="")
