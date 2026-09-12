from openai import OpenAI

from llm.config import LLMConfig
from llm.model import MessageRequest
from providers.base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI and APIs exposing the Chat Completions contract."""

    def __init__(self, config: LLMConfig, client: OpenAI | None = None) -> None:
        self.config = config
        self.client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    def generate(self, request: MessageRequest) -> str:
        messages = [
            {"role": message.role.value, "content": message.content}
            for message in request.messages
        ]
        kwargs: dict[str, object] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_new_tokens,
        }
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            # top_k is not part of the OpenAI contract; compatible APIs may accept it.
            kwargs["extra_body"] = {"top_k": request.top_k}

        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

        if not content:
            raise RuntimeError("LLM generation returned an empty response.")
        return content
