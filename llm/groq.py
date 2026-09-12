from groq import Groq
from base import LLM
from model import MessageRequest, Message, MessageRole

class GroqLLM(LLM):
    def __init__(self, api_key: str, model_name: str ):
        self.client = Groq(api_key=api_key)
        self.client.set_model(model_name)

    def generate(self, request: MessageRequest) -> str:
        messages = [{"role": message.role.value, "content": message.content} for message in request.messages]
        response = self.client.generate(
            messages=messages,
            top_p=request.top_p,
            max_new_tokens=request.max_new_tokens,
            top_k=request.top_k,
            temperature=request.temperature
        )
        return response["content"]