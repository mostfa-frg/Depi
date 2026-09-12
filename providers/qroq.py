from openai import OpenAI
from ..llm.config import LLMConfig

class Groq(OpenAI):
    def __init__(self, config: LLMConfig):
        """
        Initializes the Groq client with the given configuration.

        Args:
            config (LLMConfig): The configuration object containing API parameters.
        """
        self.client = super().__init__(
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
            timeout=config.timeout,
            max_retries=config.max_retries
        )


    def generate_response(self, request: str) -> str:
        """
        Generates a response from the Groq LLM based on the given prompt.

        Args:
            request (str): The input prompt for the LLM.
        Returns:
            str: The generated response from the LLM.
        """
        try:
            response = self.responses.create(
                input=request,
                model=self.config.MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                    "content": request
                }
                for message in request.messages if message.strip()  # Split the request into messages and filter out empty lines
                    ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p
        )
        except Exception as e:
        # Handle exceptions that may occur during the API request
        # You can log the error, raise a custom exception, or return an error message
        # For example, you might want to log the error and return a user-friendly message
        # Log the error (you can use a logging library or print to console)
        # print(f"Error generating response: {e}")
         raise RuntimeError(f"Error generating response: {e}")
    
        return response.output_text   
     