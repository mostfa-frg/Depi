from .config import LLMConfig

class LLMClient:
    """
    Client class for interacting with the LLM API.
    This class uses the configuration parameters defined in LLMConfig to make requests to the API.
    """

    def __init__(self, config: LLMConfig , provider: str = "groq"):
        """
        Initializes the LLMClient with the given configuration.

        Args:
            config (LLMConfig): The configuration object containing API parameters.
        """
        self.config = config
        self.provider = provider
        # Initialize the client with the provided configuration
        # For example, you might set up an HTTP client here using the base URL and API key

    def generate_response(self, request: str) -> str:
        """
        Generates a response from the LLM based on the given prompt.

        Args:
            prompt (str): The input prompt for the LLM.

        Returns:
            str: The generated response from the LLM.

        """
        response = self._make_request(request)

        # Implementation for generating LLM response
        pass