# abstraction class for LLM providers
from abc import ABC, abstractmethod


class LLMProvider:
    """
    Abstract base class for LLM providers.
    This class defines the interface that all LLM provider implementations must follow.
    """

    @abstractmethod
    def generate_response(self, request: str) -> str:
        """
        Generates a response from the LLM based on the given prompt.

        Args:
            request (str): The input prompt for the LLM.
        Returns:
            str: The generated response from the LLM.
        """
        pass
    