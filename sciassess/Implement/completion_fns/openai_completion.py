from typing import Any, List, Dict
import os
import openai
from .base_completion_fn import BaseCompletionFn
from openai import OpenAI

class OpenAICompletionFn(BaseCompletionFn):
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client with custom base URL if provided
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def get_completions(self, messages: List[Dict], **kwargs: Any) -> str:
        # print(kwargs)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # temperature=self.temperature,
                # max_tokens=self.max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            return "" 