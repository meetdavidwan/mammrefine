import openai
import time
import anthropic
import os

class Model:
    """
    Unified interface for different language models.
    
    Supports OpenAI and Anthropic models with retry logic and error handling.
    """
    
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if model_name in [ "gpt-4o"]:
            self.model = OpenAIModel(model_name, self.openai_api_key)
        elif model_name in ["claude-3-sonnet"]:
            self.model = AnthropicModel(model_name, self.anthropic_api_key) 
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def run(self, prompt, max_retries=3):
        """
        Run the model with retry logic and exponential backoff.
        
        Args:
            prompt (str): The input prompt to send to the model
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            str or None: Model response or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                response = self.model.run(prompt)
                if response is not None:
                    return response
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Failed after {max_retries} attempts")
                    return None
        return None

class OpenAIModel:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)
    
    def run(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        return response.choices[0].message.content

class AnthropicModel:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def run(self, prompt):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text