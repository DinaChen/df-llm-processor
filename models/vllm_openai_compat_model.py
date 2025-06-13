import requests
from openai import OpenAI
from typing import Optional, Union


class VLLMOpenAICompatModel():

    #TODO: initialize from config

    def __init__(
        self,
        api_key: Union[str, None],
        base_url: str,
        model_name: str,
        prompt: str,
        max_tokens: int = 1000,
        temperature: Optional[float] = 0.9,
    ):
        self.api_key = api_key if api_key else "dummy_dina"
        self.base_url = base_url
        self.model_name = model_name
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        
    def get_llm_response(self, query:str):
        """
        Sends a prompt to the llm server via vLLM and returns the generated text.

        Args:
            query: the input query
        Returns:
            str: The generated text response or an error message.
        """

        try:
            client = OpenAI(api_key = self.api_key, base_url = self.base_url)

            response = client.chat.completions.create(
                model = self.model_name,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens = self.max_tokens,
                temperature = self.temperature,
            )

            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error: {e}"
 
"""
if __name__ == "__main__":

    # TODO: put this config to Class attribute
    prompt = "将下列一个或多个国家代号变成国家名字返回，不要有额外解释。"
    query = "ESP, FRA,NLD"
    url = "http://localhost:8800/v1"
    model_name = "Qwen2.5-14B-Instruct"


    llm = VLLMOpenAICompatModel(None, url, model_name, prompt)
    output = llm.get_llm_response(query)

    print("Response:", output)
"""