import asyncio
import time
import json
from typing import List, Type, TypeVar, Optional, Union, Dict, Any
from pydantic import BaseModel, ValidationError
from google import genai
from openai import OpenAI
from google.genai import types
from torch.cuda import temperature  
from config.config import get_config_dict

gemini_cnf = get_config_dict()['gemini_api']
gemini_cnf['temperature'] = float(gemini_cnf.get("temperature", 0.2))
openai_cnf = get_config_dict()["openai_api"]
openai_cnf["temperature"] = float(openai_cnf.get("temperature", 0.2))

T = TypeVar("T", bound=BaseModel)

class GeminiHandler:
    def __init__(self, api_key: str = gemini_cnf['api_key1'], model: str = gemini_cnf['model'], temperature: float = gemini_cnf.get("temperature", 0.2)):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        response_model: Optional[Type[T]] = None,
        max_output_tokens: Optional[int] = 1000
    ) -> Dict[str, Any]:
        """
        Generates content and returns a dictionary containing the result 
        and detailed token usage metadata.
        """
        start = time.time()
        # Configure generation parameters
        gen_config = {
            "temperature": self.temperature,
            "max_output_tokens": max_output_tokens,
            "system_instruction": system_prompt if system_prompt else None,
        }

        if response_model:
            gen_config["response_mime_type"] = "application/json"
            gen_config["response_schema"] = response_model

        # Create the config object
        config_obj = types.GenerateContentConfig(**gen_config)

        # Execute API call
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config_obj
        )

        latency = time.time() - start
        if response.text is None:
            raise ValueError("Response text is None; cannot parse response.")

        # Extract result content
        result = response.text
        if response_model:
            result = response_model.model_validate_json(response.text)

        # Return both the content and the token metadata
        usage_metadata = {
            "prompt_tokens": 0,
            "candidates_tokens": 0,
            "total_tokens": 0
        }
        
        if response.usage_metadata:
            usage_metadata = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }
        
        return {
            "content": result,
            "usage": usage_metadata,
            "lantency_sec": latency,
        }


class OpenAIHandler:
    def __init__(
        self,
        model: str = openai_cnf.get("model", "gpt-4o-mini"),
        api_key: Optional[str] = openai_cnf.get("api_key"),
        temperature: float = openai_cnf.get("temperature", 0.2),
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    # -----------------------------
    # Core call
    # -----------------------------
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_model: Optional[Type[BaseModel]] = None,
        max_output_tokens: int = 1000,
        # extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs OpenAI call and optionally parses into a Pydantic model.
        """

        start = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.responses.create(
            model=self.model,
            input=messages,
            temperature=self.temperature,
            max_output_tokens=max_output_tokens,
            # **(extra_params or {}),
        )

        latency = time.time() - start

        raw_text = response.output_text

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        parsed_obj = None

        # -----------------------------
        # Structured parsing
        # -----------------------------
        if response_model:
            try:
                json_data = self._extract_json(raw_text)
                parsed_obj = response_model.model_validate(json_data)
            except (ValidationError, json.JSONDecodeError) as e:
                print(f"[WARN] Failed structured parse: {e}")
        
        return {
            "content": parsed_obj,
            "usage": usage,
            "latency_sec": latency
        }

    # -----------------------------
    # Helpers
    # -----------------------------
    def _extract_json(self, text: str) -> dict:
        """
        Extract JSON even if model wraps in markdown.
        """
        text = text.strip()

        # remove ```json fences
        if "```" in text:
            text = text.split("```")[-2]

        return json.loads(text)


class LLMHandler:
    def __init__(self, llm_type: str = "gemini", **kwargs):
        if llm_type == "gemini":
            self.handler = GeminiHandler(**kwargs) if kwargs else GeminiHandler()
        elif llm_type == "openai":
            self.handler = OpenAIHandler(**kwargs) if kwargs else OpenAIHandler()
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_model: Optional[Type[BaseModel]] = None,
        max_output_tokens: int = 1000,
    ) -> Dict[str, Any]:
        return self.handler.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            response_model=response_model,
            max_output_tokens=max_output_tokens
        )