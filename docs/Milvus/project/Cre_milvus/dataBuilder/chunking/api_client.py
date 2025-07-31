import requests
import json
import logging
from typing import Dict, Any, Optional, List
import time

logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, api_type: str, api_key: str, api_base: str = "", 
                 model_name: str = "gpt-3.5-turbo", max_tokens: int = 1000, 
                 temperature: float = 0.1):
        self.api_type = api_type.lower()
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        if not self.api_base:
            self.api_base = self._get_default_api_base()
        
        self.headers = self._get_headers()
    
    def _get_default_api_base(self) -> str:
        defaults = {
            "openai": "https://api.openai.com/v1",
            "qwen": "https://dashscope.aliyuncs.com/api/v1",
            "zhipu": "https://open.bigmodel.cn/api/paas/v4",
            "deepseek": "https://api.deepseek.com/v1",
            "moonshot": "https://api.moonshot.cn/v1"
        }
        return defaults.get(self.api_type, "https://api.openai.com/v1")
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        max_retries = kwargs.get("max_retries", 3)
        retry_delay = kwargs.get("retry_delay", 1)
        
        for attempt in range(max_retries):
            try:
                data = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature)
                }
                
                if self.api_type == "qwen":
                    data = {
                        "model": self.model_name,
                        "input": {"messages": messages},
                        "parameters": {
                            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                            "temperature": kwargs.get("temperature", self.temperature)
                        }
                    }
                    url = f"{self.api_base}/services/aigc/text-generation/generation"
                else:
                    url = f"{self.api_base}/chat/completions"
                
                timeout = kwargs.get("timeout", 60)
                response = requests.post(url, headers=self.headers, json=data, timeout=timeout)
                
                if response.status_code == 200:
                    result = response.json()
                    if self.api_type == "qwen":
                        return result.get("output", {}).get("text", "")
                    else:
                        choices = result.get("choices", [])
                        if choices:
                            return choices[0].get("message", {}).get("content", "")
                elif response.status_code == 429:
                    time.sleep(retry_delay * 2)
                    continue
                else:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
            except requests.exceptions.RequestException:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
        
        return None
    
    def get_probability_scores(self, prompt: str, options: List[str]) -> Dict[str, float]:
        scores = {}
        choice_prompt = f"{prompt}\n\n请从以下选项中选择一个:\n"
        for i, option in enumerate(options):
            choice_prompt += f"{i+1}. {option}\n"
        choice_prompt += "\n请只回答数字(1-{})：".format(len(options))
        
        messages = [{"role": "user", "content": choice_prompt}]
        vote_counts = {str(i+1): 0 for i in range(len(options))}
        total_calls = 3
        successful_calls = 0
        
        for call_idx in range(total_calls):
            try:
                response = self.chat_completion(messages, temperature=0.7, max_retries=2, timeout=30)
                if response and response.strip() in vote_counts:
                    vote_counts[response.strip()] += 1
                    successful_calls += 1
                time.sleep(0.5)
            except Exception:
                pass
        
        if successful_calls > 0:
            for i, option in enumerate(options):
                scores[option] = vote_counts[str(i+1)] / successful_calls
        else:
            uniform_prob = 1.0 / len(options)
            for option in options:
                scores[option] = uniform_prob
        
        return scores
    
    def binary_choice(self, prompt: str, option1: str, option2: str) -> float:
        scores = self.get_probability_scores(prompt, [option1, option2])
        return scores.get(option1, 0.5)


class MockAPIClient(APIClient):
    def __init__(self, **kwargs):
        super().__init__(api_type="mock", api_key="mock", **kwargs)
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        last_message = messages[-1].get("content", "")
        if "请回答1或2" in last_message or "Please answer 1 or 2" in last_message:
            import random
            return str(random.randint(1, 2))
        return "这是一个模拟回复。"
    
    def binary_choice(self, prompt: str, option1: str, option2: str) -> float:
        import random
        return random.random()


def create_api_client(config: Dict[str, Any]) -> Optional[APIClient]:
    try:
        model_config = config.get("model", {})
        
        if not model_config.get("enable_advanced_chunking", False):
            return None
        
        if not model_config.get("use_api", False):
            return None
        
        api_key = model_config.get("api_key", "")
        if not api_key:
            return MockAPIClient()
        
        return APIClient(
            api_type=model_config.get("api_type", "openai"),
            api_key=api_key,
            api_base=model_config.get("api_base", ""),
            model_name=model_config.get("model_name", "gpt-3.5-turbo"),
            max_tokens=model_config.get("max_tokens", 1000),
            temperature=model_config.get("temperature", 0.1)
        )
        
    except Exception:
        return None