# """
# LLM API客户端模块

# 为分块策略提供大语言模型API调用支持
# """

# import json
# import logging
# from typing import Dict, Any, List, Optional
# from abc import ABC, abstractmethod

# logger = logging.getLogger(__name__)


# class LLMAPIClient(ABC):
#     """LLM API客户端抽象基类"""
    
#     @abstractmethod
#     def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
#         """聊天完成接口"""
#         pass
    
#     @abstractmethod
#     def binary_choice(self, prompt: str, option1: str, option2: str) -> float:
#         """二元选择接口，返回选择option1的概率"""
#         pass


# class OpenAIClient(LLMAPIClient):
#     """OpenAI API客户端"""
    
#     def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", base_url: Optional[str] = None):
#         self.api_key = api_key
#         self.model = model
#         self.base_url = base_url or "https://api.openai.com/v1"
        
#         try:
#             import openai
#             self.client = openai.OpenAI(
#                 api_key=api_key,
#                 base_url=base_url
#             )
#         except ImportError:
#             raise ImportError("请安装openai库: pip install openai")
    
#     def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
#         """OpenAI聊天完成"""
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=messages,
#                 max_tokens=kwargs.get('max_tokens', 10),
#                 temperature=kwargs.get('temperature', 0.1),
#                 **kwargs
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             logger.error(f"OpenAI API调用失败: {e}")
#             raise
    
#     def binary_choice(self, prompt: str, option1: str, option2: str) -> float:
#         """OpenAI二元选择"""
#         try:
#             messages = [{"role": "user", "content": prompt}]
#             response = self.chat_completion(messages, max_tokens=5)
            
#             # 简单的概率估算
#             if option1.lower() in response.lower():
#                 return 0.8
#             elif option2.lower() in response.lower():
#                 return 0.2
#             else:
#                 return 0.5
#         except Exception as e:
#             logger.error(f"OpenAI二元选择失败: {e}")
#             return 0.5


# class ClaudeClient(LLMAPIClient):
#     """Claude API客户端"""
    
#     def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
#         self.api_key = api_key
#         self.model = model
        
#         try:
#             import anthropic
#             self.client = anthropic.Anthropic(api_key=api_key)
#         except ImportError:
#             raise ImportError("请安装anthropic库: pip install anthropic")
    
#     def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
#         """Claude聊天完成"""
#         try:
#             # 转换消息格式
#             if messages and messages[0]["role"] == "system":
#                 system_message = messages[0]["content"]
#                 user_messages = messages[1:]
#             else:
#                 system_message = "You are a helpful assistant."
#                 user_messages = messages
            
#             response = self.client.messages.create(
#                 model=self.model,
#                 max_tokens=kwargs.get('max_tokens', 10),
#                 temperature=kwargs.get('temperature', 0.1),
#                 system=system_message,
#                 messages=user_messages
#             )
#             return response.content[0].text.strip()
#         except Exception as e:
#             logger.error(f"Claude API调用失败: {e}")
#             raise
    
#     def binary_choice(self, prompt: str, option1: str, option2: str) -> float:
#         """Claude二元选择"""
#         try:
#             messages = [{"role": "user", "content": prompt}]
#             response = self.chat_completion(messages, max_tokens=5)
            
#             # 简单的概率估算
#             if option1.lower() in response.lower():
#                 return 0.8
#             elif option2.lower() in response.lower():
#                 return 0.2
#             else:
#                 return 0.5
#         except Exception as e:
#             logger.error(f"Claude二元选择失败: {e}")
#             return 0.5


# class LocalModelClient(LLMAPIClient):
#     """本地模型客户端（通过HTTP API）"""
    
#     def __init__(self, base_url: str, model: str = "local-model"):
#         self.base_url = base_url.rstrip('/')
#         self.model = model
        
#         try:
#             import requests
#             self.session = requests.Session()
#         except ImportError:
#             raise ImportError("请安装requests库: pip install requests")
    
#     def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
#         """本地模型聊天完成"""
#         try:
#             payload = {
#                 "model": self.model,
#                 "messages": messages,
#                 "max_tokens": kwargs.get('max_tokens', 10),
#                 "temperature": kwargs.get('temperature', 0.1),
#                 "stream": False
#             }
            
#             response = self.session.post(
#                 f"{self.base_url}/v1/chat/completions",
#                 json=payload,
#                 timeout=30
#             )
#             response.raise_for_status()
            
#             result = response.json()
#             return result["choices"][0]["message"]["content"].strip()
#         except Exception as e:
#             logger.error(f"本地模型API调用失败: {e}")
#             raise
    
#     def binary_choice(self, prompt: str, option1: str, option2: str) -> float:
#         """本地模型二元选择"""
#         try:
#             messages = [{"role": "user", "content": prompt}]
#             response = self.chat_completion(messages, max_tokens=5)
            
#             # 简单的概率估算
#             if option1.lower() in response.lower():
#                 return 0.8
#             elif option2.lower() in response.lower():
#                 return 0.2
#             else:
#                 return 0.5
#         except Exception as e:
#             logger.error(f"本地模型二元选择失败: {e}")
#             return 0.5


# class ZhipuClient(LLMAPIClient):
#     """智谱AI API客户端"""
    
#     def __init__(self, api_key: str, model: str = "glm-4.5-flash", api_endpoint: str = None):
#         self.api_key = api_key
#         self.model = model
#         self.api_endpoint = api_endpoint or "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        
#         try:
#             import zhipuai
#             self.client = zhipuai.ZhipuAI(api_key=api_key)
#         except ImportError:
#             logger.warning("zhipuai库未安装，尝试使用requests直接调用")
#             self.client = None
#             try:
#                 import requests
#                 self.session = requests.Session()
#             except ImportError:
#                 raise ImportError("请安装zhipuai库: pip install zhipuai 或 requests库: pip install requests")
    
#     def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
#         """智谱AI聊天完成"""
#         try:
#             if self.client:
#                 # 使用官方SDK
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=messages,
#                     max_tokens=kwargs.get('max_tokens', 10),
#                     temperature=kwargs.get('temperature', 0.1),
#                     **kwargs
#                 )
#                 return response.choices[0].message.content.strip()
#             else:
#                 # 使用requests直接调用
#                 headers = {
#                     "Authorization": f"Bearer {self.api_key}",
#                     "Content-Type": "application/json"
#                 }
                
#                 payload = {
#                     "model": self.model,
#                     "messages": messages,
#                     "max_tokens": kwargs.get('max_tokens', 10),
#                     "temperature": kwargs.get('temperature', 0.1)
#                 }
                
#                 response = self.session.post(
#                     self.api_endpoint,
#                     headers=headers,
#                     json=payload,
#                     timeout=30
#                 )
#                 response.raise_for_status()
                
#                 result = response.json()
#                 return result["choices"][0]["message"]["content"].strip()
                
#         except Exception as e:
#             logger.error(f"智谱AI API调用失败: {e}")
#             raise
    
#     def binary_choice(self, prompt: str, option1: str, option2: str) -> float:
#         """智谱AI二元选择"""
#         try:
#             messages = [{"role": "user", "content": prompt}]
#             response = self.chat_completion(messages, max_tokens=5)
            
#             # 解析响应中的选择
#             response_lower = response.lower().strip()
            
#             # 直接匹配数字
#             if "1" in response_lower and "2" not in response_lower:
#                 return 0.8  # 选择选项1的概率高
#             elif "2" in response_lower and "1" not in response_lower:
#                 return 0.2  # 选择选项2的概率高
#             elif option1.lower() in response_lower:
#                 return 0.8
#             elif option2.lower() in response_lower:
#                 return 0.2
#             else:
#                 return 0.5  # 无法确定时返回中性概率
                
#         except Exception as e:
#             logger.error(f"智谱AI二元选择失败: {e}")
#             return 0.5


# class MockClient(LLMAPIClient):
#     """模拟客户端（用于测试和降级）"""
    
#     def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
#         """模拟聊天完成"""
#         # 简单的规则基础响应
#         content = messages[-1]["content"].lower() if messages else ""
        
#         if "分割" in content or "split" in content:
#             if "1" in content or "2" in content:
#                 return "1"  # 默认选择分割
        
#         return "1"
    
#     def binary_choice(self, prompt: str, option1: str, option2: str) -> float:
#         """模拟二元选择"""
#         # 简单的启发式规则
#         prompt_lower = prompt.lower()
        
#         # 如果提到分割相关词汇，倾向于分割
#         split_keywords = ["分割", "分开", "split", "separate", "。", "！", "？"]
#         merge_keywords = ["合并", "连接", "merge", "combine", "，", "、"]
        
#         split_score = sum(1 for keyword in split_keywords if keyword in prompt_lower)
#         merge_score = sum(1 for keyword in merge_keywords if keyword in prompt_lower)
        
#         if split_score > merge_score:
#             return 0.7
#         elif merge_score > split_score:
#             return 0.3
#         else:
#             return 0.5


# def create_api_client(config: Dict[str, Any]) -> Optional[LLMAPIClient]:
#     """
#     根据配置创建API客户端
    
#     Args:
#         config: LLM配置
        
#     Returns:
#         API客户端实例或None
#     """
#     if not config:
#         return None
    
#     provider = config.get("provider", "").lower()
    
#     try:
#         if provider == "openai":
#             return OpenAIClient(
#                 api_key=config.get("api_key", ""),
#                 model=config.get("model", "gpt-3.5-turbo"),
#                 base_url=config.get("base_url")
#             )
#         elif provider == "claude":
#             return ClaudeClient(
#                 api_key=config.get("api_key", ""),
#                 model=config.get("model", "claude-3-haiku-20240307")
#             )
#         elif provider == "zhipu":
#             return ZhipuClient(
#                 api_key=config.get("api_key", ""),
#                 model=config.get("model_name", config.get("model", "glm-4.5-flash")),
#                 api_endpoint=config.get("api_endpoint")
#             )
#         elif provider == "local":
#             return LocalModelClient(
#                 base_url=config.get("base_url", "http://localhost:8000"),
#                 model=config.get("model", "local-model")
#             )
#         elif provider == "mock":
#             return MockClient()
#         else:
#             logger.warning(f"未知的LLM提供商: {provider}")
#             return MockClient()
    
#     except Exception as e:
#         logger.error(f"创建API客户端失败: {e}")
#         return MockClient()


# def get_available_providers() -> List[Dict[str, Any]]:
#     """获取可用的LLM提供商列表"""
#     return [
#         {
#             "name": "openai",
#             "display_name": "OpenAI",
#             "description": "OpenAI GPT模型",
#             "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
#             "required_fields": ["api_key"],
#             "optional_fields": ["base_url"]
#         },
#         {
#             "name": "claude",
#             "display_name": "Anthropic Claude",
#             "description": "Anthropic Claude模型",
#             "models": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
#             "required_fields": ["api_key"],
#             "optional_fields": []
#         },
#         {
#             "name": "zhipu",
#             "display_name": "智谱AI",
#             "description": "智谱AI GLM模型",
#             "models": ["glm-4.5-flash", "glm-4", "glm-3-turbo"],
#             "required_fields": ["api_key"],
#             "optional_fields": ["api_endpoint"]
#         },
#         {
#             "name": "local",
#             "display_name": "本地模型",
#             "description": "本地部署的模型（兼容OpenAI API）",
#             "models": ["local-model"],
#             "required_fields": ["base_url"],
#             "optional_fields": ["model"]
#         },
#         {
#             "name": "mock",
#             "display_name": "模拟客户端",
#             "description": "用于测试的模拟客户端",
#             "models": ["mock"],
#             "required_fields": [],
#             "optional_fields": []
#         }
#     ]