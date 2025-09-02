"""
LLM配置管理模块

提供LLM服务提供商的配置管理、验证和实例化功能
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import os

logger = logging.getLogger(__name__)

@dataclass
class LLMProvider:
    """LLM提供商信息"""
    name: str
    display_name: str
    description: str
    models: List[str]
    required_fields: List[str]
    optional_fields: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class LLMConfig:
    """LLM配置信息"""
    provider: str
    model_name: str
    api_key: str
    api_endpoint: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if result['additional_params'] is None:
            result['additional_params'] = {}
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
        return cls(
            provider=data.get('provider', ''),
            model_name=data.get('model_name', ''),
            api_key=data.get('api_key', ''),
            api_endpoint=data.get('api_endpoint'),
            additional_params=data.get('additional_params', {}),
            is_active=data.get('is_active', True)
        )


class LLMConfigManager:
    """LLM配置管理器"""
    
    def __init__(self, config_file: str = "llm_config.json"):
        self.config_file = config_file
        self.providers = self._load_providers()
        self.configs = self._load_configs()
    
    def _load_providers(self) -> Dict[str, LLMProvider]:
        """加载可用的LLM提供商"""
        providers_data = [
            {
                "name": "openai",
                "display_name": "OpenAI",
                "description": "OpenAI GPT系列模型，包括GPT-3.5和GPT-4",
                "models": [
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-16k",
                    "gpt-4",
                    "gpt-4-turbo",
                    "gpt-4-32k"
                ],
                "required_fields": ["api_key"],
                "optional_fields": ["api_endpoint", "organization"]
            },
            {
                "name": "claude",
                "display_name": "Anthropic Claude",
                "description": "Anthropic Claude系列模型，擅长长文本理解",
                "models": [
                    "claude-3-haiku-20240307",
                    "claude-3-sonnet-20240229",
                    "claude-3-opus-20240229",
                    "claude-2.1",
                    "claude-2.0"
                ],
                "required_fields": ["api_key"],
                "optional_fields": ["api_endpoint"]
            },
            {
                "name": "zhipu",
                "display_name": "智谱AI",
                "description": "智谱AI GLM系列模型，支持中文优化",
                "models": [
                    "glm-4",
                    "glm-4-air",
                    "glm-4-airx",
                    "glm-4-flash",
                    "glm-3-turbo"
                ],
                "required_fields": ["api_key"],
                "optional_fields": ["api_endpoint"]
            },
            {
                "name": "qwen",
                "display_name": "阿里云通义千问",
                "description": "阿里云通义千问系列模型",
                "models": [
                    "qwen-turbo",
                    "qwen-plus",
                    "qwen-max",
                    "qwen-max-longcontext"
                ],
                "required_fields": ["api_key"],
                "optional_fields": ["api_endpoint"]
            },
            {
                "name": "local",
                "display_name": "本地模型",
                "description": "本地部署的模型，兼容OpenAI API格式",
                "models": [
                    "local-model",
                    "llama-2-7b",
                    "llama-2-13b",
                    "chatglm-6b",
                    "baichuan-7b"
                ],
                "required_fields": ["api_endpoint"],
                "optional_fields": ["api_key", "model_name"]
            },
            {
                "name": "mock",
                "display_name": "模拟客户端",
                "description": "用于测试和开发的模拟客户端",
                "models": ["mock-model"],
                "required_fields": [],
                "optional_fields": []
            }
        ]
        
        providers = {}
        for provider_data in providers_data:
            provider = LLMProvider(**provider_data)
            providers[provider.name] = provider
        
        return providers
    
    def _load_configs(self) -> Dict[str, LLMConfig]:
        """从文件加载LLM配置"""
        if not os.path.exists(self.config_file):
            return {}
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            configs = {}
            for config_id, config_data in data.items():
                configs[config_id] = LLMConfig.from_dict(config_data)
            
            return configs
        except Exception as e:
            logger.error(f"加载LLM配置失败: {e}")
            return {}
    
    def _save_configs(self) -> bool:
        """保存LLM配置到文件"""
        try:
            data = {}
            for config_id, config in self.configs.items():
                data[config_id] = config.to_dict()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            logger.error(f"保存LLM配置失败: {e}")
            return False
    
    def get_available_providers(self) -> List[Dict[str, Any]]:
        """获取可用的LLM提供商列表"""
        return [provider.to_dict() for provider in self.providers.values()]
    
    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """获取特定提供商的信息"""
        provider = self.providers.get(provider_name)
        return provider.to_dict() if provider else None
    
    def add_llm_config(self, config_id: str, config: LLMConfig) -> bool:
        """添加LLM配置"""
        try:
            # 验证提供商是否存在
            if config.provider not in self.providers:
                raise ValueError(f"未知的LLM提供商: {config.provider}")
            
            # 验证必需字段
            validation_result = self.validate_config(config)
            if not validation_result[0]:
                raise ValueError(f"配置验证失败: {validation_result[1]}")
            
            self.configs[config_id] = config
            return self._save_configs()
        except Exception as e:
            logger.error(f"添加LLM配置失败: {e}")
            return False
    
    def update_llm_config(self, config_id: str, config: LLMConfig) -> bool:
        """更新LLM配置"""
        if config_id not in self.configs:
            return False
        
        return self.add_llm_config(config_id, config)
    
    def remove_llm_config(self, config_id: str) -> bool:
        """删除LLM配置"""
        if config_id in self.configs:
            del self.configs[config_id]
            return self._save_configs()
        return False
    
    def get_llm_config(self, config_id: str) -> Optional[LLMConfig]:
        """获取LLM配置"""
        return self.configs.get(config_id)
    
    def list_llm_configs(self) -> Dict[str, Dict[str, Any]]:
        """列出所有LLM配置"""
        result = {}
        for config_id, config in self.configs.items():
            config_dict = config.to_dict()
            # 隐藏敏感信息
            if config_dict.get('api_key'):
                config_dict['api_key'] = '*' * 8 + config_dict['api_key'][-4:]
            result[config_id] = config_dict
        return result
    
    def validate_config(self, config: LLMConfig) -> Tuple[bool, str]:
        """验证LLM配置"""
        try:
            # 检查提供商是否存在
            provider = self.providers.get(config.provider)
            if not provider:
                return False, f"未知的LLM提供商: {config.provider}"
            
            # 检查必需字段
            for field in provider.required_fields:
                if field == "api_key" and not config.api_key:
                    return False, "API密钥不能为空"
                elif field == "api_endpoint" and not config.api_endpoint:
                    return False, "API端点不能为空"
            
            # 检查模型是否在支持列表中
            if config.model_name and config.model_name not in provider.models:
                logger.warning(f"模型 {config.model_name} 不在 {config.provider} 的支持列表中")
            
            return True, "配置验证通过"
        except Exception as e:
            return False, f"配置验证失败: {str(e)}"
    
    def validate_api_key(self, provider: str, api_key: str, api_endpoint: Optional[str] = None) -> Tuple[bool, str]:
        """验证API密钥"""
        try:
            # 创建临时配置进行测试
            temp_config = LLMConfig(
                provider=provider,
                model_name="test",
                api_key=api_key,
                api_endpoint=api_endpoint
            )
            
            # 尝试创建API客户端
            from .api_client import create_api_client
            client = create_api_client({
                "provider": provider,
                "api_key": api_key,
                "api_endpoint": api_endpoint,
                "model": "test"
            })
            
            if not client:
                return False, "无法创建API客户端"
            
            # 尝试简单的API调用测试
            try:
                test_messages = [{"role": "user", "content": "测试"}]
                response = client.chat_completion(test_messages, max_tokens=1)
                return True, "API密钥验证成功"
            except Exception as api_error:
                # 某些API错误可能是正常的（如配额限制），但说明密钥有效
                error_msg = str(api_error).lower()
                if any(keyword in error_msg for keyword in ["quota", "rate limit", "billing", "usage"]):
                    return True, "API密钥有效（但可能有使用限制）"
                else:
                    return False, f"API调用失败: {str(api_error)}"
        
        except Exception as e:
            return False, f"API密钥验证失败: {str(e)}"
    
    def get_active_config(self) -> Optional[Tuple[str, LLMConfig]]:
        """获取当前激活的LLM配置"""
        for config_id, config in self.configs.items():
            if config.is_active:
                return config_id, config
        return None
    
    def set_active_config(self, config_id: str) -> bool:
        """设置激活的LLM配置"""
        if config_id not in self.configs:
            return False
        
        # 取消所有配置的激活状态
        for config in self.configs.values():
            config.is_active = False
        
        # 激活指定配置
        self.configs[config_id].is_active = True
        
        return self._save_configs()
    
    def create_api_client(self, config_id: Optional[str] = None):
        """创建API客户端实例"""
        try:
            if config_id:
                config = self.get_llm_config(config_id)
            else:
                active_config = self.get_active_config()
                config = active_config[1] if active_config else None
            
            if not config:
                return None
            
            from .api_client import create_api_client
            return create_api_client({
                "provider": config.provider,
                "api_key": config.api_key,
                "api_endpoint": config.api_endpoint,
                "model": config.model_name,
                **(config.additional_params or {})
            })
        except Exception as e:
            logger.error(f"创建API客户端失败: {e}")
            return None
    
    def export_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """导出配置（不包含敏感信息）"""
        config = self.get_llm_config(config_id)
        if not config:
            return None
        
        exported = config.to_dict()
        # 移除敏感信息
        exported.pop('api_key', None)
        return exported
    
    def import_config(self, config_id: str, config_data: Dict[str, Any], api_key: str) -> bool:
        """导入配置"""
        try:
            config_data['api_key'] = api_key
            config = LLMConfig.from_dict(config_data)
            return self.add_llm_config(config_id, config)
        except Exception as e:
            logger.error(f"导入配置失败: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        active_config = self.get_active_config()
        
        return {
            "total_configs": len(self.configs),
            "active_config": {
                "id": active_config[0] if active_config else None,
                "provider": active_config[1].provider if active_config else None,
                "model": active_config[1].model_name if active_config else None
            } if active_config else None,
            "available_providers": len(self.providers),
            "providers": list(self.providers.keys())
        }


def create_llm_config_manager(config_file: str = "llm_config.json") -> LLMConfigManager:
    """创建LLM配置管理器实例"""
    return LLMConfigManager(config_file)