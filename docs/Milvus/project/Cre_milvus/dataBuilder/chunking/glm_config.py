"""
简化GLM配置管理模块

专门处理GLM-4.5-flash模型的配置管理、验证和实例化功能
统一使用YAML配置文件管理所有配置
"""

import logging
import os
import yaml
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class GLMConfig:
    """GLM配置信息"""
    model_name: str = "glm-4.5-flash"
    api_key: str = ""
    api_endpoint: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    provider: str = "zhipu"
    is_active: bool = True
    created_at: Optional[str] = None
    last_validated: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GLMConfig':
        return cls(
            model_name=data.get('model_name', 'glm-4.5-flash'),
            api_key=data.get('api_key', ''),
            api_endpoint=data.get('api_endpoint', 'https://open.bigmodel.cn/api/paas/v4/chat/completions'),
            provider=data.get('provider', 'zhipu'),
            is_active=data.get('is_active', True),
            created_at=data.get('created_at'),
            last_validated=data.get('last_validated')
        )


class GLMConfigService:
    """GLM配置管理服务 - 基于YAML配置文件"""
    
    def __init__(self, config_file: str = None):
        # 查找config.yaml文件
        if config_file is None:
            config_file = self._find_config_file()
        
        self.config_file = config_file
        self.config: Optional[GLMConfig] = None
        self._load_config()
    
    def _find_config_file(self) -> str:
        """查找config.yaml文件"""
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml"),
            os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"),
            "config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果找不到，使用默认路径
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    

    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            if not os.path.exists(self.config_file):
                return {}
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"加载YAML配置失败: {e}")
            return {}
    
    def _save_yaml_config(self, config_data: Dict[str, Any]) -> bool:
        """保存YAML配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存YAML配置失败: {e}")
            return False
    
#     def _load_config(self) -> None:
#         """从YAML文件加载GLM配置"""
#         try:
#             yaml_config = self._load_yaml_config()
            
#             # 尝试从active_llm_config指定的配置中加载
#             active_llm = yaml_config.get('active_llm_config', 'glm_default')
#             llm_configs = yaml_config.get('llm_configs', {})
            
#             if active_llm in llm_configs:
#                 llm_config = llm_configs[active_llm]
#                 api_key = llm_config.get('api_key', '')
                
#                 # 解密API密钥
#                 if api_key:
#                     api_key = self._decrypt_api_key(api_key)
                
#                 self.config = GLMConfig(
#                     model_name=llm_config.get('model_name', 'glm-4.5-flash'),
#                     api_key=api_key,
#                     api_endpoint=llm_config.get('api_endpoint', 'https://open.bigmodel.cn/api/paas/v4/chat/completions'),
#                     provider=llm_config.get('provider', 'zhipu'),
#                     is_active=bool(api_key),
#                     created_at=llm_config.get('created_at'),
#                     last_validated=llm_config.get('last_validated')
#                 )
#             else:
#                 # 如果没有找到配置，创建默认配置
#                 self.config = GLMConfig()
            
#             logger.info("GLM配置加载成功")
            
#         except Exception as e:
#             logger.error(f"加载GLM配置失败: {e}")
#             self.config = GLMConfig()
    
#     def _save_config(self) -> bool:
#         """保存GLM配置到YAML文件"""
#         if not self.config:
#             return False
        
#         try:
#             yaml_config = self._load_yaml_config()
            
#             # 确保llm_configs存在
#             if 'llm_configs' not in yaml_config:
#                 yaml_config['llm_configs'] = {}
            
#             # 获取当前活跃配置名
#             active_llm = yaml_config.get('active_llm_config', 'glm_default')
            
#             # 加密API密钥
#             api_key = self.config.api_key
#             if api_key:
#                 api_key = self._encrypt_api_key(api_key)
            
#             # 更新配置
#             yaml_config['llm_configs'][active_llm] = {
#                 'model_name': self.config.model_name,
#                 'api_key': api_key,
#                 'api_endpoint': self.config.api_endpoint,
#                 'provider': self.config.provider,
#                 'created_at': self.config.created_at,
#                 'last_validated': self.config.last_validated
#             }
            
#             # 设置活跃配置
#             yaml_config['active_llm_config'] = active_llm
            
#             return self._save_yaml_config(yaml_config)
            
#         except Exception as e:
#             logger.error(f"保存GLM配置失败: {e}")
#             return False
    
#     def validate_api_key(self, api_key: str) -> Tuple[bool, str]:
#         """验证API密钥"""
#         if not api_key or len(api_key.strip()) == 0:
#             return False, "API密钥不能为空"
        
#         # 检查API密钥格式（智谱AI的密钥通常以特定格式开头）
#         api_key = api_key.strip()
#         if len(api_key) < 20:
#             return False, "API密钥长度不足，请检查密钥是否完整"
        
#         # 尝试连接测试
#         try:
#             from .api_client import create_api_client
            
#             client = create_api_client({
#                 "provider": "zhipu",
#                 "api_key": api_key,
#                 "api_endpoint": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
#                 "model": "glm-4.5-flash"
#             })
            
#             if not client:
#                 return False, "无法创建API客户端，请检查密钥格式"
            
#             # 尝试简单的API调用测试
#             test_messages = [{"role": "user", "content": "测试连接"}]
#             response = client.chat_completion(test_messages, max_tokens=1)
            
#             return True, "API密钥验证成功"
            
#         except Exception as e:
#             error_msg = str(e).lower()
#             if any(keyword in error_msg for keyword in ["quota", "rate limit", "billing", "usage"]):
#                 return True, "API密钥有效（但可能有使用限制）"
#             elif "unauthorized" in error_msg or "invalid" in error_msg:
#                 return False, "API密钥无效，请检查密钥是否正确"
#             elif "network" in error_msg or "connection" in error_msg:
#                 return False, "网络连接失败，请检查网络设置"
#             else:
#                 return False, f"API调用失败: {str(e)}"
    
#     def save_config(self, model_name: str, api_key: str) -> bool:
#         """保存GLM配置"""
#         try:
#             # 验证API密钥（可选，如果需要快速保存可以跳过验证）
#             # is_valid, message = self.validate_api_key(api_key)
#             # if not is_valid:
#             #     logger.error(f"API密钥验证失败: {message}")
#             #     return False
            
#             # 创建新配置
#             current_time = datetime.now().isoformat()
            
#             self.config = GLMConfig(
#                 model_name=model_name or "glm-4.5-flash",
#                 api_key=api_key.strip(),
#                 api_endpoint="https://open.bigmodel.cn/api/paas/v4/chat/completions",
#                 provider="zhipu",
#                 is_active=True,
#                 created_at=current_time,
#                 last_validated=current_time
#             )
            
#             return self._save_config()
            
#         except Exception as e:
#             logger.error(f"保存GLM配置失败: {e}")
#             return False
    
#     def get_active_config(self) -> Optional[Dict[str, str]]:
#         """获取当前激活的GLM配置"""
#         if not self.config or not self.config.is_active or not self.config.api_key:
#             return None
        
#         return {
#             "model_name": self.config.model_name,
#             "api_key": self.config.api_key,
#             "api_endpoint": self.config.api_endpoint,
#             "provider": self.config.provider
#         }
    
#     def test_connection(self) -> Tuple[bool, str]:
#         """测试当前配置的连接"""
#         if not self.config or not self.config.api_key:
#             return False, "未配置API密钥"
        
#         return self.validate_api_key(self.config.api_key)
    
#     def get_config_summary(self) -> Dict[str, Any]:
#         """获取配置摘要"""
#         if not self.config:
#             return {
#                 "configured": False,
#                 "model_name": None,
#                 "api_key_configured": False,
#                 "last_validated": None
#             }
        
#         return {
#             "configured": bool(self.config.api_key),
#             "model_name": self.config.model_name,
#             "api_key_configured": bool(self.config.api_key),
#             "api_key_preview": f"***{self.config.api_key[-4:]}" if self.config.api_key else None,
#             "last_validated": self.config.last_validated,
#             "is_active": self.config.is_active
#         }
    
#     def update_validation_time(self) -> bool:
#         """更新验证时间"""
#         if not self.config:
#             return False
        
#         self.config.last_validated = datetime.now().isoformat()
#         return self._save_config()
    
#     def clear_config(self) -> bool:
#         """清除配置"""
#         try:
#             # 清除YAML文件中的GLM配置
#             yaml_config = self._load_yaml_config()
#             active_llm = yaml_config.get('active_llm_config', 'glm_default')
            
#             if 'llm_configs' in yaml_config and active_llm in yaml_config['llm_configs']:
#                 yaml_config['llm_configs'][active_llm]['api_key'] = None
#                 yaml_config['llm_configs'][active_llm]['last_validated'] = None
#                 self._save_yaml_config(yaml_config)
            
#             self.config = GLMConfig()
#             logger.info("GLM配置已清除")
#             return True
#         except Exception as e:
#             logger.error(f"清除GLM配置失败: {e}")
#             return False


# # 全局GLM配置服务实例
# _glm_config_service = None

# def get_glm_config_service() -> GLMConfigService:
#     """获取GLM配置服务实例"""
#     global _glm_config_service
#     if _glm_config_service is None:
#         _glm_config_service = GLMConfigService()
#     return _glm_config_service

# def create_glm_api_client():
#     """创建GLM API客户端实例"""
#     try:
#         service = get_glm_config_service()
#         config = service.get_active_config()
        
#         if not config:
#             logger.debug("未找到有效的GLM配置")
#             return None
        
#         from .api_client import create_api_client
#         return create_api_client(config)
        
#     except Exception as e:
#         logger.error(f"创建GLM API客户端失败: {e}")
#         return None