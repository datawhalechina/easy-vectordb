"""
简化GLM配置管理模块

专门处理GLM-4.5-flash模型的配置管理、验证和实例化功能
"""

import json
import logging
import os
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
import base64
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class GLMConfig:
    """GLM配置信息"""
    model_name: str = "glm-4.5-flash"
    api_key: str = ""
    api_endpoint: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
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
            is_active=data.get('is_active', True),
            created_at=data.get('created_at'),
            last_validated=data.get('last_validated')
        )


class GLMConfigService:
    """GLM配置管理服务"""
    
    def __init__(self, config_file: str = "glm_config.json"):
        self.config_file = config_file
        self.config: Optional[GLMConfig] = None
        self._encryption_key = self._get_or_create_encryption_key()
        self._load_config()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """获取或创建加密密钥"""
        key_file = "glm_encryption.key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # 生成新的加密密钥
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def _encrypt_api_key(self, api_key: str) -> str:
        """加密API密钥"""
        try:
            fernet = Fernet(self._encryption_key)
            encrypted = fernet.encrypt(api_key.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"加密API密钥失败: {e}")
            return api_key  # 如果加密失败，返回原始密钥
    
    def _decrypt_api_key(self, encrypted_api_key: str) -> str:
        """解密API密钥"""
        try:
            fernet = Fernet(self._encryption_key)
            encrypted_bytes = base64.b64decode(encrypted_api_key.encode())
            decrypted = fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"解密API密钥失败: {e}")
            return encrypted_api_key  # 如果解密失败，返回原始值
    
    def _load_config(self) -> None:
        """从文件加载GLM配置"""
        if not os.path.exists(self.config_file):
            self.config = GLMConfig()
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 解密API密钥
            if 'api_key' in data and data['api_key']:
                data['api_key'] = self._decrypt_api_key(data['api_key'])
            
            self.config = GLMConfig.from_dict(data)
            logger.info("GLM配置加载成功")
            
        except Exception as e:
            logger.error(f"加载GLM配置失败: {e}")
            self.config = GLMConfig()
    
    def _save_config(self) -> bool:
        """保存GLM配置到文件"""
        if not self.config:
            return False
        
        try:
            data = self.config.to_dict()
            
            # 加密API密钥
            if data['api_key']:
                data['api_key'] = self._encrypt_api_key(data['api_key'])
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info("GLM配置保存成功")
            return True
            
        except Exception as e:
            logger.error(f"保存GLM配置失败: {e}")
            return False
    
    def validate_api_key(self, api_key: str) -> Tuple[bool, str]:
        """验证API密钥"""
        if not api_key or len(api_key.strip()) == 0:
            return False, "API密钥不能为空"
        
        # 检查API密钥格式（智谱AI的密钥通常以特定格式开头）
        api_key = api_key.strip()
        if len(api_key) < 20:
            return False, "API密钥长度不足，请检查密钥是否完整"
        
        # 尝试连接测试
        try:
            from .api_client import create_api_client
            
            client = create_api_client({
                "provider": "zhipu",
                "api_key": api_key,
                "api_endpoint": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                "model": "glm-4.5-flash"
            })
            
            if not client:
                return False, "无法创建API客户端，请检查密钥格式"
            
            # 尝试简单的API调用测试
            test_messages = [{"role": "user", "content": "测试连接"}]
            response = client.chat_completion(test_messages, max_tokens=1)
            
            return True, "API密钥验证成功"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["quota", "rate limit", "billing", "usage"]):
                return True, "API密钥有效（但可能有使用限制）"
            elif "unauthorized" in error_msg or "invalid" in error_msg:
                return False, "API密钥无效，请检查密钥是否正确"
            elif "network" in error_msg or "connection" in error_msg:
                return False, "网络连接失败，请检查网络设置"
            else:
                return False, f"API调用失败: {str(e)}"
    
    def save_config(self, model_name: str, api_key: str) -> bool:
        """保存GLM配置"""
        try:
            # 验证API密钥
            is_valid, message = self.validate_api_key(api_key)
            if not is_valid:
                logger.error(f"API密钥验证失败: {message}")
                return False
            
            # 创建新配置
            from datetime import datetime
            current_time = datetime.now().isoformat()
            
            self.config = GLMConfig(
                model_name=model_name or "glm-4.5-flash",
                api_key=api_key.strip(),
                api_endpoint="https://open.bigmodel.cn/api/paas/v4/chat/completions",
                is_active=True,
                created_at=current_time,
                last_validated=current_time
            )
            
            return self._save_config()
            
        except Exception as e:
            logger.error(f"保存GLM配置失败: {e}")
            return False
    
    def get_active_config(self) -> Optional[Dict[str, str]]:
        """获取当前激活的GLM配置"""
        if not self.config or not self.config.is_active or not self.config.api_key:
            return None
        
        return {
            "model_name": self.config.model_name,
            "api_key": self.config.api_key,
            "api_endpoint": self.config.api_endpoint,
            "provider": "zhipu"
        }
    
    def test_connection(self) -> Tuple[bool, str]:
        """测试当前配置的连接"""
        if not self.config or not self.config.api_key:
            return False, "未配置API密钥"
        
        return self.validate_api_key(self.config.api_key)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        if not self.config:
            return {
                "configured": False,
                "model_name": None,
                "api_key_configured": False,
                "last_validated": None
            }
        
        return {
            "configured": bool(self.config.api_key),
            "model_name": self.config.model_name,
            "api_key_configured": bool(self.config.api_key),
            "api_key_preview": f"***{self.config.api_key[-4:]}" if self.config.api_key else None,
            "last_validated": self.config.last_validated,
            "is_active": self.config.is_active
        }
    
    def update_validation_time(self) -> bool:
        """更新验证时间"""
        if not self.config:
            return False
        
        from datetime import datetime
        self.config.last_validated = datetime.now().isoformat()
        return self._save_config()
    
    def clear_config(self) -> bool:
        """清除配置"""
        try:
            self.config = GLMConfig()
            if os.path.exists(self.config_file):
                os.remove(self.config_file)
            logger.info("GLM配置已清除")
            return True
        except Exception as e:
            logger.error(f"清除GLM配置失败: {e}")
            return False


# 全局GLM配置服务实例
_glm_config_service = None

def get_glm_config_service() -> GLMConfigService:
    """获取GLM配置服务实例"""
    global _glm_config_service
    if _glm_config_service is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glm_config.json")
        _glm_config_service = GLMConfigService(config_path)
    return _glm_config_service

def create_glm_api_client():
    """创建GLM API客户端实例"""
    try:
        service = get_glm_config_service()
        config = service.get_active_config()
        
        if not config:
            return None
        
        from .api_client import create_api_client
        return create_api_client(config)
        
    except Exception as e:
        logger.error(f"创建GLM API客户端失败: {e}")
        return None