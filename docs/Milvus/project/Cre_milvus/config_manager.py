"""
统一配置管理系统

提供全局配置加载、验证和管理功能
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = None
        self._default_config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "milvus": {
                "host": "127.0.0.1",
                "port": "19530",
                "vector_name": "default",
                "collection_name": "Test_one",
                "index_name": "IVF_FLAT",
                "replica_num": 1,
                "index_device": "cpu"
            },
            "system": {
                "url_split": False,
                "insert_mode": "覆盖（删除原有数据）"
            },
            "search": {
                "top_k": 20,
                "col_choice": "hdbscan",
                "reorder_strategy": "distance"
            },
            "data": {
                "data_location": "./data/upload/20250729"
            },
            "chunking": {
                "strategy": "traditional",
                "chunk_length": 512,
                "ppl_threshold": 0.3,
                "language": "zh",
                "model": {
                    "enable_advanced_chunking": False,
                    "use_api": True,
                    "api_type": "openai",
                    "api_key": "",
                    "api_base": "",
                    "model_name": "gpt-3.5-turbo",
                    "max_tokens": 1000,
                    "temperature": 0.1
                }
            },
            "multimodal": {
                "enable_image": False,
                "clip_model": "ViT-B/32",
                "image_formats": ["jpg", "jpeg", "png", "bmp"]
            },
            "performance": {
                "enable_custom_metrics": True,
                "max_history": 1000,
                "monitor_interval": 1.0
            },
            "testing": {
                "enable_performance_test": True,
                "test_collection_name": "locust_test_collection",
                "test_data_size": 10000,
                "locust_config": {
                    "users": 10,
                    "spawn_rate": 1,
                    "run_time": "60s"
                }
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    if config:
                        # 合并默认配置和用户配置
                        self._config = self._merge_configs(self._default_config, config)
                        logger.info(f"配置文件加载成功: {self.config_path}")
                    else:
                        self._config = self._default_config.copy()
                        logger.warning("配置文件为空，使用默认配置")
            else:
                self._config = self._default_config.copy()
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                # 创建默认配置文件
                self.save_config()
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}，使用默认配置")
            self._config = self._default_config.copy()
        
        return self._config
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并配置"""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """更新配置"""
        try:
            if self._config is None:
                self.load_config()
            
            # 递归更新配置
            self._config = self._merge_configs(self._config, updates)
            
            # 保存到文件
            return self.save_config()
        except Exception as e:
            logger.error(f"配置更新失败: {e}")
            return False
    
    def save_config(self) -> bool:
        """保存配置到文件"""
        try:
            # 确保目录存在
            config_dir = os.path.dirname(self.config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self._config, f, allow_unicode=True, default_flow_style=False)
            
            logger.info(f"配置已保存: {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
            return False
    
    def validate_config(self) -> Dict[str, Any]:
        """验证配置有效性"""
        issues = []
        config = self.get_config()
        
        # 验证Milvus配置
        milvus_config = config.get("milvus", {})
        if not milvus_config.get("host"):
            issues.append("Milvus host未配置")
        if not milvus_config.get("port"):
            issues.append("Milvus port未配置")
        if not milvus_config.get("collection_name"):
            issues.append("Milvus collection_name未配置")
        
        # 验证数据路径
        data_location = config.get("data", {}).get("data_location")
        if data_location and not os.path.exists(data_location):
            issues.append(f"数据路径不存在: {data_location}")
        
        # 验证分块配置
        chunking_config = config.get("chunking", {})
        chunk_length = chunking_config.get("chunk_length", 512)
        if not isinstance(chunk_length, int) or chunk_length < 100:
            issues.append("chunk_length必须是大于100的整数")
        
        ppl_threshold = chunking_config.get("ppl_threshold", 0.3)
        if not isinstance(ppl_threshold, (int, float)) or not 0 <= ppl_threshold <= 1:
            issues.append("ppl_threshold必须是0-1之间的数值")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config": config
        }
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置的特定部分"""
        config = self.get_config()
        return config.get(section, {})
    
    def set_section(self, section: str, values: Dict[str, Any]) -> bool:
        """设置配置的特定部分"""
        return self.update_config({section: values})

# 全局配置管理器实例
# 使用绝对路径初始化配置管理器
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
config_manager = ConfigManager(config_path)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载配置的便捷函数"""
    global config_manager
    if config_path:
        config_manager.config_path = config_path
    else:
        # 使用默认绝对路径
        config_manager.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    return config_manager.load_config()

def get_config() -> Dict[str, Any]:
    """获取当前配置的便捷函数"""
    return config_manager.get_config()

def update_config(updates: Dict[str, Any]) -> bool:
    """更新配置的便捷函数"""
    return config_manager.update_config(updates)

def validate_config() -> Dict[str, Any]:
    """验证配置的便捷函数"""
    return config_manager.validate_config()