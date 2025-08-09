"""
简化的配置加载器
替换复杂的配置管理逻辑，提供简单直接的配置加载和访问功能
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MilvusConfig:
    """Milvus配置数据类"""
    host: str
    port: str
    collection_name: str
    vector_name: str = "default"
    index_name: str = "IVF_FLAT"
    replica_num: int = 1
    index_device: str = "cpu"
    use_lite: bool = False

@dataclass
class SystemConfig:
    """系统配置数据类"""
    insert_mode: str = "overwrite"
    url_split: bool = False

@dataclass
class ChunkingConfig:
    """分块配置数据类"""
    strategy: str = "traditional"
    chunk_length: int = 512
    overlap: int = 50
    language: str = "zh"

class SimpleConfigLoader:
    """简化的配置加载器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config: Optional[Dict[str, Any]] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """加载配置文件"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                self._config = self._get_default_config()
                return
            
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            
            # 确保配置完整性
            self._ensure_config_completeness()
            logger.info(f"✅ 配置文件加载成功: {self.config_path}")
            
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            logger.info("使用默认配置")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "milvus": {
                "host": "./milvus_lite.db",
                "port": "19530",
                "collection_name": "Test_one",
                "vector_name": "default",
                "index_name": "IVF_FLAT",
                "replica_num": 1,
                "index_device": "cpu",
                "use_lite": True
            },
            "system": {
                "insert_mode": "overwrite",
                "url_split": False
            },
            "chunking": {
                "strategy": "traditional",
                "chunk_length": 512,
                "overlap": 50,
                "language": "zh"
            },
            "search": {
                "col_choice": "hdbscan",
                "reorder_strategy": "distance",
                "top_k": 20
            },
            "multimodal": {
                "enable_image": False,
                "clip_model": "ViT-B/32",
                "image_formats": ["jpg", "jpeg", "png", "bmp"]
            }
        }
    
    def _ensure_config_completeness(self) -> None:
        """确保配置完整性，添加缺失的默认值"""
        default_config = self._get_default_config()
        
        for section, default_values in default_config.items():
            if section not in self._config:
                self._config[section] = {}
            
            for key, default_value in default_values.items():
                if key not in self._config[section]:
                    self._config[section][key] = default_value
                    logger.debug(f"添加默认配置: {section}.{key} = {default_value}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._config.copy() if self._config else {}
    
    def get_milvus_config(self) -> MilvusConfig:
        """获取Milvus配置"""
        milvus_cfg = self._config.get("milvus", {})
        
        # 检测是否使用Milvus Lite
        host = milvus_cfg.get("host", "./milvus_lite.db")
        use_lite = milvus_cfg.get("use_lite", False) or host.endswith('.db')
        
        return MilvusConfig(
            host=host,
            port=str(milvus_cfg.get("port", "19530")),
            collection_name=milvus_cfg.get("collection_name", "Test_one"),
            vector_name=milvus_cfg.get("vector_name", "default"),
            index_name=milvus_cfg.get("index_name", "IVF_FLAT"),
            replica_num=int(milvus_cfg.get("replica_num", 1)),
            index_device=milvus_cfg.get("index_device", "cpu"),
            use_lite=use_lite
        )
    
    def get_system_config(self) -> SystemConfig:
        """获取系统配置"""
        sys_cfg = self._config.get("system", {})
        return SystemConfig(
            insert_mode=sys_cfg.get("insert_mode", "overwrite"),
            url_split=bool(sys_cfg.get("url_split", False))
        )
    
    def get_chunking_config(self) -> ChunkingConfig:
        """获取分块配置"""
        chunk_cfg = self._config.get("chunking", {})
        return ChunkingConfig(
            strategy=chunk_cfg.get("strategy", "traditional"),
            chunk_length=int(chunk_cfg.get("chunk_length", 512)),
            overlap=int(chunk_cfg.get("overlap", 50)),
            language=chunk_cfg.get("language", "zh")
        )
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取指定配置段"""
        return self._config.get(section, {}).copy()
    
    def save_config(self) -> bool:
        """保存配置到文件"""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"✅ 配置文件保存成功: {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"❌ 配置文件保存失败: {e}")
            return False
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> bool:
        """更新配置段"""
        try:
            if section not in self._config:
                self._config[section] = {}
            
            self._config[section].update(updates)
            logger.info(f"✅ 配置段更新成功: {section}")
            return True
        except Exception as e:
            logger.error(f"❌ 配置段更新失败: {e}")
            return False
    
    def reload_config(self) -> bool:
        """重新加载配置文件"""
        try:
            self._load_config()
            logger.info("✅ 配置文件重新加载成功")
            return True
        except Exception as e:
            logger.error(f"❌ 配置文件重新加载失败: {e}")
            return False

# 全局配置加载器实例
_config_loader: Optional[SimpleConfigLoader] = None

def get_config_loader(config_path: str = "config.yaml") -> SimpleConfigLoader:
    """获取全局配置加载器实例"""
    global _config_loader
    if _config_loader is None:
        _config_loader = SimpleConfigLoader(config_path)
    return _config_loader

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """简单的配置加载函数"""
    loader = get_config_loader(config_path)
    return loader.get_config()

def get_milvus_config(config_path: str = "config.yaml") -> MilvusConfig:
    """获取Milvus配置"""
    loader = get_config_loader(config_path)
    return loader.get_milvus_config()

def get_system_config(config_path: str = "config.yaml") -> SystemConfig:
    """获取系统配置"""
    loader = get_config_loader(config_path)
    return loader.get_system_config()

def save_config(config: Dict[str, Any], config_path: str = "config.yaml") -> bool:
    """保存配置"""
    loader = get_config_loader(config_path)
    loader._config = config
    return loader.save_config()