#!/usr/bin/env python3
"""
新的后端API
使用预连接架构，避免连接阻塞问题
"""

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import yaml
import logging
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
from pydantic import BaseModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Cre_milvus API",
    description="向量数据库管理API，支持预连接架构",
    version="2.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class MilvusConfig(BaseModel):
    host: str
    port: str
    collection_name: str
    vector_name: str = "default"
    index_name: str = "IVF_FLAT"
    replica_num: int = 1
    index_device: str = "cpu"

class SystemConfig(BaseModel):
    url_split: bool = False
    insert_mode: str = "覆盖（删除原有数据）"

class SearchConfig(BaseModel):
    top_k: int = 20
    col_choice: str = "hdbscan"
    reorder_strategy: str = "distance"

class DataConfig(BaseModel):
    data_location: str

class ChunkingConfig(BaseModel):
    strategy: str = "traditional"
    chunk_length: int = 512
    ppl_threshold: float = 0.3
    language: str = "zh"

class ConfigRequest(BaseModel):
    milvus: MilvusConfig
    system: SystemConfig
    search: SearchConfig
    data: DataConfig
    chunking: ChunkingConfig

class CollectionStateManager:
    """集合状态管理器"""
    
    def __init__(self):
        self._collection_states = {}
        self._state_lock = {}
    
    def ensure_collection_loaded(self, collection_name: str) -> bool:
        """确保集合已加载"""
        try:
            # 检查集合是否存在
            if not self._collection_exists(collection_name):
                logger.info(f"集合 {collection_name} 不存在，尝试创建")
                return self._create_collection_if_needed(collection_name)
            
            # 检查集合是否已加载
            if not self._is_collection_loaded(collection_name):
                logger.info(f"集合 {collection_name} 未加载，开始加载")
                return self.load_collection_with_retry(collection_name)
            
            logger.info(f"集合 {collection_name} 已加载")
            return True
            
        except Exception as e:
            logger.error(f"确保集合加载失败: {e}")
            return False
    
    def get_collection_status(self, collection_name: str) -> Dict[str, Any]:
        """获取集合状态"""
        try:
            exists = self._collection_exists(collection_name)
            loaded = self._is_collection_loaded(collection_name) if exists else False
            
            status = {
                "name": collection_name,
                "exists": exists,
                "loaded": loaded,
                "last_checked": datetime.now().isoformat(),
                "status": "ready" if (exists and loaded) else "not_ready"
            }
            
            if collection_name in self._collection_states:
                status.update(self._collection_states[collection_name])
            
            return status
            
        except Exception as e:
            logger.error(f"获取集合状态失败: {e}")
            return {
                "name": collection_name,
                "exists": False,
                "loaded": False,
                "error": str(e),
                "status": "error"
            }
    
    def create_collection_if_not_exists(self, collection_name: str, schema: Dict = None) -> bool:
        """如果集合不存在则创建"""
        try:
            if self._collection_exists(collection_name):
                return True
            
            logger.info(f"创建集合: {collection_name}")
            return self._create_collection_if_needed(collection_name, schema)
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def load_collection_with_retry(self, collection_name: str, max_retries: int = 3) -> bool:
        """带重试的集合加载"""
        for attempt in range(max_retries):
            try:
                logger.info(f"尝试加载集合 {collection_name} (第 {attempt + 1} 次)")
                
                # 更新状态
                self._collection_states[collection_name] = {
                    "load_status": "loading",
                    "load_attempt": attempt + 1,
                    "last_load_time": datetime.now().isoformat()
                }
                
                # 执行加载
                success = self._load_collection(collection_name)
                
                if success:
                    self._collection_states[collection_name].update({
                        "load_status": "loaded",
                        "loaded_at": datetime.now().isoformat()
                    })
                    logger.info(f"集合 {collection_name} 加载成功")
                    return True
                else:
                    logger.warning(f"集合 {collection_name} 加载失败，尝试 {attempt + 1}/{max_retries}")
                    
            except Exception as e:
                logger.error(f"加载集合时出错 (尝试 {attempt + 1}): {e}")
                
            # 等待后重试
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
        
        # 所有重试都失败
        self._collection_states[collection_name] = {
            "load_status": "failed",
            "error": "加载重试次数已用完",
            "failed_at": datetime.now().isoformat()
        }
        return False
    
    def _collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        try:
            from milvusBuilder.fast_insert import check_collection_exists
            return check_collection_exists(collection_name)
        except ImportError:
            # 如果没有fast_insert模块，尝试其他方式
            try:
                from milvusBuilder.persistent_connection import get_persistent_connection
                from pymilvus import utility
                conn = get_persistent_connection()
                connection_alias = conn.get_connection_alias()
                if connection_alias:
                    return utility.has_collection(collection_name, using=connection_alias)
                else:
                    logger.error("没有可用的Milvus连接")
                    return False
            except Exception as e:
                logger.error(f"检查集合存在性失败: {e}")
                return False
    
    def _is_collection_loaded(self, collection_name: str) -> bool:
        """检查集合是否已加载"""
        try:
            from milvusBuilder.fast_insert import is_collection_loaded
            return is_collection_loaded(collection_name)
        except ImportError:
            # 如果没有fast_insert模块，尝试其他方式
            try:
                from milvusBuilder.persistent_connection import get_persistent_connection
                from pymilvus import Collection
                conn = get_persistent_connection()
                connection_alias = conn.get_connection_alias()
                if connection_alias:
                    # 检查集合是否存在
                    if not self._collection_exists(collection_name):
                        return False
                    # 检查集合是否已加载
                    collection = Collection(collection_name, using=connection_alias)
                    return collection.is_loaded
                else:
                    logger.error("没有可用的Milvus连接")
                    return False
            except Exception as e:
                logger.error(f"检查集合加载状态失败: {e}")
                return False
    
    def _load_collection(self, collection_name: str) -> bool:
        """加载集合"""
        try:
            from milvusBuilder.fast_insert import load_collection
            return load_collection(collection_name)
        except ImportError:
            # 如果没有fast_insert模块，尝试其他方式
            try:
                from milvusBuilder.persistent_connection import get_persistent_connection
                from pymilvus import Collection
                conn = get_persistent_connection()
                connection_alias = conn.get_connection_alias()
                if connection_alias:
                    collection = Collection(collection_name, using=connection_alias)
                    collection.load()
                    logger.info(f"集合 {collection_name} 加载成功")
                    return True
                else:
                    logger.error("没有可用的Milvus连接")
                    return False
            except Exception as e:
                logger.error(f"加载集合失败: {e}")
                return False
    
    def _create_collection_if_needed(self, collection_name: str, schema: Dict = None) -> bool:
        """创建集合（如果需要）"""
        try:
            from milvusBuilder.fast_insert import create_collection_with_schema
            return create_collection_with_schema(collection_name, schema)
        except ImportError:
            # 如果没有fast_insert模块，尝试其他方式
            try:
                from milvusBuilder.persistent_connection import get_persistent_connection
                from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
                conn = get_persistent_connection()
                connection_alias = conn.get_connection_alias()
                if connection_alias:
                    # 如果没有提供schema，使用默认schema
                    if schema is None:
                        # 创建默认schema
                        fields = [
                            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
                            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
                        ]
                        schema = CollectionSchema(fields, "Default collection schema", enable_dynamic_field=True)
                    
                    collection = Collection(name=collection_name, schema=schema, using=connection_alias)
                    logger.info(f"集合 {collection_name} 创建成功")
                    return True
                else:
                    logger.error("没有可用的Milvus连接")
                    return False
            except Exception as e:
                logger.error(f"创建集合失败: {e}")
                return False

class InsertProgressTracker:
    """插入进度跟踪器"""
    
    def __init__(self):
        self._progress_data = {}
        self._tracking_counter = 0
    
    def start_tracking(self, total_items: int) -> str:
        """开始跟踪插入进度"""
        self._tracking_counter += 1
        tracking_id = f"insert_{self._tracking_counter}_{int(time.time())}"
        
        self._progress_data[tracking_id] = {
            "tracking_id": tracking_id,
            "total_items": total_items,
            "processed_items": 0,
            "failed_items": 0,
            "start_time": datetime.now(),
            "estimated_completion": None,
            "current_status": "preparing",
            "error_details": [],
            "last_update": datetime.now()
        }
        
        logger.info(f"开始跟踪插入进度: {tracking_id}, 总项目数: {total_items}")
        return tracking_id
    
    def update_progress(self, tracking_id: str, processed: int, failed: int = 0, status: str = "inserting") -> None:
        """更新插入进度"""
        if tracking_id not in self._progress_data:
            logger.warning(f"跟踪ID {tracking_id} 不存在")
            return
        
        progress = self._progress_data[tracking_id]
        progress["processed_items"] = processed
        progress["failed_items"] = failed
        progress["current_status"] = status
        progress["last_update"] = datetime.now()
        
        # 计算预估完成时间
        if processed > 0:
            elapsed_time = (datetime.now() - progress["start_time"]).total_seconds()
            items_per_second = processed / elapsed_time
            remaining_items = progress["total_items"] - processed
            
            if items_per_second > 0:
                from datetime import timedelta
                estimated_seconds = remaining_items / items_per_second
                progress["estimated_completion"] = datetime.now() + timedelta(seconds=estimated_seconds)
        
        logger.debug(f"更新进度 {tracking_id}: {processed}/{progress['total_items']} 项已处理")
    
    def get_progress_status(self, tracking_id: str) -> Dict[str, Any]:
        """获取插入进度状态"""
        if tracking_id not in self._progress_data:
            return {
                "error": "跟踪ID不存在",
                "status": "not_found"
            }
        
        progress = self._progress_data[tracking_id]
        
        # 计算进度百分比
        progress_percentage = 0
        if progress["total_items"] > 0:
            progress_percentage = (progress["processed_items"] / progress["total_items"]) * 100
        
        # 计算处理速度
        elapsed_time = (datetime.now() - progress["start_time"]).total_seconds()
        items_per_second = progress["processed_items"] / elapsed_time if elapsed_time > 0 else 0
        
        return {
            "tracking_id": tracking_id,
            "total_items": progress["total_items"],
            "processed_items": progress["processed_items"],
            "failed_items": progress["failed_items"],
            "progress_percentage": round(progress_percentage, 2),
            "current_status": progress["current_status"],
            "start_time": progress["start_time"].isoformat(),
            "last_update": progress["last_update"].isoformat(),
            "estimated_completion": progress["estimated_completion"].isoformat() if progress["estimated_completion"] else None,
            "items_per_second": round(items_per_second, 2),
            "error_details": progress["error_details"],
            "status": "active"
        }
    
    def finish_tracking(self, tracking_id: str, success: bool, final_message: str = "") -> Dict[str, Any]:
        """完成插入进度跟踪"""
        if tracking_id not in self._progress_data:
            return {
                "error": "跟踪ID不存在",
                "status": "not_found"
            }
        
        progress = self._progress_data[tracking_id]
        progress["current_status"] = "completed" if success else "failed"
        progress["last_update"] = datetime.now()
        
        if final_message:
            progress["final_message"] = final_message
        
        # 计算最终统计
        elapsed_time = (datetime.now() - progress["start_time"]).total_seconds()
        from datetime import timedelta
        final_stats = {
            "tracking_id": tracking_id,
            "success": success,
            "total_items": progress["total_items"],
            "processed_items": progress["processed_items"],
            "failed_items": progress["failed_items"],
            "success_rate": (progress["processed_items"] / progress["total_items"]) * 100 if progress["total_items"] > 0 else 0,
            "total_time_seconds": round(elapsed_time, 2),
            "average_items_per_second": round(progress["processed_items"] / elapsed_time, 2) if elapsed_time > 0 else 0,
            "final_message": final_message,
            "completed_at": datetime.now().isoformat()
        }
        
        logger.info(f"插入跟踪完成 {tracking_id}: 成功={success}, 处理={progress['processed_items']}/{progress['total_items']}")
        
        return final_stats
    
    def add_error(self, tracking_id: str, error_message: str) -> None:
        """添加错误信息"""
        if tracking_id in self._progress_data:
            self._progress_data[tracking_id]["error_details"].append({
                "timestamp": datetime.now().isoformat(),
                "message": error_message
            })
    
    def cleanup_old_tracking(self, max_age_hours: int = 24) -> None:
        """清理旧的跟踪数据"""
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for tracking_id, progress in self._progress_data.items():
            if progress["last_update"] < cutoff_time:
                to_remove.append(tracking_id)
        
        for tracking_id in to_remove:
            del self._progress_data[tracking_id]
            logger.info(f"清理旧的跟踪数据: {tracking_id}")

class ErrorRecoveryManager:
    """错误恢复管理器"""
    
    def __init__(self):
        self._error_history = []
        self._recovery_strategies = {
            "glm_config_error": self._handle_glm_config_error,
            "chunking_error": self._handle_chunking_error,
            "collection_error": self._handle_collection_error,
            "connection_error": self._handle_connection_error
        }
    
    def handle_error(self, error_type: str, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """统一错误处理入口"""
        if context is None:
            context = {}
        
        # 记录错误
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": str(error),
            "context": context,
            "recovery_attempted": False
        }
        
        self._error_history.append(error_record)
        logger.error(f"处理错误 [{error_type}]: {str(error)}")
        
        # 尝试恢复
        recovery_action = self._attempt_recovery(error_type, error, context)
        error_record["recovery_attempted"] = True
        error_record["recovery_action"] = recovery_action
        
        return recovery_action
    
    def _attempt_recovery(self, error_type: str, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """尝试错误恢复"""
        handler = self._recovery_strategies.get(error_type)
        
        if handler:
            try:
                return handler(error, context)
            except Exception as recovery_error:
                logger.error(f"恢复策略执行失败: {recovery_error}")
                return {
                    "action": "manual_intervention_required",
                    "success": False,
                    "message": f"自动恢复失败: {str(recovery_error)}",
                    "suggestions": ["请检查系统配置", "联系技术支持"]
                }
        else:
            return {
                "action": "no_recovery_strategy",
                "success": False,
                "message": f"未找到 {error_type} 的恢复策略",
                "suggestions": ["请手动处理此错误"]
            }
    
    def _handle_glm_config_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理GLM配置错误"""
        error_msg = str(error).lower()
        
        if "api" in error_msg and "key" in error_msg:
            return {
                "action": "invalid_api_key",
                "success": False,
                "message": "API密钥无效或已过期",
                "suggestions": [
                    "检查API密钥是否正确",
                    "确认API密钥是否有足够的调用额度",
                    "尝试重新生成API密钥"
                ],
                "recovery_steps": [
                    "访问智谱AI开放平台",
                    "检查API密钥状态",
                    "更新配置中的API密钥"
                ]
            }
        elif "network" in error_msg or "connection" in error_msg:
            return {
                "action": "network_issue",
                "success": False,
                "message": "网络连接问题",
                "suggestions": [
                    "检查网络连接",
                    "确认防火墙设置",
                    "稍后重试"
                ]
            }
        else:
            return {
                "action": "general_glm_error",
                "success": False,
                "message": "GLM配置错误",
                "suggestions": [
                    "检查GLM配置是否完整",
                    "验证模型名称是否正确",
                    "重新配置GLM设置"
                ]
            }
    
    def _handle_chunking_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理分块错误"""
        error_msg = str(error).lower()
        
        if "dependency" in error_msg or "import" in error_msg:
            return {
                "action": "missing_dependencies",
                "success": False,
                "message": "分块依赖缺失",
                "suggestions": [
                    "安装缺失的Python包",
                    "检查torch是否正确安装",
                    "验证perplexity_chunking模块"
                ],
                "recovery_steps": [
                    "pip install torch",
                    "pip install nltk jieba",
                    "确保perplexity_chunking.py文件存在"
                ]
            }
        elif "glm" in error_msg or "api" in error_msg:
            return {
                "action": "fallback_to_traditional",
                "success": True,
                "message": "GLM不可用，已降级到传统分块",
                "suggestions": [
                    "配置GLM以启用高级分块功能",
                    "当前使用传统分块方法"
                ]
            }
        else:
            return {
                "action": "chunking_fallback",
                "success": True,
                "message": "分块策略已降级",
                "suggestions": [
                    "检查分块策略配置",
                    "使用传统分块作为备选方案"
                ]
            }
    
    def _handle_collection_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理集合错误"""
        error_msg = str(error).lower()
        collection_name = context.get("collection_name", "未知")
        
        if "not exist" in error_msg or "not found" in error_msg:
            return {
                "action": "create_collection",
                "success": False,
                "message": f"集合 {collection_name} 不存在",
                "suggestions": [
                    "创建新集合",
                    "检查集合名称是否正确",
                    "确认Milvus连接正常"
                ],
                "recovery_steps": [
                    "自动创建集合",
                    "使用默认schema",
                    "重新尝试操作"
                ]
            }
        elif "load" in error_msg:
            return {
                "action": "retry_load_collection",
                "success": False,
                "message": f"集合 {collection_name} 加载失败",
                "suggestions": [
                    "重试加载集合",
                    "检查Milvus服务状态",
                    "确认集合schema正确"
                ],
                "recovery_steps": [
                    "等待2秒后重试",
                    "最多重试3次",
                    "如果仍失败，请手动检查"
                ]
            }
        else:
            return {
                "action": "general_collection_error",
                "success": False,
                "message": f"集合 {collection_name} 操作失败",
                "suggestions": [
                    "检查Milvus连接状态",
                    "验证集合配置",
                    "查看Milvus日志"
                ]
            }
    
    def _handle_connection_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理连接错误"""
        return {
            "action": "retry_connection",
            "success": False,
            "message": "连接失败",
            "suggestions": [
                "检查服务是否运行",
                "验证网络连接",
                "确认端口配置正确"
            ],
            "recovery_steps": [
                "重试连接",
                "检查服务状态",
                "验证配置文件"
            ]
        }
    
    def get_error_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取错误历史"""
        return self._error_history[-limit:]
    
    def clear_error_history(self) -> None:
        """清除错误历史"""
        self._error_history.clear()
        logger.info("错误历史已清除")

# 创建FastAPI应用
app = FastAPI(title="Cre_milvus 新架构API", version="3.0.0")

# 全局状态
_app_initialized = False
_collection_manager = None
_progress_tracker = None
_error_manager = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化连接"""
    global _app_initialized, _collection_manager
    
    logger.info("=" * 60)
    logger.info("🚀 API服务启动，初始化连接...")
    logger.info("=" * 60)
    
    try:
        from System.connection_initializer import startup_initialize
        success = startup_initialize()
        
        if success:
            _app_initialized = True
            _collection_manager = CollectionStateManager()
            _progress_tracker = InsertProgressTracker()
            _error_manager = ErrorRecoveryManager()
            logger.info("✅ API服务初始化成功")
            logger.info("✅ 集合状态管理器已初始化")
            logger.info("✅ 插入进度跟踪器已初始化")
            logger.info("✅ 错误恢复管理器已初始化")
        else:
            logger.error("❌ API服务初始化失败")
            
    except Exception as e:
        logger.error(f"❌ API服务初始化异常: {e}")
        _app_initialized = False

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Cre_milvus 新架构API",
        "version": "3.0.0",
        "initialized": _app_initialized,
        "features": [
            "预连接架构",
            "快速数据插入",
            "连接状态监控",
            "无阻塞操作"
        ]
    }

@app.get("/status")
async def get_status():
    """获取系统状态"""
    try:
        from System.new_start import get_connection_status
        status = get_connection_status()
        
        return {
            "api_initialized": _app_initialized,
            "connection_status": status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "api_initialized": _app_initialized,
            "error": str(e),
            "timestamp": time.time()
        }

@app.post("/update_config")
async def update_config(request: Request):
    """
    更新配置，包括动态更新Milvus连接
    """
    try:
        config_data = await request.json()
        logger.info(f"收到配置更新请求: {config_data}")
        
        # 更新配置文件
        import yaml
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                current_config = yaml.safe_load(f) or {}
        except:
            current_config = {}
        
        # 深度合并配置
        def deep_merge(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(current_config, config_data)
        
        # 保存配置文件
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)
        
        # 如果更新了Milvus配置，动态重连
        if "milvus" in config_data:
            milvus_config = config_data["milvus"]
            if "host" in milvus_config and "port" in milvus_config:
                host = milvus_config["host"]
                port = int(milvus_config["port"])
                
                logger.info(f"检测到Milvus配置更新，尝试重新连接: {host}:{port}")
                
                from milvusBuilder.persistent_connection import get_persistent_connection
                conn = get_persistent_connection()
                
                # 检查是否需要重新连接
                if not conn.is_connection_valid_for(host, port):
                    success = conn.update_connection(host, port)
                    if success:
                        logger.info("✅ Milvus连接已更新")
                    else:
                        logger.warning("⚠️ Milvus连接更新失败，但配置已保存")
                else:
                    logger.info("✅ Milvus连接无需更新")
        
        return {
            "message": "配置更新成功",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"配置更新失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"配置更新失败: {str(e)}"
        )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), folder_name: str = Form(None)):
    """文件上传和处理"""
    try:
        if folder_name and folder_name.strip():
            upload_dir = f"./data/upload/{folder_name.strip()}"
            logger.info(f"使用指定文件夹: {upload_dir}")
        else:
            upload_dir = "./data/upload"
            logger.info(f"使用默认文件夹: {upload_dir}")
        os.makedirs(upload_dir, exist_ok=True)
        
        uploaded_files = []
        file_types = {}
        
        if file.filename:
            file_path = os.path.join(upload_dir, file.filename)
            
            content = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            uploaded_files.append(file.filename)
            file_extension = os.path.splitext(file.filename)[1].lower()
            file_types[file_extension] = file_types.get(file_extension, 0) + 1
            
            logger.info(f"文件已保存: {file.filename}")
        
        logger.info(f"文件上传完成: {len(uploaded_files)} 个文件")
        
        try:
            logger.info("开始向量化存储...")
            
            with open("config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            insert_mode = config.get("system", {}).get("insert_mode", "overwrite")
            collection_name = config.get("milvus", {}).get("collection_name", "Test_one")
            
            if insert_mode == "append":
                logger.info(f"使用append模式，检查集合 {collection_name} 状态")
                
                if _collection_manager:
                    collection_ready = _collection_manager.ensure_collection_loaded(collection_name)
                    if not collection_ready:
                        logger.error(f"集合 {collection_name} 加载失败")
                        return {
                            "message": f"成功上传 {len(uploaded_files)} 个文件，但集合加载失败",
                            "files": uploaded_files,
                            "upload_dir": upload_dir,
                            "file_types": file_types,
                            "vectorized": False,
                            "error": f"集合 {collection_name} 加载失败",
                            "status": "partial_success"
                        }
                    else:
                        logger.info(f"集合 {collection_name} 已准备就绪")
                else:
                    logger.warning("集合状态管理器未初始化")
            else:
                logger.info(f"使用overwrite模式，将重新创建集合 {collection_name}")
            
            if folder_name:
                if "data" not in config:
                    config["data"] = {}
                config["data"]["data_location"] = upload_dir
                logger.info(f"更新数据路径为: {upload_dir}")
                
                with open("config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            tracking_id = None
            if _progress_tracker:
                tracking_id = _progress_tracker.start_tracking(len(uploaded_files))
                logger.info(f"开始跟踪向量化进度: {tracking_id}")
            
            from System.new_start import fast_vector_database_build_from_config
            
            start_time = time.time()
            
            if tracking_id:
                _progress_tracker.update_progress(tracking_id, 0, 0, "开始向量化存储")
            
            try:
                result = fast_vector_database_build_from_config(config)
                end_time = time.time()
                
                logger.info(f"向量化存储完成，耗时: {end_time - start_time:.2f}秒")
                
                if tracking_id:
                    success = result.get("status") == "success"
                    final_message = f"向量化存储{'成功' if success else '失败'}，耗时: {end_time - start_time:.2f}秒"
                    _progress_tracker.finish_tracking(tracking_id, success, final_message)
                    
            except Exception as build_error:
                end_time = time.time()
                logger.error(f"向量化构建过程失败: {build_error}")
                
                if tracking_id:
                    _progress_tracker.add_error(tracking_id, str(build_error))
                    _progress_tracker.finish_tracking(tracking_id, False, f"向量化构建失败: {str(build_error)}")
                
                raise build_error
            
            if result.get("status") == "success":
                return {
                    "success": True,
                    "message": f"成功上传 {len(uploaded_files)} 个文件并完成向量化存储",
                    "filename": file.filename,
                    "size": len(content),
                    "path": file_path,
                    "folder_name": folder_name,
                    "files": uploaded_files,
                    "upload_dir": upload_dir,
                    "file_types": file_types,
                    "vectorized": True,
                    "vectorization_result": result,
                    "processing_time": end_time - start_time,
                    "tracking_id": tracking_id,
                    "status": "success"
                }
            else:
                return {
                    "success": True,
                    "message": f"成功上传 {len(uploaded_files)} 个文件，但向量化存储失败",
                    "filename": file.filename,
                    "size": len(content),
                    "path": file_path,
                    "folder_name": folder_name,
                    "files": uploaded_files,
                    "upload_dir": upload_dir,
                    "file_types": file_types,
                    "vectorized": False,
                    "error": result.get("msg", "未知错误"),
                    "tracking_id": tracking_id,
                    "status": "partial_success"
                }
                
        except Exception as vector_error:
            logger.error(f"向量化存储失败: {vector_error}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            
            return {
                "success": True,
                "message": f"成功上传 {len(uploaded_files)} 个文件，但向量化存储失败",
                "filename": file.filename,
                "size": len(content) if 'content' in locals() else 0,
                "path": file_path if 'file_path' in locals() else "",
                "folder_name": folder_name,
                "files": uploaded_files,
                "upload_dir": upload_dir,
                "file_types": file_types,
                "vectorized": False,
                "error": str(vector_error),
                "tracking_id": tracking_id if 'tracking_id' in locals() else None,
                "status": "partial_success"
            }
        
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail=f"文件上传失败: {str(e)}"
        )

@app.post("/search")
async def search_documents(request: Request):
    """
    搜索文档（增强版，包含聚类可视化数据）
    """
    if not _app_initialized:
        raise HTTPException(
            status_code=503, 
            detail="服务未初始化，请等待初始化完成"
        )
    
    try:
        data = await request.json()
        question = data.get("question", "")
        col_choice = data.get("col_choice", "hdbscan")
        collection_name = data.get("collection_name", "Test_one")
        enable_visualization = data.get("enable_visualization", True)
        
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")
        
        logger.info(f"收到搜索请求: {question}, 聚类方法: {col_choice}")
        
        # 加载配置
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 使用原有的搜索功能
        from System.start import Cre_Search
        
        start_time = time.time()
        result = Cre_Search(config, question)
        search_time = time.time() - start_time
        
        logger.info(f"基础搜索完成，耗时: {search_time:.2f}秒")
        
        # 如果启用可视化且有聚类结果，添加可视化数据
        if enable_visualization and "clusters" in result and result["clusters"]:
            try:
                from Search.clustering import create_clustering_service
                clustering_service = create_clustering_service()
                
                # 转换聚类数据格式
                clusters = []
                for cluster_data in result["clusters"]:
                    from Search.clustering import Cluster, SearchResult
                    
                    # 转换文档数据
                    documents = []
                    for doc in cluster_data.get("documents", []):
                        search_result = SearchResult(
                            id=str(doc.get("id", "")),
                            content=doc.get("content", ""),
                            url=doc.get("url"),
                            distance=float(doc.get("distance", 0.0)),
                            embedding=doc.get("embedding", []),
                            metadata=doc.get("metadata", {})
                        )
                        documents.append(search_result)
                    
                    # 创建聚类对象
                    cluster = Cluster(
                        cluster_id=cluster_data.get("cluster_id", 0),
                        documents=documents,
                        centroid=cluster_data.get("centroid"),
                        size=len(documents),
                        avg_distance=cluster_data.get("avg_distance", 0.0),
                        keywords=cluster_data.get("keywords", [])
                    )
                    clusters.append(cluster)
                
                # 生成可视化数据
                viz_start_time = time.time()
                
                scatter_plot_data = clustering_service.create_cluster_scatter_plot(clusters)
                size_chart_data = clustering_service.create_cluster_size_chart(clusters)
                heatmap_data = clustering_service.create_cluster_heatmap(clusters)
                cluster_summary = clustering_service.generate_cluster_summary(clusters)
                cluster_metrics = clustering_service.calculate_cluster_metrics(clusters)
                
                viz_time = time.time() - viz_start_time
                logger.info(f"可视化数据生成完成，耗时: {viz_time:.2f}秒")
                
                # 添加可视化数据到结果中
                result["visualization_data"] = {
                    "scatter_plot": scatter_plot_data,
                    "size_chart": size_chart_data,
                    "heatmap": heatmap_data,
                    "cluster_summary": cluster_summary,
                    "cluster_metrics": cluster_metrics
                }
                
                # 更新执行时间
                result["execution_time"] = search_time + viz_time
                result["search_time"] = search_time
                result["visualization_time"] = viz_time
                
                logger.info(f"增强搜索完成，总耗时: {result['execution_time']:.2f}秒")
                
            except Exception as viz_error:
                logger.error(f"生成可视化数据失败: {viz_error}")
                # 可视化失败不影响基础搜索结果
                result["visualization_error"] = str(viz_error)
        
        # 添加质量指标（如果不存在）
        if "quality_metrics" not in result and "clusters" in result:
            try:
                result["quality_metrics"] = _calculate_search_quality_metrics(result)
            except Exception as e:
                logger.warning(f"计算质量指标失败: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail=f"搜索失败: {str(e)}"
        )


def _calculate_search_quality_metrics(search_result: Dict[str, Any]) -> Dict[str, float]:
    """计算搜索质量指标"""
    try:
        clusters = search_result.get("clusters", [])
        if not clusters:
            return {"relevance_score": 0.0, "diversity_score": 0.0, "coverage_score": 0.0}
        
        total_docs = sum(len(cluster.get("documents", [])) for cluster in clusters)
        if total_docs == 0:
            return {"relevance_score": 0.0, "diversity_score": 0.0, "coverage_score": 0.0}
        
        # 相关性分数：基于平均距离（距离越小，相关性越高）
        total_distance = 0
        for cluster in clusters:
            for doc in cluster.get("documents", []):
                total_distance += doc.get("distance", 1.0)
        
        avg_distance = total_distance / total_docs
        relevance_score = max(0, 1 - avg_distance)  # 距离转换为相关性
        
        # 多样性分数：基于聚类数量和分布
        num_clusters = len(clusters)
        if num_clusters <= 1:
            diversity_score = 0.0
        else:
            # 计算聚类大小的标准差，标准差越小，分布越均匀，多样性越好
            cluster_sizes = [len(cluster.get("documents", [])) for cluster in clusters]
            mean_size = sum(cluster_sizes) / len(cluster_sizes)
            variance = sum((size - mean_size) ** 2 for size in cluster_sizes) / len(cluster_sizes)
            std_dev = variance ** 0.5
            
            # 归一化多样性分数
            max_possible_std = mean_size * 0.5  # 假设最大标准差为平均值的一半
            diversity_score = max(0, 1 - (std_dev / max_possible_std)) if max_possible_std > 0 else 0
        
        # 覆盖率分数：基于聚类数量相对于文档数量的比例
        coverage_ratio = num_clusters / total_docs if total_docs > 0 else 0
        coverage_score = min(1.0, coverage_ratio * 5)  # 假设理想比例是1:5
        
        return {
            "relevance_score": round(relevance_score, 3),
            "diversity_score": round(diversity_score, 3),
            "coverage_score": round(coverage_score, 3)
        }
        
    except Exception as e:
        logger.error(f"计算质量指标失败: {e}")
        return {"relevance_score": 0.0, "diversity_score": 0.0, "coverage_score": 0.0}

@app.get("/collections")
async def list_collections():
    """列出所有集合"""
    if not _app_initialized:
        raise HTTPException(
            status_code=503, 
            detail="服务未初始化，请等待初始化完成"
        )
    
    try:
        from milvusBuilder.fast_insert import list_collections
        collections = list_collections()
        
        return {
            "collections": collections,
            "count": len(collections),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"列出集合失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"列出集合失败: {str(e)}"
        )

@app.get("/collections/{collection_name}/status")
async def get_collection_status(collection_name: str):
    """获取集合状态"""
    if not _app_initialized:
        raise HTTPException(
            status_code=503, 
            detail="服务未初始化，请等待初始化完成"
        )
    
    try:
        from milvusBuilder.fast_insert import check_collection_status
        status = check_collection_status(collection_name)
        
        return {
            "collection_name": collection_name,
            "status": status
        }
        
    except Exception as e:
        logger.error(f"获取集合状态失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取集合状态失败: {str(e)}"
        )

@app.post("/chunking/process")
async def process_chunking(request: Request):
    """
    文本切分处理
    """
    try:
        data = await request.json()
        text = data.get("text", "")
        strategy = data.get("strategy", "traditional")
        params = data.get("params", {})
        
        if not text:
            raise HTTPException(status_code=400, detail="文本不能为空")
        
        logger.info(f"收到文本切分请求: 策略={strategy}, 文本长度={len(text)}")
        
        # 导入文本切分模块
        try:
            from dataBuilder.chunking.chunking_manager import ChunkingManager
            
            chunking_manager = ChunkingManager()
            chunks = chunking_manager.chunk_text(text, strategy, params)
            
            return {
                "chunks": chunks,
                "chunk_count": len(chunks),
                "strategy": strategy,
                "status": "success"
            }
            
        except ImportError:
            # 如果切分模块不可用，使用简单切分
            chunk_length = params.get("chunk_length", 512)
            overlap = params.get("overlap", 50)
            
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_length, len(text))
                chunk = text[start:end]
                chunks.append(chunk)
                start = end - overlap if end < len(text) else end
            
            return {
                "chunks": chunks,
                "chunk_count": len(chunks),
                "strategy": "simple",
                "status": "success"
            }
        
    except Exception as e:
        logger.error(f"文本切分失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"文本切分失败: {str(e)}"
        )

@app.post("/multimodal/text_to_image_search")
async def text_to_image_search(request: Request):
    """
    文搜图功能
    """
    try:
        data = await request.json()
        query_text = data.get("query_text", "")
        top_k = data.get("top_k", 10)
        collection_name = data.get("collection_name", "")
        
        if not query_text:
            raise HTTPException(status_code=400, detail="查询文本不能为空")
        
        logger.info(f"收到文搜图请求: {query_text}")
        
        # 检查多模态功能是否启用
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            multimodal_config = config.get("multimodal", {})
            if not multimodal_config.get("enable_image", False):
                return {
                    "results": [],
                    "message": "多模态功能未启用",
                    "status": "disabled"
                }
            
            # 这里应该实现实际的文搜图逻辑
            # 目前返回模拟结果
            return {
                "results": [],
                "message": "文搜图功能开发中",
                "query_text": query_text,
                "top_k": top_k,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"文搜图处理失败: {e}")
            return {
                "results": [],
                "message": f"文搜图失败: {str(e)}",
                "status": "error"
            }
        
    except Exception as e:
        logger.error(f"文搜图请求处理失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"文搜图请求处理失败: {str(e)}"
        )

@app.get("/performance/current")
async def get_performance_metrics():
    """
    获取当前性能指标
    """
    try:
        import psutil
        
        # 获取系统性能指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            "cpu": {
                "percent": cpu_percent
            },
            "memory": {
                "percent": memory.percent,
                "used": memory.used,
                "total": memory.total
            },
            "disk": {
                "percent": disk.percent,
                "used": disk.used,
                "total": disk.total
            }
        }
        
        return {
            "metrics": metrics,
            "timestamp": time.time(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取性能指标失败: {str(e)}"
        )

@app.get("/system/status")
async def get_system_status():
    """
    获取系统状态
    """
    try:
        # 获取连接状态
        try:
            from System.new_start import get_connection_status
            connection_status = get_connection_status()
            milvus_connected = connection_status.get("overall_ready", False)
        except:
            connection_status = {}
            milvus_connected = False
        
        # 获取配置信息
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except:
            config = {}
        
        # 检查LLM配置状态
        llm_configs = config.get("llm_configs", {})
        active_llm_config_id = config.get("active_llm_config")
        active_llm_config = None
        
        if active_llm_config_id and active_llm_config_id in llm_configs:
            active_llm_config = {
                "id": active_llm_config_id,
                "provider": llm_configs[active_llm_config_id].get("provider"),
                "model": llm_configs[active_llm_config_id].get("model_name")
            }
        
        # 构建状态信息，匹配前端期望的格式
        status = {
            "milvus": {
                "connected": milvus_connected
            },
            "embedding_model": {
                "available": True  # 假设嵌入模型总是可用的
            },
            "chunking_system": {
                "available": True
            },
            "clustering_service": {
                "available": True
            },
            "llm_config": {
                "available": active_llm_config is not None,
                "active_config": active_llm_config
            }
        }
        
        # 计算整体健康状态
        critical_services = [
            status["milvus"]["connected"],
            status["embedding_model"]["available"]
        ]
        
        if all(critical_services):
            overall_status = "healthy"
        elif any(critical_services):
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        health = {
            "overall_status": overall_status
        }
        
        return {
            "status": status,
            "health": health,
            "config": {
                "milvus": config.get("milvus", {}),
                "multimodal": config.get("multimodal", {}),
                "chunking": config.get("chunking", {})
            },
            "connection_status": connection_status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取系统状态失败: {str(e)}"
        )

@app.post("/visualization")
async def get_visualization_data(request: Request):
    """
    获取可视化数据
    """
    try:
        data = await request.json()
        collection_name = data.get("collection_name", "")
        
        if not collection_name:
            raise HTTPException(status_code=400, detail="集合名称不能为空")
        
        logger.info(f"收到可视化请求: {collection_name}")
        
        # 尝试获取可视化数据
        try:
            from ColBuilder.visualization import get_all_embeddings_and_texts
            import hdbscan
            from umap import UMAP
            import pandas as pd
            import numpy as np
            
            # 获取数据
            embeddings, texts, ids = get_all_embeddings_and_texts(collection_name)
            
            if not embeddings:
                return []
            
            # UMAP降维
            umap_model = UMAP(n_components=2, random_state=42)
            embeddings_2d = umap_model.fit_transform(np.array(embeddings))
            
            # HDBSCAN聚类
            clusterer = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=2)
            cluster_labels = clusterer.fit_predict(embeddings_2d)
            
            # 构建结果
            result = []
            for i, (x, y) in enumerate(embeddings_2d):
                result.append({
                    "x": float(x),
                    "y": float(y),
                    "cluster": int(cluster_labels[i]),
                    "text": texts[i][:100] if i < len(texts) else "",
                    "id": ids[i] if i < len(ids) else i
                })
            
            return result
            
        except ImportError as e:
            logger.warning(f"可视化模块导入失败: {e}")
            return []
        except Exception as e:
            logger.error(f"可视化数据生成失败: {e}")
            return []
        
    except Exception as e:
        logger.error(f"可视化请求处理失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"可视化请求处理失败: {str(e)}"
        )

@app.get("/llm/providers")
async def get_llm_providers():
    """获取LLM提供商列表"""
    try:
        providers = [
            {
                "name": "openai",
                "description": "OpenAI GPT系列模型",
                "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
            },
            {
                "name": "claude",
                "description": "Anthropic Claude系列模型",
                "models": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"]
            },
            {
                "name": "qwen",
                "description": "阿里云通义千问系列模型",
                "models": ["qwen-turbo", "qwen-plus", "qwen-max"]
            },
            {
                "name": "zhipu",
                "description": "智谱AI GLM系列模型",
                "models": ["glm-4", "glm-4-turbo", "glm-4.1v-thinking-flash"]
            },
            {
                "name": "local",
                "description": "本地部署模型",
                "models": ["custom-model"]
            }
        ]
        
        return {
            "providers": providers,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"获取LLM提供商失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取LLM提供商失败: {str(e)}"
        )

@app.get("/llm/configs")
async def get_llm_configs():
    """获取LLM配置列表"""
    try:
        # 这里应该从配置文件或数据库中读取LLM配置
        # 目前返回模拟数据
        configs = {}
        active_config = None
        
        # 尝试从配置文件读取
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                llm_configs = config.get("llm_configs", {})
                active_config_id = config.get("active_llm_config")
                
                for config_id, config_data in llm_configs.items():
                    configs[config_id] = {
                        "provider": config_data.get("provider"),
                        "model_name": config_data.get("model_name"),
                        "api_endpoint": config_data.get("api_endpoint")
                    }
                
                if active_config_id and active_config_id in configs:
                    active_config = {
                        "id": active_config_id,
                        **configs[active_config_id]
                    }
        except:
            pass
        
        return {
            "configs": configs,
            "summary": {
                "total_configs": len(configs),
                "active_config": active_config
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"获取LLM配置失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取LLM配置失败: {str(e)}"
        )



@app.post("/llm/configs")
async def save_llm_config(request: Request):
    """保存LLM配置"""
    try:
        data = await request.json()
        config_id = data.get("config_id")
        provider = data.get("provider")
        model_name = data.get("model_name")
        api_key = data.get("api_key")
        api_endpoint = data.get("api_endpoint")
        is_active = data.get("is_active", False)
        
        if not all([config_id, provider, api_key]):
            raise HTTPException(status_code=400, detail="配置ID、提供商和API密钥不能为空")
        
        # 读取现有配置
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except:
            config = {}
        
        # 更新LLM配置
        if "llm_configs" not in config:
            config["llm_configs"] = {}
        
        config["llm_configs"][config_id] = {
            "provider": provider,
            "model_name": model_name,
            "api_key": api_key,
            "api_endpoint": api_endpoint
        }
        
        # 设置激活配置
        if is_active:
            config["active_llm_config"] = config_id
        
        # 保存配置
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return {
            "message": f"LLM配置 '{config_id}' 保存成功",
            "config_id": config_id,
            "is_active": is_active,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"保存LLM配置失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"保存LLM配置失败: {str(e)}"
        )



@app.get("/chunking/strategies")
async def get_chunking_strategies():
    """获取分块策略列表"""
    try:
        strategies = [
            {
                "name": "traditional",
                "display_name": "传统固定长度切分",
                "description": "按固定长度切分文本，支持重叠",
                "parameters": ["chunk_length", "overlap"],
                "requires_llm": False
            },
            {
                "name": "meta_ppl",
                "display_name": "PPL困惑度切分",
                "description": "基于语言模型困惑度的智能切分",
                "parameters": ["chunk_length", "ppl_threshold"],
                "requires_llm": True
            },
            {
                "name": "margin_sampling",
                "display_name": "边际采样切分",
                "description": "基于边际采样的智能切分",
                "parameters": ["chunk_length", "confidence_threshold"],
                "requires_llm": True
            },
            {
                "name": "msp",
                "display_name": "MSP高级切分",
                "description": "多尺度感知切分策略",
                "parameters": ["chunk_length", "confidence_threshold"],
                "requires_llm": True
            },
            {
                "name": "semantic",
                "display_name": "语义切分",
                "description": "基于语义相似度的切分",
                "parameters": ["chunk_length", "similarity_threshold", "min_chunk_size"],
                "requires_llm": False
            }
        ]
        
        return {
            "strategies": strategies,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"获取分块策略失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取分块策略失败: {str(e)}"
        )

@app.get("/performance/export_report")
async def export_performance_report():
    """导出性能报告"""
    try:
        import psutil
        from datetime import datetime
        
        # 获取系统性能指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        report = {
            "report_time": datetime.now().isoformat(),
            "report_type": "performance_report",
            "system_metrics": {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "percent": memory.percent,
                    "used_gb": round(memory.used / (1024**3), 2),
                    "total_gb": round(memory.total / (1024**3), 2)
                },
                "disk": {
                    "percent": disk.percent,
                    "used_gb": round(disk.used / (1024**3), 2),
                    "total_gb": round(disk.total / (1024**3), 2)
                }
            },
            "connection_status": {},
            "recommendations": []
        }
        
        # 添加性能建议
        if cpu_percent > 80:
            report["recommendations"].append("CPU使用率过高，建议优化查询或增加计算资源")
        if memory.percent > 80:
            report["recommendations"].append("内存使用率过高，建议增加内存或优化内存使用")
        if disk.percent > 80:
            report["recommendations"].append("磁盘使用率过高，建议清理数据或扩容")
        
        return report
        
    except Exception as e:
        logger.error(f"导出性能报告失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"导出性能报告失败: {str(e)}"
        )

@app.post("/testing/start_load_test")
async def start_load_test(request: Request):
    """启动压力测试"""
    try:
        data = await request.json()
        
        # 模拟压力测试启动
        test_id = f"test_{int(time.time())}"
        
        return {
            "message": "压力测试已启动",
            "test_id": test_id,
            "status": "started",
            "parameters": data
        }
        
    except Exception as e:
        logger.error(f"启动压力测试失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"启动压力测试失败: {str(e)}"
        )

@app.get("/testing/list_tests")
async def list_tests():
    """列出测试"""
    try:
        # 模拟测试列表
        tests = []
        
        return {
            "tests": tests,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"列出测试失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"列出测试失败: {str(e)}"
        )

@app.post("/testing/stop_test/{test_id}")
async def stop_test(test_id: str):
    """停止测试"""
    try:
        return {
            "message": f"测试 {test_id} 已停止",
            "test_id": test_id,
            "status": "stopped"
        }
        
    except Exception as e:
        logger.error(f"停止测试失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"停止测试失败: {str(e)}"
        )

@app.post("/system/integration_test")
async def system_integration_test():
    """系统集成测试"""
    try:
        test_results = []
        
        # 测试Milvus连接
        try:
            from System.new_start import get_connection_status
            connection_status = get_connection_status()
            milvus_test = {
                "test_name": "Milvus连接测试",
                "status": "passed" if connection_status.get("overall_ready", False) else "failed",
                "details": connection_status
            }
            test_results.append(milvus_test)
        except Exception as e:
            test_results.append({
                "test_name": "Milvus连接测试",
                "status": "failed",
                "error": str(e)
            })
        
        # 测试配置文件
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            test_results.append({
                "test_name": "配置文件测试",
                "status": "passed",
                "details": "配置文件读取正常"
            })
        except Exception as e:
            test_results.append({
                "test_name": "配置文件测试",
                "status": "failed",
                "error": str(e)
            })
        
        # 计算成功率
        passed_tests = sum(1 for test in test_results if test["status"] == "passed")
        success_rate = passed_tests / len(test_results) if test_results else 0
        
        return {
            "summary": {
                "total_tests": len(test_results),
                "passed_tests": passed_tests,
                "failed_tests": len(test_results) - passed_tests,
                "success_rate": success_rate
            },
            "test_results": test_results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"系统集成测试失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"系统集成测试失败: {str(e)}"
        )

@app.post("/system/reload_config")
async def reload_config():
    """重新加载配置"""
    try:
        # 重新读取配置文件
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 这里可以添加重新初始化各个模块的逻辑
        
        return {
            "message": "系统配置已重新加载",
            "status": "success",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"重新加载配置失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"重新加载配置失败: {str(e)}"
        )

@app.post("/reinitialize")
async def reinitialize_connections():
    """重新初始化连接"""
    global _app_initialized
    
    try:
        logger.info("重新初始化连接...")
        
        from System.connection_initializer import startup_initialize
        success = startup_initialize()
        
        if success:
            _app_initialized = True
            return {
                "message": "连接重新初始化成功",
                "status": "success"
            }
        else:
            _app_initialized = False
            return {
                "message": "连接重新初始化失败",
                "status": "error"
            }
            
    except Exception as e:
        logger.error(f"重新初始化失败: {e}")
        _app_initialized = False
        raise HTTPException(
            status_code=500,
            detail=f"重新初始化失败: {str(e)}"
        )


# GLM配置管理端点
@app.post("/glm/config")
async def save_glm_config(request: Request):
    """保存GLM配置"""
    try:
        data = await request.json()
        model_name = data.get("model_name", "glm-4.5-flash")
        api_key = data.get("api_key", "")
        
        if not api_key:
            raise HTTPException(status_code=400, detail="API密钥不能为空")
        
        logger.info(f"收到GLM配置保存请求: model={model_name}")
        
        from dataBuilder.chunking.glm_config import get_glm_config_service
        service = get_glm_config_service()
        
        success = service.save_config(model_name, api_key)
        
        if success:
            # 获取配置摘要
            summary = service.get_config_summary()
            
            # 同时更新到系统配置中，集成到现有的LLM配置系统
            try:
                with open("config.yaml", "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                
                # 更新LLM配置
                if "llm_configs" not in config:
                    config["llm_configs"] = {}
                
                config["llm_configs"]["glm_default"] = {
                    "provider": "zhipu",
                    "model_name": model_name,
                    "api_key": api_key,
                    "api_endpoint": "https://open.bigmodel.cn/api/paas/v4/chat/completions"
                }
                
                # 设置为激活配置
                config["active_llm_config"] = "glm_default"
                
                # 保存配置
                with open("config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                
                logger.info("GLM配置已集成到系统配置")
                
            except Exception as e:
                logger.warning(f"集成GLM配置到系统配置失败: {e}")
            
            return {
                "status": "success",
                "message": "GLM配置保存成功",
                "config": summary
            }
        else:
            raise HTTPException(status_code=400, detail="GLM配置保存失败，请检查API密钥是否有效")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存GLM配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存GLM配置失败: {str(e)}")


@app.get("/glm/config")
async def get_glm_config():
    """获取GLM配置"""
    try:
        from dataBuilder.chunking.glm_config import get_glm_config_service
        service = get_glm_config_service()
        
        return {
            "status": "success",
            "config": service.get_config_summary()
        }
        
    except Exception as e:
        logger.error(f"获取GLM配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取GLM配置失败: {str(e)}")


@app.post("/glm/test-connection")
async def test_glm_connection():
    """测试GLM连接"""
    try:
        from dataBuilder.chunking.glm_config import get_glm_config_service
        service = get_glm_config_service()
        
        is_valid, message = service.test_connection()
        
        if is_valid:
            # 更新验证时间
            service.update_validation_time()
            
        return {
            "status": "success" if is_valid else "error",
            "connected": is_valid,
            "valid": is_valid,
            "message": message,
            "config": service.get_config_summary()
        }
        
    except Exception as e:
        logger.error(f"测试GLM连接失败: {e}")
        return {
            "status": "error",
            "connected": False,
            "valid": False,
            "message": f"连接测试失败: {str(e)}"
        }


@app.post("/glm/validate-key")
async def validate_glm_api_key(request: Request):
    """验证GLM API密钥"""
    try:
        data = await request.json()
        api_key = data.get("api_key", "")
        
        if not api_key:
            raise HTTPException(status_code=400, detail="API密钥不能为空")
        
        from dataBuilder.chunking.glm_config import get_glm_config_service
        service = get_glm_config_service()
        
        is_valid, message = service.validate_api_key(api_key)
        
        return {
            "status": "success",
            "valid": is_valid,
            "message": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"验证GLM API密钥失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证GLM API密钥失败: {str(e)}")


@app.delete("/glm/config")
async def clear_glm_config():
    """清除GLM配置"""
    try:
        from dataBuilder.chunking.glm_config import get_glm_config_service
        service = get_glm_config_service()
        
        success = service.clear_config()
        
        if success:
            return {
                "status": "success",
                "message": "GLM配置已清除"
            }
        else:
            raise HTTPException(status_code=500, detail="清除GLM配置失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清除GLM配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"清除GLM配置失败: {str(e)}")


# 压测管理端点
@app.post("/load-test/start")
async def start_load_test(request: Request):
    """启动压力测试"""
    try:
        data = await request.json()
        
        from testing.locust_manager import create_locust_test_manager
        manager = create_locust_test_manager()
        
        # 创建测试配置
        config = manager.create_test_config(data)
        
        # 启动测试
        test_id = manager.start_load_test(config)
        
        # 获取Web界面URL
        web_url = manager.get_locust_web_url(test_id)
        
        return {
            "status": "success",
            "test_id": test_id,
            "web_url": web_url,
            "message": "压力测试已启动"
        }
        
    except Exception as e:
        logger.error(f"启动压力测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动压力测试失败: {str(e)}")


@app.get("/load-test/status/{test_id}")
async def get_load_test_status(test_id: str):
    """获取压力测试状态"""
    try:
        from testing.locust_manager import create_locust_test_manager
        manager = create_locust_test_manager()
        
        status = manager.get_test_status(test_id)
        
        if status:
            return {
                "status": "success",
                "test_status": status
            }
        else:
            raise HTTPException(status_code=404, detail="测试不存在")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取测试状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取测试状态失败: {str(e)}")


@app.post("/load-test/stop/{test_id}")
async def stop_load_test(test_id: str):
    """停止压力测试"""
    try:
        from testing.locust_manager import create_locust_test_manager
        manager = create_locust_test_manager()
        
        success = manager.stop_test(test_id)
        
        if success:
            return {
                "status": "success",
                "message": "压力测试已停止"
            }
        else:
            raise HTTPException(status_code=404, detail="测试不存在或已停止")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止测试失败: {str(e)}")


@app.get("/load-test/web-url/{test_id}")
async def get_load_test_web_url(test_id: str):
    """获取Locust Web界面URL"""
    try:
        from testing.locust_manager import create_locust_test_manager
        manager = create_locust_test_manager()
        
        web_url = manager.get_locust_web_url(test_id)
        
        if web_url:
            return {
                "status": "success",
                "web_url": web_url
            }
        else:
            raise HTTPException(status_code=404, detail="测试不存在或Web界面不可用")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取Web界面URL失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取Web界面URL失败: {str(e)}")


@app.get("/load-test/list")
async def list_load_tests():
    """列出所有压力测试"""
    try:
        from testing.locust_manager import create_locust_test_manager
        manager = create_locust_test_manager()
        
        tests = manager.list_active_tests()
        
        return {
            "status": "success",
            "tests": tests,
            "count": len(tests)
        }
        
    except Exception as e:
        logger.error(f"列出测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出测试失败: {str(e)}")

@app.get("/progress/{tracking_id}")
async def get_insert_progress(tracking_id: str):
    """获取插入进度状态"""
    if not _app_initialized or not _progress_tracker:
        raise HTTPException(
            status_code=503, 
            detail="服务未初始化"
        )
    
    try:
        progress_status = _progress_tracker.get_progress_status(tracking_id)
        return progress_status
        
    except Exception as e:
        logger.error(f"获取插入进度失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取插入进度失败: {str(e)}"
        )

@app.post("/progress/cleanup")
async def cleanup_old_progress():
    """清理旧的进度跟踪数据"""
    if not _app_initialized or not _progress_tracker:
        raise HTTPException(
            status_code=503, 
            detail="服务未初始化"
        )
    
    try:
        _progress_tracker.cleanup_old_tracking()
        return {"message": "旧的进度跟踪数据已清理", "status": "success"}
        
    except Exception as e:
        logger.error(f"清理进度跟踪数据失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"清理进度跟踪数据失败: {str(e)}"
        )

@app.get("/errors/history")
async def get_error_history(limit: int = 50):
    """获取错误历史"""
    if not _app_initialized or not _error_manager:
        raise HTTPException(
            status_code=503, 
            detail="服务未初始化"
        )
    
    try:
        error_history = _error_manager.get_error_history(limit)
        return {
            "error_history": error_history,
            "count": len(error_history),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"获取错误历史失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取错误历史失败: {str(e)}"
        )

@app.post("/errors/clear")
async def clear_error_history():
    """清除错误历史"""
    if not _app_initialized or not _error_manager:
        raise HTTPException(
            status_code=503, 
            detail="服务未初始化"
        )
    
    try:
        _error_manager.clear_error_history()
        return {"message": "错误历史已清除", "status": "success"}
        
    except Exception as e:
        logger.error(f"清除错误历史失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"清除错误历史失败: {str(e)}"
        )


# ==================== 文本分块相关接口 ====================

# 导入分块相关模块
try:
    from dataBuilder.chunking import (
        ChunkingProcessRequest, ChunkingProcessResponse, ChunkingErrorResponse,
        ChunkingManager, create_success_response, create_error_response,
        calculate_chunking_metrics, validate_text_input, ProcessingTimer,
        format_error_message, get_strategy_display_name, get_available_strategies,
        ChunkingErrorHandler, ResponseFormatter, ErrorType, global_error_handler
    )
    from dataBuilder.chunking.chunk_strategies import ChunkingStrategyResolver
    from dataBuilder.chunking.meta_chunking import DependencyChecker
    CHUNKING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"分块模块导入失败: {e}")
    CHUNKING_AVAILABLE = False

# 全局分块服务变量
chunking_manager = None
dependency_checker = None

def initialize_chunking_services():
    """初始化分块服务"""
    global chunking_manager, dependency_checker
    
    if not CHUNKING_AVAILABLE:
        logger.warning("分块模块不可用，跳过初始化")
        return False
    
    try:
        # 初始化依赖检查器
        dependency_checker = DependencyChecker()
        
        # 初始化分块管理器
        chunking_manager = ChunkingManager()
        
        logger.info("分块服务初始化成功")
        return True
        
    except Exception as e:
        logger.error(f"分块服务初始化失败: {e}")
        return False

@app.on_event("startup")
async def startup_chunking_services():
    """启动时初始化分块服务"""
    if CHUNKING_AVAILABLE:
        initialize_chunking_services()

@app.get("/chunking/strategies")
async def get_chunking_strategies():
    """获取可用的分块策略"""
    if not CHUNKING_AVAILABLE:
        raise HTTPException(status_code=503, detail="分块服务不可用")
    
    try:
        strategies = get_available_strategies()
        
        # 添加策略可用性检查
        if dependency_checker:
            for strategy in strategies:
                strategy["available"] = True  # 简化处理
        
        return {
            "success": True,
            "strategies": strategies,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取策略列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略列表失败: {str(e)}")

@app.post("/chunking/process")
async def process_text_chunking(request: ChunkingProcessRequest):
    """
    处理文本分块请求
    
    支持所有五种分块策略：traditional, meta_ppl, margin_sampling, msp, semantic
    """
    if not CHUNKING_AVAILABLE or not chunking_manager:
        error_response = global_error_handler.handle_strategy_unavailable_error(
            strategy="service",
            reason="分块服务未初始化或不可用"
        )
        raise HTTPException(
            status_code=503, 
            detail=ResponseFormatter.format_error_response(error_response)
        )
    
    # 验证输入文本
    is_valid, error_msg = validate_text_input(request.text)
    if not is_valid:
        error_response = global_error_handler.handle_text_validation_error(
            text=request.text,
            validation_error=error_msg
        )
        raise HTTPException(
            status_code=400, 
            detail=ResponseFormatter.format_error_response(error_response)
        )
    
    # 开始处理
    with ProcessingTimer() as timer:
        try:
            logger.info(f"开始处理分块请求: 策略={request.strategy.value}, 文本长度={len(request.text)}")
            
            # 检查策略可用性
            if dependency_checker:
                resolver = ChunkingStrategyResolver(dependency_checker)
                actual_strategy = resolver.resolve_strategy(
                    request.strategy.value, 
                    {"glm_configured": True}  # 简化配置检查
                )
                
                if actual_strategy != request.strategy.value:
                    logger.warning(f"策略降级: {request.strategy.value} -> {actual_strategy}")
            else:
                actual_strategy = request.strategy.value
            
            # 执行分块
            chunks = chunking_manager.chunk_text(
                text=request.text,
                strategy=actual_strategy,
                **request.params
            )
            
            # 计算性能指标
            processing_time = timer.get_elapsed_time()
            metrics = None
            warnings = []
            
            # 添加策略降级警告
            if actual_strategy != request.strategy.value:
                warnings.append(f"策略已从 {request.strategy.value} 降级到 {actual_strategy}")
            
            if request.enable_metrics:
                metrics = calculate_chunking_metrics(
                    chunks=chunks,
                    processing_time=processing_time,
                    strategy_used=actual_strategy,
                    fallback_occurred=(actual_strategy != request.strategy.value)
                )
            
            # 创建成功响应
            response = create_success_response(
                request=request,
                chunks=chunks,
                actual_strategy=actual_strategy,
                processing_time=processing_time,
                warnings=warnings if warnings else None,
                metrics=metrics
            )
            
            logger.info(f"分块处理完成: {len(chunks)} 个分块, 耗时 {processing_time:.3f}s")
            
            # 格式化响应
            formatted_response = ResponseFormatter.format_success_response(response.dict())
            return formatted_response
            
        except TimeoutError as e:
            processing_time = timer.get_elapsed_time()
            error_response = global_error_handler.handle_timeout_error(
                strategy=request.strategy.value,
                timeout=request.timeout,
                actual_time=processing_time
            )
            raise HTTPException(
                status_code=408, 
                detail=ResponseFormatter.format_error_response(error_response)
            )
            
        except ValueError as e:
            # 参数错误
            error_response = global_error_handler.handle_parameter_error(
                strategy=request.strategy.value,
                invalid_params=request.params,
                validation_errors=[str(e)]
            )
            raise HTTPException(
                status_code=400, 
                detail=ResponseFormatter.format_error_response(error_response)
            )
            
        except Exception as e:
            # 内部错误
            processing_time = timer.get_elapsed_time()
            error_response = global_error_handler.handle_internal_error(
                error=e,
                context="分块处理",
                strategy=request.strategy.value
            )
            raise HTTPException(
                status_code=500, 
                detail=ResponseFormatter.format_error_response(error_response)
            )

@app.get("/chunking/status")
async def get_chunking_service_status():
    """获取分块服务状态"""
    if not CHUNKING_AVAILABLE:
        return {
            "service_available": False,
            "error": "分块模块不可用",
            "timestamp": datetime.now().isoformat()
        }
    
    status = {
        "service_available": True,
        "chunking_manager_available": chunking_manager is not None,
        "dependency_checker_available": dependency_checker is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    # 获取依赖状态
    if dependency_checker:
        try:
            status["dependencies"] = dependency_checker.check_ppl_dependencies()
            status["ppl_chunking_available"] = dependency_checker.is_ppl_chunking_available()
            status["dependency_status_message"] = dependency_checker.get_dependency_status_message()
        except Exception as e:
            status["dependency_error"] = str(e)
    
    # 获取LLM状态
    if chunking_manager:
        try:
            status["llm_status"] = chunking_manager.get_llm_status()
        except Exception as e:
            status["llm_status_error"] = str(e)
    
    return status

@app.get("/chunking/config/{strategy}")
async def get_chunking_strategy_config(strategy: str):
    """获取特定策略的配置参数"""
    if not CHUNKING_AVAILABLE or not chunking_manager:
        raise HTTPException(status_code=503, detail="分块服务不可用")
    
    try:
        config = chunking_manager.get_strategy_config(strategy)
        
        response_data = {
            "strategy": strategy,
            "display_name": get_strategy_display_name(strategy),
            "config": config
        }
        
        return ResponseFormatter.format_success_response(response_data)
        
    except Exception as e:
        logger.error(f"获取策略配置失败: {e}")
        error_response = global_error_handler.handle_parameter_error(
            strategy=strategy,
            invalid_params={"strategy": strategy},
            validation_errors=[f"获取策略配置失败: {str(e)}"]
        )
        raise HTTPException(
            status_code=400, 
            detail=ResponseFormatter.format_error_response(error_response)
        )

@app.get("/chunking/errors/statistics")
async def get_error_statistics():
    """获取错误统计信息"""
    if not CHUNKING_AVAILABLE:
        raise HTTPException(status_code=503, detail="分块服务不可用")
    
    try:
        stats = global_error_handler.get_error_statistics()
        return ResponseFormatter.format_success_response({"statistics": stats})
        
    except Exception as e:
        logger.error(f"获取错误统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取错误统计失败: {str(e)}")

@app.post("/chunking/validate")
async def validate_chunking_request(request: ChunkingProcessRequest):
    """验证分块请求参数"""
    if not CHUNKING_AVAILABLE:
        raise HTTPException(status_code=503, detail="分块服务不可用")
    
    validation_errors = []
    
    # 验证文本
    is_valid, error_msg = validate_text_input(request.text)
    if not is_valid:
        validation_errors.append(f"文本验证失败: {error_msg}")
    
    # 验证策略参数
    try:
        if chunking_manager:
            config = chunking_manager.get_strategy_config(request.strategy.value)
            # 这里可以添加更详细的参数验证逻辑
    except Exception as e:
        validation_errors.append(f"策略参数验证失败: {str(e)}")
    
    # 检查依赖
    if dependency_checker and request.strategy.value in ["meta_ppl", "msp", "margin_sampling"]:
        if not dependency_checker.is_ppl_chunking_available():
            missing_deps = dependency_checker.get_missing_dependencies()
            validation_errors.append(f"策略依赖缺失: {', '.join(missing_deps)}")
    
    validation_response = ResponseFormatter.format_validation_response(
        is_valid=len(validation_errors) == 0,
        errors=validation_errors
    )
    
    return validation_response

# ==================== 文本分块接口结束 ====================

if __name__ == "__main__":
    import uvicorn
    logger.info("启动新架构API服务...")
    uvicorn.run(app, host="0.0.0.0", port=8505)  # 使用原来的端口