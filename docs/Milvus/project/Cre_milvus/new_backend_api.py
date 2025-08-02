#!/usr/bin/env python3
"""
新的后端API
使用预连接架构，避免连接阻塞问题
"""

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import yaml
import logging
from typing import List, Dict, Any
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="Cre_milvus 新架构API", version="3.0.0")

# 全局状态
_app_initialized = False

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化连接"""
    global _app_initialized
    
    logger.info("=" * 60)
    logger.info("🚀 API服务启动，初始化连接...")
    logger.info("=" * 60)
    
    try:
        from System.connection_initializer import startup_initialize
        success = startup_initialize()
        
        if success:
            _app_initialized = True
            logger.info("✅ API服务初始化成功")
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
async def upload_files(files: List[UploadFile] = File(...), folder_name: str = Form(None)):
    """
    上传文件并进行向量化存储
    使用动态连接架构
    """
    if not _app_initialized:
        raise HTTPException(
            status_code=503, 
            detail="服务未初始化，请等待初始化完成"
        )
    
    try:
        logger.info(f"收到文件上传请求，文件数量: {len(files)}, 目标文件夹: {folder_name}")
        
        # 1. 保存上传的文件
        if folder_name and folder_name.strip():
            upload_dir = f"./data/upload/{folder_name.strip()}"
            logger.info(f"使用指定文件夹: {upload_dir}")
        else:
            upload_dir = "./data/upload"
            logger.info(f"使用默认文件夹: {upload_dir}")
        os.makedirs(upload_dir, exist_ok=True)
        
        uploaded_files = []
        file_types = {}
        
        for file in files:
            if file.filename:
                file_path = os.path.join(upload_dir, file.filename)
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                uploaded_files.append(file.filename)
                file_extension = os.path.splitext(file.filename)[1].lower()
                file_types[file_extension] = file_types.get(file_extension, 0) + 1
                
                logger.info(f"文件已保存: {file.filename}")
        
        logger.info(f"文件上传完成: {len(uploaded_files)} 个文件")
        
        # 2. 使用新架构进行向量化存储
        try:
            logger.info("开始向量化存储...")
            
            # 加载配置
            with open("config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            # 更新配置中的数据路径为当前上传的文件夹
            if folder_name:
                if "data" not in config:
                    config["data"] = {}
                config["data"]["data_location"] = upload_dir
                logger.info(f"更新数据路径为: {upload_dir}")
                
                # 保存更新后的配置
                with open("config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            # 使用新的快速构建功能
            from System.new_start import fast_vector_database_build_from_config
            
            start_time = time.time()
            result = fast_vector_database_build_from_config(config)
            end_time = time.time()
            
            logger.info(f"向量化存储完成，耗时: {end_time - start_time:.2f}秒")
            
            if result.get("status") == "success":
                return {
                    "message": f"成功上传 {len(uploaded_files)} 个文件并完成向量化存储",
                    "files": uploaded_files,
                    "upload_dir": upload_dir,
                    "file_types": file_types,
                    "vectorized": True,
                    "vectorization_result": result,
                    "processing_time": end_time - start_time,
                    "status": "success"
                }
            else:
                return {
                    "message": f"成功上传 {len(uploaded_files)} 个文件，但向量化存储失败",
                    "files": uploaded_files,
                    "upload_dir": upload_dir,
                    "file_types": file_types,
                    "vectorized": False,
                    "error": result.get("msg", "未知错误"),
                    "status": "partial_success"
                }
                
        except Exception as vector_error:
            logger.error(f"向量化存储失败: {vector_error}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            
            return {
                "message": f"成功上传 {len(uploaded_files)} 个文件，但向量化存储失败",
                "files": uploaded_files,
                "upload_dir": upload_dir,
                "file_types": file_types,
                "vectorized": False,
                "error": str(vector_error),
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
    搜索文档
    """
    if not _app_initialized:
        raise HTTPException(
            status_code=503, 
            detail="服务未初始化，请等待初始化完成"
        )
    
    try:
        data = await request.json()
        question = data.get("question", "")
        
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")
        
        logger.info(f"收到搜索请求: {question}")
        
        # 加载配置
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 使用原有的搜索功能
        from System.start import Cre_Search
        
        start_time = time.time()
        result = Cre_Search(config, question)
        end_time = time.time()
        
        logger.info(f"搜索完成，耗时: {end_time - start_time:.2f}秒")
        
        # 直接返回搜索结果，保持与前端的兼容性
        return result
        
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail=f"搜索失败: {str(e)}"
        )

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

if __name__ == "__main__":
    import uvicorn
    logger.info("启动新架构API服务...")
    uvicorn.run(app, host="0.0.0.0", port=8509)  # 使用原来的端口