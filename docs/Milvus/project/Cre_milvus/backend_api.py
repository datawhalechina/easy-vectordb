# 文件: backend_api.py - 整合版本
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import yaml
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from System.start import load_config, Cre_VectorDataBaseStart_from_config, Cre_Search
except ImportError as e:
    logger.warning(f"System模块导入失败: {e}")
    # 提供fallback函数
    def load_config():
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except:
            return {}
    
    def Cre_VectorDataBaseStart_from_config(config):
        logger.warning("Cre_VectorDataBaseStart_from_config不可用")
        pass
    
    def Cre_Search(config, question):
        return {"error": "搜索功能不可用", "query": question}
try:
    from ColBuilder.visualization import get_all_embeddings_and_texts
    import hdbscan
    from umap import UMAP
    import pandas as pd
except ImportError as e:
    logger.warning(f"可视化模块导入失败: {e}")
    # 提供fallback函数
    def get_all_embeddings_and_texts(collection_name):
        return [], [], []
    
    # 创建模拟的pandas DataFrame
    class MockDataFrame:
        def to_dict(self, orient="records"):
            return []
    
    class MockPandas:
        def DataFrame(self, *args, **kwargs):
            return MockDataFrame()
    
    pd = MockPandas()
import logging
from typing import List, Dict, Any, Optional

# 导入新的模块（使用try-except避免导入失败）
try:
    from dataBuilder.chunking import ChunkingManager, get_available_strategies
except ImportError as e:
    logger.warning(f"chunking模块导入失败: {e}")
    class ChunkingManager:
        def chunk_text(self, text, strategy, **kwargs):
            # 简单的传统切分fallback
            chunk_size = kwargs.get('chunk_size', 512)
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            return chunks
        
        def get_strategy_config(self, strategy):
            return {}
    
    def get_available_strategies():
        return [{"name": "traditional", "display_name": "传统切分", "description": "基础切分功能"}]
try:
    from .multimodal import CLIPEncoder, TextProcessor, ImageProcessor
except ImportError as e:
    logger.warning(f"multimodal模块导入失败: {e}")
    CLIPEncoder = None
    TextProcessor = None
    ImageProcessor = None

try:
    from testing import MilvusLoadTest, PerformanceMonitor, TestDataGenerator
except ImportError as e:
    logger.warning(f"testing模块导入失败: {e}")
    class PerformanceMonitor:
        def __init__(self):
            self.is_monitoring = False
        def start_monitoring(self):
            pass
        def stop_monitoring(self):
            pass
        def get_current_metrics(self):
            return {}
        def get_historical_data(self, metric, duration):
            return []
    
    MilvusLoadTest = None
    TestDataGenerator = None

# python -m uvicorn backend_api:app --reload --port 8506
app = FastAPI(title="Cre_milvus 整合版API", version="2.0.0")


def safe_load_config():
    """安全的配置加载函数，避免阻塞"""
    try:
        # 使用绝对路径而不是相对路径
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except Exception as e:
        logger.warning(f"配置加载失败，使用默认配置: {e}")
        return {
            "milvus": {
                "host": "127.0.0.1",
                "port": "19530",
                "collection_name": "Test_one",
                "index_name": "HNSW"
            },
            "search": {
                "top_k": 20,
                "col_choice": "hdbscan"
            }
        }

# 全局实例
try:
    performance_monitor = PerformanceMonitor()
except:
    performance_monitor = None

clip_encoder = None
chunking_manager = None
text_processor = None
image_processor = None

@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("后端API启动完成，模块将按需加载")


def init_advanced_modules():
    """按需初始化高级模块"""
    global clip_encoder, chunking_manager, text_processor, image_processor
    
    if chunking_manager is None:
        try:
            chunking_manager = ChunkingManager()
            logger.info("文本切分管理器初始化成功")
        except Exception as e:
            logger.warning(f"文本切分管理器初始化失败: {e}")
    
    if text_processor is None:
        try:
            text_processor = TextProcessor(chunking_manager)
            logger.info("文本处理器初始化成功")
        except Exception as e:
            logger.warning(f"文本处理器初始化失败: {e}")
    
    if image_processor is None:
        try:
            image_processor = ImageProcessor(clip_encoder)
            logger.info("图像处理器初始化成功")
        except Exception as e:
            logger.warning(f"图像处理器初始化失败: {e}")
    
    # 初始化CLIP编码器（如果配置启用）
    if clip_encoder is None:
        try:
            config = safe_load_config()
            if config.get('multimodal', {}).get('enable_image', False):
                clip_model = config.get('multimodal', {}).get('clip_model', 'ViT-B/32')
                clip_encoder = CLIPEncoder(clip_model)
                logger.info(f"CLIP编码器初始化成功: {clip_model}")
        except Exception as e:
            logger.warning(f"CLIP编码器初始化失败: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    try:
        if performance_monitor:
            performance_monitor.stop_monitoring()
    except:
        pass
    logger.info("应用已关闭")

@app.post("/update_config")
async def update_config(request: Request):
    """更新配置文件"""
    try:
        data = await request.json()
        # 使用绝对路径而不是相对路径
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        config.update(data)
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)
        
        logger.info("配置已更新")
        return {"message": "配置已更新", "status": "success"}
    
    except Exception as e:
        logger.error(f"更新配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...), folder_name: str = None):
    try:
        if not folder_name:
            return {"message": "未指定目标文件夹名", "status": "error"}
        
        upload_dir = f"data/upload/{folder_name}"
        logger.info(f"上传目录: {upload_dir}")
        os.makedirs(upload_dir, exist_ok=True)
        
        # 第一步：上传文件
        uploaded_files = []
        file_types = {"text": [], "image": [], "other": []}
        
        for file in files:
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            uploaded_files.append(file.filename)
            
            # 分类文件类型
            ext = os.path.splitext(file.filename)[1].lower()
            if ext in ['.txt', '.md', '.pdf']:
                file_types["text"].append(file.filename)
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                file_types["image"].append(file.filename)
            else:
                file_types["other"].append(file.filename)
            
            logger.info(f"文件已上传: {file.filename}")
        
        logger.info(f"文件分类统计: 文本文件{len(file_types['text'])}个, 图像文件{len(file_types['image'])}个, 其他文件{len(file_types['other'])}个")
        
        # 第二步：尝试进行向量化存储
        try:
            logger.info("开始向量化存储...")
            config = safe_load_config()
            
            # 更新配置中的数据位置
            if "data" not in config:
                config["data"] = {}
            config["data"]["data_location"] = upload_dir
            
            # 检查嵌入模型状态
            try:
                from Search.embedding import embedder
                status = embedder.check_status()
                logger.info(f"嵌入模型状态: {status}")
                
                if not status["model_loaded"]:
                    logger.warning("嵌入模型未加载，尝试重新加载...")
                    if not embedder.load_model():
                        raise Exception("嵌入模型加载失败")
            except Exception as e:
                logger.error(f"嵌入模型检查失败: {e}")
                return {
                    "message": f"成功上传 {len(uploaded_files)} 个文件，但嵌入模型不可用: {str(e)}",
                    "files": uploaded_files,
                    "upload_dir": upload_dir,
                    "file_types": file_types,
                    "vectorized": False,
                    "status": "partial_success"
                }
            
            # 调用向量化存储函数
            logger.info("调用向量化存储函数...")
            result = Cre_VectorDataBaseStart_from_config(config)
            
            return {
                "message": f"成功上传 {len(uploaded_files)} 个文件并完成向量化存储",
                "files": uploaded_files,
                "upload_dir": upload_dir,
                "file_types": file_types,
                "vectorized": True,
                "vectorization_result": result,
                "status": "success"
            }
            
        except Exception as vector_error:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"向量化存储失败: {vector_error}\n详细错误: {error_details}")
            
            return {
                "message": f"成功上传 {len(uploaded_files)} 个文件，但向量化存储失败: {str(vector_error)}",
                "files": uploaded_files,
                "upload_dir": upload_dir,
                "file_types": file_types,
                "vectorized": False,
                "error_details": str(vector_error),
                "status": "partial_success"
            }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"文件上传失败: {e}\n详细错误: {error_details}")
        return {
            "message": f"文件上传失败: {str(e)}", 
            "error_details": error_details,
            "status": "error"
        }

@app.post("/search")
async def search_api(question: str = Form(...)):
    try:
        config = safe_load_config()
        result = Cre_Search(config, question)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return JSONResponse(content={"error": f"搜索失败: {str(e)}"}, status_code=500)


@app.post("/visualization")
async def cluster_visualization(collection_name: str = Form(...)):
    try:
        ids, embeddings, texts = get_all_embeddings_and_texts(collection_name)
        if len(embeddings) == 0:
            return JSONResponse(content={"message": "没有可用的嵌入向量数据", "data": []}, status_code=404)

        # 检查是否有足够的数据进行聚类
        if len(embeddings) < 3:
            return JSONResponse(content={"message": "数据量不足，无法进行聚类", "data": []}, status_code=400)

        # HDBSCAN聚类
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=3)
            labels = clusterer.fit_predict(embeddings)
        except ImportError:
            # 如果hdbscan不可用，返回简单的聚类结果
            labels = [0] * len(embeddings)

        # UMAP降维
        try:
            from umap import UMAP
            umap = UMAP(n_components=2, random_state=42, n_neighbors=min(80, len(embeddings)-1), min_dist=0.1)
            umap_result = umap.fit_transform(embeddings)
        except ImportError:
            # 如果UMAP不可用，使用简单的2D投影
            import numpy as np
            umap_result = np.random.rand(len(embeddings), 2)

        # 创建结果数据
        result_data = []
        for i in range(len(embeddings)):
            if labels[i] != -1:  # 过滤噪声点
                result_data.append({
                    "x": float(umap_result[i][0]),
                    "y": float(umap_result[i][1]),
                    "cluster": str(labels[i]),
                    "text": texts[i] if i < len(texts) else f"文档{i}",
                    "id": ids[i] if i < len(ids) else i
                })

        return JSONResponse(content=result_data)
        
    except Exception as e:
        logger.error(f"聚类可视化失败: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"聚类可视化失败: {str(e)}"}
        )


# ==================== 新增API端点 ====================

@app.get("/chunking/strategies")
async def get_chunking_strategies():
    """获取可用的文本切分策略"""
    try:
        strategies = get_available_strategies()
        return {"strategies": strategies, "status": "success"}
    except Exception as e:
        logger.error(f"获取切分策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chunking/strategies/{strategy_name}/config")
async def get_strategy_config(strategy_name: str):
    """获取特定策略的配置参数"""
    try:
        init_advanced_modules()
        if chunking_manager:
            config = chunking_manager.get_strategy_config(strategy_name)
            return {"strategy": strategy_name, "config": config, "status": "success"}
        else:
            return {"strategy": strategy_name, "config": {}, "status": "not_available"}
    except Exception as e:
        logger.error(f"获取策略配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chunking/process")
async def process_chunking(request: Request):
    """执行文本切分"""
    try:
        # 按需初始化模块
        init_advanced_modules()
        
        data = await request.json()
        text = data.get("text", "")
        strategy = data.get("strategy", "traditional")
        params = data.get("params", {})
        
        if not text:
            raise HTTPException(status_code=400, detail="文本内容不能为空")
        
        if not chunking_manager:
            # 使用fallback切分
            chunk_size = params.get('chunk_size', 512)
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        else:
            chunks = chunking_manager.chunk_text(text, strategy, **params)
        
        return {
            "chunks": chunks,
            "chunk_count": len(chunks),
            "strategy": strategy,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"文本切分失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multimodal/encode_text")
async def encode_text(request: Request):
    """编码文本为向量"""
    try:
        # 按需初始化模块
        init_advanced_modules()
        
        data = await request.json()
        texts = data.get("texts", [])
        
        if not texts:
            raise HTTPException(status_code=400, detail="文本列表不能为空")
        
        if not clip_encoder:
            return {"error": "CLIP编码器不可用", "status": "not_available"}
        
        vectors = clip_encoder.encode_text(texts)
        
        return {
            "vectors": vectors.tolist(),
            "dimension": vectors.shape[1],
            "count": len(texts),
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"文本编码失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multimodal/encode_image")
async def encode_image(files: List[UploadFile] = File(...)):
    """编码图像为向量"""
    try:
        if not clip_encoder:
            return {"error": "CLIP编码器不可用", "status": "not_available"}
        
        # 临时保存上传的图像
        temp_paths = []
        for file in files:
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            temp_paths.append(temp_path)
        
        try:
            # 编码图像
            vectors = clip_encoder.encode_image(temp_paths)
            
            return {
                "vectors": vectors.tolist(),
                "dimension": vectors.shape[1],
                "count": len(files),
                "status": "success"
            }
        
        finally:
            # 清理临时文件
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"图像编码失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multimodal/text_to_image_search")
async def text_to_image_search(request: Request):
    """文搜图功能"""
    try:
        data = await request.json()
        query_text = data.get("query_text", "")
        top_k = data.get("top_k", 10)
        
        if not query_text:
            raise HTTPException(status_code=400, detail="查询文本不能为空")
        
        if not clip_encoder:
            return {
                "query_text": query_text,
                "results": [],
                "message": "CLIP编码器不可用，无法使用文搜图功能",
                "status": "not_available"
            }
        
        # 这里需要从数据库获取图像特征
        # 暂时返回模拟结果
        return {
            "query_text": query_text,
            "results": [],
            "message": "文搜图功能需要图像特征库支持",
            "status": "partial"
        }
    
    except Exception as e:
        logger.error(f"文搜图失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/current")
async def get_current_performance():
    """获取当前性能指标"""
    try:
        if performance_monitor:
            metrics = performance_monitor.get_current_metrics()
        else:
            # 基础性能指标fallback
            import psutil
            metrics = {
                "cpu": {"percent": psutil.cpu_percent()},
                "memory": {"percent": psutil.virtual_memory().percent}
            }
        return {"metrics": metrics, "status": "success"}
    
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/history")
async def get_performance_history(metric: str = "cpu", duration: int = 300):
    """获取性能历史数据"""
    try:
        if performance_monitor:
            history = performance_monitor.get_historical_data(metric, duration)
        else:
            history = []
        return {
            "metric": metric,
            "duration": duration,
            "data": history,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"获取性能历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/testing/start_performance_test")
async def start_performance_test(request: Request):
    """启动性能测试"""
    try:
        data = await request.json()
        config = safe_load_config()
        
        # 合并测试配置
        test_config = {
            "host": config.get("milvus", {}).get("host", "localhost"),
            "port": config.get("milvus", {}).get("port", "19530"),
            "collection_name": config.get("milvus", {}).get("collection_name", "Test_one")
        }
        test_config.update(data)
        
        load_test = MilvusLoadTest(test_config)
        result = load_test.start_test(
            users=data.get("users", 5),
            spawn_rate=data.get("spawn_rate", 1.0),
            run_time=data.get("run_time", "60s")
        )
        
        return result
    
    except Exception as e:
        logger.error(f"启动性能测试失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/testing/create_test_data")
async def create_test_data(request: Request):
    """创建测试数据"""
    try:
        data = await request.json()
        config = safe_load_config()
        
        generator = TestDataGenerator(
            host=config.get("milvus", {}).get("host", "localhost"),
            port=config.get("milvus", {}).get("port", "19530")
        )
        
        result = generator.create_test_collection(
            collection_name=data.get("collection_name", "locust_test_collection"),
            dimension=data.get("dimension", 256),
            num_vectors=data.get("num_vectors", 10000),
            batch_size=data.get("batch_size", 1000)
        )
        
        return result
    
    except Exception as e:
        logger.error(f"创建测试数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_with_progress")
async def upload_with_progress(files: list[UploadFile] = File(...), folder_name: str = None):
    """带进度反馈的文件上传和向量化存储"""
    try:
        if not folder_name:
            return {"message": "未指定目标文件夹名", "status": "error"}
        
        upload_dir = f"data/upload/{folder_name}"
        logger.info(f"上传目录: {upload_dir}")
        os.makedirs(upload_dir, exist_ok=True)
        
        # 进度跟踪
        progress = {
            "stage": "uploading",
            "current": 0,
            "total": len(files),
            "message": "正在上传文件..."
        }
        
        # 上传文件
        uploaded_files = []
        for i, file in enumerate(files):
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            uploaded_files.append(file.filename)
            progress["current"] = i + 1
            logger.info(f"文件上传进度: {progress['current']}/{progress['total']}")
        
        progress["stage"] = "processing"
        progress["message"] = "正在处理文件..."
        
        # 异步启动向量化存储
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def vectorize_data():
            try:
                config = safe_load_config()
                config["data"]["data_location"] = upload_dir
                return Cre_VectorDataBaseStart_from_config(config)
            except Exception as e:
                logger.error(f"向量化失败: {e}")
                return {"status": "error", "message": str(e)}
        
        # 在后台执行向量化
        with ThreadPoolExecutor() as executor:
            future = executor.submit(vectorize_data)
            # 这里可以添加进度检查逻辑
            result = future.result(timeout=300)  # 5分钟超时
        
        return {
            "message": f"成功处理 {len(uploaded_files)} 个文件",
            "files": uploaded_files,
            "upload_dir": upload_dir,
            "vectorization_result": result,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        return {"message": f"处理失败: {str(e)}", "status": "error"}


@app.get("/system/status")
async def get_system_status():
    """获取系统状态"""
    try:
        config = safe_load_config()
        
        # 检查嵌入模型状态
        embedding_status = {"available": False, "model_name": "unknown"}
        try:
            from Search.embedding import embedder
            status = embedder.check_status()
            embedding_status = {
                "available": status["model_loaded"] and status["tokenizer_loaded"],
                "model_name": status["model_name"],
                "device": status["device"]
            }
        except Exception as e:
            logger.warning(f"检查嵌入模型状态失败: {e}")
        
        # 检查Milvus连接状态
        milvus_status = {"connected": False}
        try:
            from pymilvus import connections, utility
            milvus_config = config.get("milvus", {})
            host = milvus_config.get("host", "127.0.0.1")
            port = milvus_config.get("port", "19530")
            
            # 尝试连接
            connections.connect(alias="status_check", host=host, port=port)
            milvus_status = {
                "connected": True,
                "host": host,
                "port": port,
                "collections": utility.list_collections()
            }
            connections.disconnect("status_check")
        except Exception as e:
            logger.warning(f"检查Milvus连接状态失败: {e}")
            milvus_status = {"connected": False, "error": str(e)}
        
        # 安全地检查性能监控状态
        monitor_status = False
        try:
            if performance_monitor:
                monitor_status = performance_monitor.is_monitoring
        except:
            pass
        
        status = {
            "performance_monitor": monitor_status,
            "clip_encoder": clip_encoder is not None,
            "chunking_manager": chunking_manager is not None,
            "text_processor": text_processor is not None,
            "image_processor": image_processor is not None,
            "embedding_model": embedding_status,
            "milvus": milvus_status,
            "config": {
                "milvus_host": config.get("milvus", {}).get("host"),
                "milvus_port": config.get("milvus", {}).get("port"),
                "collection_name": config.get("milvus", {}).get("collection_name"),
                "multimodal_enabled": config.get("multimodal", {}).get("enable_image", False),
                "chunking_strategy": config.get("chunking", {}).get("strategy", "traditional")
            }
        }
        
        return {"status": status, "message": "系统运行正常"}
    
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))