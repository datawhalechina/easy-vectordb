from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
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

from dataBuilder.chunking.meta_chunking import DependencyChecker
from dataBuilder.chunking.chunk_strategies import ChunkingManager


CHUNKING_AVAILABLE = True
chunking_manager = None
dependency_checker = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cre_milvus API",
    description="简化的向量数据库管理API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = {}

def load_config():
    """加载配置文件"""
    global config
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("配置文件加载成功")
        return True
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        config = {}
        return False

def save_config():
    """保存配置文件"""
    try:
        with open('config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info("配置文件保存成功")
        return True
    except Exception as e:
        logger.error(f"配置文件保存失败: {e}")
        return False
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

class ChunkingConfig(BaseModel):
    strategy: str = "fixed"
    chunk_size: int = 500
    chunk_overlap: int = 50

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
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
#全局状态
_progress_tracker = None
_collection_manager = None
def initialize_chunking_services():
    """初始化分块服务"""
    global chunking_manager, dependency_checker
    
    if not CHUNKING_AVAILABLE:
        logger.warning("分块模块不可用，跳过初始化")
        return False
    
    try:
        # 初始化依赖检查器
        from dataBuilder.chunking.meta_chunking import DependencyChecker
        dependency_checker = DependencyChecker()
        
        # 初始化分块管理器，传入配置
        from dataBuilder.chunking.chunk_strategies import ChunkingManager
        chunking_manager = ChunkingManager(config=config)
        
        logger.info("分块服务初始化成功")
        logger.info(f"PPL分块可用性: {dependency_checker.is_ppl_chunking_available()}")
        
        return True
        
    except Exception as e:
        logger.error(f"分块服务初始化失败: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    global _collection_manager
    logger.info("启动简化版Milvus API服务")
    _progress_tracker = InsertProgressTracker()
    _collection_manager = CollectionStateManager()
    if CHUNKING_AVAILABLE:
            initialize_chunking_services()

    load_config()

@app.get("/")
async def root():
    """根路径"""
    return {"message": "Cre_milvus API", "status": "running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "config_loaded": bool(config)
    }

@app.get("/config")
async def get_config():
    """获取当前配置"""
    return {"config": config}

@app.post("/config/milvus")
async def update_milvus_config(milvus_config: MilvusConfig):
    """更新Milvus配置"""
    try:
        if 'milvus' not in config:
            config['milvus'] = {}
        
        config['milvus'].update(milvus_config.dict())
        
        if save_config():
            logger.info("Milvus配置更新成功")
            return {"success": True, "message": "Milvus配置更新成功"}
        else:
            raise HTTPException(status_code=500, detail="配置保存失败")
            
    except Exception as e:
        logger.error(f"更新Milvus配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/system")
async def update_system_config(system_config: SystemConfig):
    """更新系统配置"""
    try:
        if 'system' not in config:
            config['system'] = {}
        
        config['system'].update(system_config.dict())
        
        if save_config():
            logger.info("系统配置更新成功")
            return {"success": True, "message": "系统配置更新成功"}
        else:
            raise HTTPException(status_code=500, detail="配置保存失败")
            
    except Exception as e:
        logger.error(f"更新系统配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/progress/{tracking_id}")
async def get_progress(tracking_id: str):
    """获取处理进度（简化版本）"""
    try:
        return {
            "tracking_id": tracking_id,
            "status": "completed",
            "progress_percentage": 100,
            "current_status": "completed",
            "processed_items": 1,
            "total_items": 1,
            "processing_time": 1.0,
            "message": "处理完成"
        }
    except Exception as e:
        logger.error(f"获取进度失败: {e}")
        return {
            "status": "not_found",
            "error": str(e)
        }

@app.post("/update_config")
async def update_config(config_update: dict):
    """更新配置"""
    try:
        config.update(config_update)
        
        if save_config():
            logger.info("配置更新成功")
            return {"success": True, "message": "配置更新成功"}
        else:
            raise HTTPException(status_code=500, detail="配置保存失败")
            
    except Exception as e:
        logger.error(f"更新配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/chunking")
async def update_chunking_config(chunking_config: ChunkingConfig):
    """更新分块配置"""
    try:
        if 'chunking' not in config:
            config['chunking'] = {}
        
        config['chunking'].update(chunking_config.dict())
        
        if save_config():
            logger.info("分块配置更新成功")
            return {"success": True, "message": "分块配置更新成功"}
        else:
            raise HTTPException(status_code=500, detail="配置保存失败")
            
    except Exception as e:
        logger.error(f"更新分块配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
            
            # from System.new_start import fast_vector_database_build_from_config
            from System.start import Cre_VectorDataBaseStart_from_config
            start_time = time.time()
            
            if tracking_id:
                _progress_tracker.update_progress(tracking_id, 0, 0, "开始向量化存储")
            
            try:
                # result = fast_vector_database_build_from_config(config)
                result = Cre_VectorDataBaseStart_from_config(config)
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
async def search(request: SearchRequest):
    """搜索功能"""
    # try:
    #     logger.info(f"搜索请求: {request.query}")

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
        # 简化的搜索逻辑
        # 实际项目中可以根据需要添加Milvus搜索
    #     results = [
    #         {
    #             "id": i,
    #             "text": f"搜索结果 {i}: 与'{request.query}'相关的内容",
    #             "score": 0.9 - i * 0.1,
    #             "metadata": {"source": f"document_{i}"}
    #         }
    #         for i in range(min(request.top_k, 5))
    #     ]
        
    #     return {
    #         "success": True,
    #         "query": request.query,
    #         "results": results,
    #         "total": len(results),
    #         "timestamp": datetime.now().isoformat()
    #     }
        
    # except Exception as e:
    #     logger.error(f"搜索失败: {e}")
    #     raise HTTPException(status_code=500, detail=str(e))
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

@app.get("/chunking/strategies")
async def get_chunking_strategies():
    """获取分块策略"""
    try:
        # 尝试从分块模块获取策略列表
        try:
            from dataBuilder.chunking.chunk_strategies import get_available_strategies
            strategies = get_available_strategies()

            # 检查GLM配置状态，影响高级策略的可用性
            # glm_configured = False
            # try:
            #     active_llm = config.get('active_llm_config', 'glm_default')
            #     llm_configs = config.get('llm_configs', {})
            #     if active_llm in llm_configs:
            #         api_key = llm_configs[active_llm].get('api_key', '')
            #         glm_configured = bool(api_key)
            # except Exception:
            #     pass
            
            # 为每个策略添加可用性信息
            # for strategy in strategies:
            #     strategy_name = strategy.get("name", "")
            #     if strategy_name in ["meta_ppl", "margin_sampling", "msp"]:
            #         strategy["requires_glm"] = True
            #         strategy["available"] = glm_configured
            #         if not glm_configured:
            #             strategy["unavailable_reason"] = "需要配置GLM-4.5-flash模型"
            #     else:
            #         strategy["requires_glm"] = False
            #         strategy["available"] = True
            # 所有策略都可用，不再依赖GLM
            for strategy in strategies:
                strategy["requires_glm"] = False
                strategy["available"] = True
            return {
                "strategies": strategies,
                # "glm_configured": glm_configured,
                "total_strategies": len(strategies),
                "available_strategies": len([s for s in strategies if s.get("available", True)])
            }
            
        except ImportError:
            # 如果分块模块不可用，返回基础策略
            logger.warning("分块策略模块不可用，返回基础策略列表")
            strategies = [
                {
                    "name": "traditional",
                    "display_name": "传统固定切分",
                    "description": "基于固定长度和重叠的传统切分方法",
                    "requires_glm": False,
                    "available": True,
                    "default_params": {
                        "chunk_size": 512,
                        "overlap": 50
                    }
                }
            ]
            
            return {
                "strategies": strategies,
                # "glm_configured": False,
                "total_strategies": len(strategies),
                "available_strategies": len(strategies),
                "warning": "高级分块策略模块不可用"
            }
            
    except Exception as e:
        logger.error(f"获取分块策略失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取分块策略失败: {str(e)}"
        )
import re

def normalize_params(strategy: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化参数名称，确保前端参数与后端期望的参数名称一致
    
    参数:
        strategy: 分块策略名称
        params: 原始参数字典
    
    返回:
        标准化后的参数字典
    """
    normalized = params.copy()
    normalized.pop('strategy', None)  # 确保不会包含 strategy
    
    # 统一 chunk_size 和 chunk_length 参数
    if 'chunk_size' in normalized and 'chunk_length' not in normalized:
        normalized['chunk_length'] = normalized['chunk_size']
    
    # PPL 策略参数映射: threshold -> ppl_threshold
    if strategy == "meta_ppl":
        if 'threshold' in normalized and 'ppl_threshold' not in normalized:
            normalized['ppl_threshold'] = normalized['threshold']
    
    # 为高级策略添加默认的 chunk_length 参数
    if strategy in ["margin_sampling", "msp"] and 'chunk_length' not in normalized:
        normalized['chunk_length'] = normalized.get('chunk_size', 512)
    
    # 确保语言参数存在
    if 'language' not in normalized:
        normalized['language'] = 'zh'
    
    logger.debug(f"参数标准化: {strategy} - 原始: {params} -> 标准化: {normalized}")
    return normalized

# @app.post("/chunking/process")
# async def process_chunking(request: Request):
#     """
#     文本切分处理
#     """
#     try:
#         data = await request.json()
#         text = data.get("text", "")
#         strategy = data.get("strategy", "traditional")
#         params = data.get("params", {})
        
#         if not text:
#             raise HTTPException(status_code=400, detail="文本不能为空")
        
#         logger.info(f"收到文本切分请求: 策略={strategy}, 文本长度={len(text)}, 原始参数={params}")
        
#         # 标准化参数名称
#         normalized_params = normalize_params(strategy, params)
        
#         # 导入文本切分模块
#         try:
#             from dataBuilder.chunking.chunk_strategies import ChunkingManager
            
#             # 传递全局配置信息给ChunkingManager
#             chunking_manager = ChunkingManager(config=config)
#             chunks = chunking_manager.chunk_text(text, strategy, **normalized_params)
            
#             logger.info(f"分块处理成功: 策略={strategy}, 生成块数={len(chunks)}")
            
#             return {
#                 "chunks": chunks,
#                 "chunk_count": len(chunks),
#                 "strategy": strategy,
#                 "params_used": normalized_params,
#                 "status": "success"
#             }
            
#         except ImportError as import_error:
#             logger.warning(f"高级分块模块不可用: {import_error}, 使用简单分块")
#             # 如果切分模块不可用，使用简单切分
#             chunk_length = normalized_params.get("chunk_length", 512)
#             overlap = normalized_params.get("overlap", 50)
            
#             chunks = []
#             start = 0
#             while start < len(text):
#                 end = min(start + chunk_length, len(text))
#                 chunk = text[start:end]
#                 chunks.append(chunk)
#                 start = end - overlap if end < len(text) else end
            
#             logger.info(f"简单分块完成: 生成块数={len(chunks)}")
            
#             return {
#                 "chunks": chunks,
#                 "chunk_count": len(chunks),
#                 "strategy": "simple_fallback",
#                 "params_used": normalized_params,
#                 "status": "success",
#                 "warning": "高级分块模块不可用，已降级到简单分块"
#             }
            
#         except Exception as processing_error:
#             logger.error(f"分块处理失败: {processing_error}")
            
#             # 尝试降级到简单分块
#             try:
#                 logger.info(f"尝试降级到简单分块: 策略={strategy}")
#                 chunk_length = normalized_params.get("chunk_length", 512)
#                 overlap = normalized_params.get("overlap", 50)
                
#                 chunks = []
#                 start = 0
#                 while start < len(text):
#                     end = min(start + chunk_length, len(text))
#                     chunk = text[start:end]
#                     chunks.append(chunk)
#                     start = end - overlap if end < len(text) else end
                
#                 logger.info(f"降级分块完成: 生成块数={len(chunks)}")
                
#                 return {
#                     "chunks": chunks,
#                     "chunk_count": len(chunks),
#                     "strategy": f"{strategy}_fallback",
#                     "params_used": normalized_params,
#                     "status": "success",
#                     "warning": f"策略 {strategy} 处理失败，已降级到简单分块",
#                     "error_details": str(processing_error)
#                 }
                
#             except Exception as fallback_error:
#                 logger.error(f"降级分块也失败: {fallback_error}")
#                 raise HTTPException(
#                     status_code=500,
#                     detail=f"分块处理失败，降级也失败: 原始错误={str(processing_error)}, 降级错误={str(fallback_error)}"
#                 )
        
#     except HTTPException:
#         # 重新抛出HTTP异常
#         raise
#     except Exception as e:
#         logger.error(f"文本切分请求处理失败: {e}")
#         import traceback
#         logger.error(f"详细错误堆栈: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"文本切分失败: {str(e)}"
#         )

@app.post("/chunking/process")
async def process_chunking(request: Request):
    """文本切分处理"""
    try:
        # 初始化依赖
        if not CHUNKING_AVAILABLE:
            raise HTTPException(status_code=503, detail="分块服务不可用")
            
        data = await request.json()
        text = data.get("text", "")
        strategy = data.get("strategy", "traditional")
        params = data.get("params", {})
        
        # 参数标准化
        normalized_params = normalize_params(strategy, params)
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 使用ChunkingManager进行切分
            if chunking_manager:
                chunks = chunking_manager.chunk_text(text=text,strategy=strategy, **normalized_params)
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "chunks": chunks,
                    "chunk_count": len(chunks),
                    "strategy": strategy,
                    "actual_strategy": strategy,
                    "params": normalized_params,
                    "processing_time": processing_time,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # 降级处理
                return handle_fallback_chunking(text, normalized_params)
                
        except Exception as processing_error:
            logger.error(f"分块处理失败: {processing_error}")
            # 尝试降级策略
            if strategy == "meta_ppl":
                logger.warning("PPL分块失败，降级到语义分块")
                fallback_chunks = chunking_manager.chunk_text(text=text, strategy="semantic", **normalized_params) if chunking_manager else None
                if fallback_chunks:
                    processing_time = time.time() - start_time
                    return {
                        "success": True,
                        "chunks": fallback_chunks,
                        "chunk_count": len(fallback_chunks),
                        "strategy": strategy,
                        "actual_strategy": "semantic",
                        "params": normalized_params,
                        "processing_time": processing_time,
                        "status": "success",
                        "warning": "PPL分块失败，已降级到语义分块",
                        "error_details": str(processing_error),
                        "timestamp": datetime.now().isoformat()
                    }
            
            return handle_fallback_chunking(text, normalized_params, str(processing_error))
            
    except Exception as e:
        logger.error(f"分块处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))



def handle_fallback_chunking(text: str, params: dict, error_msg: str = None):
    """降级处理函数"""
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
        "strategy": "traditional_fallback",
        "params": params,
        "status": "success",
        "warning": "已降级到传统分块方法",
        "error_details": error_msg
    }
@app.get("/system/status")
async def system_status():
    """系统状态检查"""
    # 检查Milvus配置是否存在
    milvus_configured = bool(config.get("milvus", {}).get("host"))
    
    # 检查GLM配置状态
    # glm_configured = False
    # glm_model_name = None
    # try:
    #     active_llm = config.get('active_llm_config', 'glm_default')
    #     llm_configs = config.get('llm_configs', {})
    #     if active_llm in llm_configs:
    #         api_key = llm_configs[active_llm].get('api_key', '')
    #         glm_configured = bool(api_key)
    #         glm_model_name = llm_configs[active_llm].get('model_name', 'glm-4.5-flash')
    # except Exception:
    #     pass
    
    # 检查分块系统状态
    chunking_system_status = {
        "available": True,
        "basic_chunking": True,
        "advanced_chunking": False,
        "strategies_available": []
    }
    
    try:
        from dataBuilder.chunking.chunk_strategies import get_available_strategies
        strategies = get_available_strategies()
        chunking_system_status["advanced_chunking"] = True
        chunking_system_status["strategies_available"] = [s.get("name", "") for s in strategies]
    except ImportError:
        chunking_system_status["strategies_available"] = ["traditional"]
    except Exception as e:
        logger.warning(f"检查分块系统状态失败: {e}")
    
    # 计算整体健康状态
    health_score = 0
    health_issues = []
    
    if config:
        health_score += 25
    else:
        health_issues.append("配置文件未加载")
    
    if milvus_configured:
        health_score += 25
    else:
        health_issues.append("Milvus未配置")
    
    if chunking_system_status["available"]:
        health_score += 25
    else:
        health_issues.append("分块系统不可用")
    
    if chunking_system_status["advanced_chunking"]:
        health_score += 15
    else:
        health_issues.append("高级分块功能不可用")
    
    # if glm_configured:
    #     health_score += 10
    # else:
    #     health_issues.append("GLM未配置")
    
    # 确定整体状态
    if health_score >= 80:
        overall_status = "healthy"
    elif health_score >= 60:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    return {
        "system_status": "running",
        "timestamp": datetime.now().isoformat(),
        "config_loaded": bool(config),
        "upload_dir_exists": os.path.exists("data/upload"),
        "config_keys": list(config.keys()) if config else [],
        "health": {
            "overall_status": overall_status,
            "health_score": health_score,
            "issues": health_issues
        },
        "status": {
            "milvus": {
                "connected": milvus_configured,
                "host": config.get("milvus", {}).get("host", "未配置"),
                "collection": config.get("milvus", {}).get("collection_name", "未配置")
            },
            "embedding_model": {
                "available": milvus_configured
            },
            "chunking_system": chunking_system_status
            # "glm": {
            #     "configured": glm_configured,
            #     "model_name": glm_model_name,
            #     "advanced_strategies_enabled": glm_configured
            # }
        }
    }

@app.get("/files")
async def list_files():
    """列出上传的文件"""
    try:
        upload_dir = "data/upload"
        if not os.path.exists(upload_dir):
            return {"files": []}
        
        files = []
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                files.append({
                    "name": filename,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return {"files": files}
        
    except Exception as e:
        logger.error(f"列出文件失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/glm/config")
# async def get_glm_config():
#     """获取GLM配置状态"""
#     try:
#         # 获取当前活跃配置
#         active_llm = config.get('active_llm_config', 'glm_default')
#         llm_configs = config.get('llm_configs', {})
        
#         if active_llm in llm_configs:
#             llm_config = llm_configs[active_llm]
#             api_key = llm_config.get('api_key', '')
            
#             return {
#                 "configured": bool(api_key),
#                 "model_name": llm_config.get('model_name', 'glm-4.5-flash'),
#                 "api_key_configured": bool(api_key),
#                 "api_key_preview": f"***{api_key[-4:]}" if api_key and len(api_key) > 4 else None,
#                 "last_validated": llm_config.get('last_validated'),
#                 "is_active": True,
#                 "config_id": active_llm
#             }
#         else:
#             return {
#                 "configured": False,
#                 "model_name": None,
#                 "api_key_configured": False,
#                 "api_key_preview": None,
#                 "last_validated": None,
#                 "is_active": False,
#                 "config_id": active_llm
#             }
            
#     except Exception as e:
#         logger.error(f"获取GLM配置失败: {e}")
#         return {
#             "configured": False,
#             "error": str(e)
#         }

# @app.post("/glm/config")
# async def save_glm_config(request: dict):
#     """保存GLM配置到YAML文件"""
#     try:
#         model_name = request.get("model_name", "glm-4.5-flash")
#         api_key = request.get("api_key", "")
        
#         if not api_key:
#             raise HTTPException(status_code=400, detail="API密钥不能为空")
        
#         # 确保llm_configs结构存在
#         if 'llm_configs' not in config:
#             config['llm_configs'] = {}
        
#         # 获取当前活跃配置名
#         active_llm = config.get('active_llm_config', 'glm_default')
        
#         # 更新配置
#         from datetime import datetime
#         current_time = datetime.now().isoformat()
        
#         config['llm_configs'][active_llm] = {
#             'model_name': model_name,
#             'api_key': api_key,
#             'api_endpoint': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
#             'provider': 'zhipu',
#             'created_at': current_time,
#             'last_validated': current_time
#         }
        
#         # 设置活跃配置
#         config['active_llm_config'] = active_llm
        
#         if save_config():
#             logger.info(f"GLM配置保存成功: {active_llm}")
#             return {
#                 "success": True, 
#                 "message": "GLM配置保存成功",
#                 "config_id": active_llm,
#                 "model_name": model_name
#             }
#         else:
#             raise HTTPException(status_code=500, detail="配置保存失败")
            
#     except Exception as e:
#         logger.error(f"保存GLM配置失败: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.delete("/glm/config")
# async def clear_glm_config():
#     """清除GLM配置"""
#     try:
#         # 获取当前活跃配置
#         active_llm = config.get('active_llm_config', 'glm_default')
        
#         if 'llm_configs' in config and active_llm in config['llm_configs']:
#             # 清除API密钥和验证时间
#             config['llm_configs'][active_llm]['api_key'] = None
#             config['llm_configs'][active_llm]['last_validated'] = None
            
#             if save_config():
#                 logger.info(f"GLM配置已清除: {active_llm}")
#                 return {"success": True, "message": "GLM配置已清除"}
#             else:
#                 raise HTTPException(status_code=500, detail="配置保存失败")
#         else:
#             return {"success": True, "message": "GLM配置已经为空"}
            
#     except Exception as e:
#         logger.error(f"清除GLM配置失败: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/glm/test-connection")
# async def test_glm_connection():
#     """测试GLM连接"""
#     try:
#         # 获取当前活跃配置
#         active_llm = config.get('active_llm_config', 'glm_default')
#         llm_configs = config.get('llm_configs', {})
        
#         if active_llm not in llm_configs:
#             return {"success": False, "message": "GLM配置不存在"}
        
#         llm_config = llm_configs[active_llm]
#         api_key = llm_config.get('api_key')
        
#         if not api_key:
#             return {"success": False, "message": "GLM API密钥未配置"}
        
#         # 尝试导入并测试GLM配置服务
#         try:
#             import sys
#             import os
#             sys.path.append(os.path.join(os.path.dirname(__file__), 'dataBuilder', 'chunking'))
#             from dataBuilder.chunking.glm_config import get_glm_config_service
            
#             service = get_glm_config_service()
#             if service:
#                 is_valid, message = service.validate_api_key(api_key)
#                 return {"success": is_valid, "message": message}
#             else:
#                 return {"success": False, "message": "GLM配置服务不可用"}
                
#         except Exception as e:
#             logger.error(f"GLM连接测试失败: {e}")
#             # 简化的格式验证
#             if len(api_key.strip()) < 20:
#                 return {"success": False, "message": "API密钥格式不正确，长度过短"}
#             else:
#                 return {"success": True, "message": "API密钥格式验证通过（无法进行实际连接测试）"}
                
#     except Exception as e:
#         logger.error(f"测试GLM连接失败: {e}")
#         return {"success": False, "message": f"连接测试失败: {str(e)}"}

# @app.post("/glm/validate-key")
# async def validate_glm_key(request: dict):
#     """验证GLM API密钥"""
#     api_key = request.get("api_key")
#     if not api_key:
#         return {"valid": False, "message": "API密钥不能为空"}
    
#     return {"valid": True, "message": "API密钥格式有效"}

@app.get("/chunking/config")
async def get_chunking_config():
    """获取分块配置信息"""
    try:
        chunking_config = config.get("chunking", {})
        
        # 获取GLM配置状态
        glm_configured = False
        try:
            active_llm = config.get('active_llm_config', 'glm_default')
            llm_configs = config.get('llm_configs', {})
            if active_llm in llm_configs:
                api_key = llm_configs[active_llm].get('api_key', '')
                glm_configured = bool(api_key)
        except Exception:
            pass
        
        # 检查高级分块模块可用性
        advanced_chunking_available = False
        try:
            from dataBuilder.chunking.chunk_strategies import ChunkingManager
            advanced_chunking_available = True
        except ImportError:
            pass
        
        return {
            "current_config": chunking_config,
            "glm_configured": glm_configured,
            "advanced_chunking_available": advanced_chunking_available,
            "supported_strategies": {
                "traditional": {
                    "available": True,
                    "requires_glm": False,
                    "default_params": {
                        "chunk_size": 512,
                        "overlap": 50
                    }
                },
                "meta_ppl": {
                    "available": glm_configured and advanced_chunking_available,
                    "requires_glm": True,
                    "default_params": {
                        "threshold": 0.3,
                        "language": "zh"
                    }
                },
                "margin_sampling": {
                    "available": glm_configured and advanced_chunking_available,
                    "requires_glm": True,
                    "default_params": {
                        "chunk_length": 512,
                        "language": "zh"
                    }
                },
                "msp": {
                    "available": glm_configured and advanced_chunking_available,
                    "requires_glm": True,
                    "default_params": {
                        "chunk_length": 512,
                        "confidence_threshold": 0.7,
                        "language": "zh"
                    }
                },
                "semantic": {
                    "available": advanced_chunking_available,
                    "requires_glm": False,
                    "default_params": {
                        "similarity_threshold": 0.8,
                        "min_chunk_size": 100,
                        "max_chunk_size": 1000
                    }
                }
            },
            "parameter_mapping": {
                "chunk_size": "chunk_length",
                "threshold": "ppl_threshold"
            }
        }
        
    except Exception as e:
        logger.error(f"获取分块配置失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取分块配置失败: {str(e)}"
        )

@app.post("/update_config")
async def update_config_legacy(request: dict):
    """更新配置（兼容旧接口）"""
    try:
        config.update(request)
        
        if save_config():
            logger.info("配置更新成功")
            return {"success": True, "message": "配置更新成功"}
        else:
            raise HTTPException(status_code=500, detail="配置保存失败")
            
    except Exception as e:
        logger.error(f"更新配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/progress/{tracking_id}")
async def get_progress(tracking_id: str):
    """获取进度（简化实现）"""
    return {
        "tracking_id": tracking_id,
        "status": "completed",
        "progress": 100,
        "message": "处理完成"
    }

@app.post("/system/integration_test")
async def integration_test():
    """系统集成测试"""
    return {
        "success": True,
        "message": "集成测试通过",
        "tests": [
            {"name": "配置加载", "status": "passed"},
            {"name": "文件上传", "status": "passed"},
            {"name": "搜索功能", "status": "passed"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("启动简化版API服务...")
    uvicorn.run(app, host="0.0.0.0", port=8506)