
import os
import time
import json
import uuid
import subprocess
import threading
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import signal

logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """压测配置"""
    test_id: str
    users: int
    spawn_rate: float
    run_time: str
    host: str
    port: str
    collection_name: str
    test_scenarios: List[str]
    search_params: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class LoadTestResult:
    """压测结果"""
    test_id: str
    status: str  # running, completed, failed, stopped
    start_time: datetime
    end_time: Optional[datetime]
    config: LoadTestConfig
    metrics: Dict[str, Any]
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # 处理datetime序列化
        result['start_time'] = self.start_time.isoformat() if self.start_time else None
        result['end_time'] = self.end_time.isoformat() if self.end_time else None
        result['config'] = self.config.to_dict()
        return result


class LocustTestManager:
    """Locust压测管理器"""
    
    def __init__(self):
        self.active_tests: Dict[str, LoadTestResult] = {}
        self.test_processes: Dict[str, subprocess.Popen] = {}
        self.test_lock = threading.Lock()
        self.results_dir = "test_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def create_test_config(self, params: Dict[str, Any]) -> LoadTestConfig:
        """
        创建测试配置
        
        Args:
            params: 测试参数
            
        Returns:
            测试配置对象
        """
        test_id = str(uuid.uuid4())[:8]
        
        # 默认测试场景
        default_scenarios = [
            "single_vector_search",
            "batch_vector_search", 
            "high_precision_search",
            "fast_search"
        ]
        
        # 默认搜索参数
        default_search_params = {
            "metric_type": "L2",
            "nprobe": 16,
            "top_k": 10
        }
        
        config = LoadTestConfig(
            test_id=test_id,
            users=params.get('users', 10),
            spawn_rate=params.get('spawn_rate', 1.0),
            run_time=params.get('run_time', '60s'),
            host=params.get('host', 'localhost'),
            port=params.get('port', '19530'),
            collection_name=params.get('collection_name', 'Test_one'),
            test_scenarios=params.get('test_scenarios', default_scenarios),
            search_params={**default_search_params, **params.get('search_params', {})}
        )
        
        return config
    
    def start_load_test(self, config: LoadTestConfig) -> str:
        """
        启动压力测试
        
        Args:
            config: 测试配置
            
        Returns:
            测试ID
        """
        with self.test_lock:
            # 检查是否已有相同的测试在运行
            for test_id, result in self.active_tests.items():
                if result.status == "running":
                    logger.warning(f"已有测试 {test_id} 在运行中")
            
            # 创建测试结果对象
            test_result = LoadTestResult(
                test_id=config.test_id,
                status="starting",
                start_time=datetime.now(),
                end_time=None,
                config=config,
                metrics={},
                errors=[]
            )
            
            self.active_tests[config.test_id] = test_result
            
            try:
                # 创建Locust测试文件
                locust_file = self._create_locust_file(config)
                
                # 启动Locust进程
                process = self._start_locust_process(config, locust_file)
                
                if process:
                    self.test_processes[config.test_id] = process
                    test_result.status = "running"
                    logger.info(f"压测 {config.test_id} 启动成功")
                    
                    # 启动监控线程
                    monitor_thread = threading.Thread(
                        target=self._monitor_test,
                        args=(config.test_id,),
                        daemon=True
                    )
                    monitor_thread.start()
                    
                else:
                    test_result.status = "failed"
                    test_result.errors.append("无法启动Locust进程")
                    
            except Exception as e:
                test_result.status = "failed"
                test_result.errors.append(f"启动测试失败: {str(e)}")
                logger.error(f"启动测试失败: {e}")
            
            return config.test_id
    
    def _create_locust_file(self, config: LoadTestConfig) -> str:
        """创建Locust测试文件"""
        locust_content = f'''
"""
动态生成的Locust测试文件
测试ID: {config.test_id}
"""

import time
import random
import logging
import numpy as np
from locust import User, task, between, events
from pymilvus import connections, Collection, utility
import threading

logger = logging.getLogger(__name__)

# 全局连接管理
_connection_initialized = False
_shared_connection = "test_{config.test_id}"
_shared_collection = None
_shared_dimension = 256
_connection_lock = threading.Lock()

def init_shared_connection():
    global _connection_initialized, _shared_collection, _shared_dimension
    
    with _connection_lock:
        if _connection_initialized:
            return
        
        try:
            connections.connect(
                alias=_shared_connection,
                host="{config.host}",
                port=int("{config.port}"),
                timeout=10
            )
            
            if not utility.has_collection("{config.collection_name}", using=_shared_connection):
                raise Exception(f"集合 '{config.collection_name}' 不存在")
            
            _shared_collection = Collection("{config.collection_name}", using=_shared_connection)
            
            # 获取向量维度
            schema = _shared_collection.schema
            for field in schema.fields:
                if field.name in ["embedding", "vector"]:
                    _shared_dimension = field.params.get('dim', 256)
                    break
            
            _connection_initialized = True
            logger.info(f"连接成功！向量维度: {{_shared_dimension}}")
            
        except Exception as e:
            logger.error(f"连接失败: {{e}}")
            raise

class MilvusUser(User):
    wait_time = between(0.5, 2.0)
    
    def on_start(self):
        init_shared_connection()
        self.collection = _shared_collection
        self.dimension = _shared_dimension
    
    def generate_random_vector(self):
        vector = np.random.normal(0, 1, self.dimension).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
'''

        # 添加测试场景
        if "single_vector_search" in config.test_scenarios:
            locust_content += '''
    @task(10)
    def single_vector_search(self):
        self._perform_search("single_search", batch_size=1, top_k=10)
'''

        if "batch_vector_search" in config.test_scenarios:
            locust_content += '''
    @task(5)
    def batch_vector_search(self):
        batch_size = random.randint(2, 5)
        self._perform_search("batch_search", batch_size=batch_size, top_k=10)
'''

        if "high_precision_search" in config.test_scenarios:
            locust_content += '''
    @task(3)
    def high_precision_search(self):
        self._perform_search("high_precision_search", batch_size=1, top_k=50)
'''

        if "fast_search" in config.test_scenarios:
            locust_content += '''
    @task(2)
    def fast_search(self):
        self._perform_search("fast_search", batch_size=1, top_k=5)
'''

        # 添加搜索执行方法
        locust_content += f'''
    def _perform_search(self, name: str, batch_size: int = 1, top_k: int = 10):
        start_time = time.time()
        exception = None
        result_count = 0
        
        try:
            query_vectors = []
            for _ in range(batch_size):
                query_vectors.append(self.generate_random_vector())
            
            search_params = {{
                "metric_type": "{config.search_params.get('metric_type', 'L2')}",
                "params": {{"nprobe": {config.search_params.get('nprobe', 16)}}}
            }}
            
            results = self.collection.search(
                data=query_vectors,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                timeout=30
            )
            
            result_count = sum(len(result) for result in results if result is not None)
            
        except Exception as e:
            exception = e
            logger.error(f"搜索失败: {{e}}")
        
        response_time = (time.time() - start_time) * 1000
        
        events.request.fire(
            request_type="SEARCH",
            name=name,
            response_time=response_time,
            response_length=result_count,
            exception=exception
        )

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("Milvus性能测试开始")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    if _connection_initialized and connections.has_connection(_shared_connection):
        connections.disconnect(_shared_connection)
'''

        # 保存到临时文件
        temp_file = os.path.join(tempfile.gettempdir(), f"locust_test_{config.test_id}.py")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(locust_content)
        
        return temp_file
    
    def _start_locust_process(self, config: LoadTestConfig, locust_file: str) -> Optional[subprocess.Popen]:
        """启动Locust进程（支持Web界面）"""
        try:
            # 动态分配端口
            web_port = self._find_available_port(12089)
            
            # 构建Locust命令（启用Web界面）
            cmd = [
                "locust",
                "-f", locust_file,
                "--web-host", "0.0.0.0",
                "--web-port", str(web_port),
                "--users", str(config.users),
                "--spawn-rate", str(config.spawn_rate),
                "--run-time", config.run_time,
                "--csv", os.path.join(self.results_dir, f"test_{config.test_id}"),
                "--html", os.path.join(self.results_dir, f"test_{config.test_id}.html"),
                "--autostart"  # 自动开始测试
            ]
            
            # 启动进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 更新测试结果中的Web信息
            if config.test_id in self.active_tests:
                self.active_tests[config.test_id].config.web_port = web_port
                self.active_tests[config.test_id].web_url = f"http://localhost:{web_port}"
                self.active_tests[config.test_id].process_id = process.pid
            
            logger.info(f"Locust进程启动: PID {process.pid}, Web端口: {web_port}")
            return process
            
        except Exception as e:
            logger.error(f"启动Locust进程失败: {e}")
            return None
    
    def _find_available_port(self, start_port: int = 12089) -> int:
        """查找可用端口"""
        import socket
        
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        
        # 如果找不到可用端口，返回默认端口
        return start_port
    
    def get_locust_web_url(self, test_id: str) -> Optional[str]:
        """获取Locust Web界面URL"""
        test_result = self.active_tests.get(test_id)
        if test_result and hasattr(test_result, 'web_url'):
            return test_result.web_url
        return None
    
    def _monitor_test(self, test_id: str):
        """监控测试进程"""
        try:
            process = self.test_processes.get(test_id)
            test_result = self.active_tests.get(test_id)
            
            if not process or not test_result:
                return
            
            # 等待进程结束
            stdout, stderr = process.communicate()
            
            # 更新测试结果
            test_result.end_time = datetime.now()
            
            if process.returncode == 0:
                test_result.status = "completed"
                # 读取测试结果
                self._collect_test_results(test_id)
            else:
                test_result.status = "failed"
                if stderr:
                    test_result.errors.append(f"进程错误: {stderr}")
            
            # 保存测试结果到文件
            self._save_test_results(test_id)
            
            # 清理
            if test_id in self.test_processes:
                del self.test_processes[test_id]
            
            logger.info(f"测试 {test_id} 完成，状态: {test_result.status}")
            
        except Exception as e:
            logger.error(f"监控测试 {test_id} 失败: {e}")
            if test_id in self.active_tests:
                self.active_tests[test_id].status = "failed"
                self.active_tests[test_id].errors.append(f"监控失败: {str(e)}")
    
    def _collect_test_results(self, test_id: str):
        """收集测试结果"""
        try:
            test_result = self.active_tests.get(test_id)
            if not test_result:
                return
            
            # 读取CSV结果文件
            stats_file = os.path.join(self.results_dir, f"test_{test_id}_stats.csv")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # 跳过标题行
                        # 解析统计数据
                        stats_data = lines[1].strip().split(',')
                        if len(stats_data) >= 10:
                            test_result.metrics = {
                                "total_requests": int(stats_data[1]) if stats_data[1].isdigit() else 0,
                                "failures": int(stats_data[2]) if stats_data[2].isdigit() else 0,
                                "avg_response_time": float(stats_data[5]) if stats_data[5].replace('.', '').isdigit() else 0.0,
                                "min_response_time": float(stats_data[6]) if stats_data[6].replace('.', '').isdigit() else 0.0,
                                "max_response_time": float(stats_data[7]) if stats_data[7].replace('.', '').isdigit() else 0.0,
                                "requests_per_second": float(stats_data[9]) if stats_data[9].replace('.', '').isdigit() else 0.0
                            }
            
        except Exception as e:
            logger.error(f"收集测试结果失败: {e}")
            test_result.errors.append(f"结果收集失败: {str(e)}")
    
    def _save_test_results(self, test_id: str):
        """保存测试结果到JSON文件"""
        try:
            test_result = self.active_tests.get(test_id)
            if not test_result:
                return
            
            # 确保结果目录存在
            os.makedirs(self.results_dir, exist_ok=True)
            
            # 保存测试结果
            result_file = os.path.join(self.results_dir, f"test_{test_id}.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(test_result.to_dict(), f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"测试结果已保存: {result_file}")
            
        except Exception as e:
            logger.error(f"保存测试结果失败: {e}")
            if test_id in self.active_tests:
                self.active_tests[test_id].errors.append(f"结果保存失败: {str(e)}")
    
    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """获取测试状态"""
        test_result = self.active_tests.get(test_id)
        if test_result:
            return test_result.to_dict()
        return None
    
    def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """获取测试结果"""
        return self.get_test_status(test_id)
    
    def stop_test(self, test_id: str) -> bool:
        """停止测试"""
        try:
            process = self.test_processes.get(test_id)
            if process and process.poll() is None:  # 进程还在运行
                process.terminate()
                time.sleep(2)
                if process.poll() is None:  # 如果还没结束，强制杀死
                    process.kill()
                
                # 更新状态
                if test_id in self.active_tests:
                    self.active_tests[test_id].status = "stopped"
                    self.active_tests[test_id].end_time = datetime.now()
                
                logger.info(f"测试 {test_id} 已停止")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"停止测试失败: {e}")
            return False
    
    def list_active_tests(self) -> List[Dict[str, Any]]:
        """列出活跃的测试"""
        return [result.to_dict() for result in self.active_tests.values()]
    
    def cleanup_completed_tests(self, max_age_hours: int = 24):
        """清理完成的测试"""
        current_time = datetime.now()
        to_remove = []
        
        for test_id, result in self.active_tests.items():
            if result.status in ["completed", "failed", "stopped"]:
                if result.end_time:
                    age = (current_time - result.end_time).total_seconds() / 3600
                    if age > max_age_hours:
                        to_remove.append(test_id)
        
        for test_id in to_remove:
            del self.active_tests[test_id]
            # 清理相关文件
            try:
                temp_file = os.path.join(tempfile.gettempdir(), f"locust_test_{test_id}.py")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        
        logger.info(f"清理了 {len(to_remove)} 个过期测试")


def create_locust_test_manager() -> LocustTestManager:
    """创建Locust测试管理器实例"""
    return LocustTestManager()