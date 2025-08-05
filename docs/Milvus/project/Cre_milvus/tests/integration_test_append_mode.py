#!/usr/bin/env python3
"""
Append模式数据插入的集成测试
"""

import requests
import time
import sys
import os
import json
import tempfile

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AppendModeIntegrationTest:
    """Append模式集成测试类"""
    
    def __init__(self, backend_url="http://localhost:8509"):
        self.backend_url = backend_url
        self.test_results = []
        self.test_collection_name = "test_append_integration"
        self.test_data = [
            {
                "text": "这是第一个测试文档，用于验证append模式的数据插入功能。",
                "metadata": {"source": "test1", "type": "integration_test"}
            },
            {
                "text": "这是第二个测试文档，包含不同的内容以验证数据多样性。",
                "metadata": {"source": "test2", "type": "integration_test"}
            },
            {
                "text": "第三个测试文档用于验证批量插入的稳定性和正确性。",
                "metadata": {"source": "test3", "type": "integration_test"}
            }
        ]
    
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """记录测试结果"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message
        })
    
    def test_backend_connection(self):
        """测试后端连接"""
        try:
            response = requests.get(f"{self.backend_url}/", timeout=5)
            success = response.status_code == 200
            message = f"状态码: {response.status_code}" if success else "连接失败"
            self.log_test("后端连接测试", success, message)
            return success
        except Exception as e:
            self.log_test("后端连接测试", False, str(e))
            return False
    
    def test_collection_management_api(self):
        """测试集合管理API"""
        try:
            # 测试获取集合列表
            response = requests.get(f"{self.backend_url}/collections", timeout=10)
            success = response.status_code == 200
            
            if success:
                collections = response.json().get("collections", [])
                message = f"获取到 {len(collections)} 个集合"
            else:
                message = f"获取集合列表失败，状态码: {response.status_code}"
            
            self.log_test("集合管理API测试", success, message)
            return success
        except Exception as e:
            self.log_test("集合管理API测试", False, str(e))
            return False
    
    def test_collection_creation(self):
        """测试集合创建功能"""
        try:
            # 首先删除测试集合（如果存在）
            self.cleanup_test_collection()
            
            # 创建测试集合
            create_data = {
                "collection_name": self.test_collection_name,
                "dimension": 768,
                "description": "Integration test collection for append mode"
            }
            
            response = requests.post(
                f"{self.backend_url}/create-collection",
                json=create_data,
                timeout=30
            )
            
            success = response.status_code == 200
            
            if success:
                result = response.json()
                message = f"集合创建成功: {result.get('message', '无消息')}"
            else:
                message = f"集合创建失败，状态码: {response.status_code}"
                if response.text:
                    message += f", 错误: {response.text}"
            
            self.log_test("集合创建功能", success, message)
            return success
        except Exception as e:
            self.log_test("集合创建功能", False, str(e))
            return False
    
    def test_collection_loading(self):
        """测试集合加载功能"""
        try:
            # 检查集合状态
            response = requests.get(
                f"{self.backend_url}/collection-status/{self.test_collection_name}",
                timeout=10
            )
            
            if response.status_code == 200:
                status_data = response.json()
                is_loaded = status_data.get("loaded", False)
                
                if not is_loaded:
                    # 加载集合
                    load_response = requests.post(
                        f"{self.backend_url}/load-collection",
                        json={"collection_name": self.test_collection_name},
                        timeout=30
                    )
                    
                    success = load_response.status_code == 200
                    message = "集合加载成功" if success else f"集合加载失败，状态码: {load_response.status_code}"
                else:
                    success = True
                    message = "集合已处于加载状态"
            else:
                success = False
                message = f"获取集合状态失败，状态码: {response.status_code}"
            
            self.log_test("集合加载功能", success, message)
            return success
        except Exception as e:
            self.log_test("集合加载功能", False, str(e))
            return False
    
    def test_append_mode_data_insertion(self):
        """测试append模式数据插入"""
        try:
            # 创建临时文件用于测试
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                json.dump(self.test_data, f, ensure_ascii=False, indent=2)
                temp_file_path = f.name
            
            try:
                # 准备插入数据
                insert_data = {
                    "collection_name": self.test_collection_name,
                    "data_source": "json",
                    "file_path": temp_file_path,
                    "insert_mode": "append",
                    "chunking_strategy": "traditional",
                    "chunk_size": 200,
                    "overlap": 50
                }
                
                # 执行数据插入
                response = requests.post(
                    f"{self.backend_url}/insert-data",
                    json=insert_data,
                    timeout=60
                )
                
                success = response.status_code == 200
                
                if success:
                    result = response.json()
                    inserted_count = result.get("inserted_count", 0)
                    message = f"数据插入成功，插入 {inserted_count} 条记录"
                else:
                    message = f"数据插入失败，状态码: {response.status_code}"
                    if response.text:
                        try:
                            error_data = response.json()
                            message += f", 错误: {error_data.get('error', response.text)}"
                        except:
                            message += f", 错误: {response.text}"
                
                self.log_test("Append模式数据插入", success, message)
                return success
                
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except Exception as e:
            self.log_test("Append模式数据插入", False, str(e))
            return False
    
    def test_data_verification(self):
        """测试插入数据的验证"""
        try:
            # 获取集合统计信息
            response = requests.get(
                f"{self.backend_url}/collection-stats/{self.test_collection_name}",
                timeout=10
            )
            
            if response.status_code == 200:
                stats = response.json()
                entity_count = stats.get("entity_count", 0)
                
                success = entity_count > 0
                message = f"集合包含 {entity_count} 条记录" if success else "集合为空"
            else:
                success = False
                message = f"获取集合统计失败，状态码: {response.status_code}"
            
            self.log_test("数据验证", success, message)
            return success
        except Exception as e:
            self.log_test("数据验证", False, str(e))
            return False
    
    def test_search_functionality(self):
        """测试搜索功能"""
        try:
            # 执行搜索测试
            search_data = {
                "collection_name": self.test_collection_name,
                "query_text": "测试文档",
                "top_k": 3,
                "search_params": {"metric_type": "COSINE", "params": {"nprobe": 10}}
            }
            
            response = requests.post(
                f"{self.backend_url}/search",
                json=search_data,
                timeout=30
            )
            
            success = response.status_code == 200
            
            if success:
                results = response.json()
                hits = results.get("results", [])
                message = f"搜索成功，返回 {len(hits)} 个结果"
            else:
                message = f"搜索失败，状态码: {response.status_code}"
            
            self.log_test("搜索功能测试", success, message)
            return success
        except Exception as e:
            self.log_test("搜索功能测试", False, str(e))
            return False
    
    def test_append_mode_incremental_insertion(self):
        """测试append模式增量插入"""
        try:
            # 获取当前记录数
            stats_response = requests.get(
                f"{self.backend_url}/collection-stats/{self.test_collection_name}",
                timeout=10
            )
            
            if stats_response.status_code != 200:
                self.log_test("Append模式增量插入", False, "无法获取初始统计信息")
                return False
            
            initial_count = stats_response.json().get("entity_count", 0)
            
            # 准备增量数据
            incremental_data = [
                {
                    "text": "这是增量插入的第一个文档，用于测试append模式的增量功能。",
                    "metadata": {"source": "incremental1", "type": "append_test"}
                },
                {
                    "text": "增量插入的第二个文档，验证数据能够正确追加到现有集合中。",
                    "metadata": {"source": "incremental2", "type": "append_test"}
                }
            ]
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                json.dump(incremental_data, f, ensure_ascii=False, indent=2)
                temp_file_path = f.name
            
            try:
                # 执行增量插入
                insert_data = {
                    "collection_name": self.test_collection_name,
                    "data_source": "json",
                    "file_path": temp_file_path,
                    "insert_mode": "append",
                    "chunking_strategy": "traditional",
                    "chunk_size": 200,
                    "overlap": 50
                }
                
                response = requests.post(
                    f"{self.backend_url}/insert-data",
                    json=insert_data,
                    timeout=60
                )
                
                if response.status_code != 200:
                    self.log_test("Append模式增量插入", False, f"增量插入失败，状态码: {response.status_code}")
                    return False
                
                # 验证记录数增加
                time.sleep(2)  # 等待插入完成
                final_stats_response = requests.get(
                    f"{self.backend_url}/collection-stats/{self.test_collection_name}",
                    timeout=10
                )
                
                if final_stats_response.status_code == 200:
                    final_count = final_stats_response.json().get("entity_count", 0)
                    success = final_count > initial_count
                    message = f"记录数从 {initial_count} 增加到 {final_count}" if success else f"记录数未增加: {initial_count} -> {final_count}"
                else:
                    success = False
                    message = "无法获取最终统计信息"
                
                self.log_test("Append模式增量插入", success, message)
                return success
                
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except Exception as e:
            self.log_test("Append模式增量插入", False, str(e))
            return False
    
    def cleanup_test_collection(self):
        """清理测试集合"""
        try:
            # 删除测试集合
            response = requests.delete(
                f"{self.backend_url}/collection/{self.test_collection_name}",
                timeout=30
            )
            # 不管成功失败都继续，因为集合可能不存在
            return True
        except:
            return True
    
    def run_all_tests(self):
        """运行所有集成测试"""
        print("=" * 60)
        print("🧪 Append模式数据插入集成测试")
        print("=" * 60)
        
        tests = [
            self.test_backend_connection,
            self.test_collection_management_api,
            self.test_collection_creation,
            self.test_collection_loading,
            self.test_append_mode_data_insertion,
            self.test_data_verification,
            self.test_search_functionality,
            self.test_append_mode_incremental_insertion
        ]
        
        try:
            for test in tests:
                test()
                time.sleep(1)  # 测试间隔
        finally:
            # 清理测试数据
            print("\n🧹 清理测试数据...")
            self.cleanup_test_collection()
        
        # 统计结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        
        print("=" * 60)
        print(f"📊 测试结果: {passed_tests}/{total_tests} 通过")
        
        if passed_tests == total_tests:
            print("✅ 所有Append模式集成测试通过！")
            return True
        else:
            print("❌ 部分Append模式集成测试失败！")
            
            # 显示失败的测试
            failed_tests = [result for result in self.test_results if not result["success"]]
            for failed in failed_tests:
                print(f"   - {failed['test']}: {failed['message']}")
            
            return False


if __name__ == '__main__':
    tester = AppendModeIntegrationTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)