#!/usr/bin/env python3
"""
运行所有集成测试的脚本
"""

import sys
import os
import subprocess
import time
import requests

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class IntegrationTestRunner:
    """集成测试运行器"""
    
    def __init__(self, backend_url="http://localhost:8509"):
        self.backend_url = backend_url
        self.test_results = {}
    
    def check_backend_availability(self):
        """检查后端服务是否可用"""
        print("🔍 检查后端服务可用性...")
        try:
            response = requests.get(f"{self.backend_url}/", timeout=5)
            if response.status_code == 200:
                print("✅ 后端服务可用")
                return True
            else:
                print(f"❌ 后端服务响应异常，状态码: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 无法连接到后端服务: {e}")
            print(f"   请确保后端服务在 {self.backend_url} 运行")
            return False
    
    def run_test_script(self, script_name, test_description):
        """运行单个测试脚本"""
        print(f"\n{'='*60}")
        print(f"🧪 运行 {test_description}")
        print(f"{'='*60}")
        
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        
        try:
            # 运行测试脚本
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            # 输出测试结果
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("错误输出:", result.stderr)
            
            success = result.returncode == 0
            self.test_results[test_description] = {
                "success": success,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            return success
            
        except subprocess.TimeoutExpired:
            print(f"❌ 测试超时: {test_description}")
            self.test_results[test_description] = {
                "success": False,
                "error": "测试超时"
            }
            return False
        except Exception as e:
            print(f"❌ 运行测试失败: {e}")
            self.test_results[test_description] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def run_all_integration_tests(self):
        """运行所有集成测试"""
        print("🚀 开始运行Milvus系统关键修复集成测试")
        print("=" * 80)
        
        # 检查后端服务
        if not self.check_backend_availability():
            print("\n❌ 后端服务不可用，无法运行集成测试")
            print("请先启动后端服务，然后重新运行测试")
            return False
        
        # 定义要运行的测试
        tests = [
            ("integration_test_glm_config.py", "GLM配置前置功能集成测试"),
            ("integration_test_ppl_chunking.py", "PPL分块依赖修复集成测试"),
            ("integration_test_append_mode.py", "Append模式数据插入集成测试"),
            ("integration_test_end_to_end.py", "端到端综合集成测试")
        ]
        
        # 运行每个测试
        all_passed = True
        for script_name, test_description in tests:
            success = self.run_test_script(script_name, test_description)
            if not success:
                all_passed = False
            
            # 测试间隔
            time.sleep(2)
        
        # 输出总结
        self.print_summary()
        
        return all_passed
    
    def run_specific_test(self, test_name):
        """运行特定的集成测试"""
        test_mapping = {
            "glm": ("integration_test_glm_config.py", "GLM配置前置功能集成测试"),
            "ppl": ("integration_test_ppl_chunking.py", "PPL分块依赖修复集成测试"),
            "append": ("integration_test_append_mode.py", "Append模式数据插入集成测试"),
            "e2e": ("integration_test_end_to_end.py", "端到端综合集成测试")
        }
        
        if test_name not in test_mapping:
            print(f"❌ 未知的测试名称: {test_name}")
            print(f"可用的测试: {', '.join(test_mapping.keys())}")
            return False
        
        script_name, test_description = test_mapping[test_name]
        
        # 检查后端服务（除了PPL测试，它不需要后端）
        if test_name != "ppl":
            if not self.check_backend_availability():
                print(f"\n❌ 后端服务不可用，无法运行 {test_description}")
                return False
        
        success = self.run_test_script(script_name, test_description)
        self.print_summary()
        
        return success
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 80)
        print("📊 集成测试总结")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["success"])
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"{status} {test_name}")
            
            if not result["success"] and "error" in result:
                print(f"     错误: {result['error']}")
        
        print("=" * 80)
        print(f"总计: {passed_tests}/{total_tests} 测试通过")
        
        if passed_tests == total_tests:
            print("🎉 所有集成测试通过！")
        else:
            print("⚠️  部分集成测试失败，请检查上述错误信息")
        
        return passed_tests == total_tests


def main():
    """主函数"""
    runner = IntegrationTestRunner()
    
    if len(sys.argv) > 1:
        # 运行特定测试
        test_name = sys.argv[1].lower()
        success = runner.run_specific_test(test_name)
    else:
        # 运行所有测试
        success = runner.run_all_integration_tests()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()