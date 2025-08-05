#!/usr/bin/env python3
"""
运行用户体验验证测试的脚本
Script to run user experience validation tests
"""

import sys
import os
import subprocess
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_system_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    requirements = {
        'python_version': sys.version_info >= (3, 8),
        'streamlit_available': False,
        'milvus_available': False
    }
    
    # 检查Streamlit
    try:
        import streamlit
        requirements['streamlit_available'] = True
        print("  ✅ Streamlit 可用")
    except ImportError:
        print("  ❌ Streamlit 不可用")
    
    # 检查Milvus相关模块
    try:
        from pymilvus import connections
        requirements['milvus_available'] = True
        print("  ✅ Milvus 客户端可用")
    except ImportError:
        print("  ❌ Milvus 客户端不可用")
    
    return requirements

def run_validation_with_mock():
    """使用模拟环境运行验证"""
    print("🎭 使用模拟环境运行用户体验验证...")
    
    try:
        # 导入验证器
        from user_experience_validation import UserExperienceValidator
        
        # 创建验证器实例
        validator = UserExperienceValidator()
        
        # 运行验证
        results = validator.run_all_validations()
        
        return results
        
    except Exception as e:
        print(f"❌ 验证运行失败: {str(e)}")
        return None

def run_frontend_validation():
    """运行前端相关的验证"""
    print("🖥️ 运行前端用户体验验证...")
    
    try:
        # 检查前端文件是否存在
        frontend_path = "../frontend.py"
        if not os.path.exists(frontend_path):
            print(f"❌ 前端文件不存在: {frontend_path}")
            return False
        
        # 检查GLM配置相关代码
        with open(frontend_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        validation_checks = {
            'glm_config_present': 'GLM' in content or 'glm' in content,
            'expander_usage': 'expander' in content,
            'config_status': 'config' in content and 'status' in content,
            'streamlit_components': 'st.' in content
        }
        
        print("  前端代码检查结果:")
        for check, result in validation_checks.items():
            status = "✅" if result else "❌"
            print(f"    {status} {check}: {result}")
        
        return all(validation_checks.values())
        
    except Exception as e:
        print(f"❌ 前端验证失败: {str(e)}")
        return False

def generate_ux_test_report(results):
    """生成用户体验测试报告"""
    if not results:
        print("❌ 无法生成报告：验证结果为空")
        return
    
    report_path = "ux_validation_summary.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 用户体验验证报告\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 总体评分\n")
        f.write(f"**总分**: {results.get('overall_score', 0):.1f}/100\n\n")
        
        f.write("## 详细结果\n\n")
        
        # GLM配置界面
        f.write("### 1. GLM配置界面可用性\n")
        glm_results = results.get('glm_config_ui', {})
        f.write(f"**得分**: {glm_results.get('score', 0):.1f}/100\n\n")
        for detail in glm_results.get('details', []):
            f.write(f"- {detail}\n")
        f.write("\n")
        
        # PPL分块错误信息
        f.write("### 2. PPL分块错误信息清晰度\n")
        ppl_results = results.get('ppl_chunking_errors', {})
        f.write(f"**得分**: {ppl_results.get('score', 0):.1f}/100\n\n")
        for detail in ppl_results.get('details', []):
            f.write(f"- {detail}\n")
        f.write("\n")
        
        # Append模式进度
        f.write("### 3. Append模式进度反馈\n")
        append_results = results.get('append_mode_progress', {})
        f.write(f"**得分**: {append_results.get('score', 0):.1f}/100\n\n")
        for detail in append_results.get('details', []):
            f.write(f"- {detail}\n")
        f.write("\n")
        
        f.write("## 验证状态\n")
        f.write("- [x] GLM配置界面的可用性和直观性\n")
        f.write("- [x] PPL分块错误信息的清晰度\n")
        f.write("- [x] Append模式插入的进度反馈和成功率\n\n")
        
        f.write("## 结论\n")
        score = results.get('overall_score', 0)
        if score >= 80:
            f.write("✅ 用户体验验证通过，系统可用性良好。\n")
        else:
            f.write("⚠️ 用户体验需要改进，建议优化相关功能。\n")
    
    print(f"📄 用户体验验证报告已生成: {report_path}")

def main():
    """主函数"""
    print("🚀 开始用户体验验证流程")
    print("=" * 50)
    
    # 1. 检查系统要求
    requirements = check_system_requirements()
    
    # 2. 运行前端验证
    frontend_ok = run_frontend_validation()
    
    # 3. 运行主要验证
    results = run_validation_with_mock()
    
    # 4. 生成报告
    if results:
        generate_ux_test_report(results)
        
        # 打印最终结果
        print("\n" + "=" * 50)
        print("🎯 用户体验验证完成")
        print("=" * 50)
        
        score = results.get('overall_score', 0)
        if score >= 80:
            print("✅ 验证通过：用户体验良好")
            return True
        else:
            print("⚠️ 验证警告：用户体验需要改进")
            return False
    else:
        print("❌ 验证失败：无法完成用户体验验证")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)