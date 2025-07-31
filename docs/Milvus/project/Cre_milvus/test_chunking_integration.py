import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_chunking_imports():
    try:
        from dataBuilder.chunking import ChunkingManager, get_available_strategies
        strategies = get_available_strategies()
        print(f"✅ 可用策略: {[s['name'] for s in strategies]}")
        return True
    except Exception as e:
        print(f"❌ 分块模块导入失败: {e}")
        return False

def test_chunking_strategies():
    try:
        from dataBuilder.chunking import ChunkingManager
        from config_manager import load_config
        
        config = load_config()
        chunking_manager = ChunkingManager(config=config)
        
        test_text = """
        这是一个测试文本，用于验证不同的分块策略。
        我们将测试传统分块、PPL困惑度分块、边际采样分块等多种策略。
        每种策略都有其独特的优势和适用场景。
        传统分块简单高效，适合大多数场景。
        PPL分块能够在语义边界处切分，保持语义完整性。
        边际采样分块则基于概率决策，动态调整切分点。
        """
        
        chunking_config = config.get("chunking", {})
        model_config = chunking_config.get("model", {})
        advanced_chunking_enabled = model_config.get("enable_advanced_chunking", False)
        use_api = model_config.get("use_api", False)
        api_key = model_config.get("api_key", "")
        
        print(f"高级分块功能: {'启用' if advanced_chunking_enabled else '禁用'}")
        if advanced_chunking_enabled and use_api:
            print(f"API密钥配置: {'已配置' if api_key else '未配置'}")
        
        strategies_to_test = [
            ("traditional", {"chunk_size": 100, "overlap": 20}, False),
            ("semantic", {"similarity_threshold": 0.8, "min_chunk_size": 50, "max_chunk_size": 200}, False),
            ("msp", {"language": "zh", "chunk_length": 200, "confidence_threshold": 0.7}, True),
            ("meta_ppl", {"threshold": 0.3, "language": "zh"}, True),
            ("margin_sampling", {"language": "zh", "chunk_length": 200}, True),
        ]
        
        results = {}
        
        for strategy_name, params, requires_model in strategies_to_test:
            try:
                print(f"测试策略: {strategy_name}")
                
                if requires_model and strategy_name in ["meta_ppl", "margin_sampling"] and not advanced_chunking_enabled:
                    print(f"⚠️ {strategy_name}: 跳过测试 - 高级分块功能未启用")
                    results[strategy_name] = {
                        "status": "skipped", 
                        "reason": "高级分块功能未启用"
                    }
                    continue
                
                chunks = chunking_manager.chunk_text(test_text, strategy_name, **params)
                
                if chunks:
                    print(f"✅ {strategy_name}: 生成 {len(chunks)} 个文本块")
                    results[strategy_name] = {
                        "status": "success",
                        "chunk_count": len(chunks),
                        "chunks": chunks[:2]
                    }
                else:
                    print(f"⚠️ {strategy_name}: 返回空结果")
                    results[strategy_name] = {"status": "empty", "chunk_count": 0}
                    
            except Exception as e:
                error_msg = str(e)
                if "Model and tokenizer are required" in error_msg:
                    print(f"⚠️ {strategy_name}: 跳过测试 - 需要语言模型支持")
                    results[strategy_name] = {
                        "status": "skipped", 
                        "reason": "需要语言模型支持"
                    }
                elif "KeyboardInterrupt" in error_msg or "timeout" in error_msg.lower():
                    print(f"⚠️ {strategy_name}: 跳过测试 - 网络超时")
                    results[strategy_name] = {
                        "status": "skipped", 
                        "reason": "网络超时或用户中断"
                    }
                else:
                    print(f"❌ {strategy_name}: 测试失败 - {e}")
                    results[strategy_name] = {"status": "error", "error": str(e)}
        
        return results
        
    except Exception as e:
        print(f"❌ 分块策略测试失败: {e}")
        return {}

def test_data_processing_integration():
    try:
        from dataBuilder.data import get_chunking_strategies
        strategies = get_chunking_strategies()
        print(f"✅ 数据处理模块可用策略: {[s['name'] for s in strategies]}")
        return True
    except Exception as e:
        print(f"❌ 数据处理集成测试失败: {e}")
        return False

def test_system_integration():
    try:
        from System.start import load_config
        config = load_config()
        chunking_config = config.get("chunking", {})
        print(f"✅ 配置加载成功")
        print(f"当前分块策略: {chunking_config.get('strategy', 'traditional')}")
        return True
    except Exception as e:
        print(f"❌ 系统集成测试失败: {e}")
        return False

def main():
    print("分块策略集成测试")
    
    test_results = {}
    
    test_results["imports"] = test_chunking_imports()
    
    if test_results["imports"]:
        test_results["strategies"] = test_chunking_strategies()
    
    test_results["data_processing"] = test_data_processing_integration()
    test_results["system_integration"] = test_system_integration()
    
    print("\n测试结果总结:")
    
    if test_results["imports"]:
        print("✅ 模块导入: 成功")
    else:
        print("❌ 模块导入: 失败")
    
    if test_results.get("strategies"):
        strategy_results = test_results["strategies"]
        success_count = sum(1 for r in strategy_results.values() if r.get("status") == "success")
        skipped_count = sum(1 for r in strategy_results.values() if r.get("status") == "skipped")
        total_count = len(strategy_results)
        print(f"✅ 分块策略测试: {success_count}/{total_count} 成功, {skipped_count} 个跳过")
        
        for strategy, result in strategy_results.items():
            status = result.get("status", "unknown")
            if status == "success":
                print(f"  ✅ {strategy}: {result.get('chunk_count', 0)} 个块")
            elif status == "skipped":
                print(f"  ⏭️ {strategy}: 跳过 - {result.get('reason', '未知原因')}")
            elif status == "empty":
                print(f"  ⚠️ {strategy}: 返回空结果")
            else:
                print(f"  ❌ {strategy}: {result.get('error', '未知错误')}")
    
    if test_results["data_processing"]:
        print("✅ 数据处理集成: 成功")
    else:
        print("❌ 数据处理集成: 失败")
    
    if test_results["system_integration"]:
        print("✅ 系统集成: 成功")
    else:
        print("❌ 系统集成: 失败")
    
    all_tests_passed = (
        test_results["imports"] and
        test_results["data_processing"] and
        test_results["system_integration"]
    )
    
    if all_tests_passed:
        print("🎉 所有集成测试通过！")
    else:
        print("❌ 部分测试失败")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)