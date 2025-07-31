import sys
import os
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def test_zhipu_api_connection():
    api_key = input("请输入智谱AI的API密钥: ").strip()
    if not api_key:
        print("API密钥不能为空")
        return False
    
    try:
        from dataBuilder.chunking.api_client import APIClient
        
        client = APIClient(
            api_type="zhipu",
            api_key=api_key,
            model_name="glm-4",
            max_tokens=1000,
            temperature=0.1
        )
        
        messages = [{"role": "user", "content": "你好，请简单介绍一下你自己。"}]
        response = client.chat_completion(messages)
        
        if response:
            print("✅ 基本聊天测试成功")
            print(f"模型回复: {response[:100]}...")
        else:
            print("❌ 基本聊天测试失败")
            return False
        
        text1 = "人工智能是计算机科学的一个分支。"
        text2 = "它致力于创造能够执行通常需要人类智能的任务的机器。"
        
        prompt = f"""请判断以下两个文本片段是否应该分开处理：

文本1：{text1}
文本2：{text2}

如果这两个文本片段在语义上应该分开处理，请回答"分开"；
如果应该合并在一起，请回答"合并"。

请只回答"分开"或"合并"："""
        
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(messages)
        
        if response:
            print("✅ 文本分块判断测试成功")
            print(f"分块判断结果: {response}")
        else:
            print("❌ 文本分块判断测试失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 智谱AI连接测试失败: {e}")
        return False

def test_zhipu_chunking_integration():
    api_key = input("请输入智谱AI的API密钥: ").strip()
    
    try:
        from dataBuilder.chunking import ChunkingManager
        
        test_config = {
            "chunking": {
                "strategy": "traditional",
                "chunk_length": 512,
                "language": "zh",
                "model": {
                    "enable_advanced_chunking": True,
                    "use_api": True,
                    "api_type": "zhipu",
                    "api_key": api_key,
                    "model_name": "glm-4",
                    "max_tokens": 1000,
                    "temperature": 0.1
                }
            }
        }
        
        chunking_manager = ChunkingManager(config=test_config)
        
        test_text = """
        人工智能技术正在快速发展，深刻改变着我们的生活和工作方式。
        机器学习作为人工智能的核心技术，通过算法让计算机能够从数据中学习和改进。
        深度学习是机器学习的一个重要分支，它模拟人脑神经网络的工作原理。
        自然语言处理技术让计算机能够理解和生成人类语言，这是实现人机交互的关键。
        计算机视觉技术使机器能够识别和理解图像内容，广泛应用于各个领域。
        """
        
        try:
            chunks = chunking_manager.chunk_text(test_text, "meta_ppl", threshold=0.3, language="zh")
            if chunks:
                print(f"✅ PPL分块成功，生成 {len(chunks)} 个文本块")
                for i, chunk in enumerate(chunks):
                    print(f"  块{i+1}: {chunk.strip()[:80]}...")
        except Exception as e:
            print(f"❌ PPL分块测试失败: {e}")
        
        try:
            chunks = chunking_manager.chunk_text(test_text, "margin_sampling", language="zh", chunk_length=200)
            if chunks:
                print(f"✅ 边际采样分块成功，生成 {len(chunks)} 个文本块")
                for i, chunk in enumerate(chunks):
                    print(f"  块{i+1}: {chunk.strip()[:80]}...")
        except Exception as e:
            print(f"❌ 边际采样分块测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 智谱AI分块集成测试失败: {e}")
        return False

def test_zhipu_config_setup():
    try:
        from config_manager import load_config, update_config
        
        api_key = input("请输入要保存到配置文件的智谱AI API密钥: ").strip()
        if not api_key:
            print("未提供API密钥，跳过配置保存")
            return True
        
        config_updates = {
            "chunking": {
                "model": {
                    "enable_advanced_chunking": True,
                    "use_api": True,
                    "api_type": "zhipu",
                    "api_key": api_key,
                    "model_name": "glm-4",
                    "max_tokens": 1000,
                    "temperature": 0.1
                }
            }
        }
        
        success = update_config(config_updates)
        
        if success:
            print("✅ 智谱AI配置保存成功")
            config = load_config()
            model_config = config.get("chunking", {}).get("model", {})
            print(f"API类型: {model_config.get('api_type')}")
            print(f"模型名称: {model_config.get('model_name')}")
            return True
        else:
            print("❌ 配置保存失败")
            return False
            
    except Exception as e:
        print(f"❌ 智谱AI配置设置失败: {e}")
        return False

def show_zhipu_info():
    info = """
智谱AI (GLM) 模型信息:
官网: https://open.bigmodel.cn/
文档: https://open.bigmodel.cn/dev/api

支持的模型:
- glm-4: 最新的GLM-4模型
- glm-4v: 支持视觉的多模态模型
- glm-3-turbo: 轻量级高速模型

获取API密钥:
1. 访问 https://open.bigmodel.cn/
2. 注册并登录账号
3. 在控制台创建API密钥
"""
    print(info)

def main():
    print("智谱AI模型连接测试工具")
    show_zhipu_info()
    
    while True:
        print("\n请选择测试项目:")
        print("1. 测试智谱AI API连接")
        print("2. 测试智谱AI分块集成")
        print("3. 设置智谱AI配置")
        print("4. 显示智谱AI信息")
        print("5. 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == "1":
            test_zhipu_api_connection()
        elif choice == "2":
            test_zhipu_chunking_integration()
        elif choice == "3":
            test_zhipu_config_setup()
        elif choice == "4":
            show_zhipu_info()
        elif choice == "5":
            print("测试结束")
            break
        else:
            print("无效选择，请重新输入")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试已取消")
        sys.exit(0)