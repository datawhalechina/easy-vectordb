import os
import yaml
from rag.utils import initialize_rag, query_rag

def load_config():
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    
    docs_dir = config["data"]["path"]
    if not os.path.isabs(docs_dir):
        docs_dir = os.path.join(os.getcwd(), docs_dir)
        
    if not os.path.isdir(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        print(f"创建文档目录: {docs_dir}，请放入 .txt 文件。")
        return

    print("正在初始化 RAG 系统（加载模型并构建索引）...")
    rag_context = initialize_rag(docs_dir, config)
    print("初始化完成！现在可以开始对话了（输入 'exit' 退出）。")

    while True:
        question = input("\n请输入问题: ").strip()
        if not question:
            continue
        if question.lower() in ["exit", "quit", "退出"]:
            break
        
        print("正在检索并生成回答...\n")
        result = query_rag(
            question=question,
            rag_context=rag_context,
            top_k=config["rag"].get("top_k", 3),
            stream=True
        )

        if not rag_context["llm"]:
            print("==== 检索上下文 ====")
            print(result["contexts"])
            print("\n提示：未配置有效的 LLM API Key，仅显示检索结果。")
        else:
            if not result["answer"]:
                print("==== 检索上下文 ====")
                print(result["contexts"])

if __name__ == "__main__":
    main()
