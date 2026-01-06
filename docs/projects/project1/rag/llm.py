"""LLM 适配：仅保留 DeepSeek 支持，使用 OpenAI SDK。"""
from openai import OpenAI

def load_model(config: dict):
    """根据配置加载 DeepSeek 大模型客户端"""
    model_name = config["llm"]["model"]
    api_key = config["llm"]["api_key"]
    base_url = "https://api.deepseek.com"

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_name

def stream_chat(client_info, prompt: str) -> str:
    """流式生成，将增量输出打印到控制台，并返回完整文本"""
    client, model_name = client_info
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    response_text = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            response_text += content
            print(content, end="", flush=True)
    print("\n", end="")
    return response_text
