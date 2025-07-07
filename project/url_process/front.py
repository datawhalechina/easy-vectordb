import os
import logging
from urllib.parse import urlparse
from zhipuai import ZhipuAI
from pymilvus import connections, Collection
import gradio as gr
import re

# 设置日志级别
logging.basicConfig(level=logging.INFO)

class RAGApplication:
    def __init__(self, api_key, milvus_host="localhost", milvus_port="19530", collection_name='video_push'):
        self.api_key = api_key
        self.zhipuai_client = ZhipuAI(api_key=api_key)
        self.collection_name = collection_name
        
        # 初始化 Milvus 连接
        connections.connect("default", host=milvus_host, port=milvus_port)
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def is_valid_url(self, string):
        # 尝试解析字符串
        try:
            result = urlparse(string)
            # 如果解析成功，返回scheme和netloc是否都存在
            return all([result.scheme, result.netloc])
        except:
            # 如果解析失败，返回False
            return False

    def extract_urls(self, text):
        url_pattern = r'(https?://[^\s\)\]\}>]+)'
        return [m.group(0) for m in re.finditer(url_pattern, text)]

    def generate_text(self, prompt):
        try:
            response = self.zhipuai_client.chat.completions.create(
                model="glm-4-plus",  # 使用需要调用的模型编码
                messages=[
                    {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
                    {"role": "user", "content": f"{prompt}"}
                ],
            )
            if response and response.choices and len(response.choices) > 0:
                message_content = response.choices[0].message.content.strip() if hasattr(response.choices[0].message, 'content') else ''
                return message_content or "No response from the model."
            else:
                logging.warning("Unexpected response structure from ZhipuAI API.")
                return "Error: Unable to generate text."
        except Exception as e:
            logging.error(f"An error occurred during the API request: {e}")
            return "Error: An unexpected error occurred."

    def query_rag_model(self, user_query, min_score):  
        try:
            # 调用 ZhipuAI 嵌入创建 API 获取查询的嵌入向量
            response = self.zhipuai_client.embeddings.create(model="embedding-2", input=user_query)
            query_embedding = response.data[0].embedding
            
            # 在 Milvus 中搜索最相似的视频
            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=5,  # 返回前5个最相似的结果
                output_fields=["id", "url", "description"] 
            )

            print(results)
            # 整理搜索结果，并筛选出分数大于等于min_score的数据
            relevant_data = []
            urls = []
            for hits in results:
                print(f"hits:{hits}")
                for hit in hits:
                    if hit.distance > min_score:
                        video_info = {field: getattr(hit.entity, field) for field in ["id", "url", "description"]}
                        relevant_data.append(video_info["description"])
                        print(f"hitId:{hit.id},relevant_data:{relevant_data}")
                        if self.is_valid_url(video_info["url"]):
                            urls.append(video_info["url"])

            # 如果没有找到相关性足够高的数据，则不构建context
            if not relevant_data:
                print("没有找到相关性足够高的数据，不构建context！")
                final_answer = self.generate_text(user_query)
            else:
                # 使用 ZhipuAI 的语言模型生成综合回答
                context = "\n".join(relevant_data)
                prompt = f"Based on the following context:\n{context}\nAnswer the question: {user_query}"
                final_answer = self.generate_text(prompt)

            # 构建最终回复
            if urls:
                urls_str = "\n".join(urls)
                final_answer += f"\n\nFor more information, please refer to the following URLs:\n{urls_str}"

            return final_answer
        except Exception as e:
            logging.error(f"An error occurred during query: {e}")
            return f"Error: {str(e)}"

if __name__ == "__main__":
    # 从环境变量获取API密钥
    api_key = ""
    if not api_key:
        raise ValueError("Environment variable ZHIPUAI_API_KEY is not set.")

    app = RAGApplication(api_key)

    # 创建 Gradio 界面
    with gr.Blocks() as demo:
        gr.Markdown("# RAG Application")
        
        with gr.Row():
            input_text = gr.Textbox(label="Enter your query:", placeholder="Type your query here...")
        
        with gr.Row():
            output_text = gr.Textbox(label="Response:")
        
        submit_button = gr.Button("Submit Query")
        
        # 设置事件处理程序
        submit_button.click(fn=lambda x: app.query_rag_model(x, min_score=0.6), inputs=input_text, outputs=output_text)

    # 启动 Gradio 应用
    demo.launch()
