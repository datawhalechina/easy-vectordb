import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from Search.embedding import embedder
from System.monitor import log_event
import traceback

def process_txt(txt_path, url_split):
    """
    处理 TXT 文件并返回一个包含 id、content、embedding 和 URL 的列表。
    
    参数:
        txt_path (str): TXT 文件路径。
        url_split (bool): 是否将 URL 分割成单独的字段。
    返回:
        list: 包含 id、content、embedding 和 URL 的列表，格式为 [{'id': id, 'content': content, 'embedding': embedding, 'urls': [url1, url2, ...]}, ...]。
    """
    try:
        # 定义正则表达式清洗函数
        def clean_content(content):
            pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
            content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), content)
            content = content.replace('•', '').replace(' ', '').replace('\n\n', '\n')
            return content

        # 提取 URL 的函数
        def extract_urls_with_positions(text):
            url_pattern = r'(https?://[^\s\)\]\}>]+)'
            return [(m.group(0), m.span()) for m in re.finditer(url_pattern, text)]

        # 验证文件存在
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"文件 {txt_path} 不存在")
        
        # 读取 TXT 文件内容
        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        log_event(f"已加载TXT文件: {txt_path}, 长度: {len(content)} 字符")

        # 清洗内容
        cleaned_content = clean_content(content)

        # 定义文本分割器
        CHUNK_SIZE = 256
        OVERLAP_SIZE = 64
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE
        )

        # 分割文本
        split_docs = text_splitter.split_text(cleaned_content)
        log_event(f"已分割TXT文件: {txt_path}, 得到 {len(split_docs)} 个文本块")

        # 收集所有文本块和相关信息
        all_texts = []
        text_info = []  # 保存每个文本块的相关信息 (urls, id)
        current_id = 1

        # 处理每个文本块
        for doc in split_docs:
            urls_with_positions = extract_urls_with_positions(doc)
            non_url_text = doc

            # 移除 URL 并清理多余空白
            for url, (start, end) in reversed(urls_with_positions):
                non_url_text = non_url_text[:start] + non_url_text[end:]
            non_url_text = re.sub(r'\s+', ' ', non_url_text).strip()
            
            # 保存文本和相关信息
            all_texts.append(non_url_text)
            text_info.append({
                "urls": [url for url, _ in urls_with_positions],
                "id": current_id
            })
            current_id += 1
        
        # 批量生成嵌入向量
        status = embedder.check_status()
        print(f"嵌入生成器状态: {status}")
        
        if not status["model_loaded"] or not status["tokenizer_loaded"]:
            print("嵌入生成器初始化失败，无法处理文件")
            return []
        embeddings = embedder.batch_get_embedding(all_texts) if all_texts else []
        
        # 构建结果列表
        results = []
        for i, text in enumerate(all_texts):
            if url_split:
                results.append({
                    'id': text_info[i]["id"],
                    'content': text,
                    'embedding': embeddings[i],
                    'urls': text_info[i]["urls"]
                })
            else:
                results.append({
                    'id': text_info[i]["id"],
                    'content': text,
                    'embedding': embeddings[i],
                })
        
        log_event(f"TXT处理完成: {txt_path}, 生成 {len(results)} 条记录")
        return results
    
    except Exception as e:
        error_msg = f"处理TXT失败: {txt_path}\n错误: {str(e)}\n{traceback.format_exc()}"
        log_event(error_msg)
        print(error_msg)
        return []