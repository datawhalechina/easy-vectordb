# 添加全局开关以禁用图像处理功能
DISABLE_IMAGE_PROCESSING = True

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .tools.csvmake import process_csv
from .tools.mdmake import process_md
from .tools.pdfmake import process_pdf
from .tools.txtmake import process_txt

logger = logging.getLogger(__name__)

if not DISABLE_IMAGE_PROCESSING:
    from .tools.imgmake import process_img, process_image_directory, get_image_statistics
    from multimodal.clip_encoder import CLIPEncoder

from .chunking.meta_chunking import MetaChunking
from System.monitor import log_event
import traceback


ADVANCED_CHUNKING_AVAILABLE = True

def data_process(data_location, url_split, chunking_strategy="traditional", chunking_params=None, enable_multimodal=True):
    """
    data_location: 用户上传的文件夹路径
    url_split: 是否对文本做url切分
    chunking_strategy: 分块策略 ("traditional", "meta_ppl", "margin_sampling")
    chunking_params: 分块策略参数
    enable_multimodal: 是否启用多模态图像处理
    自动识别文件夹下的csv、md、pdf、txt、图像文件并多线程处理
    """
    dataList = []
    folder = Path(data_location)
    
    files = [f for f in folder.rglob("*") if f.is_file()]
    print(f"找到 {len(files)} 个文件")
    
    chunking_manager = None
    if chunking_strategy != "traditional":
        try:
            chunking_manager = MetaChunking()
            print(f"使用高级分块策略: {chunking_strategy}")
        except Exception as e:
            print(f"高级分块策略初始化失败，使用传统方法: {e}")
            chunking_manager = None
    
    if chunking_params is None:
        chunking_params = {}

    valid_extensions = {".csv", ".md", ".pdf", ".txt"}
    if not DISABLE_IMAGE_PROCESSING:
        valid_extensions.update({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"})
    
    tasks = []
    image_files = []
    
    for file in files:
        ext = file.suffix.lower()
        if ext in valid_extensions:
            if not DISABLE_IMAGE_PROCESSING and ext in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}:
                image_files.append(str(file))
                tasks.append(("img", str(file)))
            else:
                tasks.append((ext[1:], str(file)))  
    
    print(f"准备处理 {len(tasks)} 个有效文件，其中图像文件 {len(image_files)} 个")
    
    if not DISABLE_IMAGE_PROCESSING and image_files:
        try:
            img_stats = get_image_statistics(data_location)
            print(f"图像统计: {img_stats['total_images']} 张图像，总大小 {img_stats['total_size_mb']} MB")
            log_event(f"图像统计信息: {img_stats}")
        except Exception as e:
            print(f"获取图像统计信息失败: {e}")
    
    def process_one(task):
        file_type, file_path = task
        print(f"开始处理文件: {file_path}, 类型: {file_type}")
        try:
            if file_type == "csv":
                return process_csv(csv_path=file_path)
            elif file_type == "md":
                # 传递分块策略参数
                return process_md_with_strategy(file_path, url_split, chunking_strategy, chunking_params, chunking_manager)
            elif file_type == "pdf":
                return process_pdf_with_strategy(file_path, url_split, chunking_strategy, chunking_params, chunking_manager)
            elif file_type == "txt":
                return process_txt_with_strategy(file_path, url_split, chunking_strategy, chunking_params, chunking_manager)
            elif file_type == "img" and not DISABLE_IMAGE_PROCESSING:
                # 初始化CLIP编码器
                clip_encoder = None
                if enable_multimodal:
                    try:
                        clip_encoder = CLIPEncoder()
                        log_event("CLIP编码器初始化成功")
                    except Exception as e:
                        log_event(f"CLIP编码器初始化失败: {e}")
                        enable_multimodal = False
                return process_img(img_path=file_path, use_clip=enable_multimodal)
        except Exception as e:
            print(f"处理文件出错: {file_path}\n错误详情: {traceback.format_exc()}")
            return []  
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(process_one, tasks):
            if isinstance(result, list):
                dataList.extend(result)
            elif result is not None:
                dataList.append(result)
    
    print(f"处理完成，共处理了 {len(dataList)} 条数据。")
    log_event(f"数据处理完成，共处理 {len(tasks)} 个文件，生成 {len(dataList)} 条数据")
    return dataList


# def get_chunking_config():
#     """获取分块配置"""
#     try:
#         from .chunking.glm_config import get_glm_config_service
#         service = get_glm_config_service()
#         glm_config = service.get_active_config()
#         return {
#             "glm_configured": glm_config is not None,
#             "glm_config": glm_config
#         }
#     except Exception as e:
#         logger.warning(f"获取GLM配置失败: {e}")
#         return {"glm_configured": False}


def process_txt_with_strategy(txt_path, url_split, chunking_strategy, chunking_params, chunking_manager):
    """
    使用指定分块策略处理TXT文件
    """
    if chunking_strategy == "traditional" or not chunking_manager:
        return process_txt(txt_path, url_split)
    
    try:
        from Search.embedding import embedder
        import re
        import os
        
        # 验证文件存在
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"文件 {txt_path} 不存在")
        
        # 读取文件内容
        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        log_event(f"已加载TXT文件: {txt_path}, 长度: {len(content)} 字符")
        
        # 清洗内容
        def clean_content(content):
            pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
            content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), content)
            content = content.replace('•', '').replace(' ', '').replace('\n\n', '\n')
            return content
        
        cleaned_content = clean_content(content)
        
        # 使用新的分块策略
        # config = get_chunking_config()  # 从实际配置中获取
        config = {"glm_configured": False}  
        split_docs = chunking_manager.smart_chunking(cleaned_content, chunking_strategy, config, **chunking_params)
        log_event(f"使用 {chunking_strategy} 策略分割TXT文件: {txt_path}, 得到 {len(split_docs)} 个文本块")
        
        # 提取URL的函数
        def extract_urls_with_positions(text):
            url_pattern = r'(https?://[^\s\)\]\}>]+)'
            return [(m.group(0), m.span()) for m in re.finditer(url_pattern, text)]
        
        # 处理每个文本块
        all_texts = []
        text_info = []
        current_id = 1
        
        for doc in split_docs:
            urls_with_positions = extract_urls_with_positions(doc)
            non_url_text = doc
            
            # 移除URL并清理多余空白
            for url, (start, end) in reversed(urls_with_positions):
                non_url_text = non_url_text[:start] + non_url_text[end:]
            non_url_text = re.sub(r'\s+', ' ', non_url_text).strip()
            
            all_texts.append(non_url_text)
            text_info.append({
                "urls": [url for url, _ in urls_with_positions],
                "id": current_id
            })
            current_id += 1
        
        # 批量生成嵌入向量
        status = embedder.check_status()
        if not status["model_loaded"] or not status["tokenizer_loaded"]:
            print("嵌入生成器初始化失败，无法处理文件")
            return []
        
        embeddings = embedder.batch_get_embedding(all_texts) if all_texts else []
        
        # 构建结果列表
        results = []
        for i, text in enumerate(all_texts):
            result = {
                'id': text_info[i]["id"],
                'content': text,
                'embedding': embeddings[i],
                'chunking_strategy': chunking_strategy,
                'chunking_params': chunking_params
            }
            
            if url_split:
                result['urls'] = text_info[i]["urls"]
            
            results.append(result)
        
        log_event(f"TXT处理完成: {txt_path}, 生成 {len(results)} 条记录")
        return results
        
    except Exception as e:
        error_msg = f"使用策略 {chunking_strategy} 处理TXT失败: {txt_path}\n错误: {str(e)}"
        log_event(error_msg)
        print(error_msg)
        # 降级到传统方法
        return process_txt(txt_path, url_split)


def process_pdf_with_strategy(pdf_path, url_split, chunking_strategy, chunking_params, chunking_manager):
    """
    使用指定分块策略处理PDF文件
    """
    if chunking_strategy == "traditional" or not chunking_manager:
        return process_pdf(pdf_path, url_split)
    
    try:
        from Search.embedding import embedder
        from langchain_community.document_loaders import PyMuPDFLoader
        import re
        
        # 加载PDF文件
        loader = PyMuPDFLoader(pdf_path)
        pdf_pages = loader.load()
        log_event(f"已加载PDF: {pdf_path}, 共 {len(pdf_pages)} 页")
        
        # 清洗内容函数
        def clean_content(content):
            pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
            content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), content)
            content = content.replace('•', '').replace(' ', '').replace('\n\n', '\n')
            return content
        
        # 合并所有页面内容
        full_content = ""
        for page in pdf_pages:
            full_content += clean_content(page.page_content) + "\n"
        
        # 使用新的分块策略
        # config = get_chunking_config()  # 从实际配置中获取
        config = {"glm_configured": False}  # 添加这行
        split_docs = chunking_manager.smart_chunking(full_content, chunking_strategy, config, **chunking_params)
        log_event(f"使用 {chunking_strategy} 策略分割PDF文件: {pdf_path}, 得到 {len(split_docs)} 个文本块")
        
        # 提取URL的函数
        def extract_urls_with_positions(text):
            url_pattern = r'(https?://[^\s\)\]\}>]+)'
            return [(m.group(0), m.span()) for m in re.finditer(url_pattern, text)]
        
        # 处理每个文本块
        all_texts = []
        text_info = []
        current_id = 1
        
        for doc in split_docs:
            urls_with_positions = extract_urls_with_positions(doc)
            non_url_text = doc
            
            # 移除URL并清理多余空白
            for url, (start, end) in reversed(urls_with_positions):
                non_url_text = non_url_text[:start] + non_url_text[end:]
            non_url_text = re.sub(r'\s+', ' ', non_url_text).strip()
            
            all_texts.append(non_url_text)
            text_info.append({
                "urls": [url for url, _ in urls_with_positions],
                "id": current_id
            })
            current_id += 1
        
        # 批量生成嵌入向量
        status = embedder.check_status()
        if not status["model_loaded"] or not status["tokenizer_loaded"]:
            print("嵌入生成器初始化失败，无法处理文件")
            return []
        
        embeddings = embedder.batch_get_embedding(all_texts) if all_texts else []
        
        # 构建结果列表
        results = []
        for i, text in enumerate(all_texts):
            result = {
                'id': text_info[i]["id"],
                'content': text,
                'embedding': embeddings[i],
                'chunking_strategy': chunking_strategy,
                'chunking_params': chunking_params
            }
            
            if url_split:
                result['urls'] = text_info[i]["urls"]
            
            results.append(result)
        
        log_event(f"PDF处理完成: {pdf_path}, 生成 {len(results)} 条记录")
        return results
        
    except Exception as e:
        error_msg = f"使用策略 {chunking_strategy} 处理PDF失败: {pdf_path}\n错误: {str(e)}"
        log_event(error_msg)
        print(error_msg)
        # 降级到传统方法
        return process_pdf(pdf_path, url_split)


def process_md_with_strategy(md_path, url_split, chunking_strategy, chunking_params, chunking_manager):
    """
    使用指定分块策略处理Markdown文件
    """
    if chunking_strategy == "traditional" or not chunking_manager:
        return process_md(md_path, url_split)
    
    try:
        from Search.embedding import embedder
        import re
        import os
        
        # 验证文件存在
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"文件 {md_path} 不存在")
        
        # 读取文件内容
        with open(md_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        log_event(f"已加载Markdown文件: {md_path}, 长度: {len(content)} 字符")
        
        # 清洗内容
        def clean_content(content):
            pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
            content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), content)
            content = content.replace('•', '').replace(' ', '').replace('\n\n', '\n')
            return content
        
        cleaned_content = clean_content(content)
        
        # 使用新的分块策略
        # config = get_chunking_config()  # 从实际配置中获取
        config = {"glm_configured": False}  # 添加这行
        split_docs = chunking_manager.smart_chunking(cleaned_content, chunking_strategy, config, **chunking_params)
        log_event(f"使用 {chunking_strategy} 策略分割Markdown文件: {md_path}, 得到 {len(split_docs)} 个文本块")
        
        # 提取URL的函数
        def extract_urls_with_positions(text):
            url_pattern = r'(https?://[^\s\)\]\}>]+)'
            return [(m.group(0), m.span()) for m in re.finditer(url_pattern, text)]
        
        # 处理每个文本块
        all_texts = []
        text_info = []
        current_id = 1
        
        for doc in split_docs:
            urls_with_positions = extract_urls_with_positions(doc)
            non_url_text = doc
            
            # 移除URL并清理多余空白
            for url, (start, end) in reversed(urls_with_positions):
                non_url_text = non_url_text[:start] + non_url_text[end:]
            non_url_text = re.sub(r'\s+', ' ', non_url_text).strip()
            
            all_texts.append(non_url_text)
            text_info.append({
                "urls": [url for url, _ in urls_with_positions],
                "id": current_id
            })
            current_id += 1
        
        # 批量生成嵌入向量
        status = embedder.check_status()
        if not status["model_loaded"] or not status["tokenizer_loaded"]:
            print("嵌入生成器初始化失败，无法处理文件")
            return []
        
        embeddings = embedder.batch_get_embedding(all_texts) if all_texts else []
        
        # 构建结果列表
        results = []
        for i, text in enumerate(all_texts):
            result = {
                'id': text_info[i]["id"],
                'content': text,
                'embedding': embeddings[i],
                'chunking_strategy': chunking_strategy,
                'chunking_params': chunking_params
            }
            
            if url_split:
                result['urls'] = text_info[i]["urls"]
            
            results.append(result)
        
        log_event(f"Markdown处理完成: {md_path}, 生成 {len(results)} 条记录")
        return results
        
    except Exception as e:
        error_msg = f"使用策略 {chunking_strategy} 处理Markdown失败: {md_path}\n错误: {str(e)}"
        log_event(error_msg)
        print(error_msg)
        # 降级到传统方法
        return process_md(md_path, url_split)


def get_chunking_strategies():
    """
    获取可用的分块策略列表
    """
    if ADVANCED_CHUNKING_AVAILABLE:
        return get_available_strategies()
    else:
        return [
            {
                "name": "traditional",
                "display_name": "传统切分",
                "description": "基于固定长度和重叠的传统切分方法"
            }
        ]


def batch_process_images(image_directory, use_clip=True):
    """
    批量处理图像目录
    """
    try:
        return process_image_directory(image_directory, use_clip)
    except Exception as e:
        error_msg = f"批量处理图像失败: {image_directory}\n错误: {str(e)}"
        log_event(error_msg)
        print(error_msg)
        return []