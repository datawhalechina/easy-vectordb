# Meta-Chunking: 基于逻辑感知的高效文本分割学习
# 本应用提供了一个基于大语言模型的智能文本分块Gradio界面

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import torch.nn.functional as F

# 初始化用于分块决策的语言模型
# 使用Qwen2-1.5B-Instruct模型进行高效文本分析
model_name_or_path= 'Qwen2-1.5B-Instruct'   
device_map = "auto"  # 自动在可用设备间分配模型
small_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)  
small_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map) 
small_model.eval()  # 设置模型为评估模式

def get_prob_subtract(model,tokenizer,sentence1,sentence2,language):
    """
    使用边际采样计算分块决策的概率差值。
    
    该函数使用语言模型通过比较选择选项1（分割）和选项2（保持一起）的概率
    来确定两个句子是否应该分割或保持在一起。
    
    参数:
        model: 用于推理的语言模型
        tokenizer: 模型的分词器
        sentence1: 第一个句子片段
        sentence2: 第二个句子片段
        language: 语言代码（'zh'表示中文，'en'表示英文）
        
    返回:
        prob_subtract: 概率差值 (P(选项2) - P(选项1))
                      正值倾向于保持句子在一起
                      负值倾向于分割句子
    """
    if language=='zh':
        # 中文分块决策提示
        query='''这是一个文本分块任务.你是一位文本分析专家，请根据提供的句子的逻辑结构和语义内容，从下面两种方案中选择一种分块方式：
        1. 将“{}”分割成“{}”与“{}”两部分；
        2. 将“{}”不进行分割，保持原形式；
        请回答1或2。'''.format(sentence1+sentence2,sentence1,sentence2,sentence1+sentence2)
        prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        input_ids=prompt_ids
        output_ids = tokenizer.encode(['1','2'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]  # Get logits for next token
            token_probs = F.softmax(next_token_logits, dim=-1)  # Convert to probabilities
        # Get probability for option '1' (split)
        next_token_id_0 = output_ids[:, 0].unsqueeze(0)
        next_token_prob_0 = token_probs[:, next_token_id_0].item()      
        # Get probability for option '2' (keep together)
        next_token_id_1 = output_ids[:, 1].unsqueeze(0)
        next_token_prob_1 = token_probs[:, next_token_id_1].item()  
        # Calculate probability difference (higher means prefer keeping together)
        prob_subtract=next_token_prob_1-next_token_prob_0
    else:
        # 英文分块决策提示
        query='''This is a text chunking task. You are a text analysis expert. Please choose one of the following two options based on the logical structure and semantic content of the provided sentence:
        1. Split "{}" into "{}" and "{}" two parts;
        2. Keep "{}" unsplit in its original form;
        Please answer 1 or 2.'''.format(sentence1+' '+sentence2,sentence1,sentence2,sentence1+' '+sentence2)
        # 使用聊天模板格式化提示
        prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        input_ids=prompt_ids
        # 编码可能的答案'1'和'2'
        output_ids = tokenizer.encode(['1','2'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]  # Get logits for next token
            token_probs = F.softmax(next_token_logits, dim=-1)  # Convert to probabilities
        # Get probability for option '1' (split)
        next_token_id_0 = output_ids[:, 0].unsqueeze(0)
        next_token_prob_0 = token_probs[:, next_token_id_0].item()      
        # Get probability for option '2' (keep together)
        next_token_id_1 = output_ids[:, 1].unsqueeze(0)
        next_token_prob_1 = token_probs[:, next_token_id_1].item()  
        # Calculate probability difference (higher means prefer keeping together)
        prob_subtract=next_token_prob_1-next_token_prob_0
    return prob_subtract

# 从chunk_rag模块导入分块函数
from chunk_rag import extract_by_html2text_db_nolist,split_text_by_punctuation

def meta_chunking(original_text,base_model,language,ppl_threshold,chunk_length):
    """
    应用智能文本分割的主要元分块函数。
    
    该函数支持两种分块方法：
    1. PPL分块：使用chunk_rag模块中基于困惑度的分块
    2. 边际采样分块：使用基于概率的决策制定
    
    参数:
        original_text: 待分块的输入文本
        base_model: 分块方法（'PPL Chunking'或'Margin Sampling Chunking'）
        language: 语言代码（'zh'表示中文，'en'表示英文）
        ppl_threshold: 基于困惑度分块的阈值
        chunk_length: 最终块的最大长度
        
    返回:
        final_text: 用双换行符分隔段落的分块文本
    """
    chunk_length=int(chunk_length)
    
    if base_model=='PPL Chunking':
        # 使用基于困惑度的分块方法
        final_chunks=extract_by_html2text_db_nolist(original_text,small_model,small_tokenizer,ppl_threshold,language=language)
    else:
        # 使用边际采样分块方法
        # 基于标点符号将文本分割为句子
        full_segments = split_text_by_punctuation(original_text,language)
        tmp=''  # 当前正在构建的块
        threshold=0  # 分块决策的动态阈值
        threshold_list=[]  # 概率差值的历史记录
        final_chunks=[]  # 最终的块列表
        
        for sentence in full_segments:
            if tmp=='':
                # 块中的第一个句子
                tmp+=sentence
            else:
                # 计算分块决策的概率差值
                prob_subtract=get_prob_subtract(small_model,small_tokenizer,tmp,sentence,language)    
                threshold_list.append(prob_subtract)
                
                if prob_subtract>threshold:
                    # 保持句子在一起（添加到当前块）
                    tmp+=' '+sentence
                else:
                    # 在此处分割（开始新块）
                    final_chunks.append(tmp)
                    tmp=sentence
                    
            # 使用最近5次决策的移动平均更新动态阈值
            if len(threshold_list)>=5:
                last_ten = threshold_list[-5:]  
                avg = sum(last_ten) / len(last_ten)
                threshold=avg
                
        # 如果存在最后一个块，则添加它
        if tmp!='':
            final_chunks.append(tmp)
            
    # 合并块以遵守最大块长度约束
    merged_paragraphs = []
    current_paragraph = ""  
    
    if language=='zh':
        # 对于中文：计算字符数
        for paragraph in final_chunks:  
            if len(current_paragraph) + len(paragraph) <= chunk_length:  
                current_paragraph +=paragraph  
            else:  
                merged_paragraphs.append(current_paragraph)  
                current_paragraph = paragraph    
    else:
        # 对于英文：计算单词数
        for paragraph in final_chunks:  
            if len(current_paragraph.split()) + len(paragraph.split()) <= chunk_length:  
                current_paragraph +=' '+paragraph  
            else:  
                merged_paragraphs.append(current_paragraph)   
                current_paragraph = paragraph 
                
    # 添加最后一个合并的段落
    if current_paragraph:  
        merged_paragraphs.append(current_paragraph) 
        
    # 用双换行符连接所有块以实现清晰分离
    final_text='\n\n'.join(merged_paragraphs)
    return final_text

with open('data/examples.json', 'r') as f:
    examples = json.load(f)
original_prompt_list = [[s["original_text"]] for s in examples]

title = "Meta-Chunking"

header = """# Meta-Chunking: 基于逻辑感知的高效文本分割学习
        """

theme = "soft"
css = """#anno-img .mask {opacity: 0.5; transition: all 0.2s ease-in-out;}
            #anno-img .mask.active {opacity: 0.7}"""


with gr.Blocks(title=title, css=css) as app:
    gr.Markdown(header)
    
    with gr.Row():
        with gr.Column(scale=3):
            original_text = gr.Textbox(value='', label="Original Text", lines=10, max_lines=10, interactive=True)
            chunking_result = gr.Textbox(value='', label="Chunking Result", lines=10, max_lines=10, interactive=False)
            
        with gr.Column(scale=1):
            base_model = gr.Radio(["PPL Chunking", "Margin Sampling Chunking"], 
                                label="Chunking Method", value="PPL Chunking", interactive=True)
            language = gr.Radio(["en", "zh"], label="Text Language", value="en", interactive=True)
            # 困惑度阈值滑块（用于PPL分块）
            ppl_threshold = gr.Slider(minimum=0, maximum=1.0, step=0.1, value=0, 
                                    label="Threshold", interactive=True)
            chunk_length = gr.Textbox(lines=1, label="Chunk length", interactive=True)
    
    button = gr.Button("⚡Click to Chunking")

    button.click(fn=meta_chunking,
                 inputs=[original_text,base_model,language,ppl_threshold,chunk_length],
                 outputs=[chunking_result])
    
    gr.Markdown("## 示例 (点击选择)")
    dataset = gr.Dataset(label="Meta-Chunking",
                         components=[gr.Textbox(visible=False, max_lines=3)],
                         samples=original_prompt_list,
                         type="index")

    dataset.select(fn=lambda idx: (examples[idx]["original_text"],examples[idx]["base_model"],
                                 examples[idx]["language"],examples[idx]["ppl_threshold"],
                                 examples[idx]["chunk_length"]),
                   inputs=[dataset],
                   outputs=[original_text,base_model,language,ppl_threshold,chunk_length])

# 启动Gradio应用程序
# - queue: 启用请求队列，最多10个并发请求
# - api_open: 出于安全考虑禁用API访问
# - show_api: 隐藏API文档
# - server_port: 在端口7080上运行
app.queue(max_size=10, api_open=False).launch(show_api=False,server_port=7080)