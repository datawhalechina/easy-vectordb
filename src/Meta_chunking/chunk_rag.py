"""
chunk_rag.py - PPL困惑度分块的核心实现文件

- split_text_by_punctuation(): 按标点符号分割文本为句子
- find_minima(): 寻找困惑度序列的局部极小值点
- extract_by_html2text_db_nolist(): 标准PPL分块实现
- extract_by_html2text_db_dynamic(): 动态阈值分块
- extract_by_html2text_db_dynamic_batch(): 支持长文本的批处理分块

"""

from perplexity_chunking import Chunking
from typing import List, Dict
import re
import math 
from nltk.tokenize import sent_tokenize
import jieba 
import torch

def split_text_by_punctuation(text, language): 
    """
    按标点符号将文本分割为句子
    
    这是文本预处理的第一步，将长文本分割为句子级别的片段。
    不同语言使用不同的分割策略。
    
    参数:
        text: 输入的原始文本
        language: 语言类型 ('zh'中文 或 'en'英文)
    
    返回:
        sentences: 分割后的句子列表
    """
    if language == 'zh': 
        # 中文文本处理
        # 使用jieba进行分词，不使用全模式分词
        sentences = jieba.cut(text, cut_all=False)  
        sentences_list = list(sentences)  
        sentences = []  
        temp_sentence = ""  
        
        # 重新组合句子：遇到句末标点符号时结束一个句子
        for word in sentences_list:  
            if word in ["。", "！", "？", "；"]:  # 中文句末标点
                sentences.append(temp_sentence.strip() + word)  
                temp_sentence = ""  
            else:  
                temp_sentence += word  
        
        # 处理最后一个句子（可能没有标点结尾）
        if temp_sentence:   
            sentences.append(temp_sentence.strip())  
        
        return sentences
    else:
        # 英文文本处理
        # 使用NLTK的句子分割器
        full_segments = sent_tokenize(text)
        ret = []
        
        # 限制句子长度，防止单个句子过长影响模型处理
        for item in full_segments:
            item_l = item.strip().split(' ')  # 按空格分割单词
            
            # 如果句子太长，进行截断
            if len(item_l) > 512:
                if len(item_l) > 1024:
                    # 超长句子截断到256个单词
                    item = ' '.join(item_l[:256]) + "..."
                else:
                    # 长句子截断到512个单词
                    item = ' '.join(item_l[:512]) + "..."
            ret.append(item)
        return ret


def find_minima(values, threshold):  
    """
    在困惑度序列中寻找局部极小值点
    
    这是PPL分块算法的核心：通过识别困惑度的局部极小值来确定文本的语义边界。
    
    参数:
        values: 困惑度值序列（每个句子的平均困惑度）
        threshold: 阈值，用于过滤微小的波动，只保留显著的极小值点
    
    返回:
        minima_indices: 局部极小值点的索引列表
    """
    minima_indices = []  
    for i in range(1, len(values) - 1):  
        # 情况1：标准的局部极小值（两边都比当前值大）
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            # 检查下降幅度是否显著（左边或右边的下降超过阈值）
            if (values[i - 1] - values[i] >= threshold) or (values[i + 1] - values[i] >= threshold):
                minima_indices.append(i)  
        # 情况2：左边下降，右边持平的情况
        elif values[i] < values[i - 1] and values[i] == values[i + 1]:
            # 只要左边的下降超过阈值就认为是有效的分割点
            if values[i - 1] - values[i] >= threshold:
                minima_indices.append(i) 
    return minima_indices

def find_minima_dynamic(values, threshold, threshold_zlist):  
    """
    动态阈值版本的局部极小值检测函数
    
    与find_minima函数类似，但增加了动态阈值调整机制。
    通过记录历史的下降幅度，动态调整阈值以适应不同文本的特性。
    
    参数:
        values: 困惑度值序列
        threshold: 当前阈值
        threshold_zlist: 历史阈值记录列表
    
    返回:
        minima_indices: 局部极小值点的索引列表
        threshold: 更新后的阈值
        threshold_zlist: 更新后的历史阈值记录
    
    """
    minima_indices = []  
    for i in range(1, len(values) - 1):  
        # 标准局部极小值检测
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            if (values[i - 1] - values[i] >= threshold) or (values[i + 1] - values[i] >= threshold):
                minima_indices.append(i)
                # 记录较小的下降幅度（更保守的估计）
                threshold_zlist.append(min(values[i - 1] - values[i], values[i + 1] - values[i]))  
        # 左边下降，右边持平的情况
        elif values[i] < values[i - 1] and values[i] == values[i + 1]:
            if values[i - 1] - values[i] >= threshold:
                minima_indices.append(i) 
                threshold_zlist.append(values[i - 1] - values[i])
        
        # 动态阈值调整：当有足够的历史数据时
        if len(threshold_zlist) >= 100:
            # 使用历史记录中的最小值作为新阈值（更严格的标准）
            avg = min(threshold_zlist)
            threshold = avg
    
    return minima_indices, threshold, threshold_zlist

def extract_by_html2text_db_chongdie(sub_text, model, tokenizer, threshold, language='zh') -> List[str]:   
    """
    PPL困惑度分块函数（重叠版本）
    
    这个函数与extract_by_html2text_db_nolist类似，但在分块时采用了不同的策略。
    主要区别在于分块边界的处理方式，可能会产生重叠的文本块。
    
    参数:
        sub_text: 输入的原始文本
        model: 语言模型，用于计算困惑度
        tokenizer: 分词器
        threshold: 困惑度阈值，用于过滤微小波动
        language: 语言类型 ('zh'中文 或 'en'英文)
    
    返回:
        final_chunks: 分块后的文本列表
    
    """
    temp_para = sub_text

    # ==================== 文本预处理 ====================
    if language == 'zh':
        cleaned_text = temp_para
    else:
        cleaned_text = temp_para
 
    # ==================== 句子分割和编码 ====================
    segments = split_text_by_punctuation(cleaned_text, language)
    segments = [item for item in segments if item.strip()]  
    ch = Chunking(model, tokenizer)
    len_sentences = []
    input_ids = torch.tensor([[]], device=model.device, dtype=torch.long)  
    attention_mask = torch.tensor([[]], device=model.device, dtype=torch.long)  
    
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)

    # ==================== 计算困惑度 ====================
    loss, past_key_values = ch.get_ppl_batch( 
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    
    # ==================== 计算句子级平均困惑度 ====================
    first_cluster_ppl = []
    index = 0
    for i in range(len(len_sentences)):
        if i == 0:
            first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
            index += len_sentences[i] - 1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            index += len_sentences[i]
        
    # ==================== 寻找分割点并生成文本块 ====================
    minima_indices = find_minima(first_cluster_ppl, threshold)
    first_chunk_indices = []
    first_chunk_sentences = []
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    
    # 重叠版本的分块策略：包含分割点本身
    for i in range(len(split_points)-1):
        tmp_index = []
        tmp_sentence = []
        
        # 从当前分割点到下一个分割点（包含两端）
        for sp_index in range(split_points[i], split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    
    # 生成最终文本块
    final_chunks = []
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    
    print('重叠分块结果 - 句子索引:', first_chunk_indices)

    return final_chunks

def extract_by_html2text_db_nolist(sub_text, model, tokenizer, threshold, language='zh') -> List[str]:  
    """
    PPL困惑度分块的核心实现函数
    
    这是Meta-Chunking中PPL分块方法的主要实现，通过计算每个句子的困惑度，
    找到困惑度的局部极小值点作为分割点，实现语义连贯的文本分块。
    
    参数:
        sub_text: 输入的原始文本
        model: 语言模型，用于计算困惑度
        tokenizer: 分词器
        threshold: 困惑度阈值，用于过滤微小波动
        language: 语言类型 ('zh'中文 或 'en'英文)
    
    返回:
        final_chunks: 分块后的文本列表，每个元素是一个语义连贯的文本块
    
    """
    temp_para = sub_text

    # ==================== 第一步：文本预处理 ====================
    if language == 'zh':
        # 中文文本预处理（这里保持原样，可以根据需要添加清理逻辑）
        cleaned_text = temp_para
    else:
        # 英文文本预处理
        cleaned_text = temp_para
 
    # ==================== 第二步：句子分割 ====================
    # 使用标点符号将文本分割为句子
    segments = split_text_by_punctuation(cleaned_text, language)
    segments = [item for item in segments if item.strip()]  # 过滤空句子
    
    # ==================== 第三步：准备困惑度计算 ====================
    # 初始化困惑度计算器
    ch = Chunking(model, tokenizer)
    
    # 记录每个句子的token长度，用于后续分割困惑度序列
    len_sentences = []
    
    # 初始化空的token序列和注意力掩码
    input_ids = torch.tensor([[]], device=model.device, dtype=torch.long)  
    attention_mask = torch.tensor([[]], device=model.device, dtype=torch.long)  
    
    # 将所有句子编码并拼接成一个长序列
    for context in segments:
        # 对每个句子进行tokenization（不添加特殊token）
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        
        # 拼接到总的token序列中
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        len_sentences.append(input_id.shape[1])  # 记录当前句子的token数量
        
        # 拼接注意力掩码
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)

    # ==================== 第四步：批量计算困惑度 ====================
    # 使用困惑度计算器计算整个序列的token级困惑度
    loss, past_key_values = ch.get_ppl_batch( 
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    
    # ==================== 第五步：计算句子级平均困惑度 ====================
    first_cluster_ppl = []  # 存储每个句子的平均困惑度
    index = 0  # 当前处理到的token位置
    
    for i in range(len(len_sentences)):
        if i == 0:
            # 第一个句子：从位置0到len_sentences[i]-1
            first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
            index += len_sentences[i] - 1
        else:
            # 其他句子：从当前index到index+len_sentences[i]
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            index += len_sentences[i]
        
    # ==================== 第六步：寻找局部极小值点 ====================
    # 在困惑度序列中找到局部极小值，这些点通常对应语义边界
    minima_indices = find_minima(first_cluster_ppl, threshold)
    
    # ==================== 第七步：根据分割点生成文本块 ====================
    first_chunk_indices = []  # 存储每个块包含的句子索引
    first_chunk_sentences = []  # 存储每个块包含的句子内容
    
    # 构建分割点列表：开始点 + 极小值点 + 结束点
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    
    # 根据分割点生成文本块
    for i in range(len(split_points)-1):
        tmp_index = []
        tmp_sentence = []
        
        if i == 0:
            # 第一个块：包含第一个句子
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        
        # 添加当前分割区间内的所有句子
        for sp_index in range(split_points[i]+1, split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
            
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    
    # ==================== 第八步：生成最终文本块 ====================
    final_chunks = []
    for sent_list in first_chunk_sentences:
        # 将每个块中的句子拼接成完整的文本
        final_chunks.append(''.join(sent_list))
    
    # 调试输出：显示分块的句子索引
    print('分块结果 - 句子索引:', first_chunk_indices)

    return final_chunks

def extract_by_html2text_db_dynamic(sub_text, model, tokenizer, threshold, threshold_zlist, language='zh') -> List[str]:  
    """
    动态阈值PPL分块函数
    
    这个函数与extract_by_html2text_db_nolist类似，但使用动态阈值调整机制。
    通过记录历史的困惑度下降幅度，自适应地调整分割阈值。
    
    参数:
        sub_text: 输入的原始文本
        model: 语言模型，用于计算困惑度
        tokenizer: 分词器
        threshold: 初始困惑度阈值
        threshold_zlist: 历史阈值记录列表，用于动态调整
        language: 语言类型 ('zh'中文 或 'en'英文)
    
    返回:
        final_chunks: 分块后的文本列表
        threshold: 更新后的动态阈值
        threshold_zlist: 更新后的历史阈值记录

    """
    temp_para = sub_text 
    
    # ==================== 文本预处理 ====================
    if language == 'zh':
        cleaned_text = temp_para
    else:
        cleaned_text = temp_para

    # ==================== 句子分割和编码 ====================
    segments = split_text_by_punctuation(cleaned_text, language)
    segments = [item for item in segments if item.strip()]  
    ch = Chunking(model, tokenizer)
    len_sentences = []
    input_ids = torch.tensor([[]], device=model.device, dtype=torch.long)  
    attention_mask = torch.tensor([[]], device=model.device, dtype=torch.long)  
    
    # 将所有句子编码并拼接
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)

    # ==================== 计算困惑度 ====================
    loss, past_key_values = ch.get_ppl_batch( 
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    
    # ==================== 计算句子级平均困惑度 ====================
    first_cluster_ppl = []
    index = 0
    for i in range(len(len_sentences)):
        if i == 0:
            first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
            index += len_sentences[i] - 1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            index += len_sentences[i]
        
    # ==================== 动态阈值分块 ====================
    # 使用动态阈值寻找局部极小值点，同时更新阈值
    minima_indices, threshold, threshold_zlist = find_minima_dynamic(first_cluster_ppl, threshold, threshold_zlist)
    
    # ==================== 生成文本块 ====================
    first_chunk_indices = []
    first_chunk_sentences = []
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    
    for i in range(len(split_points)-1):
        tmp_index = []
        tmp_sentence = []
        if i == 0:
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i]+1, split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    
    final_chunks = []
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    
    print('动态阈值分块结果 - 句子索引:', first_chunk_indices)
    
    return final_chunks, threshold, threshold_zlist

def extract_by_html2text_db_dynamic_batch(sub_text, model, tokenizer, threshold, threshold_zlist, language='zh', past_key_values=None) -> List[str]:
    """
    支持长文本批处理和KV缓存的动态PPL分块函数
    
    这是处理超长文本的优化版本，通过批处理和KV缓存机制来处理可能超出模型上下文长度的文本。
    
    参数:
        sub_text: 输入的原始文本
        model: 语言模型
        tokenizer: 分词器
        threshold: 困惑度阈值
        threshold_zlist: 历史阈值记录列表
        language: 语言类型
        past_key_values: 预先存在的KV缓存
    
    返回:
        final_chunks: 分块后的文本列表
        threshold: 更新后的阈值
        threshold_zlist: 更新后的历史阈值记录
    """
    temp_para = sub_text

    # ==================== 文本预处理 ====================
    if language == 'zh':
        cleaned_text = temp_para
    else:
        cleaned_text = temp_para
 
    # 句子分割和编码准备
    segments = split_text_by_punctuation(cleaned_text, language)
    segments = [item for item in segments if item.strip()]  
    ch = Chunking(model, tokenizer)
    len_sentences = []
    input_ids = torch.tensor([[]], device=model.device, dtype=torch.long)  
    attention_mask = torch.tensor([[]], device=model.device, dtype=torch.long)  
    
    # 将所有句子编码并拼接
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)

    # ==================== 批处理困惑度计算 ====================
    batch_size = 4096  # 每批处理的token数量，可根据GPU内存调整
    total_batches = math.ceil(input_ids.shape[1] / batch_size)   
    loss = torch.tensor([], device=model.device, dtype=torch.long)
    
    # 分批处理长序列
    for i in range(total_batches): 
        start = i * batch_size
        end = start + batch_size
        input_ids_tmp = input_ids[:, start:end]

        # 构建当前批次的注意力掩码
        attention_mask_tmp = attention_mask[:, :end]
        
        # 在序列前添加空格token，用于上下文连接
        input_ids_tmp = torch.cat([tokenizer(' ', return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device), input_ids_tmp], dim=-1)
        attention_mask_tmp = torch.cat([attention_mask_tmp, torch.ones((1, i+1), device=model.device, dtype=torch.long)], dim=-1)
        
        size = input_ids_tmp.shape[1]
        
        # ==================== KV缓存管理 ====================
        # 当注意力掩码长度超过阈值时，进行缓存裁剪
        if attention_mask_tmp.shape[1] > 24576:  # 最大上下文长度限制
            # 裁剪过早的KV缓存，保留最近的上下文
            past_key_values = [  
                [k[:, :, size+1:], v[:, :, size+1:]]  
                for k, v in past_key_values  
            ]
            # 相应地调整注意力掩码
            attention_mask_tmp = attention_mask_tmp[:, attention_mask_tmp.shape[1]-size-past_key_values[0][0].shape[2]:]
        
        # 计算当前批次的困惑度
        loss_tmp, past_key_values = ch.get_ppl_batch( 
            input_ids_tmp,
            attention_mask_tmp,
            past_key_values=past_key_values,
            return_kv=True
        )
        loss = torch.cat([loss, loss_tmp], dim=-1)
            
    # ==================== 计算句子级困惑度 ====================
    first_cluster_ppl = []
    index = 0
    for i in range(len(len_sentences)):
        if i == 0:
            # 第一个句子：跳过第一个token（空格token）
            first_cluster_ppl.append(loss[1:len_sentences[i]].mean().item())
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
        index += len_sentences[i]
        
    # ==================== 动态分块 ====================
    # 使用动态阈值寻找局部极小值点
    minima_indices, threshold, threshold_zlist = find_minima_dynamic(first_cluster_ppl, threshold, threshold_zlist)
    
    # 根据分割点生成文本块
    first_chunk_indices = []
    first_chunk_sentences = []
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    
    for i in range(len(split_points)-1):
        tmp_index = []
        tmp_sentence = []
        if i == 0:
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i]+1, split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    
    # 生成最终文本块
    final_chunks = []
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    
    print('批处理分块结果 - 句子索引:', first_chunk_indices)
    
    return final_chunks, threshold, threshold_zlist

def extract_by_html2text_db_bench(sub_text, model, tokenizer, threshold, language='zh', past_key_values=None) -> List[str]:  
    """
    基准测试版本的PPL分块函数
    
    这是用于性能基准测试的PPL分块实现，使用较大的批处理大小（8192）来优化处理速度。
    主要用于评估不同批处理大小对分块性能的影响。
    
    参数:
        sub_text: 输入的原始文本
        model: 语言模型，用于计算困惑度
        tokenizer: 分词器
        threshold: 困惑度阈值，用于过滤微小波动
        language: 语言类型 ('zh'中文 或 'en'英文)
        past_key_values: 预先存在的KV缓存
    
    返回:
        final_chunks: 分块后的文本列表
    """
    temp_para = sub_text
    
    # ==================== 文本预处理 ====================
    if language == 'zh':
        cleaned_text = temp_para
    else:
        cleaned_text = temp_para

    # ==================== 句子分割和编码 ====================
    segments = split_text_by_punctuation(cleaned_text, language)
    segments = [item for item in segments if item.strip()]  
    ch = Chunking(model, tokenizer)
    len_sentences = []
    input_ids = torch.tensor([[]], device=model.device, dtype=torch.long)  
    attention_mask = torch.tensor([[]], device=model.device, dtype=torch.long)  
    
    # 将所有句子编码并拼接
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)
  
    # ==================== 大批处理困惑度计算 ====================
    batch_size = 8192  # 大批处理大小，适合高性能GPU
    total_batches = math.ceil(input_ids.shape[1] / batch_size)   
    
    # 调试输出：显示总token数量
    print('总token数量:', input_ids.shape[1])
    
    loss = torch.tensor([], device=model.device, dtype=torch.long)
    
    # 分批处理长序列
    for i in range(total_batches): 
        start = i * batch_size
        end = start + batch_size
        input_ids_tmp = input_ids[:, start:end]
        attention_mask_tmp = attention_mask[:, :end]
        
        # 在序列前添加空格token，用于上下文连接
        input_ids_tmp = torch.cat([tokenizer(' ', return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device), input_ids_tmp], dim=-1)
        attention_mask_tmp = torch.cat([attention_mask_tmp, torch.ones((1, i+1), device=model.device, dtype=torch.long)], dim=-1)
        
        size = input_ids_tmp.shape[1]
        
        # ==================== KV缓存管理 ====================
        # 当注意力掩码长度超过阈值时，进行缓存裁剪
        if attention_mask_tmp.shape[1] > 24576:  # 最大上下文长度限制
            # 裁剪过早的KV缓存，保留最近的上下文
            past_key_values = [  
                [k[:, :, size+1:], v[:, :, size+1:]]  
                for k, v in past_key_values  
            ]
            # 相应地调整注意力掩码
            attention_mask_tmp = attention_mask_tmp[:, attention_mask_tmp.shape[1]-size-past_key_values[0][0].shape[2]:]
        
        # 计算当前批次的困惑度
        loss_tmp, past_key_values = ch.get_ppl_batch( 
            input_ids_tmp,
            attention_mask_tmp,
            past_key_values=past_key_values,
            return_kv=True
        )
        loss = torch.cat([loss, loss_tmp], dim=-1)
            
    # ==================== 计算句子级困惑度 ====================
    first_cluster_ppl = []
    index = 0
    for i in range(len(len_sentences)):
        if i == 0:
            # 第一个句子：跳过第一个token（空格token）
            first_cluster_ppl.append(loss[1:len_sentences[i]].mean().item())
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
        index += len_sentences[i]
    
    # ==================== 寻找分割点并生成文本块 ====================
    minima_indices = find_minima(first_cluster_ppl, threshold)
    first_chunk_indices = []
    first_chunk_sentences = []
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    
    for i in range(len(split_points)-1):
        tmp_index = []
        tmp_sentence = []
        if i == 0:
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i]+1, split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    
    # 生成最终文本块
    final_chunks = []
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    
    # 调试输出：显示分块的句子索引
    print('基准测试分块结果 - 句子索引:', first_chunk_indices)

    return final_chunks
 