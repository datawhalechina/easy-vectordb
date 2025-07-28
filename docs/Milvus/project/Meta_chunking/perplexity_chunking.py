"""
perplexity_chunking.py - 困惑度计算的核心实现模块

这个文件实现了困惑度(Perplexity)的高效批量计算，是PPL分块方法的核心基础组件。
困惑度是衡量语言模型对文本序列预测能力的重要指标，值越低表示模型对该文本的理解越好。
"""

import torch 


class Chunking:
    """
    困惑度计算类
    
    用于批量计算文本序列中每个token的困惑度，支持KV缓存优化。
    这是PPL分块算法的核心组件。
    """
    
    def __init__(self, model, tokenizer) -> None:
        """
        初始化困惑度计算器
        
        参数:
            model: 预训练的语言模型（如Qwen2-1.5B-Instruct）
            tokenizer: 对应的分词器
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def get_ppl_batch(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_kv=False,
        end=None
    ):
        """
        批量计算困惑度的核心函数
        
        这个函数实现了高效的困惑度计算，支持KV缓存来处理长文本。
        
        参数:
            input_ids: 输入的token序列 [batch_size, seq_len]
            attention_mask: 注意力掩码，标识哪些位置是有效的token
            past_key_values: KV缓存，用于加速长文本处理
            return_kv: 是否返回KV缓存
            end: 计算结束位置，默认为序列末尾
            
        返回:
            loss: 每个token的交叉熵损失（困惑度的对数形式）
            past_key_values: KV缓存（如果return_kv=True）
        """
        past_length = 0
        if end is None:
            end = input_ids.shape[1]
            
        # 使用torch.no_grad()禁用梯度计算，节省内存和计算
        with torch.no_grad():
            # 模型前向推理
            response = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,  # 使用KV缓存加速
                use_cache=True,  # 启用缓存
            )
            past_key_values = response.past_key_values
        
        # ==================== 计算交叉熵损失 ====================
        # shift_logits: 模型输出的logits，去掉最后一个位置
        # 因为最后一个位置没有对应的"下一个token"来计算损失
        shift_logits = response.logits[..., :-1, :].contiguous()  
        
        # shift_labels: 真实的token序列，去掉第一个位置
        # 因为第一个位置没有"前一个token"的上下文
        shift_labels = input_ids[..., past_length + 1 : end].contiguous() 
        
        # ==================== 处理注意力掩码 ====================
        # 只计算有效token的损失（attention_mask=1的位置）
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
        
        # 展平张量，准备计算损失
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        
        # ==================== 计算交叉熵损失 ====================
        # reduction="none" 表示返回每个token的单独损失，而不是平均值
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)  
        
        # 损失值就是困惑度的对数形式
        # 真正的困惑度 = exp(loss) 或 2^loss
        res = loss
        
        # 根据需要返回结果
        return (res, past_key_values) if return_kv else res