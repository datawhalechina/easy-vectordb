#记一次 Qwen3-0.6B 微调 内容提取

> 基于@不要葱姜蒜 的self-llm项目[self-llm/models/Qwen3/08-Qwen3_0_6B的小模型有什么用.md at master · datawhalechina/self-llm](https://github.com/datawhalechina/self-llm/blob/master/models/Qwen3/08-Qwen3_0_6B%E7%9A%84%E5%B0%8F%E6%A8%A1%E5%9E%8B%E6%9C%89%E4%BB%80%E4%B9%88%E7%94%A8.md)点击可访问源文章地址。

## 需求

对于zf发布的政策文章，全量爬取的数据包含很多多余的内容，比如下面的内容中：

```

登录  注册
繁體版  智能机器人  无障碍  关怀版  手机版 网站支持IPv6

搜索您想了解的政策/资讯/服务
 
首页 政务公开 政务服务 互动交流 走进重庆
 当前位置： 首页>政务公开>政策文件库>其他文件
【字体：小 大】分享到：
索引号 ：	11500000009275780L/2025-00040	主题分类 ：	建设规划
发布机构 ：	市政府	成文日期 ：	2025-05-11	发布日期 ：	2025-05-16
标题 ：	重庆市人民政府关于核准重庆北碚经济技术开发区规划范围的批复
发文字号 ：	渝府〔2025〕17号	有 效 性 ：	有效
重庆市人民政府

关于核准重庆北碚经济技术开发区

规划范围的批复

渝府〔2025〕17号



xxxxxxxxxx这里是内容



重庆市人民政府　　　　

2025年5月11日　　　　



（此件公开发布）

文件下载
重庆市人民政府关于核准重庆北碚经济技术开发区规划范围的批复.doc
文件下载（图片版）
重庆市人民政府关于核准重庆北碚经济技术开发区规划范围的批复.pdf

中国政府网国务院部门网站地方政府网站市政府部门网站区县政府网站其他网站公共服务单位网站新闻媒体网站
   网站地图 版权声明 联系我们
重庆市政府网
微信公众号

 
重庆市政府网
新浪微博

 
“渝快办”
移动客户端

 
城市手册
微信小程序

版权所有：重庆市人民政府网站 主办：重庆市人民政府办公厅 网站标识：5000000095 ICP备案：渝ICP备05003300号 国际联网备案：渝公网安备 50010302000814号
```

我们需要的是标题、发布时间、内容三个方面，由于不同网站的样式各不相同，所以采取市面上的提取算法获得的效果不尽人意。

## 微调

> 源Colab地址@不要葱姜蒜 宋博大佬的代码[点击这里](https://colab.research.google.com/drive/18ByY11KVhIy6zWx1uKUjSzqeHTme-TtU?usp=drive_link)

下面是我自己的一个记录：

```shell
!pip install datasets swanlab -q
!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1a0sf5C209CLW5824TJkUM4olMy0zZWpg' -O fake_sft.json
```

```python
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch

# 将JSON文件转换为CSV文件
df = pd.read_json('fake_sft.json')
ds = Dataset.from_pandas(df)
ds[:3]

model_id = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
# 加载模型
```

我们需要的数据格式为：

```json
{
  "instruction": "将文本中的title、publishtime、content提取出来，以json格式输出，字段为title、publishtime、content，值为文本中提取出来的内容。",
  "input": "登录  注册 繁體版  智能机器人  无障碍  关怀版  手机版 网站支持IPv6 渝府〔2025〕17号 20250818 关于什么什么的通知 xxxxxxxxxx这里是内容 版权所有：重庆市人民政府网站 主办：重庆市人民政府办公厅 网站标识：5000000095 ICP备案：渝ICP备05003300号 国际联网备案：渝公网安备 50010302000814号",
  "output": "{
      "title": "关于什么什么的通知",
      "publishtime": "2025-08-18",
      "content": "xxxxxxxxxx这里是内容",
  }"
}
```

instruction为用户的指令

Qwen3采用的`Chat Template`格式：

```python
messages = [
    {"role": "system", "content": "You are a helpful AI"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm fine, think you. and you?"},
]

text = tokenizer.apply_chat_template(
    messages,# 模板
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False #关闭思考
)
print(text)
```

```
<|im_start|>system
You are a helpful AI<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
<think>

</think>

I'm fine, think you. and you?<|im_end|>
<|im_start|>assistant
<think>

</think>
```

LoRA(Low-Rank Adaption)训练的数据是需要经过格式化、编码之后再输入给模型，将输入文本编码为input_ids，将输出文本编码为labels，编码之后的结果是向量

```python
"""
该方法将作用于每一个训练样本，编码其输入、输出文本，并返回一个编码后的字典。
"""
def process_func(example):
    MAX_LENGTH = 1024 # 设置最大序列长度为1024个token
    input_ids, attention_mask, labels = [], [], [] # 初始化返回值
    # 适配chat_template
    instruction = tokenizer(
        f"<s><|im_start|>system\n{example['system']}<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    # 将instructio部分和response部分的input_ids拼接，并在末尾添加eos token作为标记结束的token
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # 注意力掩码，表示模型需要关注的位置
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # 对于instruction，使用-100表示这些位置不计算loss（即模型不需要预测这部分）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 超出最大序列长度截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
```

```python
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenizer.decode(tokenized_id[0]['input_ids'])
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))
```

### 加载模型

```python
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",torch_dtype=torch.bfloat16)
```

```
Qwen3ForCausalLM(
  (model): Qwen3Model( 模型名称
    (embed_tokens): Embedding(151936, 1024) 将每个输入的token转换为1024的向量，这里的151936为模型最大token数量
    (layers): ModuleList( 28层解码器层
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention( 自注意力机制 让模型在处理当前token时能够关注其他的前面的token
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False) 生成Query向量
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False) 生成Key向量
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False) 生成Value向量
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False) 将注意力结果映射回原始维度
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06) 对Query做归一化
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06) 对Key做归一化
        )
        (mlp): Qwen3MLP( 前馈神经网络 进一步处理注意力机制输出的信息
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False) 和gate_proj一起，将数据升维到3072
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False) 再降维回1024
          (act_fn): SiLU() 激活函数
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06) 使用RMSNorm归一化
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06) 最后的归一化层 对所有层的输出做一个归一化处理 稳定训练和推理
    (rotary_emb): Qwen3RotaryEmbedding() 旋转位置编码，给模型加上位置信息，让它知道token的顺序
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False) 语言模型头，将最后一层的输出（1024维度）映射回词汇表大小（151936），预测下一个token
)
```

```python
model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
model.dtype
```

### 配置Lora Config

* `task_type`：模型类型，现在绝大部分 `decoder_only` 的模型都是因果语言模型 `CAUSAL_LM`
* `target_modules`：需要训练的模型层的名字，主要就是 `attention`部分的层，不同的模型对应的层的名字不同
* `r`：`LoRA` 的秩，决定了低秩矩阵的维度，较小的 `r` 意味着更少的参数
* `lora_alpha`：缩放参数，与 `r` 一起决定了 `LoRA` 更新的强度。实际缩放比例为`lora_alpha/r`，在当前示例中是 `32 / 8 = 4` 倍
* `lora_dropout`：应用于 `LoRA` 层的 `dropout rate`，用于防止过拟合

```python
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)

model = get_peft_model(model, config)

model.print_trainable_parameters()  # 模型参数训练量只有0.8395%
```

模型参数训练量只有0.8395%表示，整个模型中只有大约0.8395%的参数是可训练的，该策略只需要更新模型的一小部分参数即可实现良好的性能提升。

### Training Arguments

- `output_dir`：模型的输出路径
- `per_device_train_batch_size`：每张卡上的 `batch_size`
- `gradient_accumulation_steps`: 梯度累计
- `num_train_epochs`：顾名思义 `epoch`

```python
args = TrainingArguments(
    output_dir="Qwen3_instruct_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=1,
    num_train_epochs=3,
    save_steps=50,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)
```

### Swanlab记录

```python
import swanlab
from swanlab.integration.transformers import SwanLabCallback

# 实例化SwanLabCallback
swanlab_callback = SwanLabCallback(
    project="Qwen3-Lora",  # 你的项目名称
    experiment_name="Qwen3-8B-LoRA-experiment"  # 你的实验名称
)
```

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback]
)
trainer.train()
```

![](https://gitee.com/Liuxiaomj/Liuxiaomj/raw/main/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20250528113818.png)

### `train/loss`

- **含义**：这是训练过程中的损失（Loss）曲线，表示模型在训练数据上的预测误差。

- 解读

  ：

  - 损失值越低，说明模型对训练数据的拟合越好。
  - 图中显示损失值从初始较高的值迅速下降，然后趋于平稳，这表明模型在训练初期快速学习，并逐渐收敛到一个较低的误差水平。
  - 如果损失值在后期出现波动或上升，可能意味着过拟合或其他问题。

### `train/grad_norm`

- **含义**：梯度范数（Gradient Norm），表示参数更新时梯度的大小。

- 解读

  ：

  - 梯度范数反映了模型参数更新的幅度，通常希望其保持在一个合理的范围内。
  - 图中梯度范数在训练初期有一个较大的峰值，随后逐渐减小并趋于稳定，这表明模型在训练初期进行了较大的参数调整，之后调整幅度逐渐减小。
  - 过大的梯度范数可能导致梯度爆炸，而过小则可能导致梯度消失，都不利于模型训练。

### `train/learning_rate`

- **含义**：学习率（Learning Rate），控制参数更新的速度。

- 解读

  ：

  - 学习率决定了每次迭代中参数更新的步长，图中显示学习率随着时间逐步减小，这是一种常见的策略，称为学习率衰减。
  - 初始较高的学习率有助于模型快速接近最优解，随后降低学习率可以使模型更精细地调整参数，以达到更好的收敛效果。

### `train/epoch`

- **含义**：当前训练轮次（Epoch），表示模型已经完整遍历训练数据集的次数。

- 解读

  ：

  - 随着训练的进行，轮次逐渐增加，图中显示模型已经完成了大约3个轮次的训练。
  - 通过观察轮次与其它指标的关系，可以了解模型在不同训练阶段的表现。

### `train/global_step`

- **含义**：全局步数（Global Step），表示模型已经执行了多少次参数更新。

- 解读

  ：

  - 全局步数随着训练的进行线性增加，反映了模型训练的进度。
  - 通过对比全局步数与其他指标的变化，可以分析模型在不同训练阶段的学习动态。

### 测试

```python
prompt = "内容"

messages = [
    {"role": "system", "content": "将文本中的title、publishtime、content提取出来，以json格式输出，字段为title、publishtime、content，值为文本中提取出来的内容。"},
    {"role": "user", "content": prompt}
]

inputs = tokenizer.apply_chat_template(messages,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True,
                                       enable_thinking=False).to('cuda')

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```











































