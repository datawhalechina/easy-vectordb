# SwanLab+Qwen2.5-Coder-7B-Instruct Lora微调

参考DataWhale项目self-llm[原文地址]([self-llm/models/Qwen2.5-Coder/05-Qwen2.5-Coder-7B-Instruct Lora 微调 SwanLab 可视化记录版.md at master · datawhalechina/self-llm](https://github.com/datawhalechina/self-llm/blob/master/models/Qwen2.5-Coder/05-Qwen2.5-Coder-7B-Instruct%20Lora%20%E5%BE%AE%E8%B0%83%20SwanLab%20%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%B0%E5%BD%95%E7%89%88.md))

记录自己的一次微调

使用的数据集是中文法律问答数据集[DISC-Law-SFT](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT)，使用SwanLab监控训练过程与评估模型效果

## 关于LoRA

LoRA（Low-Rank Adaptation）是一种高效的微调方法，专为大型预训练语言模型设计。它通过引入低秩矩阵来调整模型的权重，而不是直接更新所有参数，从而大大减少了需要优化的参数数量。

1. **超参数控制**:
   - LoRA引入了几个重要的超参数来控制其行为：
     - **秩 (rank)**: 决定了低秩矩阵 AA 和 BB 的大小。较低的秩意味着更少的额外参数，但可能会限制表达能力。
     - **alpha**: 控制LoRA更新强度的比例因子。较大的alpha值表示更强的更新效果。
     - **dropout**: 应用于LoRA层以防止过拟合的一种正则化手段。

### 应用场景

LoRA特别适用于那些希望利用大规模预训练模型的强大功能，但在**特定领域或任务上有独特需求 **的情况。例如，在医疗、法律等领域，可能有专门的知识库或者特殊的文本格式，这时可以通过LoRA快速定制一个适合该领域的专用模型，而不必从头训练整个模型。

### 微调方案如何选择

#### 根据任务类型

* 如果你的任务有标注好的数据集（如分类、回归），可以选择传统的微调方法。
* 如果目标是将通用语言模型调整为特定领域的专家（例如医学、法律），则应优先考虑那些能够保留预训练知识同时专注于领域特性的方法，如LoRA（低秩适配）或P-tuning。

#### 根据数据量

* 当可用数据较少时，推荐采用参数高效的微调技术，如LoRA。这些方法通过引入少量新参数来避免过拟合，并能更好地利用有限的数据进行有效训练。
* 如果有足够的高质量数据，则可以直接对整个模型进行全参数微调，以获得最佳性能

#### 根据算力

- 如果拥有强大的硬件支持（如多GPU集群），那么可以考虑更耗资源但可能带来更好结果的方法，比如全参数微调加上数据增强。
- 在计算资源受限的情况下，应该倾向于选择轻量级且高效的微调方案，如LoRA或者蒸馏

## 关于微调跟踪工具SwanLab

1. 提供实验跟踪与管理，可视化机器学习实验的结果，自动记录日志、系统硬件环境和环境配置信息，如显卡型号、python版本等，在项目集成SwanLab后，运行项目时按照要求填写密钥，此时监控平台将自动获取项目运行环境信息并给予展示。
2. 友好的API和界面，结合了超参数跟踪、指标记录、在线协作、实验连接分享、实时消息通知等功能，（图表对比视图各个图的含义）
   1. **train/loss**：
      - **含义**：这个图表展示了在训练过程中损失函数的值随时间（通常是迭代次数或批次）的变化情况。
      - **具体意义**：损失函数是衡量模型预测值与真实值之间差异的指标。train/loss图表可以帮助研究人员观察模型是否正在学习，以及学习的效果如何。一般来说，随着训练的进行，损失值应该逐渐下降，表示模型正在改进。如果损失值波动很大或者不下降，可能意味着模型需要调整，比如调整学习率或修改网络结构。
   2. **train/grad_norm**：
      - **含义**：这个图表展示了训练过程中梯度范数的变化。
      - **具体意义**：梯度范数是指模型参数更新的幅度。这个指标可以反映模型的学习速度和稳定性。如果梯度范数过大，可能导致训练不稳定（梯度爆炸）；如果梯度范数过小，可能意味着学习过程过于缓慢或者模型已经接近收敛。通过观察train/grad_norm，研究人员可以调整优化器的参数，以保持合适的更新幅度。
   3. **train/learning_rate**：
      - **含义**：这个图表展示了训练过程中学习率的变化。
      - **具体意义**：学习率是决定模型权重更新幅度的超参数。train/learning_rate图表对于监控学习率调度策略非常有用。常见的学习率调度包括固定学习率、学习率衰减、循环学习率等。通过观察学习率的变化，研究人员可以评估调度策略的有效性，并根据需要调整学习率。
   4. **train/epoch**：
      - **含义**：这个图表展示了训练过程中的周期（epoch）数。
      - **具体意义**：一个周期（epoch）通常指的是模型完整地遍历一次训练数据集。train/epoch图表可以简单地显示训练的进度，即已经完成了多少个周期。虽然这个图表通常不如其他指标图表信息丰富，但它提供了训练进度的直观了解。
3. 全面的框架集成，支持多种机器学习框架
4. 允许实验用户在一个项目下交流协作

### 项目环境：

```
ubuntu 22.04
python 3.12
cuda 12.1
pytorch 2.3.0
```

```
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.20.0
pip install transformers==4.46.2
pip install accelerate==1.1.1
pip install peft==0.13.2
pip install datasets==3.1.0
pip install swanlab==0.3.25

```

下列代码实现了一个基于Transformers库的微调过程，特别是针对一个大型语言模型（LLM）进行低秩适应（LoRA, Low-Rank Adaptation）微调

### 流程

1. 导入相关操作库：pandas数据加载与处理，swanlab训练监控，peft用于将一个预训练模型与LoRAConfig结合进行特定任务的微调，
2. 分词器处理文本数据，将用户的输入和输出都通过分词器处理为token id列表，创建适当的注意力掩码，并准备好用于监督学习的标签
3. 定义一个预测函数接收用户消息作为输入，通过模型生成响应。它首先构建一个对话模板，然后对文本进行编码，传递给模型以生成新的 tokens，最后解码这些 tokens 得到人类可读的文本。
4. 下载模型并使用 Hugging Face 的 AutoTokenizer 和 AutoModelForCausalLM 加载模型权重。
5. 读取提前在hugging face上下载的数据，pandas进行处理
6. 配置LoRA的具体参数，与基础模型一起构建一个适配后的模型peft_model
7. 设置训练参数
8. 创建回调函数，继承 `SwanLabCallback` 类，重写其方法以实现特定的训练开始前和每个epoch结束时的行为。这包括在训练开始前和每个 epoch 结束时打印一些样本预测结果，并向 SwanLab 发送日志。
9. 初始化Trainer实例开始训练，当 `trainer.train()` 被调用时，训练过程正式开始。`Trainer` 将按照指定的参数执行训练循环，期间会自动应用 LoRA 来更新模型参数，并通过 `swanlab_callback` 监控训练状态
10. `swanlab.finish()` 来停止 SwanLab 的记录服务

### 代码详解

```python
import json
import pandas as pd#数据分析和操作库，用于加载和处理数据集
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback#swanlab监控训练进度，记录日志
from peft import LoraConfig, TaskType, get_peft_model#参数效率微调，lora是peft的一种，通过仅训练少量额外参数（低秩矩阵）来实现。
from transformers import (
    AutoModelForCausalLM,#自动选择适用于因果语言建模任务的模型架构
    TrainingArguments,#定义训练过程中需要的各种超参数
    Trainer,#提供了一个高级的API来简化循环的管理
    DataCollatorForSeq2Seq,#数据整理器
)
import swanlab
```

```python
#用于对数据集中的每个样本进行预处理
#example: 一个字典对象，表示数据集中的单个样本。它通常包含至少两个键：instruction 和 input，以及一个代表模型应生成的输出的键 output。
#函数返回一个新的字典，该字典包含了三个键：input_ids, attention_mask, 和 labels。这些是模型训练所需的张量。
def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384//序列的最大长度，超过的将被截断
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(#使用分词器将系统指令instruction和用户输入input转换为tokenID列表
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)#使用特殊标记的字符串格式化来构造prompt,false表示不自动添加特殊标记
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

```python
#用于根据给定的对话消息列表 messages，利用已经训练好的模型 model 和分词器 tokenizer 来生成回复
def predict(messages, model, tokenizer):
    device = "cuda"#指定使用CUDA设备（GPU）
    text = tokenizer.apply_chat_template(#apply_chat_template方法构建对话的提示串,tokenize=false表示不进行分词操作，add_generation_prompt=true表示添加用于引导生成的提示
        messages, tokenize=False, add_generation_prompt=True
    )
    #将构建好的提示词通过分词器转换为张量格式，并且移动到指定设备（GPU）
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    #进行文本生成
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    #遍历生成的id列表，把原始在内的所有的内容去除，只保留生成的新文本
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    #使用batch_decode方法将生成的tokenid列表转换为字符串形式的文本,skip_special_tokens=True表示：确保特殊标记不会出现在最终的输出中。	
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response
```

```python
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,#指定任务类型为因果语言模型（Causal Language Model），这通常是针对生成式任务，如文本补全或对话系统
    target_modules=[#查询（q）、键（k）、值（v）
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=False,  # 训练模式 false表示当前配置是用于训练而不是推理
    r=64,  # Lora 秩 较低的秩可以减少需要学习的参数量，同时保持模型性能
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理，较大的alpha值意味着更强烈的更新
    lora_dropout=0.1,  # Dropout 比例 用于防止过拟合
)

#将原始模型与LoRA配置结合起来，创建了一个新的适配后的模型实例（peft_model），该模型将在指定的模块上应用LoRA技术进行微调
peft_model = get_peft_model(model, config)

args = TrainingArguments(
  #训练期间保存检查点和日志文件的目录地址
    output_dir="./output/Qwen2.5-Coder-7b",
  
  #每个设备（GPU/CPU）上的批量大小。这里设置为2，意味着每个训练步骤处理2个样本。
    per_device_train_batch_size=2,
  
  #梯度累积步数。当批量大小受限于硬件时，可以通过累积多个小批量的梯度来模拟更大的批量。
    gradient_accumulation_steps=8,
  
  #每多少个训练步骤记录一次日志信息。
    logging_steps=10,
  
  #总共训练的轮数。
    num_train_epochs=1,
  
  #每多少个训练步骤保存一次模型检查点。
    save_steps=100,
  
  #学习率，即优化器调整模型参数的速度。
    learning_rate=1e-4,
  
  #是否在分布式训练中的每个节点上都保存模型。这对于多节点训练环境特别有用。
    save_on_each_node=True,
  
#启用梯度检查点技术以节省内存。此技术通过重新计算前向传递的部分来减少内存占用，但可能会增加计算时间。
    gradient_checkpointing=True,
  
  #指定报告训练进度的方式。设置为 "none" 表示不使用任何外部监控工具。
    report_to="none",
)
```

```python
def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model, **kwargs)
            
        print("训练开始")
        print("未开始微调，先取3条主观评测：")
        test_text_list = []
        for index, row in test_df[:3].iterrows():
            instruction = row["instruction"]
            input_value = row["input"]

            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"},
            ]

            #使用 predict 函数基于当前模型生成回复。此时，模型尚未经过微调，因此这些预测代表了预训练模型的表现。
            response = predict(messages, peft_model, tokenizer)
            
            
            messages.append({"role": "assistant", "content": f"{response}"})    
            result_text = f"【Q】{messages[1]['content']}\n【LLM】{messages[2]['content']}\n"
            print(result_text)
            
            #将每条预测结果作为 swanlab.Text 对象添加到 test_text_list 中，附带一个标题（caption），即生成的回复文本。
            test_text_list.append(swanlab.Text(result_text, caption=response))

        swanlab.log({"Prediction": test_text_list}, step=0)
```

### 微调完整代码

```python
import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import swanlab
        
        
def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download("Qwen/Qwen2.5-Coder-7B-Instruct", cache_dir="/root/autodl-tmp", revision="master")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Qwen/Qwen2___5-Coder-7B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/Qwen/Qwen2___5-Coder-7B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 处理数据集
train_jsonl_path = "DISC-Law-SFT-Pair-QA-released-new.jsonl"
train_df = pd.read_json(train_jsonl_path, lines=True)[5:5000]
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
test_df = pd.read_json(train_jsonl_path, lines=True)[:5]

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

peft_model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="./output/Qwen2.5-Coder-7b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

class HuanhuanSwanLabCallback(SwanLabCallback):   
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model, **kwargs)
            
        print("训练开始")
        print("未开始微调，先取3条主观评测：")
        test_text_list = []
        for index, row in test_df[:3].iterrows():
            instruction = row["instruction"]
            input_value = row["input"]

            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"},
            ]

            response = predict(messages, peft_model, tokenizer)
            messages.append({"role": "assistant", "content": f"{response}"})
                
            result_text = f"【Q】{messages[1]['content']}\n【LLM】{messages[2]['content']}\n"
            print(result_text)
            
            test_text_list.append(swanlab.Text(result_text, caption=response))

        swanlab.log({"Prediction": test_text_list}, step=0)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # ===================测试阶段======================
        test_text_list = []
        for index, row in test_df.iterrows():
            instruction = row["instruction"]
            input_value = row["input"]
            ground_truth = row["output"]#正确答案

            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"},
            ]

            response = predict(messages, peft_model, tokenizer)#预测答案
            messages.append({"role": "assistant", "content": f"{response}"})
            
            if index == 0:
                print("epoch", round(state.epoch), "主观评测：")#epoch可能是浮点数，用round取整
                
            result_text = f"【Q】{messages[1]['content']}\n【LLM】{messages[2]['content']}\n【GT】 {ground_truth}"
            print(result_text)
            
            test_text_list.append(swanlab.Text(result_text, caption=response))

        swanlab.log({"Prediction": test_text_list}, step=round(state.epoch))
        
        
#Swanlab监控回调
swanlab_callback = HuanhuanSwanLabCallback(
  #你的项目名称
    project="Qwen2.5-Coder-LoRA-Law",
  #你的实验名称
    experiment_name="7b",
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2.5-Coder-7B-Instruct",
        "dataset": "https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT",
        "github": "https://github.com/datawhalechina/self-llm",
        "system_prompt": "你是一个法律专家，请根据用户的问题给出专业的回答",
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
)


trainer = Trainer(
  #之前通过LoRA配置得到的适配后模型
    model=peft_model,
  #训练参数
    args=args,
    train_dataset=train_dataset,
  #数据整理器，用于将不同长度的序列填充到相同的长度，以便于批处理
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
  #在训练的不同阶段触发相应的回调方法
    callbacks=[swanlab_callback],
)

trainer.train()

# 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
swanlab.finish()
```

