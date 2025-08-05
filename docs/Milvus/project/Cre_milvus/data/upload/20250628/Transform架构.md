# Transform架构

> 图片来自[Happy-llm]([happy-llm/docs/chapter2/第二章 Transformer架构.md at main · datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter2/%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Transformer%E6%9E%B6%E6%9E%84.md))，若加载不出来，请开梯子

## 注意力机制

* 前馈神经网络

  > 每一层的神经元都与上下两层的每一个神经元完全连接![图片描述](https://haluki.oss-cn-hangzhou.aliyuncs.com/happyllm/1-0.png)
  >
  > 数据在其中只向前流动，用于处理静态的数据，进行图像识别或者分类，但是该网络没有记忆能力，数据在它里面没有循环。

- 卷积神经网络（Convolutional Neural Network，CNN），即训练参数量远小于前馈神经网络的卷积层来进行特征提取和学习

> ![图片描述](https://haluki.oss-cn-hangzhou.aliyuncs.com/happyllm/1-1.png)
>
> 只要用于处理具有网格结构的数据，比如时间序列或者图像数据，通过局部感知野（即每个神经元只需要“看到”输入数据的一部分）和权值共享（同一特征检测器可以用于整个输入空间）来减少参数数量，从而能够更高效地进行训练和预测。该网络适用于图像识别、语音信号处理

- 循环神经网络（Recurrent Neural Network，RNN），能够使用历史信息作为输入、包含环和自重复的网络

> ![图片描述](https://haluki.oss-cn-hangzhou.aliyuncs.com/happyllm/1-2.png)
>
> 网络中的节点按链式方式连接形成一个有向图，允许信息持久存在，可以用于语言模型，理解上下文的场景。

​	由于RNN在处理长序列时是顺序进行输入和计算，会有梯度消失或者爆炸的问题，还浪费了GPU的并行能力，Vaswani等人借鉴了计算机视觉（CV）领域提出的注意力机制，并创新地构建了一个完全基于注意力机制的神经网络——Transformer。

> **梯度消失、爆炸**
>
> * 梯度消失：在深层网络中，靠近输入层的权重由于梯度接近于0而难以得到有效的更新，是使得网络无法学习到有用的特征表示，尤其是那些自然语言处理（有上下文，长距离任务）
> * 梯度爆炸：指相反的情况，即梯度变得异常大，导致权重更新幅度巨大，使得网络不稳定，甚至可能导致数值溢出错误。训练过程不稳定，网络性能下降。

* Transformer允许更大的并行化处理，显著提高了计算效率。
* 通过自注意力机制（self-attention mechanism），Transformer能够更有效地捕捉到序列中任意位置间的依赖关系，不受限于序列长度。

###  **注意力机制**

​	核心思想是模型在处理每个位置的输入时，动态的关注到某些部分的信息，就像我们看照片，不可能全部细节都看，在自然语言处理中，我们可以将注意力集中在一个或者几个token上，给予不同元素不同的权重，从而获取到更加高质量的计算效果。

​	注意力有三个核心变量：**Query**（查询值）、**Key**（键值）和 **Value**（真值），

​	假设有如下简化的新闻报道：“开营会议马上召开，时间为2025年6月9日。”

- Query可能是“时间在哪里？”
- Keys将是每个词（”开营“，“会议”，“马上”，“召开”，“时间”，“为”，“2025年”，“6月”，“9日”）的向量表示。
- Values也是这些词的向量表示。

key可以视为标识符，用于和query进行比较，确定哪些部分与query最相关。value就是后续的输出。通过计算Query与所有Key之间的相似度或相关性来确定对每个词的关注程度，结果是一组权重，反映了从Query出发，对文本中每一个token应该分配多少注意力，**然后，利用这些权重对相应的Values进行加权求和，从而获得最终的输出结果。这意味着，那些与Query高度相关的部分（如包含时间信息的词）将对最终结果产生更大的影响。**

Happy-llm提到了如何计算注意力分数，使用词向量，进行点积运算，首先对query进行向量化，然后对key中的某一个进行向量化，然后进行点积运算，获取一个相似度值，然后选中key的下一个，继续这样操作，计算 Query 和每一个键的相似程度。

然后我们通过一个Softmax层将其转化为和为1的权重$$ \text{softmax}(x)*i = \frac{e^{xi}}{\sum*{j}e^{x_j}} $$

**这样，得到的向量就能够反映 Query 和每一个 Key 的相似程度，同时又相加权重为 1，也就是我们的注意力分数了。**

不过，此时的值还是一个标量，同时，我们此次只查询了一个 Query。我们可以将同时一次性查询多个 Query，同样将多个 Query 对应的词向量堆叠在一起形成矩阵 Q，这个矩阵的每一行都是一个query：

> 对于这个Q矩阵，比如句子“I love NLP”有3个词，就会有3个Query，分别对应每个词的向量。我们可以把这些Query堆叠成一个矩阵 QQ，一次性计算所有Query对应的注意力结果。这样就
>
> - 不需要一个一个地单独计算每个Query，节省大量时间。
> - 充分利用GPU的并行计算能力。
>
> 并且！模型可以关注到整个句子的各个词的相关性，对于长难句或者长距离依赖的处理更加好。

1. 对于每个查询，计算它与所有Key之间的相似度。这可以通过矩阵乘法实现，即 QKTQKT，其结果是一个矩阵，其中每个元素表示对应Query和Key之间的相似度。
2. **权重转换**：同样地，我们需要使用Softmax函数将这些相似度转换为权重。这里的Softmax是对每个查询分别应用的，以确保每组权重加起来等于1。
3. **加权求和**：最后，我们利用这些权重对Values进行加权求和，得到针对每个查询的输出。这一步也是通过矩阵乘法完成的。

**这样处理有什么用呢**，比如在语言处理场景下，你要把英文句子 “The cat is on the mat.” 翻译成中文，需要知道每个中文词应该“关注”英文句子中的哪个部分。比如，猫这个字就需要关注英文语句中的cat，垫子这个词就要关注mat。注意力机制就像让模型在处理一个目标词时，自动选择性关注源语言中最相关的部分。

### **自注意力**

​	自注意力机制的关键在于认识到它允许每个位置的token能够直接与其他所有位置的token进行交互，这与传统的RNN或者LSTM不同，RNN只能顺序的处理，浪费了GPU并行计算的能力。

​	在自注意力机制中，输入序列的每一个token都会生成三个向量：Q、K、V。这三个向量是通过原始向量分别乘以不同的权重矩阵而得到的。Q就是当前关注点的信息，K与Q进行相似度计算，找到相关性，V包含实际要传递的信息。

​	在Transformer的Encoder部分，对于输入序列中的每个token，我们使用相同的输入向量来计算Q、K、V，但通过不同的权重矩阵变换它们。这意味着：

$$Q=XWq $$

$$K=XWk$$

$$V=XWv$$

其中，X 是输入序列的嵌入表示（embedding），$$Wq$$、$$Wk$$和 $$Wv$$ 分别是学习到的参数矩阵。

​	通过自注意力机制，可以找到一段文本中每一个token与其他所有的token的相关关系大小，并根据它们的相关性调整自己的表示。

> 自注意力机制：
>
> * 提高了模型处理长距离依赖的关系
> * 使得模型结构更加适合并行化处理，提高了训练效率

### **掩码自注意力**

​	指的是使用掩码来遮蔽掉特定位置的token，主要控制哪些部分的信息可以被模型看到，哪些部分应该被忽略或者屏蔽，这对于确保模型不会“作弊”（幻觉，利用了没有看到的信息进行预测），有很好的作用。

​	在标准的自注意力机制中，每个token能够与序列中所有的其他token交互，包括哪些**在实际应用场景中未出现的未来token**。然而，在模型训练中，我们希望模型只能依赖于已见过的信息来做出预测，而不是未来的信息。这就需要使用掩码来阻止模型访问未来的token。

> 例如，对于句子“我喜欢猫。”
>
> 首先基于“我”预测出“喜欢”，然后基于“我喜欢”预测出“猫”。

​	在这个过程中，我们不希望模型提前知道喜欢和猫这俩词来预测，但是这样就是并行计算了，浪费了GPU。所以就有了掩码自注意力的方法，掩码自注意力机制会生成一串掩码，来屏蔽未来信息。例如，我们待学习的文本序列仍然是 【BOS】I like you【EOS】，我们使用的注意力掩码是【MASK】，那么模型的输入为：

```txt
<BOS> 【MASK】【MASK】【MASK】【MASK】
<BOS>    I   【MASK】 【MASK】【MASK】
<BOS>    I     like  【MASK】【MASK】
<BOS>    I     like    you  【MASK】
<BoS>    I     like    you   </EOS>
```

​	在每一行输入中，模型仍然是只看到前面的 token，预测下一个 token。但是注意，上述输入不再是串行的过程，而可以一起并行地输入到模型中，模型只需要每一个样本根据未被遮蔽的 token 来预测下一个 token 即可，从而实现了并行的语言模型。

> 下面的关于掩码的实现将直接引用表达非常完美的原文

在具体实现中，我们通过以下代码生成 Mask 矩阵：

```python
# 创建一个上三角矩阵，用于遮蔽未来信息。
# 先通过 full 函数创建一个 1 * seq_len * seq_len 的矩阵
mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
# triu 函数的功能是创建一个上三角矩阵
mask = torch.triu(mask, diagonal=1)
```

生成的 Mask 矩阵会是一个上三角矩阵，上三角位置的元素均为 -inf，其他位置的元素置为0。

在注意力计算时，我们会将计算得到的注意力分数与这个掩码做和，再进行 Softmax 操作：

```python
# 此处的 scores 为计算得到的注意力分数，mask 为上文生成的掩码矩阵
scores = scores + mask[:, :seqlen, :seqlen]
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```

通过做求和，上三角区域（也就是应该被遮蔽的 token 对应的位置）的注意力分数结果都变成了 `-inf`，而下三角区域的分数不变。再做 Softmax 操作，`-inf` 的值在经过 Softmax 之后会被置为 0，从而忽略了上三角区域计算的注意力分数，从而实现了注意力遮蔽。

### **多头注意力机制**

​	注意力机制在一段序列中，很难拟合全部的相关关系，所以有了多头注意力机制，即**对同一个语料进行多次注意力的计算**，每次注意力计算都能拟合不同的关系，将最后的多次结果拼接起来作为最后的输出，即可更全面深入地拟合语言信息。

​	事实上，所谓的多头注意力机制其实就是将原始的输入序列进行多组的自注意力处理；然后再将每一组得到的自注意力结果拼接起来，再通过一个线性层进行处理。

* 多头注意力允许模型从不同的角度理解和处理输入数据。每个头可以专注于不同类型的关系或特征，从而提供更丰富和多样化的信息表示。
* 由于不同头可以关注序列中的不同部分，因此它们能够更好地捕捉长距离依赖以及复杂的语法和语义结构。
* 模型能够更精确地捕捉细微的区别和关联。
* 看起来计算量很多，但每个的注意力计算都是独立的，充分利用了GPU的并行能力

## Encoder-Decoder（编码-解码器）

### Seq2Seq模型

​	Transformer 是一个经典的 Seq2Seq 模型，即模型的输入为文本序列，输出为另一个文本序列。例如，我们的输入可能是“今天天气真好”，输出是“Today is a good day.”。

​	对于 Seq2Seq 任务，一般的思路是对自然语言序列进行编码再解码。

> 编码：
>
> 将输入的自然语言序列通过隐藏层编码成能够表征语义的向量（或矩阵），可以简单理解为更复杂的词向量表示。
>
> 解码：
>
> 就是对输入的自然语言序列编码得到的向量或矩阵通过隐藏层输出，再解码成对应的自然语言目标序列。

![图片描述](https://haluki.oss-cn-hangzhou.aliyuncs.com/happyllm/2-0.jpg)

​	Transformer 由 Encoder 和 Decoder 组成，每一个 Encoder（Decoder）又由 6个 Encoder（Decoder）Layer 组成。输入源序列会进入 Encoder 进行编码，到 Encoder Layer 的最顶层再将编码结果输出给 Decoder Layer 的每一层，通过 Decoder 解码后就可以得到输出目标序列了。

### 前馈神经网络FFN

​	FNN：每一层的神经元都和上下两层的每一个神经元完全连接的网络结构。

​	每一个 Encoder Layer 都包含一个上文讲的注意力机制和一个前馈神经网络。

​	**FNN的作用是什么**

* **引入非线性**：尽管自注意力机制可以捕捉输入序列中不同位置之间的复杂依赖关系，但它本质上是一个线性操作（加上Softmax函数）。为了增强模型的学习能力和表达力，需要引入非线性变换。前馈神经网络通过使用激活函数（如ReLU或GELU），为模型引入了必要的非线性。
* **特征转换**：前馈神经网络允许模型学习更复杂的特征表示。具体来说，它由两层线性变换组成，中间夹有一个非线性激活函数。**这使得模型可以在不同的特征空间之间进行映射，从而可能发现原始输入中未直接显现的模式或结构** 。
* **维度变换**：在Transformer中，前馈神经网络通常会增加维度大小（即所谓的“扩展维度”）。例如，如果输入的隐藏层维度是512，那么前馈神经网络的第一个线性层可能会将其映射到一个更高的维度（比如2048），然后通过第二个线性层再映射回原来的维度（512）。这种维度上的先增后减有助于模型捕捉更丰富的信息。

### 层归一化Layer Normalization

​	归一化核心是为了**让不同层输入的取值范围或者分布能够比较一致**。由于深度神经网络中每一层的输入都是上一层的输出，因此**多层传递下，对网络中较高的层，之前的所有神经层的参数变化会导致其输入的分布发生较大的改变 **。各层的输出分布差异随着网络深度的增大而增大。但是，需要预测的条件分布始终是相同的，从而也就造成了预测的误差。**归一化操作可以帮助稳定每一层的输入分布，从而加速训练并提高模型的表现。** 

### 批归一化Batch Normalization

​	在每个mini-batch上进行归一化。

通过Layer Norm，可以有效解决Batch Norm的一些局限性：

* 当batch size较小时，计算出的均值和方差可能不能很好地代表整个数据集的分布。
* 由于RNN处理的是变长序列，不同时间步的分布可能差异很大，导致Batch Norm的效果不佳。
* 在训练过程中需要保存每一步的统计信息，在推理时则需要使用这些统计量，这对变长序列特别麻烦。

### 残差连接

​	随着神经网络层数的增加，传统的深层网络面临两个主要问题：**梯度消失/爆炸**和**退化问题**。这些问题限制了模型的深度和性能。

> 对于非常深的网络，如果每层的梯度都非常小或非常大，那么经过多层后，梯度可能会变得极小（梯度消失）或极大（梯度爆炸），从而导致训练不稳定甚至无法收敛。

> 即使解决了梯度消失/爆炸的问题，更深的网络并不总是表现得更好。在某些情况下，更深的网络可能比浅层网络表现更差。这是因为深层网络难以优化，即使有足够的训练时间，网络也可能陷入较差的局部最优解。

​	残差连接，即**下一层的输入不仅是上一层的输出，还包括上一层的输入**。

* 残差连接的出现，很有效的**保持了梯度的稳定性和强度，从而支持更深的网络训练** 。

### Encoder

​	Encoder由N个Encoder Layer组成，每一个Encoder Layer包括一个注意力层和一个前馈神经网络。

> 多头注意力机制用于捕获输入序列中不同token之间的依赖关系
>
> 前馈神经网络FNN用于提供一个非线性变化，增强模型表达能力

​	此外，在每个子层周围还应用了残差连接，并跟随一个层归一化。

> 残差连接在每个子层之后添加输入的直接连接，即`output = layer(input) + input`，这有助于缓解深层网络中的梯度消失问题。
>
> 层归一化，对每个样本的所有特征维度进行归一化处理，确保每一层的输入分布保持稳定。

### Decoder

​	与Encoder不同的是，Decoder是由两个注意力层和一个前馈神经网络组成。

1. 第一个注意力层是一个掩码自注意力层，使用Mask的注意力计算，保证每一个token只能使用该token之前的注意力分数。
2. 第二个注意力层是一个多头注意力层，使用第一个注意力层的输出作为query，使用Encoder的输出作为key和value来计算注意力分数。
3. 最后在经过前馈神经网络。

Encoder的主要任务是将输入序列转换为一个连续的表示形式，这个表示形式可以被后续的Decoder使用来生成输出序列。

* 首先接受原始的数据，将其转换为模型可以理解的形式
* 再通过多层的Encoder层，让模型捕捉输入序列中的复杂模式和依赖关系，每一层包括：
  * 多头注意力机制：每个位置的token都可以关注到其他各个位置的token，从而捕获全局信息
  * 前馈神经网络FNN：提供线性变化，增强模型的表达能力
  * 残差连接与层归一化：缓解梯度消失，稳定训练
* 输出一个与输入序列长度相同的序列，包含了输入序列的重要信息，可以被Decoder用来生成目标序列。

Decoder的任务是基于Encoder层提供的上下文信息生成输出序列Decoder通常从一个特殊的开始标记（如<BOS>）开始解码过程。

1. 对于每一个时间步，Decoder执行以下操作：
   - **多头自注意力机制**：类似于Encoder中的自注意力，但这里的Query来自于当前Decoder层的上一层，而Key和Value既可以来自同一层也可以来自Encoder的最后一层。这使得Decoder可以同时关注到之前生成的所有token以及整个输入序列的信息。
   - **编码器-解码器注意力机制**：除了自身的自注意力外，Decoder还会进行一次编码器-解码器之间的注意力计算，其中Query来自Decoder，而Key和Value则来自Encoder的输出。这种方式让Decoder能够“查看”输入序列中的所有信息，以指导其生成下一个token。
   - **前馈神经网络**：与Encoder类似，也包含了一个前馈神经网络来引入非线性变换。
   - **输出预测**：经过上述步骤后，Decoder会输出一个分布，从中选择最有可能的下一个token加入到已生成的序列中。
2. **终止条件**：当生成了特殊的结束标记（如<EOS>）或者达到了预设的最大序列长度时，解码过程结束。

## 搭建Transformer

### Embedding层

​	Embedding层的作用就是将这些符号转化为高维空间中的向量表示，这样就可以被神经网络处理了。

1. Embedding层内部实际上是一个可训练的权重矩阵，其形状为`(vocab_size, embedding_dim)`。这里的`vocab_size`是词汇表的大小，而`embedding_dim`是你希望每个词向量具有的维度。
2. 当给定一个包含整数索引的输入时，Embedding层会根据这些索引来查找对应的行（即词向量），并将其作为输出的一部分。
3. 对于输入中的每一个索引，Embedding层都会找到对应的词向量，并将这些词向量按顺序拼接起来形成最终的输出张量。

```python
import torch
import torch.nn as nn

# 假设我们的词汇表大小是4，嵌入维度是5
vocab_size = 4
embedding_dim = 5

# 创建一个Embedding层
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# 输入是一个形状为(batch_size, seq_len)的张量，这里我们只有一个样本，序列长度为3
input_indices = torch.tensor([[0, 1, 2]])  # 形状为(1, 3)

# 使用Embedding层进行转换
embedded_output = embedding_layer(input_indices)

print("Input Indices:\n", input_indices)
print("\nEmbedded Output Shape:", embedded_output.shape)
print("\nEmbedded Output:\n", embedded_output)
```

- `vocab_size=4` 表示词汇表中有4个不同的词。
- `embedding_dim=5` 表示每个词会被映射到一个5维的向量空间中。
- `input_indices` 是输入的索引，形状为 `(1, 3)`，表示有1个样本，序列长度为3，每个位置上的整数代表词汇表中的某个词。
- `embedded_output` 是经过Embedding层后的输出，形状为 `(1, 3, 5)`，即 `(batch_size, seq_len, embedding_dim)`。

```python
Input Indices:
 tensor([[0, 1, 2]])

Embedded Output Shape: torch.Size([1, 3, 5])

Embedded Output:
 tensor([[[-0.7986, -0.4993, -0.8005, -0.7126, -0.6747],
          [ 0.4580,  0.7640,  0.4576, -0.5646, -0.4797],
          [-0.6667,  0.3355,  0.3609, -0.4949,  0.6763]]], grad_fn=<EmbeddingBackward>)
```

### 位置编码

​	在注意力机制的计算过程中，**对于序列中的每一个 token，其他各个位置对其来说都是平等的，即“我喜欢你”和“你喜欢我”在注意力机制看来是完全相同的**，因此，为使用序列顺序信息，保留序列中的相对位置信息，Transformer 采用了位置编码机制，该机制也在之后被多种模型沿用。

​	位置编码，即根据序列中 token 的相对位置对其进行编码，再将位置编码加入词向量编码中。位置编码的方式有很多，Transformer 使用了正余弦函数来进行位置编码（绝对位置编码Sinusoidal），其编码方式为：

假设 $ pos $ 表示位置，$ i $ 表示维度索引，而 $ d_{model} $ 是模型的隐藏层维度大小。

对于偶数维度（$2i$）的位置编码使用正弦函数：

$$
 PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right) 
$$


对于奇数维度（$2i+1$）的位置编码使用余弦函数：

$$
 PE(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right) 
$$
这样的位置编码主要有两个好处：

1. 使 PE 能够适应比训练集里面所有句子更长的句子，假设训练集里面最长的句子是有 20 个单词，突然来了一个长度为 21 的句子，则使用公式计算的方法可以计算出第 21 位的 Embedding。
2. 可以让模型容易地计算出相对位置，对于固定长度的间距 k，PE(pos+k) 可以用 PE(pos) 计算得到。因为 Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B), Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B)。

我们也可以通过严谨的数学推导证明该编码方式的优越性。原始的 Transformer Embedding 可以表示为：

$$ \begin{equation}f(\cdots,\boldsymbol{x}_m,\cdots,\boldsymbol{x}_n,\cdots)=f(\cdots,\boldsymbol{x}_n,\cdots,\boldsymbol{x}_m,\cdots)\end{equation} $$

很明显，这样的函数是不具有不对称性的，也就是无法表征相对位置信息。我们想要得到这样一种编码方式：

f~(⋯,\boldsymbolxm,⋯,\boldsymbolxn,⋯)=f(⋯,\boldsymbolxm+\boldsymbolpm,⋯,\boldsymbolxn+\boldsymbolpn,⋯)

这里加上的 pm，$p_n$ 就是位置编码。接下来我们将 f(...,xm+pm,...,xn+pn) 在 m,n 两个位置上做泰勒展开：

$$ \begin{equation}\tilde{f}\approx f + \boldsymbol{p}_m^{\top} \frac{\partial f}{\partial \boldsymbol{x}_m} + \boldsymbol{p}_n^{\top} \frac{\partial f}{\partial \boldsymbol{x}_n} + \frac{1}{2}\boldsymbol{p}_m^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_m^2}\boldsymbol{p}_m + \frac{1}{2}\boldsymbol{p}_n^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_n^2}\boldsymbol{p}_n + \underbrace{\boldsymbol{p}_m^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_m \partial \boldsymbol{x}_n}\boldsymbol{p}*n}*{\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n}\end{equation} $$

可以看到第1项与位置无关，2～5项仅依赖单一位置，第6项（f 分别对 m、n 求偏导）与两个位置有关，所以我们希望第六项（$p_m^THp_n$）表达相对位置信息，即求一个函数 g 使得:

pmTHpn=g(m−n)

我们假设 H 是一个单位矩阵，则：

pmTHpn=pmTpn=⟨\boldsymbolpm,\boldsymbolpn⟩=g(m−n)

通过将向量 [x,y] 视为复数 x+yi，基于复数的运算法则构建方程:

⟨\boldsymbolpm,\boldsymbolpn⟩=Re[\boldsymbolpm\boldsymbolpn∗]

再假设存在复数 qm−n 使得：

$$ \begin{equation}\boldsymbol{p}_m \boldsymbol{p}*n^\* = \boldsymbol{q}*{m-n}\end{equation} $$

使用复数的指数形式求解这个方程，得到二维情形下位置编码的解：

\boldsymbolpm=eimθ⇔\boldsymbolpm=(cos⁡mθ sin⁡mθ)

由于内积满足线性叠加性，所以更高维的偶数维位置编码，我们可以表示为多个二维位置编码的组合：

$$ \begin{equation}\boldsymbol{p}*m = \begin{pmatrix}e^{\text{i}m\theta_0} \ e^{\text{i}m\theta_1} \ \vdots \ e^{\text{i}m\theta*{d/2-1}}\end{pmatrix}\quad\Leftrightarrow\quad \boldsymbol{p}*m=\begin{pmatrix}\cos m\theta_0 \ \sin m\theta_0 \ \cos m\theta_1 \ \sin m\theta_1 \ \vdots \ \cos m\theta*{d/2-1} \ \sin m\theta_{d/2-1} \end{pmatrix}\end{equation} $$

再取 θi=10000−2i/d（该形式可以使得随着|m−n|的增大，⟨pm,pn⟩有着趋于零的趋势，这一点可以通过对位置编码做积分来证明，而 base 取为 10000 是实验结果），就得到了上文的编码方式。

当 H 不是一个单位矩阵时，因为模型的 Embedding 层所形成的 d 维向量之间任意两个维度的相关性比较小，满足一定的解耦性，我们可以将其视作对角矩阵，那么使用上述编码：

$$ \begin{equation}\boldsymbol{p}*m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}n=\sum{i=1}^{d/2} \boldsymbol{\mathcal{H}}*{2i,2i} \cos m\theta_i \cos n\theta_i + \boldsymbol{\mathcal{H}}_{2i+1,2i+1} \sin m\theta_i \sin n\theta_i\end{equation} $$

通过积化和差：

$$ \begin{equation}\sum_{i=1}^{d/2} \frac{1}{2}\left(\boldsymbol{\mathcal{H}}*{2i,2i} + \boldsymbol{\mathcal{H}}*{2i+1,2i+1}\right) \cos (m-n)\theta_i + \frac{1}{2}\left(\boldsymbol{\mathcal{H}}*{2i,2i} - \boldsymbol{\mathcal{H}}*{2i+1,2i+1}\right) \cos (m+n)\theta_i \end{equation} $$

说明该编码仍然可以表示相对位置。

![图片描述](https://haluki.oss-cn-hangzhou.aliyuncs.com/happyllm/3-0.png)

基于上述原理，我们实现一个位置编码层：

```
class PositionalEncoding(nn.Module):
    '''位置编码模块'''

    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        self.dropout = nn.Dropout(p=args.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # 计算 theta
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
```

### 完整的Transformer

![图片描述](https://haluki.oss-cn-hangzhou.aliyuncs.com/happyllm/3-1.png)



```powershell
pip install torch torchvision torchaudio
pip install numpy
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 参数配置类
class Args:
    # 初始化参数类
    def __init__(self):
        # 词汇表大小
        self.vocab_size = 10000
        # 块大小
        self.block_size = 512
        # 嵌入维度
        self.n_embd = 768
        # dropout率
        self.dropout = 0.1
        # 层数
        self.n_layer = 6

# 辅助类定义
class PositionalEncoding(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 初始化dropout层
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, x):
        # 简化实现
        # 返回dropout后的输入
        return self.dropout(x)

class Encoder(nn.Module):
    # 定义一个编码器类，继承自nn.Module
    def __init__(self, args):
        # 初始化函数，接收一个参数args
        super().__init__()
        # 调用父类的初始化函数
        self.layers = nn.ModuleList([nn.Linear(args.n_embd, args.n_embd) for _ in range(args.n_layer)])
        
        # 创建一个ModuleList，其中包含n_layer个全连接层，输入和输出的维度都是args.n_embd
    def forward(self, x):
        # 定义前向传播函数，接收一个参数x
        for layer in self.layers:
            # 遍历layers中的每个层
            x = layer(x)
            # 将x传入当前层，得到新的x
        return x

class Decoder(nn.Module):
    # 定义解码器类，继承自nn.Module
    def __init__(self, args):
        # 初始化函数，接收参数args
        super().__init__()
        # 调用父类的初始化函数
        self.layers = nn.ModuleList([nn.Linear(args.n_embd, args.n_embd) for _ in range(args.n_layer)])
        
        # 定义一个nn.ModuleList，其中包含n_layer个nn.Linear层，每个层的输入和输出维度都是args.n_embd
    def forward(self, x, enc_out):
        # 定义前向传播函数，接收输入x和编码器输出enc_out
        for layer in self.layers:
            # 遍历layers中的每个层
            x = layer(x)
            # 将输入x通过当前层进行处理，得到新的输出x
        return x

# Transformer 模型
class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 检查参数是否为空
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        # 定义transformer模块
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.n_embd),
            wpe = PositionalEncoding(args),
            drop = nn.Dropout(args.dropout),
            encoder = Encoder(args),
            decoder = Decoder(args),
        ))
        # 定义语言模型头
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        # 初始化权重
        self.apply(self._init_weights)
        # 打印参数数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=False):
        # 计算参数数量
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # 如果不计算embedding参数，则减去embedding参数的数量
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        # 初始化权重
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        # 前向传播
        device = idx.device
        b, t = idx.size()
        # 检查序列长度是否超过最大长度
        assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"

        # 计算token嵌入
        tok_emb = self.transformer.wte(idx)
        # 计算位置嵌入
        pos_emb = self.transformer.wpe(tok_emb) 
        # 添加dropout
        x = self.transformer.drop(pos_emb)
        # 计算编码器输出
        enc_out = self.transformer.encoder(x)
        # 计算解码器输出
        x = self.transformer.decoder(x, enc_out)

        # 如果有目标，计算损失
        if targets is not None:
            # 计算logits
            logits = self.lm_head(x)
            # 计算损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 如果没有目标，计算logits
            logits = self.lm_head(x[:, [-1], :])
            # 没有损失
            loss = None

        # 返回logits和损失
        return logits, loss

# 测试运行
if __name__ == "__main__":
    args = Args()
    model = Transformer(args)
    
    # 创建测试输入
    input_ids = torch.randint(0, args.vocab_size, (2, args.block_size))
    targets = torch.randint(0, args.vocab_size, (2, args.block_size))
    
    # 前向传播
    logits, loss = model(input_ids, targets)
    print("Logits shape:", logits.shape)
    print("Loss:", loss)
```

注意，上述代码除去搭建了整个 Transformer 结构外，我们还额外实现了三个函数：

- get_num_params：用于统计模型的参数量
- _init_weights：用于对模型所有参数进行随机初始化
- forward：前向计算函数

