# Chapter 5 Milvus的AI应用开发：图像检索应用实战
本章节采用ipynb的方式进行操作，按照指示运行代码，手把手带你实现一个文搜图应用。

点击->[Text_search_pic代码](https://github.com/datawhalechina/easy-vectordb/blob/main/docs/Milvus/chapter5/1_build_text_image_search_engine.ipynb) 进行下载。

下面是Markdown格式的介绍
## 相关技术介绍

### Towhere

Towhee 是一个开源的 **多模态数据处理框架**，专注于高效生成非结构化数据（如文本、图像、音频、视频等）的向量表示（Embeddings），并支持构建端到端的 AI 流水线（Pipeline）。它旨在简化从原始数据到向量化表示再到实际应用（如搜索、推荐、问答系统）的开发流程，尤其适用于需要处理多模态数据的场景。

---

### **一、Towhee 的核心功能**
1. **多模态 Embedding 生成**  
   - 支持文本、图像、音频、视频等非结构化数据的向量化。
   - 内置丰富的预训练模型（如 BERT、CLIP、ViT、ResNet、Whisper 等），可直接调用。
   - 支持自定义模型集成，灵活适配业务需求。

2. **流水线（Pipeline）构建**  
   - 提供声明式 API，通过链式调用快速组合数据处理步骤（如数据加载、预处理、模型推理、后处理等）。
   - 示例：一个图像搜索流水线可以包含 `图像解码 → 特征提取 → 向量归一化 → 存储到向量数据库`。

3. **高性能与可扩展性**  
   - 支持批量处理（Batch Processing）和 GPU 加速。
   - 分布式计算能力，适合大规模数据处理。
   - 通过算子（Operator）机制，可灵活扩展新功能。

4. **与向量数据库无缝集成**  
   - 深度兼容 Milvus、Elasticsearch、FAISS 等向量数据库，简化数据存储与检索流程。

---

## 准备
确保系统有GPU（可以使用魔搭社区提供的NoteBook），并且python版本为3.10，当前不支持python3.12

### 下载依赖

```python 

import subprocess
import sys
import os

def install_package(package_name, version=None, use_mirror=True):
    try:
        if version:
            package = f"{package_name}{version}"
        else:
            package = package_name
        
        print(f"正在安装 {package}...")
        cmd = [sys.executable, "-m", "pip", "install", package]
        if use_mirror:
            cmd += ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple/", "--trusted-host", "pypi.tuna.tsinghua.edu.cn"]
        cmd.append("--quiet")
        
        subprocess.check_call(cmd)
        print(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package_name} 安装失败: {e}")
        return False

def check_and_install_dependencies():    
    required_packages = [
        ("torch", ">=2.0.0"),
        ("torchvision", None),
        ("transformers", ">=4.21.0"),
        ("open_clip_torch", None),
        ("Pillow", None),
        ("opencv-python", None),
        ("pandas", None),
        ("numpy", None),
        ("gradio", ">=4.0.0"),
        ("scikit-learn", None)
    ]

    for package_name, version in required_packages:
        import_name = package_name
        if package_name == "Pillow":
            import_name = "PIL"
        elif package_name == "open_clip_torch":
            import_name = "open_clip"
        elif package_name == "opencv-python":
            import_name = "cv2"
        else:
            import_name = package_name.replace('-', '_').split('[')[0]

        try:
            __import__(import_name)
            print(f"✅ {package_name} 已安装")
        except ImportError:
            print(f"❌ {package_name} 未安装，正在安装...")
            install_package(package_name, version, use_mirror=True)
check_and_install_dependencies()
```

### 环境检查

在开始之前，让我们检查一下运行环境：
- Python 版本应该是 3.8 或更高
- 如果有 GPU，确保 CUDA 可用
- 确保有足够的磁盘空间下载模型和数据

### 准备数据
数据集包含100个图像类别，每个类别中包含10张图片。数据集可通过Github下载： [Github](https://github.com/towhee-io/examples/releases/download/data/reverse_image_search.zip). 

数据集包含如下三个部分：
- **train**: 候选图片目录;
- **test**: 测试图片目录;
- **reverse_image_search.csv**: csv文件，每张图片包含： ***id***, ***path***,  ***label*** ;

```python 
import os
import urllib.request
import zipfile
from pathlib import Path

def download_dataset():
    """下载并解压数据集"""
    
    # 检查必要的文件是否存在
    required_files = ['reverse_image_search.csv']
    required_dirs = ['train', 'test']
    
    all_exist = all(os.path.exists(f) for f in required_files) and all(os.path.exists(d) for d in required_dirs)
    
    if all_exist:
        print("✅ 数据集已存在")
        return True
    
    print("📥 开始下载数据集...")
    
    # 尝试多个下载源
    download_urls = [
        "https://github.com/towhee-io/examples/releases/download/data/reverse_image_search.zip",
    ]
    
    for i, url in enumerate(download_urls):
        try:
            print(f"尝试从源 {i+1} 下载: {url}")
            
            # 下载文件
            urllib.request.urlretrieve(url, "reverse_image_search.zip")
            print("下载完成，正在解压...")
            
            # 解压文件
            with zipfile.ZipFile("reverse_image_search.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # 清理压缩文件
            os.remove("reverse_image_search.zip")
            
            print("✅ 数据集下载并解压完成")
            return True
            
        except Exception as e:
            print(f"❌ 从源 {i+1} 下载失败: {e}")
            if i < len(download_urls) - 1:
                print("尝试下一个下载源...")
            continue
    
    print("❌ 所有下载源都失败了")
    print("\n📋 手动下载说明:")
    print("1. 访问: https://github.com/towhee-io/examples/releases/download/data/reverse_image_search.zip")
    print("2. 下载 reverse_image_search.zip")
    print("3. 解压到当前目录")

    return False

# 执行数据集下载
download_success = download_dataset()

if download_success:
    # 检查数据集内容
    import pandas as pd
    
    try:
        df = pd.read_csv('reverse_image_search.csv')
        print(f"\n📊 数据集信息:")
        print(f"- 总图片数量: {len(df)}")
        print(f"- 图片类别数: {df['label'].nunique()}")
        print(f"- 数据列: {list(df.columns)}")
        
        # 显示前几行数据
        print("\n📋 数据样例:")
        print(df.head())
        
    except Exception as e:
        print(f"❌ 读取数据集失败: {e}")
else:
    print("⚠️  请手动下载数据集后再继续")
```

```python 
 # 解压文件
with zipfile.ZipFile("reverse_image_search.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

# 清理压缩文件
# os.remove("reverse_image_search.zip")
```


### 数据集结构说明

我们使用的数据集包含以下结构：

```
reverse_image_search/
├── reverse_image_search.csv  # 图片索引文件
├── train/                    # 训练图片目录
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/                     # 测试图片目录
    ├── class1/
    ├── class2/
    └── ...
```

- **CSV文件**: 包含每张图片的ID、路径和标签信息
- **图片目录**: 按类别组织的图片文件
- **总计**: 约1000张图片，100个类别，每类10张图片


```python 
# 验证数据集完整性
import pandas as pd
import os
from pathlib import Path

def validate_dataset():
    """验证数据集的完整性"""
    
    if not os.path.exists('reverse_image_search.csv'):
        print("❌ CSV文件不存在，请先下载数据集")
        return False
    
    # 读取CSV文件
    df = pd.read_csv('reverse_image_search.csv')
    print(f"📊 CSV文件包含 {len(df)} 条记录")
    
    # 检查图片文件是否存在
    missing_files = []
    existing_files = 0
    
    for idx, row in df.iterrows():
        if os.path.exists(row['path']):
            existing_files += 1
        else:
            missing_files.append(row['path'])
        
        # 只检查前100个文件以节省时间
        if idx >= 100:
            break
    
    print(f"✅ 检查了前100个文件，{existing_files}个存在")
    
    if missing_files:
        print(f"⚠️  发现 {len(missing_files)} 个缺失文件")
        print("前几个缺失文件:", missing_files[:5])
    
    return len(missing_files) == 0

# 执行验证
is_valid = validate_dataset()

if is_valid:
    print("\n✅ 数据集验证通过，可以继续下一步")
    
    # 显示数据样例
    df = pd.read_csv('reverse_image_search.csv')
    print("\n📋 数据集前5行:")
    display(df.head())
    
    print(f"\n📈 数据集统计:")
    print(f"- 总记录数: {len(df)}")
    print(f"- 唯一标签数: {df['label'].nunique()}")
    print(f"- 标签分布:")
    print(df['label'].value_counts().head(10))
else:
    print("\n❌ 数据集验证失败，请检查数据完整性")
```

下面的fuction是作为text-image search的辅助
- **read_images(results)**: 通过图片ID读入图片，返回图片列表;


```python 
# 图像处理辅助函数
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 创建图片ID到路径的映射
if 'df' in locals():
    id_img = df.set_index('id')['path'].to_dict()
    print(f"✅ 创建了 {len(id_img)} 个图片的ID映射")
else:
    print("❌ 请先运行数据集验证代码")

def read_images_from_ids(image_ids):
    """根据图片ID列表读取图片"""
    images = []
    valid_paths = []
    
    for img_id in image_ids:
        if img_id in id_img:
            path = id_img[img_id]
            if os.path.exists(path):
                try:
                    # 使用PIL读取图片
                    img = Image.open(path).convert('RGB')
                    images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"⚠️  读取图片失败 {path}: {e}")
            else:
                print(f"⚠️  图片文件不存在: {path}")
        else:
            print(f"⚠️  图片ID不存在: {img_id}")
    
    return images, valid_paths

def display_images(images, titles=None, max_images=5):
    """显示图片列表"""
    if not images:
        print("没有图片可显示")
        return
    
    n_images = min(len(images), max_images)
    fig, axes = plt.subplots(1, n_images, figsize=(15, 3))
    
    if n_images == 1:
        axes = [axes]
    
    for i in range(n_images):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=10)
    
    plt.tight_layout()
    plt.show()

# 测试图片读取功能
if 'df' in locals() and len(df) > 0:
    print("\n🧪 测试图片读取功能...")
    test_ids = df['id'].head(3).tolist()
    test_images, test_paths = read_images_from_ids(test_ids)
    
    if test_images:
        print(f"✅ 成功读取 {len(test_images)} 张测试图片")
        display_images(test_images, [f"ID: {test_ids[i]}" for i in range(len(test_images))])
    else:
        print("❌ 测试图片读取失败")
```
### 创建Milvus链接

为了防止版本冲突情况，确保grpcio的版本限制在如下的范围内，下面还引入了Milvus，是因为源码中没有启动Milvus，所以需要手动安装milvus然后启动milvus服务

```python
import subprocess
import sys

# 安装 Milvus 相关依赖
try:
    print("正在安装 Milvus 依赖...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "grpcio>=1.49.1,<=1.53.0", "pymilvus", "milvus"])
    print("Milvus 依赖安装完成 ✓")
except Exception as e:
    print(f"安装失败: {e}")
    print("如果遇到版本冲突，请先卸载 pymilvus: pip uninstall pymilvus -y")
    print("然后重新安装: pip install pymilvus")
```
如果你已经安装了pymilvus导致了版本冲突问题，请运行如下代码，重新安装pymilvus

```shell
! pip uninstall pymilvus -y
```
现在创建一个 `text_image_search` 的milvus collection，使用 [L2 distance metric](https://milvus.io/docs/metric.md#Euclidean-distance-L2) 和 [IVF_FLAT index](https://milvus.io/docs/index.md#IVF_FLAT)索引.

```python
from milvus import default_server  
default_server.start()  
```

```python 
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

def create_milvus_collection(collection_name, dim):
    connections.connect("default",host='localhost', port='19530')
    
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
    FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', is_primary=True, auto_id=False),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='text image search')
    collection = Collection(name=collection_name, schema=schema)

    # 为集合创建 IVF_FLAT 索引.
    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":512}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

# collection = create_milvus_collection('text_image_search', 512)
```
## Text Image Search

使用 [Towhee](https://towhee.io/), 建立一个文本图像搜索引擎。

### 使用CLIP模型对文本和图片进行向量化

使用 [CLIP](https://openai.com/blog/clip/) 提取图像或文本的特征，该模型能够通过联合训练图像编码器和文本编码器来最大化余弦相似度，从而生成文本和图像的嵌入表示。

```shell
! pip install towhee
```


```python
from towhee import ops, pipe, DataCollection
import numpy as np
```

### 从魔搭社区下载模型
下面的两段代码是从魔搭社区下载模型，建议自己手动下载clip-vit-base-patch16，放到model文件夹下


```python
import os
import subprocess
import sys

# 检查模型是否已存在
model_path = "./model"
if not os.path.exists(model_path) or not os.listdir(model_path):
    print("正在下载 CLIP 模型...")
    try:
        # 安装 modelscope
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
        
        # 下载模型
        subprocess.check_call([
            "modelscope", "download", 
            "--model", "openai-mirror/clip-vit-base-patch16", 
            "--local_dir", model_path
        ])
        print("模型下载完成 ✓")
        
    except Exception as e:
        print(f"模型下载失败: {e}")
        print("请手动下载 clip-vit-base-patch16 模型到 ./model 文件夹")
        print("或者使用 Hugging Face 模型: openai/clip-vit-base-patch16")
else:
    print("模型已存在 ✓")
```

```shell
! pip install safetensors
```

```python 
from transformers import CLIPModel

# 直接使用 safetensors 加载（不需要 torch >= 2.6）
model = CLIPModel.from_pretrained("./model", use_safetensors=True)

# 保存为新的 safetensors 模型目录
model.save_pretrained("./model-safetensors", safe_serialization=True)
```
```python
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from towhee import register, ops, pipe  #
from towhee.operator import PyOperator
from pydantic import ConfigDict

@register
class CustomClipOperator(PyOperator):
    model_config = ConfigDict(protected_namespaces=())
    
    def __init__(self, model_path='./model'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        
    def __call__(self, img):
        inputs = self.processor(images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.get_image_features(**inputs)
        return outputs.cpu().detach().numpy()[0]

p = (
    pipe.input('path')
    .map('path', 'img', ops.image_decode.cv2('rgb'))
    .map('img', 'vec', CustomClipOperator(model_path='./model'))
    .map('vec', 'vec', lambda x: x / np.linalg.norm(x))
    .output('img', 'vec')
)
```


```python

# 检查 PyTorch 版本
import torch
print(torch.__version__)  # 应该 >= 2.6.0

# 检查模型加载
from transformers import CLIPModel
model = CLIPModel.from_pretrained('./model')
print("模型加载成功！")
```

```shell
! pip install torch>=2.6 --upgrade
```

```python
p2 = (
    pipe.input('text')
    .map('text', 'vec', ops.image_text_embedding.clip(model_name='model', modality='text'))
    .map('vec', 'vec', lambda x: x / np.linalg.norm(x))
    .output('text', 'vec')
)

DataCollection(p2("A teddybear on a skateboard in Times Square.")).show()
```
下面是代码释意:

- `map('path', 'img', ops.image_decode.cv2_rgb('rgb'))`: 对于数据的每一行, 读取并且decode `path`下的数据然后放到 `img`中;

- `map('img', 'vec', ops.image_text_embedding.clip(model_name='model', modality='image'/'text'))`：使用 `ops.image_text_embedding.clip` 提取图像或文本的嵌入特征，该操作符来自 [Towhee hub](https://towhee.io/image-text-embedding/clip)。此操作符支持多种模型，包括 `clip_vit_base_patch16`、`clip_vit_base_patch32`、`clip_vit_large_patch14`、`clip_vit_large_patch14_336` 等。


### 将图片向量数据导入Milvus中

我们首先将已经由 `clip_vit_base_patch16` 模型处理好的图片向量化数据插入Milvus中用于后面的检索。 Towhee 提供了[method-chaining style API](https://towhee.readthedocs.io/en/main/index.html) 因此，用户可以使用这些操作符组装一个数据处理管道。这意味着用户可以根据自己的需求，将不同的操作符（如图像和文本嵌入提取操作符）组合起来，创建复杂的数据处理流程，以实现特定的功能或任务。例如，在图像检索、文本匹配或其他涉及多模态数据处理的应用场景中，通过这种方式可以灵活地构建解决方案。

```python
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from towhee import pipe, ops, DataCollection, register
from towhee.operator import PyOperator
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
import time
import csv
import os

# ==============================
# Step 0: 确保 Milvus 已连接并创建集合（先运行一次）
# ==============================


# # 创建集合
collection = create_milvus_collection('text_image_search', 512)
# collection.load()  # 加载到内存


# ==============================
# Step 1: 读取 CSV 文件
# ==============================
def read_csv(csv_path, encoding='utf-8-sig'):
    with open(csv_path, 'r', encoding=encoding) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield int(row['id']), row['path']


# ==============================
# Step 2: 自定义 CLIP 编码 Operator
# ==============================
@register(name='custom_clip_encoder')
class CustomClipOperator(PyOperator):
    def __init__(self, model_path='./model', device=0):
        self.device = "cuda" if device >= 0 and torch.cuda.is_available() else "cpu"
        print(f"Loading model on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.eval()

    def __call__(self, img_path: str):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        import cv2
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("cv2 could not decode image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        vec = outputs.cpu().numpy()[0]
        return vec / np.linalg.norm(vec)


# ==============================
# Step 3: 构建 Towhee Pipeline
# ==============================
p3 = (
    pipe.input('csv_file')  # 输入字段名
         .flat_map('csv_file', ('id', 'path'), read_csv)
         .map('path', 'vec', CustomClipOperator(model_path='./model', device=0))
         .output('id', 'vec')  # 输出 id 和向量
)

# ==============================
# Step 4: 执行管道并插入 Milvus
# ==============================
start = time.time()
success_count = 0

try:
    # 使用 DataCollection 来处理管道结果
    dc = DataCollection(p3('./reverse_image_search.csv'))
    
    batch_ids = []
    batch_vecs = []

    for result in dc:
        if result is None:
            continue
        id_val = result['id']
        vec = result['vec']

        batch_ids.append(id_val)
        batch_vecs.append(vec)

        # 批量插入（提升性能）
        if len(batch_ids) >= 100:  # 每 100 条插入一次
            collection.insert([batch_ids, batch_vecs])
            success_count += len(batch_ids)
            print(f"Inserted batch of {len(batch_ids)} records.")
            batch_ids, batch_vecs = [], []

    # 插入剩余数据
    if batch_ids:
        collection.insert([batch_ids, batch_vecs])
        success_count += len(batch_ids)
        print(f"Inserted final batch of {len(batch_ids)} records.")
        

except Exception as e:
    print("Pipeline execution error:", str(e))
    import traceback
    traceback.print_exc()
# ==============================
# Step 5: 统计结果
# ==============================
print(f"插入完成! 成功: {success_count} 条记录")
print(f"耗时: {time.time() - start:.2f} 秒")
print(f"集合中的实体数量: {collection.num_entities}")
```

```python
collection.flush()

collection.load()

print('Total number of inserted data is {}.'.format(collection.num_entities))
```

### 开始向量化检索

现在，候选图像的嵌入向量已经插入到 Milvus 中，我们可以对其进行最近邻查询。同样，我们使用 Towhee 来加载输入文本、计算嵌入向量，并将该向量作为 Milvus 的查询条件。由于 Milvus 仅返回图像 ID 和距离值，我们提供了一个 `read_images` 函数，根据 ID 获取原始图像并进行展示。

```python
import pandas as pd
import cv2

def read_image(image_ids):
    df = pd.read_csv('reverse_image_search.csv')
    id_img = df.set_index('id')['path'].to_dict()
    imgs = []
    decode = ops.image_decode.cv2('rgb')
    for image_id in image_ids:
        path = id_img[image_id]
        imgs.append(decode(path))
    return imgs


p4 = (
    pipe.input('text')
    .map('text', 'vec', ops.image_text_embedding.clip(model_name='model', modality='text'))
    .map('vec', 'vec', lambda x: x / np.linalg.norm(x))
    .map('vec', 'result', ops.ann_search.milvus_client(host='127.0.0.1', port='19530', collection_name='text_image_search', limit=5))
    .map('result', 'image_ids', lambda x: [item[0] for item in x])
    .map('image_ids', 'images', read_image)
    .output('text', 'images')
)

DataCollection(p4("A white dog")).show()
DataCollection(p4("A black dog")).show()
```
## 使用Gradio构建一个应用
```python
search_pipeline = (
    pipe.input('text')
    .map('text', 'vec', ops.image_text_embedding.clip(model_name='model', modality='text'))
    .map('vec', 'vec', lambda x: x / np.linalg.norm(x))
    .map('vec', 'result', ops.ann_search.milvus_client(host='127.0.0.1', port='19530', collection_name='text_image_search', limit=5))
    .map('result', 'image_ids', lambda x: [item[0] for item in x])
    .output('image_ids')
)

def search(text):
    df = pd.read_csv('reverse_image_search.csv')
    id_img = df.set_index('id')['path'].to_dict()
    imgs = []
    image_ids = search_pipeline(text).to_list()[0][0]
    return [id_img[image_id] for image_id in image_ids]

```

在高版本的gradio中，已经不支持gradio.inputs.xxx和gradio.outputs.xxx，可直接使用gradio.TextBox或者gradio.Image
你可以使用如下代码更新一下你的gradio

```shell
! pip install --upgrade gradio
```

```python
import gradio

interface = gradio.Interface(search, 
                             gradio.Textbox(lines=1),
                             [gradio.Image(type="filepath", label=None) for _ in range(5)]
                            )
# 记得搜索的时候用英文！比如我要搜索蓝色的天空，那我就输入blue sky
interface.launch(inline=True, share=True)
```



