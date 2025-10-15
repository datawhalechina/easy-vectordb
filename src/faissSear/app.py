# app.py
import os
import uuid
import numpy as np
import faiss
import time
from flask import Flask, request, jsonify, render_template
from modelscope import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局变量
embedding_pipeline = None
indices = {}
documents = {}
current_device = 'gpu'
performance_history = []

# 模型ID常量
MODEL_ID = "iic/nlp_gte_sentence-embedding_chinese-base"

# 初始化 embedding pipeline
def init_model():
    global embedding_pipeline
    if embedding_pipeline is None:
        print("加载 ModelScope 文本向量化模型...")
        try:
            embedding_pipeline = pipeline(
                Tasks.sentence_embedding,
                model=MODEL_ID,
                sequence_length=512,
                model_revision='master'
            )
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            # 创建备用嵌入器
            embedding_pipeline = DummyEmbedder()
    return embedding_pipeline

# 备用嵌入器
class DummyEmbedder:
    def __init__(self):
        self.dimension = 768
        
    def __call__(self, inputs):
        if isinstance(inputs, dict) and 'source_sentence' in inputs:
            texts = inputs['source_sentence']
            if isinstance(texts, str):
                texts = [texts]
            
            # 生成确定性随机向量
            embeddings = []
            for text in texts:
                np.random.seed(hash(text) % 10000)
                embedding = np.random.random(self.dimension).astype('float32')
                # 归一化
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                embeddings.append(embedding)
            
            return {'text_embedding': np.array(embeddings)}
        return {'text_embedding': np.random.random((1, self.dimension)).astype('float32')}

# 支持的索引类型
SUPPORTED_INDEXES = {
    'FlatL2': {
        'name': '精确搜索 (FlatL2)',
        'description': '暴力搜索，100%准确率，适合小数据集'
    },
    'IVFFlat': {
        'name': '倒排索引 (IVFFlat)',
        'description': '平衡精度和速度，适合中等数据集'
    },
    'IVFPQ': {
        'name': '乘积量化 (IVFPQ)',
        'description': '内存优化，适合大规模数据集'
    }
}

@app.route('/')
def index():
    return render_template('index.html', indexes=SUPPORTED_INDEXES)

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理多文件上传和向量化"""
    if 'files' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': '没有选择文件'}), 400
    
    valid_files = [file for file in files if file and allowed_file(file.filename)]
    if not valid_files:
        return jsonify({'error': '没有有效的文件'}), 400
    
    # 生成唯一ID
    file_id = str(uuid.uuid4())
    all_chunks = []
    all_vectors = []
    file_info = []
    
    try:
        for file in valid_files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
            file.save(filepath)
            
            # 处理单个文件
            chunks, vectors = process_file(filepath)
            
            # 记录文件信息
            file_info.append({
                'filename': filename,
                'chunk_count': len(chunks)
            })
            
            # 合并所有文件的块和向量
            all_chunks.extend(chunks)
            if len(all_vectors) == 0:
                all_vectors = vectors
            else:
                all_vectors = np.vstack([all_vectors, vectors])
            
            print(f"处理文件: {filename}, 块数: {len(chunks)}")
        
        # 存储合并后的文档块
        documents[file_id] = {
            'files': file_info,
            'chunks': all_chunks,
            'vectors': all_vectors
        }
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'files': file_info,
            'total_chunks': len(all_chunks),
            'vector_dim': all_vectors.shape[1]
        })
        
    except Exception as e:
        return jsonify({'error': f'处理文件失败: {str(e)}'}), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'txt', 'md'}

def process_file(filepath):
    """处理文件内容并生成向量"""
    # 读取文件
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # 简单的文本分块
    chunks = text_chunking(content)
    
    # 生成向量
    pipeline = init_model()
    
    # 批量处理以提高效率
    batch_size = 32
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_vectors = pipeline({"source_sentence": batch_chunks})['text_embedding']
        vectors.extend(batch_vectors)
    
    return chunks, np.array(vectors)

def text_chunking(text, chunk_size=250, overlap=50):
    """将文本分割成块"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        
        # 尝试在句子边界分割
        while end > start and text[end] not in {'.', '!', '?', '\n', ' '}:
            end -= 1
        
        if end == start:  # 没有找到合适的分割点
            end = start + chunk_size
        
        chunk = text[start:end].strip()
        if chunk:  # 只添加非空块
            chunks.append(chunk)
        
        start = end - overlap if end - overlap > start else end
    
    return chunks

@app.route('/build_index', methods=['POST'])
def build_index():
    """构建Faiss索引"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '无效的JSON数据'}), 400
            
        file_id = data.get('file_id')
        index_type = data.get('index_type')
        device = data.get('device', 'cpu')
        
        if not file_id or file_id not in documents:
            return jsonify({'error': '文件ID不存在'}), 400
        
        if index_type not in SUPPORTED_INDEXES:
            return jsonify({'error': '不支持的索引类型'}), 400
        
        global current_device
        current_device = device
        
        # 获取向量
        vectors = documents[file_id]['vectors']
        dimension = vectors.shape[1]
        n_vectors = vectors.shape[0]
        # 根据设备选择资源
        if device == 'gpu' and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
        else:
            res = None
        
        # 创建索引
        start_time = time.perf_counter()
        index = create_faiss_index(index_type, dimension, res, device,n_vectors)
        
        # 训练索引 (如果需要)
        if hasattr(index, 'is_trained') and not index.is_trained:
            print("训练索引...")
            index.train(vectors)
        
        # 添加向量
        index.add(vectors)
        build_time = time.perf_counter() - start_time
        
        # 存储索引
        indices[file_id] = {
            'index': index,
            'index_type': index_type,
            'device': device
        }
        
        return jsonify({
            'success': True,
            'index_type': index_type,
            'device': device,
            'vector_count': index.ntotal,
            'build_time': round(build_time, 4)
        })
        
    except Exception as e:
        return jsonify({'error': f'构建索引失败: {str(e)}'}), 500

def create_faiss_index(index_type, dimension, res, device,n_vectors):
    """创建指定类型的Faiss索引"""
    if index_type == 'FlatL2':
        cpu_index = faiss.IndexFlatL2(dimension)
        if device == 'gpu' and res is not None:
            return faiss.index_cpu_to_gpu(res, 0, cpu_index)
        return cpu_index
        
    elif index_type == 'IVFFlat':
        nlist = min(100, max(1, n_vectors // 10))  # 取样本数量的1/10，但不超100
        quantizer = faiss.IndexFlatL2(dimension)
        cpu_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        if device == 'gpu' and res is not None:
            return faiss.GpuIndexIVFFlat(res, dimension, nlist, faiss.METRIC_L2)
        return cpu_index
        
    elif index_type == 'IVFPQ':
        nlist = min(100, max(1, n_vectors // 10))
        m = min(8,dimension)  # 子量化器数量
        bits = 8  # 每个量化器的比特数
        quantizer = faiss.IndexFlatL2(dimension)
        cpu_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits)
        
        if device == 'gpu' and res is not None:
            config = faiss.GpuIndexIVFPQConfig()
            return faiss.GpuIndexIVFPQ(res, dimension, nlist, m, bits, faiss.METRIC_L2, config)
        return cpu_index
    
    else:
        raise ValueError(f"不支持的索引类型: {index_type}")

@app.route('/search', methods=['POST'])
def search():
    """执行向量搜索"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '无效的JSON数据'}), 400
            
        file_id = data.get('file_id')
        query = data.get('query')
        top_k = data.get('top_k', 5)
        
        if not file_id or file_id not in indices:
            return jsonify({'error': '请先构建索引'}), 400
        
        if not query or not query.strip():
            return jsonify({'error': '查询不能为空'}), 400
        
        # 生成查询向量
        pipeline = init_model()
        
        embed_start = time.perf_counter()
        query_vector = np.array([pipeline({"source_sentence": [query]})['text_embedding'][0]])
        embed_time = time.perf_counter() - embed_start
        
        # 获取索引
        index_info = indices[file_id]
        index = index_info['index']
        
        # 设置搜索参数 (对于IVF索引)
        if hasattr(index, 'nprobe'):
            index.nprobe = min(10, getattr(index, 'nlist', 100) // 4)
        
        # 执行搜索 (多次运行取平均值以获得更准确的时间)
        search_times = []
        for _ in range(3):  # 运行3次取平均
            search_start = time.perf_counter()
            distances, indices_result = index.search(query_vector, top_k)
            search_times.append(time.perf_counter() - search_start)
        
        search_time = min(search_times)  # 取最快的一次，避免系统干扰
        
        total_time = embed_time + search_time
        
        # 获取相关文档块
        results = []
        doc_chunks = documents[file_id]['chunks']
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices_result[0])):
            if 0 <= idx < len(doc_chunks):
                results.append({
                    'rank': i + 1,
                    'score': float(1 / (1 + distance)),
                    'content': doc_chunks[idx],
                    'filename': get_filename_by_chunk_index(file_id,idx),
                    'distance': float(distance)
                })
        
        # 记录性能数据
        performance_data = {
            'timestamp': time.time(),
            'index_type': index_info['index_type'],
            'device': index_info['device'],
            'embed_time': round(embed_time * 1000, 4),  # Convert to milliseconds
            'search_time': round(search_time * 1000, 4),  # Convert to milliseconds
            'total_time': round(total_time * 1000, 4),  # Convert to milliseconds
            'query_length': len(query),
            'results_count': len(results)
        }
        performance_history.append(performance_data)
        
        # 只保留最近50条记录
        if len(performance_history) > 50:
            performance_history.pop(0)
        
        return jsonify({
            'success': True,
            'results': results,
            'index_type': index_info['index_type'],
            'device': index_info['device'],
            'timing': {
                'embedding': round(embed_time * 1000, 4),  # milliseconds
                'search': round(search_time * 1000, 4),  # milliseconds
                'total': round(total_time * 1000, 4)  # milliseconds
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'搜索失败: {str(e)}'}), 500

def get_filename_by_chunk_index(file_id, chunk_index):
    """根据块索引找到对应的文件名"""
    if file_id not in documents:
        return "未知文件"
    
    file_info = documents[file_id]['files']
    
    current_index = 0
    for file_data in file_info:
        chunk_count = file_data['chunk_count']
        if current_index <= chunk_index < current_index + chunk_count:
            return file_data['filename']
        current_index += chunk_count
    
    return "未知文件"
        
    
@app.route('/performance_stats')
def performance_stats():
    """获取性能统计"""
    if not performance_history:
        return jsonify({'message': '暂无性能数据'})
    
    # 按索引类型和设备分组统计
    stats = {}
    for record in performance_history:
        key = f"{record['index_type']}_{record['device']}"
        if key not in stats:
            stats[key] = {
                'index_type': record['index_type'],
                'device': record['device'],
                'count': 0,
                'total_search_time': 0,
                'total_embed_time': 0,
                'search_times': [],
                'embed_times': []
            }
        
        stats[key]['count'] += 1
        stats[key]['total_search_time'] += record['search_time']
        stats[key]['total_embed_time'] += record['embed_time']
        stats[key]['search_times'].append(record['search_time'])
        stats[key]['embed_times'].append(record['embed_time'])
    
    # 计算平均值和最佳值
    for key in stats:
        stats[key]['avg_search_time'] = stats[key]['total_search_time'] / stats[key]['count']
        stats[key]['avg_embed_time'] = stats[key]['total_embed_time'] / stats[key]['count']
        stats[key]['best_search_time'] = min(stats[key]['search_times'])
        stats[key]['best_embed_time'] = min(stats[key]['embed_times'])
    
    return jsonify({
        'stats': list(stats.values()),
        'recent': performance_history[-10:]  # 最近10次搜索
    })

@app.route('/system_info')
def system_info():
    """获取系统信息"""
    gpu_available = torch.cuda.is_available()
    gpu_info = {}
    
    if gpu_available:
        gpu_info = {
            'device_name': torch.cuda.get_device_name(0),
            'device_count': torch.cuda.device_count(),
            'memory': torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    import pkg_resources
    faiss_version = pkg_resources.get_distribution("faiss-gpu-cu12").version
# 或者如果是gpu版本
# faiss_version = pkg_resources.get_distribution("faiss-gpu").version
    return jsonify({
        'gpu_available': gpu_available,
        'gpu_info': gpu_info,
        'faiss_version': faiss_version,
        'current_device': current_device,
        'performance_records': len(performance_history)
    })

if __name__ == '__main__':
    print("初始化Faiss Web应用...")
    # 检查 GPU 支持
    if hasattr(faiss, 'StandardGpuResources'):
        print("GPU 版本可用")
    else:
        print("当前是 CPU 版本")
    init_model()
    print("应用启动完成，访问 http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)