import csv
import numpy as np
import os
from towhee import pipe, ops
from System.monitor import log_event
import traceback

def process_csv(csv_path):
    """
    处理 CSV 文件中的图片数据并返回图片 ID 和向量数据的列表。

    参数:
        csv_path (str): CSV 文件路径。
    返回:
        list: 包含图片 ID 和向量数据的列表，格式为 [(id, vector), ...]。
    """
    try:
        # 验证文件存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        
        # 检查文件扩展名
        if not csv_path.endswith('.csv'):
            log_event(f"警告: 文件 {csv_path} 不是CSV文件")
            return []
        
        log_event(f"开始处理CSV文件: {csv_path}")

        # 定义 CSV 文件读取逻辑
        def read_csv(csv_file):
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    try:
                        # 获取ID和图片路径
                        img_id = int(line['id'])
                        img_path = line['path']
                        
                        # 验证图片路径存在
                        if not os.path.exists(img_path):
                            log_event(f"警告: 图片路径不存在: {img_path}")
                            continue
                            
                        yield img_id, img_path
                    except KeyError as e:
                        log_event(f"CSV行缺少必要字段: {line}, 错误: {e}")
                    except ValueError as e:
                        log_event(f"ID格式错误: {line.get('id', '')}, 错误: {e}")

        # 定义处理管道
        results = []

        def collect_results(id, vec):
            results.append((id, vec))

        # 创建处理管道
        p3 = (
            pipe.input('csv_file')
            .flat_map('csv_file', ('id', 'path'), read_csv)
            .map('path', 'img', ops.image_decode.cv2('rgb'))
            .map('img', 'vec', ops.image_text_embedding.clip(
                model_name='clip-vit-base-patch16', 
                modality='image', 
                device=0))
            .map('vec', 'vec', lambda x: x / np.linalg.norm(x))
            .map(('id', 'vec'), (), collect_results)  # 收集结果到列表
            .output()
        )

        # 执行管道
        p3(csv_path)
        
        log_event(f"CSV文件处理完成: {csv_path}, 处理了 {len(results)} 条记录")
        return results
    
    except Exception as e:
        error_msg = f"处理CSV失败: {csv_path}\n错误: {str(e)}\n{traceback.format_exc()}"
        log_event(error_msg)
        print(error_msg)
        return []