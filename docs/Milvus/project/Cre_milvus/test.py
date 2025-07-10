# check_environment.py
import torch
import transformers
import onnxruntime
import platform

print("="*50)
print("环境诊断报告")
print("="*50)

# 系统信息
print("\n系统信息:")
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"Python版本: {platform.python_version()}")

# PyTorch 信息
print("\nPyTorch信息:")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"设备数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")

# Transformers 信息
print("\nTransformers信息:")
print(f"Transformers版本: {transformers.__version__}")

# ONNX 信息
print("\nONNX信息:")
print(f"ONNX Runtime版本: {onnxruntime.__version__}")

# 模型加载测试
print("\n模型加载测试:")
try:
    from transformers import AutoModel
    print("尝试加载小型模型...")
    model = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5")
    print("小型模型加载成功!")
except Exception as e:
    print(f"小型模型加载失败: {str(e)}")

print("\n诊断完成")