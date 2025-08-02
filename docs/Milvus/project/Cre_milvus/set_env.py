#!/usr/bin/env python3
"""
环境设置脚本
在启动主程序前设置环境变量以减少警告
"""

import os
import warnings

# 设置环境变量
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 过滤特定警告
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.distributed')
warnings.filterwarnings('ignore', category=UserWarning, module='gevent')

# 在导入其他模块前进行monkey patching
try:
    from gevent import monkey
    monkey.patch_all(ssl=False)  # 不patch SSL以避免警告
except ImportError:
    pass

print("环境变量已设置，警告已过滤")
