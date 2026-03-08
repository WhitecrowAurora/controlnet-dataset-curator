# ControlNet GUI Package
"""
ControlNet 数据集筛选工具
支持流式数据源和本地文件，多线程处理，自动评分筛选
"""

import os

# Disable torch formatter in datasets to avoid torch DLL loading issues
# This allows datasets library to work without torch
os.environ['HF_DATASETS_DISABLE_TORCH'] = '1'

__version__ = "1.0.0"
__author__ = "ControlNet GUI"
