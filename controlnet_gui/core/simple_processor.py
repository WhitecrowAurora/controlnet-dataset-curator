"""
ControlNet GUI - 简化的处理器
生成Canny控制图并评分，返回GUI友好的格式
"""

import os
from typing import List, Dict, Optional
from PIL import Image
import cv2
import numpy as np

from .scorer import CannyScorer


# 默认Canny阈值预设
DEFAULT_CANNY_PRESETS = [
    {"name": "light", "low": 30, "high": 120},    # 细节多（细线风）
    {"name": "medium", "low": 50, "high": 150},   # 平衡（最常用）
    {"name": "clean", "low": 80, "high": 180},    # 干净（厚涂风）
    {"name": "strong", "low": 120, "high": 220},  # 极简（Q版/简笔）
]


class ControlNetProcessor:
    """
    ControlNet处理器
    生成多阈值Canny图并评分
    """

    def __init__(
        self,
        enable_canny: bool = True,
        enable_openpose: bool = False,
        enable_depth: bool = False,
        canny_presets: Optional[List[Dict]] = None
    ):
        """
        初始化处理器

        Args:
            enable_canny: 是否启用Canny
            enable_openpose: 是否启用OpenPose（暂未实现）
            enable_depth: 是否启用Depth（暂未实现）
            canny_presets: Canny阈值预设
        """
        self.enable_canny = enable_canny
        self.enable_openpose = enable_openpose
        self.enable_depth = enable_depth
        self.canny_presets = canny_presets or DEFAULT_CANNY_PRESETS
        self.canny_scorer = CannyScorer()

    def process(self, sample) -> dict:
        """
        处理单个样本

        Args:
            sample: 来自data_source的样本，可以是ImageData对象或字典

        Returns:
            处理结果字典
        """
        # 兼容ImageData对象和字典
        if hasattr(sample, 'image'):
            original_image = sample.image
            original_path = sample.image_path
            basename = sample.basename
        else:
            original_image = sample.get('image')
            original_path = sample.get('path', '')
            basename = sample.get('basename', 'unknown')

        if original_image is None:
            return self._empty_result()

        # 转换为numpy数组
        np_image = np.array(original_image)
        if len(np_image.shape) == 2:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
        elif np_image.shape[2] == 4:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)

        # 生成多阈值Canny图
        variants = []
        for preset in self.canny_presets:
            canny_img = self._generate_canny(
                np_image,
                low_threshold=preset["low"],
                high_threshold=preset["high"]
            )

            # 评分
            score_result = self.canny_scorer.calculate_score(canny_img)

            # 转换为PIL图像
            canny_pil = Image.fromarray(canny_img)

            variants.append({
                'image': canny_pil,
                'score': score_result.total_score,
                'preset': preset,
                'score_details': {
                    'white_ratio': score_result.white_ratio,
                    'noise_count': score_result.noise_count,
                    'avg_area': score_result.avg_area,
                    'ratio_score': score_result.ratio_score,
                    'noise_penalty': score_result.noise_penalty,
                    'thickness_bonus': score_result.thickness_bonus
                }
            })

        return {
            'original_image': original_image,
            'original_path': original_path,
            'basename': basename,
            'variants': variants,
            'pose_image': None,
            'depth_image': None,
            'pose_score': None,
            'depth_score': None
        }

    def _generate_canny(
        self,
        image: np.ndarray,
        low_threshold: int,
        high_threshold: int
    ) -> np.ndarray:
        """生成Canny边缘图（返回灰度图）"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 应用Canny边缘检测
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        return edges

    def _empty_result(self) -> dict:
        """返回空结果"""
        return {
            'original_image': None,
            'original_path': '',
            'basename': 'error',
            'variants': [],
            'pose_image': None,
            'depth_image': None,
            'pose_score': None,
            'depth_score': None
        }
