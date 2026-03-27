"""
ControlNet GUI - 图像处理模块
生成Canny、OpenPose、Depth控制图
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from controlnet_aux import (
        CannyDetector,
        LineartAnimeDetector,
        DWposeDetector,
        ZoeDetector
    )
    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False

from .scorer import CannyScorer, CannyScoreResult, OpenPoseScorer, OpenPoseScoreResult, DepthScorer, DepthScoreResult
from .data_source import ImageData


@dataclass
class ProcessingResult:
    """处理结果"""
    source_data: ImageData          # 原始图像数据
    canny_images: List[np.ndarray]  # 多阈值Canny图
    canny_scores: List[CannyScoreResult]  # Canny评分结果
    canny_presets: List[Dict]       # Canny阈值预设
    pose_image: Optional[np.ndarray]  # OpenPose图
    pose_result: Optional[OpenPoseScoreResult]  # Pose评分
    depth_image: Optional[np.ndarray]  # Depth图
    depth_result: Optional[DepthScoreResult]  # Depth评分
    best_canny_index: int           # 最佳Canny索引
    error: Optional[str] = None     # 错误信息


# 默认Canny阈值预设
DEFAULT_CANNY_PRESETS = [
    {"name": "light", "low": 30, "high": 120},    # 细节多（细线风）
    {"name": "medium", "low": 50, "high": 150},   # 平衡（最常用）
    {"name": "clean", "low": 80, "high": 180},    # 干净（厚涂风）
    {"name": "strong", "low": 120, "high": 220},  # 极简（Q版/简笔）
]


class ImageProcessor:
    """
    图像处理器
    负责生成各种ControlNet控制图并评分
    """
    
    def __init__(
        self,
        canny_presets: Optional[List[Dict]] = None,
        enable_pose: bool = True,
        enable_depth: bool = True,
        device: Optional[str] = None,
        canny_scorer: Optional[CannyScorer] = None,
        pose_scorer: Optional[OpenPoseScorer] = None,
        depth_scorer: Optional[DepthScorer] = None
    ):
        """
        初始化图像处理器
        
        Args:
            canny_presets: Canny阈值预设列表
            enable_pose: 是否启用OpenPose
            enable_depth: 是否启用Depth
            device: 计算设备 (cuda/cpu)
            canny_scorer: Canny评分器
            pose_scorer: OpenPose评分器
            depth_scorer: Depth评分器
        """
        self.canny_presets = canny_presets or DEFAULT_CANNY_PRESETS
        self.enable_pose = enable_pose
        self.enable_depth = enable_depth
        
        # 设置设备
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # 评分器
        self.canny_scorer = canny_scorer or CannyScorer()
        self.pose_scorer = pose_scorer or OpenPoseScorer()
        self.depth_scorer = depth_scorer or DepthScorer()
        
        # 检测器（延迟加载）
        self._canny_detector = None
        self._pose_detector = None
        self._depth_detector = None
        self._initialized = False
    
    def _initialize_detectors(self):
        """延迟初始化检测器"""
        if self._initialized:
            return
        
        # Canny检测器（不需要模型）
        self._canny_detector = CannyDetector() if CONTROLNET_AUX_AVAILABLE else None
        
        # Pose检测器
        if self.enable_pose and CONTROLNET_AUX_AVAILABLE:
            try:
                self._pose_detector = DWposeDetector(device=self.device)
            except Exception as e:
                print(f"警告: 无法加载OpenPose检测器: {e}")
                self._pose_detector = None
        
        # Depth检测器
        if self.enable_depth and CONTROLNET_AUX_AVAILABLE:
            try:
                self._depth_detector = ZoeDetector.from_pretrained("lllyasviel/Annotators")
            except Exception as e:
                print(f"警告: 无法加载Depth检测器: {e}")
                self._depth_detector = None
        
        self._initialized = True
    
    def process(self, image_data: ImageData) -> ProcessingResult:
        """
        处理单张图片
        
        Args:
            image_data: 图像数据
        
        Returns:
            ProcessingResult: 处理结果
        """
        self._initialize_detectors()
        
        try:
            np_image = np.array(image_data.image)
            canny_images, canny_scores = self._build_canny_variants(np_image)
            best_canny_index = self._best_canny_index(canny_scores)
            pose_image, pose_result = self._generate_optional_pose(np_image)
            depth_image, depth_result = self._generate_optional_depth(np_image)

            return self._build_success_result(
                image_data=image_data,
                canny_images=canny_images,
                canny_scores=canny_scores,
                best_canny_index=best_canny_index,
                pose_image=pose_image,
                pose_result=pose_result,
                depth_image=depth_image,
                depth_result=depth_result,
            )
             
        except Exception as e:
            return self._build_error_result(image_data, str(e))

    def _build_canny_variants(self, np_image: np.ndarray) -> tuple[List[np.ndarray], List[CannyScoreResult]]:
        canny_images = []
        canny_scores = []
        for preset in self.canny_presets:
            canny_img = self._generate_canny(
                np_image,
                low_threshold=preset["low"],
                high_threshold=preset["high"]
            )
            canny_images.append(canny_img)
            canny_scores.append(self.canny_scorer.calculate_score(canny_img))
        return canny_images, canny_scores

    @staticmethod
    def _best_canny_index(canny_scores: List[CannyScoreResult]) -> int:
        return max(range(len(canny_scores)), key=lambda i: canny_scores[i].total_score)

    def _generate_optional_pose(self, np_image: np.ndarray):
        if self.enable_pose and self._pose_detector:
            return self._generate_pose(np_image)
        return None, None

    def _generate_optional_depth(self, np_image: np.ndarray):
        if self.enable_depth and self._depth_detector:
            return self._generate_depth(np_image)
        return None, None

    def _build_success_result(
        self,
        *,
        image_data: ImageData,
        canny_images: List[np.ndarray],
        canny_scores: List[CannyScoreResult],
        best_canny_index: int,
        pose_image,
        pose_result,
        depth_image,
        depth_result,
    ) -> ProcessingResult:
        return ProcessingResult(
            source_data=image_data,
            canny_images=canny_images,
            canny_scores=canny_scores,
            canny_presets=self.canny_presets,
            pose_image=pose_image,
            pose_result=pose_result,
            depth_image=depth_image,
            depth_result=depth_result,
            best_canny_index=best_canny_index
        )

    def _build_error_result(self, image_data: ImageData, error: str) -> ProcessingResult:
        return ProcessingResult(
            source_data=image_data,
            canny_images=[],
            canny_scores=[],
            canny_presets=self.canny_presets,
            pose_image=None,
            pose_result=None,
            depth_image=None,
            depth_result=None,
            best_canny_index=0,
            error=error
        )
    
    def _generate_canny(
        self,
        image: np.ndarray,
        low_threshold: int,
        high_threshold: int
    ) -> np.ndarray:
        """生成Canny边缘图（返回灰度图）"""
        if self._canny_detector:
            # 使用controlnet_aux的CannyDetector
            pil_image = Image.fromarray(image)
            result = self._canny_detector(pil_image, low_threshold=low_threshold, high_threshold=high_threshold)
            result_np = np.array(result)
            # 确保返回灰度图
            if len(result_np.shape) == 3:
                result_np = cv2.cvtColor(result_np, cv2.COLOR_RGB2GRAY)
            return result_np
        else:
            # 使用OpenCV
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            return edges
    
    def _generate_pose(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[OpenPoseScoreResult]]:
        """生成OpenPose图"""
        if not self._pose_detector:
            return None, None
        
        try:
            pil_image = Image.fromarray(image)
            pose_img = self._pose_detector(pil_image, include_hands=True, include_face=True)
            pose_np = np.array(pose_img)
            
            # TODO: 从检测结果提取关键点进行评分
            # 这里简化处理，暂时不评分
            return pose_np, None
        except Exception as e:
            print(f"Pose检测失败: {e}")
            return None, None
    
    def _generate_depth(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[DepthScoreResult]]:
        """生成Depth图"""
        if not self._depth_detector:
            return None, None
        
        try:
            pil_image = Image.fromarray(image)
            depth_img = self._depth_detector(pil_image)
            depth_np = np.array(depth_img)
            
            # 评分
            score_result = self.depth_scorer.calculate_score(depth_np)
            
            return depth_np, score_result
        except Exception as e:
            print(f"Depth估计失败: {e}")
            return None, None
    
    def process_batch(
        self,
        image_data_list: List[ImageData],
        max_workers: int = 1
    ) -> List[ProcessingResult]:
        """
        批量处理图片

        Args:
            image_data_list: 图像数据列表
            max_workers: 最大线程数（保留用于未来扩展，当前GPU模型不适合多线程）

        Returns:
            处理结果列表
        """
        # GPU模型不适合多线程，通常使用单线程
        # max_workers参数保留用于未来CPU模式的并行处理
        results = []
        for img_data in image_data_list:
            result = self.process(img_data)
            results.append(result)
        return results


class ProcessingWorker:
    """
    后台处理工作器
    用于在后台线程中处理图像
    """
    
    def __init__(self, processor: ImageProcessor):
        self.processor = processor
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
    
    def process_stream(
        self,
        data_source,
        callback,
        batch_size: int = 1,
        preload_count: int = 15
    ):
        """
        流式处理数据源中的图像
        
        Args:
            data_source: 数据源
            callback: 回调函数，接收ProcessingResult
            batch_size: 批次大小
            preload_count: 预加载数量
        """
        self._stop_flag.clear()
        self._pause_flag.clear()
        
        processed_count = 0
        
        for image_data in data_source:
            if self._stop_flag.is_set():
                break
            
            # 暂停处理
            while self._pause_flag.is_set():
                if self._stop_flag.is_set():
                    return
                import time
                time.sleep(0.1)
            
            result = self.processor.process(image_data)
            callback(result)
            processed_count += 1
            
            # 检查预加载数量
            if processed_count >= preload_count:
                pass  # 让回调决定是否继续
    
    def stop(self):
        """停止处理"""
        self._stop_flag.set()
    
    def pause(self):
        """暂停处理"""
        self._pause_flag.set()
    
    def resume(self):
        """恢复处理"""
        self._pause_flag.clear()
    
    def is_stopped(self) -> bool:
        return self._stop_flag.is_set()
    
    def is_paused(self) -> bool:
        return self._pause_flag.is_set()
