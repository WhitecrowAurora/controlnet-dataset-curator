"""
ControlNet GUI - 评分系统模块
包含Canny、OpenPose、Depth的质量评估算法
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class CannyScoreResult:
    """Canny评分结果"""
    total_score: float
    white_ratio: float
    noise_count: int
    avg_area: float
    ratio_score: float
    noise_penalty: float
    thickness_bonus: float
    is_valid: bool


class CannyScorer:
    """
    Canny边缘图质量评分器
    
    评分维度:
    1. 白色线条占比 (white_ratio): 理想范围 8%~22%
    2. 小噪点惩罚 (noise_penalty): 面积<8像素的连通块
    3. 线条粗细奖励 (thickness_bonus): 平均连通块面积
    """
    
    def __init__(
        self,
        target_white_ratio: float = 0.15,
        white_ratio_min: float = 0.08,
        white_ratio_max: float = 0.22,
        noise_threshold: int = 8,
        weight_ratio: float = 50.0,
        weight_noise: float = 30.0,
        weight_thickness: float = 20.0
    ):
        self.target_white_ratio = target_white_ratio
        self.white_ratio_min = white_ratio_min
        self.white_ratio_max = white_ratio_max
        self.noise_threshold = noise_threshold
        self.weight_ratio = weight_ratio
        self.weight_noise = weight_noise
        self.weight_thickness = weight_thickness
    
    def calculate_score(self, canny_image: np.ndarray) -> CannyScoreResult:
        """
        计算Canny边缘图的质量分数
        
        Args:
            canny_image: 二值化的Canny边缘图 (0或255)
        
        Returns:
            CannyScoreResult: 包含详细评分信息的结果对象
        """
        if canny_image is None or canny_image.size == 0:
            return CannyScoreResult(0, 0, 0, 0, 0, 0, 0, False)
        
        # 确保是灰度图
        if len(canny_image.shape) == 3:
            canny_image = cv2.cvtColor(canny_image, cv2.COLOR_BGR2GRAY)
        
        # 二值化确保只有0和255
        _, binary = cv2.threshold(canny_image, 127, 255, cv2.THRESH_BINARY)
        
        total_pixels = binary.shape[0] * binary.shape[1]
        white_pixels = np.sum(binary > 0)
        white_ratio = white_pixels / total_pixels
        
        # 连通组件分析
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels <= 1:
            return CannyScoreResult(0, white_ratio, 0, 0, 0, 0, 0, False)
        
        # 获取所有连通块面积（排除背景，索引0）
        areas = stats[1:, cv2.CC_STAT_AREA]
        
        # 统计噪点数量（面积小于阈值的连通块）
        noise_count = int(np.sum(areas < self.noise_threshold))
        
        # 计算平均连通块面积
        avg_area = float(np.mean(areas))
        
        # === 评分计算 ===
        total_score = 0.0
        ratio_score = 0.0
        noise_penalty = 0.0
        thickness_bonus = 0.0
        
        # 1. 白色占比评分 (最高 weight_ratio 分)
        if self.white_ratio_min <= white_ratio <= self.white_ratio_max:
            # 越接近目标值得分越高
            deviation = abs(white_ratio - self.target_white_ratio)
            max_deviation = max(
                self.target_white_ratio - self.white_ratio_min,
                self.white_ratio_max - self.target_white_ratio
            )
            # 防止除零错误
            if max_deviation > 0:
                ratio_score = self.weight_ratio * max(0, 1 - deviation / max_deviation)
            else:
                ratio_score = self.weight_ratio  # 完美匹配
            total_score += ratio_score
        
        # 2. 噪点惩罚 (最高扣 weight_noise 分)
        noise_penalty = min(noise_count * 0.3, self.weight_noise)
        total_score -= noise_penalty
        
        # 3. 线条粗细奖励 (最高 weight_thickness 分)
        # 平均面积越大，线条越粗实
        thickness_bonus = min(avg_area / 100 * self.weight_thickness, self.weight_thickness)
        total_score += thickness_bonus
        
        # 确保分数非负
        total_score = max(0, total_score)
        
        # 判断是否有效
        is_valid = (self.white_ratio_min <= white_ratio <= self.white_ratio_max and 
                    total_score > 0)
        
        return CannyScoreResult(
            total_score=total_score,
            white_ratio=white_ratio,
            noise_count=noise_count,
            avg_area=avg_area,
            ratio_score=ratio_score,
            noise_penalty=noise_penalty,
            thickness_bonus=thickness_bonus,
            is_valid=is_valid
        )


@dataclass
class OpenPoseScoreResult:
    """OpenPose评分结果"""
    is_valid: bool
    confidence_score: float
    avg_confidence: float
    anatomy_score: float
    root_confidences: list
    core_keypoints_count: int
    error: Optional[str] = None


class OpenPoseScorer:
    """
    OpenPose关键点质量评估器
    
    评估维度:
    1. 核心骨架完整性 (躯干、肩部、髋部)
    2. 关键点置信度均值
    3. 解剖合理性检查
    """
    
    # COCO 18关键点索引
    NOSE = 0
    NECK = 1
    R_SHOULDER = 2
    R_ELBOW = 3
    R_WRIST = 4
    L_SHOULDER = 5
    L_ELBOW = 6
    L_WRIST = 7
    R_HIP = 8
    R_KNEE = 9
    R_ANKLE = 10
    L_HIP = 11
    L_KNEE = 12
    L_ANKLE = 13
    R_EYE = 14
    L_EYE = 15
    R_EAR = 16
    L_EAR = 17
    
    # 核心骨架索引 (排除易失效的面部和踝关节)
    CORE_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
    # 根节点索引 (脖子、双肩必须存在)
    ROOT_INDICES = [1, 2, 5]
    
    def __init__(
        self,
        min_core_confidence: float = 0.4,
        root_confidence_threshold: float = 0.1
    ):
        self.min_core_confidence = min_core_confidence
        self.root_confidence_threshold = root_confidence_threshold
    
    def calculate_score(
        self, 
        keypoints: Optional[np.ndarray]
    ) -> OpenPoseScoreResult:
        """
        计算OpenPose关键点的质量分数
        
        Args:
            keypoints: 关键点数组，形状为 (N, 3)，格式为 [x, y, confidence]
        
        Returns:
            OpenPoseScoreResult: 包含详细评分信息的结果对象
        """
        if keypoints is None:
            return OpenPoseScoreResult(
                is_valid=False,
                confidence_score=0,
                avg_confidence=0,
                anatomy_score=0,
                root_confidences=[],
                core_keypoints_count=0,
                error="no_keypoints"
            )
        
        # 确保是numpy数组
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        
        if len(keypoints) < 18:
            return OpenPoseScoreResult(
                is_valid=False,
                confidence_score=0,
                avg_confidence=0,
                anatomy_score=0,
                root_confidences=[],
                core_keypoints_count=len(keypoints),
                error="insufficient_keypoints"
            )
        
        # 提取核心关键点
        core_keypoints = keypoints[self.CORE_INDICES]
        core_confidences = core_keypoints[:, 2]
        
        # 根节点检查 (脖子、双肩必须存在)
        root_confidences = keypoints[self.ROOT_INDICES, 2].tolist()
        
        if any(c < self.root_confidence_threshold for c in root_confidences):
            return OpenPoseScoreResult(
                is_valid=False,
                confidence_score=0,
                avg_confidence=float(np.mean(core_confidences)),
                anatomy_score=0,
                root_confidences=root_confidences,
                core_keypoints_count=len(keypoints),
                error="missing_root_keypoints"
            )
        
        # 计算核心置信度
        avg_confidence = float(np.mean(core_confidences))
        
        # 解剖合理性检查
        anatomy_score = self._check_anatomy_plausibility(keypoints)
        
        # 综合评分
        confidence_score = avg_confidence * 100
        
        is_valid = (avg_confidence >= self.min_core_confidence and 
                    anatomy_score > 0.5)
        
        return OpenPoseScoreResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            avg_confidence=avg_confidence,
            anatomy_score=anatomy_score,
            root_confidences=root_confidences,
            core_keypoints_count=len(keypoints)
        )
    
    def _check_anatomy_plausibility(self, keypoints: np.ndarray) -> float:
        """
        检查解剖合理性
        - 肩宽与髋宽比例
        - 四肢长度对称性
        """
        try:
            # 检查肩宽和髋宽
            r_shoulder = keypoints[self.R_SHOULDER, :2]
            l_shoulder = keypoints[self.L_SHOULDER, :2]
            r_hip = keypoints[self.R_HIP, :2]
            l_hip = keypoints[self.L_HIP, :2]
            
            # 检查关键点是否有效
            if (keypoints[self.R_SHOULDER, 2] < 0.3 or 
                keypoints[self.L_SHOULDER, 2] < 0.3 or
                keypoints[self.R_HIP, 2] < 0.3 or 
                keypoints[self.L_HIP, 2] < 0.3):
                return 0.5  # 默认中等分数
            
            shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
            hip_width = np.linalg.norm(r_hip - l_hip)
            
            if shoulder_width > 0 and hip_width > 0:
                ratio = shoulder_width / hip_width
                # 正常人体肩宽约为髋宽的1.2-1.8倍
                if 0.8 <= ratio <= 2.5:
                    return 1.0
                elif 0.5 <= ratio <= 3.0:
                    return 0.7
                else:
                    return 0.3
            
            return 0.5
        except Exception:
            return 0.5


@dataclass
class DepthScoreResult:
    """Depth评分结果"""
    is_valid: bool
    quality_score: float
    std_dev: float
    dynamic_range: float
    valid_ratio: float
    error: Optional[str] = None


class DepthScorer:
    """
    深度图质量评估器
    
    评估维度:
    1. 标准差 (std_dev): 评估深度变化丰富度
    2. 动态范围 (dynamic_range): 评估前景到背景的过渡
    3. 有效深度占比: 排除纯黑/纯白区域
    """
    
    def __init__(
        self,
        min_std: float = 15.0,
        min_dynamic_range: float = 50.0,
        valid_min: int = 10,
        valid_max: int = 245
    ):
        self.min_std = min_std
        self.min_dynamic_range = min_dynamic_range
        self.valid_min = valid_min
        self.valid_max = valid_max
    
    def calculate_score(self, depth_map: np.ndarray) -> DepthScoreResult:
        """
        计算深度图的质量分数
        
        Args:
            depth_map: 深度图，可以是单通道灰度图或彩色图
        
        Returns:
            DepthScoreResult: 包含详细评分信息的结果对象
        """
        if depth_map is None or depth_map.size == 0:
            return DepthScoreResult(
                is_valid=False,
                quality_score=0,
                std_dev=0,
                dynamic_range=0,
                valid_ratio=0,
                error="null_depth_map"
            )
        
        # 确保是灰度图
        if len(depth_map.shape) == 3:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
        
        depth_float = depth_map.astype(np.float32)
        
        # 1. 标准差评估
        std_dev = float(np.std(depth_float))
        
        # 2. 动态范围评估 (使用百分位数排除极端值)
        p95 = float(np.percentile(depth_float, 95))
        p05 = float(np.percentile(depth_float, 5))
        dynamic_range = p95 - p05
        
        # 3. 有效深度占比
        valid_mask = (depth_float > self.valid_min) & (depth_float < self.valid_max)
        valid_ratio = float(np.sum(valid_mask) / depth_float.size)
        
        # === 评分计算 ===
        score = 0.0
        
        # 标准差评分 (最高40分)
        if std_dev >= self.min_std:
            score += min(std_dev / 50 * 40, 40)
        
        # 动态范围评分 (最高40分)
        if dynamic_range >= self.min_dynamic_range:
            score += min(dynamic_range / 100 * 40, 40)
        
        # 有效占比评分 (最高20分)
        score += valid_ratio * 20
        
        is_valid = (std_dev >= self.min_std and 
                    dynamic_range >= self.min_dynamic_range)
        
        return DepthScoreResult(
            is_valid=is_valid,
            quality_score=score,
            std_dev=std_dev,
            dynamic_range=dynamic_range,
            valid_ratio=valid_ratio
        )


class CompositeScorer:
    """
    综合评分器：整合Canny、OpenPose、Depth评分
    """
    
    def __init__(
        self,
        canny_scorer: Optional[CannyScorer] = None,
        openpose_scorer: Optional[OpenPoseScorer] = None,
        depth_scorer: Optional[DepthScorer] = None
    ):
        self.canny_scorer = canny_scorer or CannyScorer()
        self.openpose_scorer = openpose_scorer or OpenPoseScorer()
        self.depth_scorer = depth_scorer or DepthScorer()
    
    def score_canny_batch(
        self, 
        canny_images: list
    ) -> Tuple[list, int]:
        """
        对多张Canny图进行评分，返回最佳索引
        
        Args:
            canny_images: Canny图像列表
        
        Returns:
            (scores, best_index): 评分列表和最佳图像索引
        """
        scores = []
        for img in canny_images:
            result = self.canny_scorer.calculate_score(img)
            scores.append(result.total_score)
        
        best_index = int(np.argmax(scores)) if scores else 0
        return scores, best_index
