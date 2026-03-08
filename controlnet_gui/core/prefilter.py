"""
ControlNet GUI - 预筛选策略模块
自动分类图片质量，决定哪些需要人工审核
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

import numpy as np

from .scorer import CannyScoreResult, OpenPoseScoreResult, DepthScoreResult


class QualityLevel(Enum):
    """质量等级"""
    AUTO_ACCEPT = "auto_accept"     # 自动通过
    AUTO_REJECT = "auto_reject"     # 自动废弃
    NEED_REVIEW = "need_review"     # 需要人工审核


@dataclass
class PreFilterResult:
    """预筛选结果"""
    quality_level: QualityLevel
    best_canny_index: int
    canny_scores: List[float]
    pose_score: Optional[float]
    depth_score: Optional[float]
    total_score: float
    reason: str


class PreFilter:
    """
    预筛选器：决定哪些图片需要人工审核

    策略逻辑:
    - 高分图片（≥80分）：自动通过，保存最佳Canny版本
    - 低分图片（≤40分）：自动废弃
    - 中等图片（40-80分）：需要人工审核
    """

    def __init__(
        self,
        auto_accept_threshold: float = 80.0,
        auto_reject_threshold: float = 40.0
    ):
        """
        初始化预筛选器

        Args:
            auto_accept_threshold: 自动通过阈值
            auto_reject_threshold: 自动废弃阈值
        """
        self.auto_accept_threshold = auto_accept_threshold
        self.auto_reject_threshold = auto_reject_threshold

        if auto_reject_threshold >= auto_accept_threshold:
            raise ValueError("auto_reject_threshold 必须小于 auto_accept_threshold")

    def evaluate(self, processing_result: dict) -> dict:
        """
        评估处理结果并添加预筛选信息

        Args:
            processing_result: 来自processor的处理结果

        Returns:
            添加了预筛选信息的结果字典
        """
        variants = processing_result.get('variants', [])

        if not variants:
            processing_result['quality_level'] = QualityLevel.AUTO_REJECT.value
            processing_result['total_score'] = 0
            processing_result['reason'] = "无有效变体"
            return processing_result

        # 获取所有Canny分数
        canny_scores = [v.get('score', 0) for v in variants]
        best_index = int(np.argmax(canny_scores))
        best_score = canny_scores[best_index]

        # 标记最佳变体
        for i, variant in enumerate(variants):
            variant['is_best'] = (i == best_index)

        # 获取pose和depth分数（如果有）
        pose_score = processing_result.get('pose_score')
        depth_score = processing_result.get('depth_score')

        # 计算综合分数
        total_score = self._calculate_total_score(best_score, pose_score, depth_score)

        # 判断质量等级
        if total_score >= self.auto_accept_threshold:
            quality_level = QualityLevel.AUTO_ACCEPT
            reason = f"高分自动通过（{total_score:.1f}分）"
        elif total_score <= self.auto_reject_threshold:
            quality_level = QualityLevel.AUTO_REJECT
            reason = f"低分自动废弃（{total_score:.1f}分）"
        else:
            quality_level = QualityLevel.NEED_REVIEW
            reason = f"需要人工审核（{total_score:.1f}分）"

        # 添加预筛选信息
        processing_result['quality_level'] = quality_level.value
        processing_result['total_score'] = total_score
        processing_result['best_canny_index'] = best_index
        processing_result['reason'] = reason

        return processing_result

    def _calculate_total_score(
        self,
        canny_score: float,
        pose_score: Optional[float],
        depth_score: Optional[float]
    ) -> float:
        """计算综合分数（加权平均）"""
        weights = [1.0]  # Canny权重为1
        scores = [min(max(canny_score, 0.0), 100.0)]

        if pose_score is not None:
            weights.append(0.3)
            scores.append(min(max(pose_score, 0.0), 100.0))

        if depth_score is not None:
            weights.append(0.2)
            scores.append(min(max(depth_score, 0.0), 100.0))

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_score = sum(s * w for s, w in zip(scores, weights))
        return weighted_score / total_weight
