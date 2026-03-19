"""
Pre-filter - First layer quality check.
"""
import cv2
import numpy as np
from PIL import Image
from enum import Enum

from .fusion_score_filter import FusionScoreFilter


class SharpnessLevel(Enum):
    """Sharpness quality levels"""
    HIGH = "high_sharpness"
    MEDIUM = "medium_sharpness"
    LOW = "low_sharpness"


class ImagePreFilter:
    """Pre-filter for input image quality checks."""

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dict with blur detection and score filter settings.
        """
        if 'blur_detection' in config:
            blur_config = config.get('blur_detection', {})
            score_filter_config = config.get('score_filter', {})
        else:
            prefilter_config = config.get('prefilter', {})
            blur_config = prefilter_config.get('blur_detection', {})
            score_filter_config = prefilter_config.get('score_filter', {})

        self.enabled = blur_config.get('enabled', True)
        self.threshold_high = blur_config.get('laplacian_threshold_high', 100.0)
        self.threshold_medium = blur_config.get('laplacian_threshold_medium', 50.0)
        self.score_filter = FusionScoreFilter(score_filter_config)

    def has_active_score_filter(self) -> bool:
        return self.score_filter.is_enabled()

    def get_score_filter_status(self, force_refresh: bool = False) -> dict:
        return self.score_filter.get_probe(force_refresh=force_refresh)

    def prepare_score_filter(self) -> tuple[bool, str]:
        return self.score_filter.ensure_runtime_ready()

    def check_sharpness(self, image: Image.Image) -> tuple:
        """
        Check image sharpness using Laplacian variance

        Args:
            image: PIL Image

        Returns:
            (sharpness_level, blur_score)
        """
        if not self.enabled:
            return SharpnessLevel.HIGH, 100.0

        # Convert to grayscale
        img_array = np.array(image.convert('L'))

        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
        blur_score = laplacian.var()

        # Classify sharpness level
        if blur_score >= self.threshold_high:
            level = SharpnessLevel.HIGH
        elif blur_score >= self.threshold_medium:
            level = SharpnessLevel.MEDIUM
        else:
            level = SharpnessLevel.LOW

        return level, blur_score

    def evaluate(self, image: Image.Image) -> dict:
        """Evaluate image quality."""
        sharpness_level, blur_score = self.check_sharpness(image)

        result = {
            'sharpness_level': sharpness_level.value,
            'blur_score': blur_score,
            'quality_warning': sharpness_level == SharpnessLevel.LOW,
            'skip_processing': False,
        }

        if self.score_filter.is_enabled():
            score_result = self.score_filter.evaluate(image)
            result['score_filter'] = score_result
            result['skip_processing'] = bool(score_result.get('should_reject', False))

        return result
