"""
Pre-filter - First layer quality check (blur detection)
"""
import cv2
import numpy as np
from PIL import Image
from enum import Enum


class SharpnessLevel(Enum):
    """Sharpness quality levels"""
    HIGH = "high_sharpness"
    MEDIUM = "medium_sharpness"
    LOW = "low_sharpness"


class ImagePreFilter:
    """Pre-filter for input image quality check"""

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dict with blur detection settings
        """
        # Support both nested and flat config structures
        if 'blur_detection' in config:
            blur_config = config.get('blur_detection', {})
        else:
            blur_config = config.get('prefilter', {}).get('blur_detection', {})

        self.enabled = blur_config.get('enabled', True)
        self.threshold_high = blur_config.get('laplacian_threshold_high', 100.0)
        self.threshold_medium = blur_config.get('laplacian_threshold_medium', 50.0)

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
        """
        Evaluate image quality

        Args:
            image: PIL Image

        Returns:
            dict with quality information
        """
        sharpness_level, blur_score = self.check_sharpness(image)

        return {
            'sharpness_level': sharpness_level.value,
            'blur_score': blur_score,
            'quality_warning': sharpness_level == SharpnessLevel.LOW
        }
