"""
Score Converter - Convert raw scores to 1-10 scale
"""


def canny_to_10_scale(raw_score: float) -> float:
    """
    Convert Canny score (0-100) to 1-10 scale

    Args:
        raw_score: Raw Canny score (0-100)

    Returns:
        Score on 1-10 scale
    """
    # Simple linear mapping: 0-100 -> 1-10
    # But ensure minimum is 1.0
    score = (raw_score / 100.0) * 9.0 + 1.0
    return round(score, 1)


def openpose_to_10_scale(visibility_ratio: float, is_valid: bool, warning: str = None, is_vitpose: bool = False) -> float:
    """
    Convert OpenPose metrics to 1-10 scale

    Args:
        visibility_ratio: Ratio of visible pose pixels (0-1)
        is_valid: Whether pose is valid
        warning: Warning message if any
        is_vitpose: Whether using ViTPose (finer lines)

    Returns:
        Score on 1-10 scale
    """
    if not is_valid:
        return 1.0

    # Adjust thresholds based on model type
    if is_vitpose:
        min_optimal = 0.01  # ViTPose has finer lines
        max_optimal = 0.35
    else:
        min_optimal = 0.05  # DWpose has thicker lines
        max_optimal = 0.35

    # Base score from visibility
    if visibility_ratio < min_optimal:
        base_score = 1.0
    elif visibility_ratio > max_optimal:
        # Over-detected, penalize
        base_score = max(1.0, 10.0 - (visibility_ratio - max_optimal) * 20)
    else:
        # Optimal range: map to 6-10
        normalized = (visibility_ratio - min_optimal) / (max_optimal - min_optimal)
        base_score = 6.0 + normalized * 4.0

    # Penalize if there's a warning
    if warning:
        base_score = max(1.0, base_score - 2.0)

    return round(base_score, 1)


def depth_to_10_scale(metrics: dict, is_valid: bool, warning: str = None) -> float:
    """
    Convert Depth metrics to 1-10 scale (adjusted for Depth Anything V2)

    Args:
        metrics: Dict with 'std', 'dynamic_range', 'grad_mean'
        is_valid: Whether depth is valid
        warning: Warning message if any

    Returns:
        Score on 1-10 scale
    """
    if not is_valid:
        return 1.0

    std = metrics.get('std', 0)
    dynamic_range = metrics.get('dynamic_range', 0)
    grad_mean = metrics.get('grad_mean', 0)

    # Score components (each 0-1) - adjusted for Depth Anything V2
    # Standard deviation: optimal > 70, minimum 40
    std_score = min(1.0, max(0.0, (std - 40.0) / 40.0))

    # Dynamic range: optimal > 200, minimum 120
    dr_score = min(1.0, max(0.0, (dynamic_range - 120.0) / 100.0))

    # Gradient mean: optimal 2-6, minimum 1.5, maximum 10
    if grad_mean < 1.5:
        grad_score = 0.0
    elif grad_mean > 10.0:
        grad_score = max(0.0, 1.0 - (grad_mean - 10.0) / 15.0)
    elif grad_mean < 2.0:
        grad_score = (grad_mean - 1.5) / 0.5
    elif grad_mean > 6.0:
        grad_score = 1.0 - (grad_mean - 6.0) / 4.0
    else:
        grad_score = 1.0

    # Weighted average: std 30%, dr 30%, grad 40%
    combined = std_score * 0.3 + dr_score * 0.3 + grad_score * 0.4

    # Map 0-1 to 1-10
    base_score = 1.0 + combined * 9.0

    # Penalize if there's a warning
    if warning:
        base_score = max(1.0, base_score - 1.5)

    return round(base_score, 1)


def bbox_to_10_scale(raw_score: float) -> float:
    """
    Convert BBox score (0-100) to 1-10 scale.
    """
    score = (float(raw_score) / 100.0) * 9.0 + 1.0
    score = max(1.0, min(10.0, score))
    return round(score, 1)


def calculate_overall_score(canny_score: float, openpose_score: float = None,
                           depth_score: float = None) -> float:
    """
    Calculate overall score (1-10) from individual control scores

    Args:
        canny_score: Canny score (1-10)
        openpose_score: OpenPose score (1-10), optional
        depth_score: Depth score (1-10), optional

    Returns:
        Overall score on 1-10 scale
    """
    scores = [canny_score]
    weights = [0.6]  # Canny base weight

    if openpose_score is not None:
        scores.append(openpose_score)
        weights.append(0.2)

    if depth_score is not None:
        scores.append(depth_score)
        weights.append(0.2)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Weighted average
    overall = sum(s * w for s, w in zip(scores, weights))

    return round(overall, 1)
