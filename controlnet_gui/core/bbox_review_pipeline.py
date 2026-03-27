"""
BBox review pipeline for center-quality verification.

Design goals:
- Always provide a lightweight local geometry review.
- Optionally add model-based reviewers (pose and SAM).
- Never hard-fail the main detection path when optional dependencies are missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _clamp_box(xyxy: List[float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(xyxy, (list, tuple)) or len(xyxy) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1i = int(max(0, min(width - 1, round(x1))))
    y1i = int(max(0, min(height - 1, round(y1))))
    x2i = int(max(0, min(width - 1, round(x2))))
    y2i = int(max(0, min(height - 1, round(y2))))
    if x2i - x1i < 2 or y2i - y1i < 2:
        return None
    return x1i, y1i, x2i, y2i


def _box_center(box_xyxy: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _normalized_center_distance(
    center_a: Tuple[float, float],
    center_b: Tuple[float, float],
    box_xyxy: Tuple[int, int, int, int],
) -> float:
    x1, y1, x2, y2 = box_xyxy
    diag = math.sqrt(float((x2 - x1) ** 2 + (y2 - y1) ** 2))
    if diag <= 1e-6:
        return 1.0
    return float(math.dist(center_a, center_b) / diag)


def _distance_to_score(norm_dist: float, good_threshold: float = 0.15, bad_threshold: float = 0.30) -> float:
    d = float(max(0.0, norm_dist))
    good = float(max(1e-6, good_threshold))
    bad = float(max(good + 1e-6, bad_threshold))
    if d <= good:
        return 1.0
    if d >= bad:
        return 0.0
    return float(1.0 - (d - good) / (bad - good))


@dataclass
class ReviewerScore:
    source: str
    score: float
    center_xy: Optional[Tuple[float, float]]
    detail: Dict


class GeometryCenterReviewer:
    """Local non-ML center estimator using saliency inside bbox."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.good_threshold = float(cfg.get("good_threshold", 0.15))
        self.bad_threshold = float(cfg.get("bad_threshold", 0.30))
        self.min_edge_pixels = int(cfg.get("min_edge_pixels", 24))

    def review(self, rgb: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> ReviewerScore:
        x1, y1, x2, y2 = box_xyxy
        crop = rgb[y1:y2, x1:x2]
        box_center = _box_center(box_xyxy)
        if crop.size == 0:
            return ReviewerScore(
                source="geometry",
                score=0.0,
                center_xy=box_center,
                detail={"ok": False, "reason": "empty_crop"},
            )

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)
        flat = mag.reshape(-1)
        if flat.size <= 0:
            sal_center = box_center
            score = 0.0
            detail = {"ok": False, "reason": "empty_magnitude"}
            return ReviewerScore("geometry", score, sal_center, detail)

        q = float(np.percentile(flat, 75))
        mask = mag >= q
        edge_count = int(np.count_nonzero(mask))
        if edge_count < self.min_edge_pixels:
            # Fallback: treat center as unknown but not catastrophic.
            sal_center = box_center
            score = 0.55
            detail = {
                "ok": False,
                "reason": "low_edge_signal",
                "edge_pixels": edge_count,
            }
            return ReviewerScore("geometry", score, sal_center, detail)

        weights = mag * mask.astype(np.float32)
        ys, xs = np.indices(weights.shape)
        total = float(np.sum(weights))
        if total <= 1e-6:
            sal_center = box_center
            score = 0.5
            detail = {"ok": False, "reason": "zero_weight_sum"}
            return ReviewerScore("geometry", score, sal_center, detail)

        cx = float(np.sum(xs * weights) / total) + x1
        cy = float(np.sum(ys * weights) / total) + y1
        sal_center = (cx, cy)

        norm_dist = _normalized_center_distance(box_center, sal_center, box_xyxy)
        score = _distance_to_score(norm_dist, self.good_threshold, self.bad_threshold)
        detail = {
            "ok": True,
            "edge_pixels": edge_count,
            "center_distance_norm": norm_dist,
            "box_center": [float(box_center[0]), float(box_center[1])],
            "saliency_center": [float(sal_center[0]), float(sal_center[1])],
        }
        return ReviewerScore("geometry", score, sal_center, detail)


class PoseCenterReviewer:
    """Optional ViTPose center estimator (ONNX runtime)."""

    def __init__(self, config: Optional[Dict] = None, device: str = "cpu"):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.device = str(device or "cpu").lower()
        self.model_path = str(cfg.get("vitpose_path", "models/vitpose/vitpose-l-wholebody.onnx") or "").strip()
        self.min_keypoint_conf = float(cfg.get("min_keypoint_confidence", 0.25))
        self.good_threshold = float(cfg.get("good_threshold", 0.12))
        self.bad_threshold = float(cfg.get("bad_threshold", 0.24))
        self._session = None
        self._session_error = ""
        if self.enabled:
            self._init_session()

    def _init_session(self) -> None:
        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception as exc:
            self._session = None
            self._session_error = str(exc)

    def review(self, rgb: np.ndarray, box_xyxy: Tuple[int, int, int, int], target_part: str) -> ReviewerScore:
        if not self.enabled:
            return ReviewerScore("pose", 0.0, None, {"ok": False, "reason": "disabled"})
        if self._session is None:
            return ReviewerScore("pose", 0.0, None, {"ok": False, "reason": "session_unavailable", "error": self._session_error})

        x1, y1, x2, y2 = box_xyxy
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return ReviewerScore("pose", 0.0, None, {"ok": False, "reason": "empty_crop"})

        keypoints = self._infer_keypoints(crop)
        center_local = self._center_from_keypoints(keypoints, target_part)
        if center_local is None:
            return ReviewerScore("pose", 0.0, None, {"ok": False, "reason": "insufficient_keypoints"})

        center_global = (float(center_local[0] + x1), float(center_local[1] + y1))
        box_center = _box_center(box_xyxy)
        norm_dist = _normalized_center_distance(box_center, center_global, box_xyxy)
        score = _distance_to_score(norm_dist, self.good_threshold, self.bad_threshold)
        detail = {
            "ok": True,
            "center_distance_norm": norm_dist,
            "center_xy": [float(center_global[0]), float(center_global[1])],
            "target_part": target_part,
        }
        return ReviewerScore("pose", score, center_global, detail)

    def _infer_keypoints(self, crop_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop_rgb, (192, 256))
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        input_name = self._session.get_inputs()[0].name
        heatmaps = self._session.run(None, {input_name: blob})[0][0]  # [17, H, W]

        h_out, w_out = int(heatmaps.shape[1]), int(heatmaps.shape[2])
        h_in, w_in = int(crop_rgb.shape[0]), int(crop_rgb.shape[1])
        kpts = np.zeros((17, 3), dtype=np.float32)
        for i in range(17):
            hm = heatmaps[i]
            idx = int(np.argmax(hm))
            y_hm, x_hm = divmod(idx, w_out)
            conf = float(hm[y_hm, x_hm])
            x = float(x_hm * (w_in / max(1, w_out)))
            y = float(y_hm * (h_in / max(1, h_out)))
            kpts[i] = [x, y, conf]
        return kpts

    def _center_from_keypoints(self, kpts: np.ndarray, target_part: str) -> Optional[Tuple[float, float]]:
        conf = kpts[:, 2]
        c = self.min_keypoint_conf
        # COCO indices:
        # nose0 le1 re2 lear3 rear4 lsh5 rsh6 lel7 rel8 lw9 rw10 lhip11 rhip12 lk13 rk14 la15 ra16
        if target_part == "head":
            ids = [0, 1, 2, 3, 4]
        elif target_part == "upper_body":
            ids = [5, 6, 11, 12]
        elif target_part == "lower_body":
            ids = [11, 12, 13, 14, 15, 16]
        else:
            ids = [5, 6, 11, 12]
        valid = [i for i in ids if conf[i] >= c]
        if len(valid) < 2:
            return None
        pts = kpts[valid, :2]
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


class SamCenterReviewer:
    """Optional SAM-based center estimator via ultralytics."""

    def __init__(self, config: Optional[Dict] = None, device: str = "cpu"):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.model_path = str(cfg.get("model_path", "models/sam/sam2_b.pt") or "").strip()
        self.device = str(device or "cpu").lower()
        self.good_threshold = float(cfg.get("good_threshold", 0.10))
        self.bad_threshold = float(cfg.get("bad_threshold", 0.22))
        self._model = None
        self._model_error = ""
        if self.enabled:
            self._init_model()

    def _init_model(self) -> None:
        try:
            from ultralytics import SAM
            self._model = SAM(self.model_path)
        except Exception as exc:
            self._model = None
            self._model_error = str(exc)

    def review(self, rgb: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> ReviewerScore:
        if not self.enabled:
            return ReviewerScore("sam", 0.0, None, {"ok": False, "reason": "disabled"})
        if self._model is None:
            return ReviewerScore("sam", 0.0, None, {"ok": False, "reason": "model_unavailable", "error": self._model_error})
        x1, y1, x2, y2 = box_xyxy
        box_center = _box_center(box_xyxy)
        try:
            results = self._model.predict(source=rgb, bboxes=[[x1, y1, x2, y2]], device=self.device, verbose=False)
            masks = getattr(results[0], "masks", None) if results else None
            data = getattr(masks, "data", None) if masks is not None else None
            if data is None or len(data) <= 0:
                return ReviewerScore("sam", 0.0, None, {"ok": False, "reason": "no_mask"})
            mask = data[0].cpu().numpy() if hasattr(data[0], "cpu") else np.asarray(data[0])
            ys, xs = np.where(mask > 0.5)
            if len(xs) == 0:
                return ReviewerScore("sam", 0.0, None, {"ok": False, "reason": "empty_mask"})
            center = (float(np.mean(xs)), float(np.mean(ys)))
            norm_dist = _normalized_center_distance(box_center, center, box_xyxy)
            score = _distance_to_score(norm_dist, self.good_threshold, self.bad_threshold)
            detail = {
                "ok": True,
                "center_distance_norm": norm_dist,
                "center_xy": [float(center[0]), float(center[1])],
                "mask_pixels": int(len(xs)),
            }
            return ReviewerScore("sam", score, center, detail)
        except Exception as exc:
            return ReviewerScore("sam", 0.0, None, {"ok": False, "reason": "sam_infer_failed", "error": str(exc)})


class BBoxReviewPipeline:
    """Review detections and produce center-quality verdicts."""

    def __init__(self, config: Optional[Dict] = None, device: str = "cpu"):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.override_score = bool(cfg.get("override_score", False))
        self.pass_threshold = float(cfg.get("pass_threshold", 0.72))
        self.fail_threshold = float(cfg.get("fail_threshold", 0.45))
        self.weights = {
            "geometry": float(cfg.get("weight_geometry", 0.35)),
            "pose": float(cfg.get("weight_pose", 0.35)),
            "sam": float(cfg.get("weight_sam", 0.30)),
        }

        self.geometry_reviewer = GeometryCenterReviewer(cfg.get("geometry", {}))
        self.pose_reviewer = PoseCenterReviewer(cfg.get("pose", {}), device=device)
        self.sam_reviewer = SamCenterReviewer(cfg.get("sam", {}), device=device)

    def review(self, rgb: np.ndarray, detections: List[Dict], target_part: str = "person") -> Dict:
        if not self.enabled:
            return {
                "enabled": False,
                "detections": detections,
                "summary": {
                    "aggregate_score": 0.0,
                    "pass_count": 0,
                    "review_count": len(detections or []),
                    "fail_count": 0,
                    "warning": "bbox_review_disabled",
                },
            }

        h, w = rgb.shape[:2]
        reviewed: List[Dict] = []
        center_scores: List[float] = []
        pass_count = 0
        fail_count = 0
        review_count = 0

        for det in detections or []:
            box = _clamp_box(det.get("xyxy", []), w, h)
            if box is None:
                det_out = dict(det)
                det_out["review"] = {"decision": "fail", "center_score": 0.0, "reason": "invalid_box"}
                reviewed.append(det_out)
                fail_count += 1
                center_scores.append(0.0)
                continue

            candidates: List[ReviewerScore] = [
                self.geometry_reviewer.review(rgb, box),
                self.pose_reviewer.review(rgb, box, target_part),
                self.sam_reviewer.review(rgb, box),
            ]

            weighted_sum = 0.0
            total_weight = 0.0
            details = {}
            for item in candidates:
                details[item.source] = item.detail
                if item.center_xy is None:
                    continue
                weight = float(max(0.0, self.weights.get(item.source, 0.0)))
                if weight <= 0.0:
                    continue
                weighted_sum += float(item.score) * weight
                total_weight += weight

            if total_weight <= 1e-6:
                center_score = float(max(0.0, min(1.0, candidates[0].score)))
            else:
                center_score = float(max(0.0, min(1.0, weighted_sum / total_weight)))

            if center_score >= self.pass_threshold:
                decision = "pass"
                pass_count += 1
            elif center_score <= self.fail_threshold:
                decision = "fail"
                fail_count += 1
            else:
                decision = "review"
                review_count += 1

            center_scores.append(center_score)
            det_out = dict(det)
            det_out["review"] = {
                "decision": decision,
                "center_score": round(float(center_score), 4),
                "sources": details,
                "target_part": target_part,
            }
            reviewed.append(det_out)

        aggregate = float(np.mean(center_scores)) if center_scores else 0.0
        warning = None
        if fail_count > 0:
            warning = "bbox_center_review_failed"
        elif review_count > 0:
            warning = "bbox_center_review_needed"

        return {
            "enabled": True,
            "detections": reviewed,
            "summary": {
                "aggregate_score": round(aggregate, 4),
                "pass_count": int(pass_count),
                "review_count": int(review_count),
                "fail_count": int(fail_count),
                "warning": warning,
            },
        }

