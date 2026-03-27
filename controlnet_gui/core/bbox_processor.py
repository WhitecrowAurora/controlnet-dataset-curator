"""
BBox Processor based on Ultralytics YOLO.
Generates a visualized bbox map and structured detection metadata.
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .bbox_review_pipeline import BBoxReviewPipeline


class BBoxProcessor:
    """Simple wrapper around YOLO object detection for dataset curation."""

    def __init__(
        self,
        model_source: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 20,
        person_only: bool = True,
        target_part: str = "person",
        use_specialized_channels: bool = False,
        specialized_model_sources: Optional[Dict[str, str]] = None,
        review_config: Optional[Dict] = None,
        device: str = "cpu",
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for BBox detection. Install with: pip install ultralytics") from exc

        self._YOLO = YOLO
        self.model_source = str(model_source or "").strip()
        self.model = YOLO(self.model_source)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.person_only = bool(person_only)
        self.target_part = self._normalize_target_part(target_part)
        self.use_specialized_channels = bool(use_specialized_channels)
        self.specialized_model_sources = self._normalize_specialized_model_sources(specialized_model_sources)
        self.specialized_models: Dict[str, object] = {}
        if self.use_specialized_channels:
            self._load_specialized_models()
        self.review_pipeline = BBoxReviewPipeline(review_config or {}, device=device)

    @staticmethod
    def _normalize_target_part(target_part: str) -> str:
        key = str(target_part or "person").strip().lower()
        mapping = {
            "person": "person",
            "full": "person",
            "full_body": "person",
            "head": "head",
            "face": "head",
            "upper": "upper_body",
            "upper_body": "upper_body",
            "torso": "upper_body",
            "lower": "lower_body",
            "lower_body": "lower_body",
            "legs": "lower_body",
        }
        return mapping.get(key, "person")

    @staticmethod
    def _specialized_part_keys() -> Tuple[str, str, str]:
        return ("head", "upper_body", "lower_body")

    @classmethod
    def _normalize_specialized_model_sources(cls, raw_sources: Optional[Dict[str, str]]) -> Dict[str, str]:
        normalized = {part: "" for part in cls._specialized_part_keys()}
        if not isinstance(raw_sources, dict):
            return normalized
        for part in cls._specialized_part_keys():
            value = raw_sources.get(part, "")
            if isinstance(value, dict):
                value = value.get("custom_path") or value.get("path") or ""
            normalized[part] = str(value or "").strip()
        return normalized

    def _load_specialized_models(self) -> None:
        for part, source in self.specialized_model_sources.items():
            if not source:
                continue
            try:
                self.specialized_models[part] = self._YOLO(source)
                print(f"[INFO] BBox specialized model loaded for {part}: {source}")
            except Exception as exc:
                print(f"[WARN] Failed to load specialized BBox model for {part} ({source}): {exc}")

    def _resolve_detector_channel(self) -> Tuple[object, str]:
        if self.use_specialized_channels and self.target_part in self._specialized_part_keys():
            detector = self.specialized_models.get(self.target_part)
            if detector is not None:
                return detector, "specialized"
        return self.model, "main"

    def _names(self, model=None) -> Dict[int, str]:
        detector = model or self.model
        names = getattr(detector, "names", {}) or {}
        if isinstance(names, list):
            return {i: str(name) for i, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {}

    def _is_person_label(self, class_id: int, label: str) -> bool:
        lowered = (label or "").strip().lower()
        if lowered == "person":
            return True
        if lowered in {"human", "people"}:
            return True
        # COCO person class id.
        if class_id == 0:
            return True
        return False

    def _compute_score(self, detections: List[Dict]) -> float:
        if not detections:
            return 0.0
        confidences = [float(item.get("confidence", 0.0)) for item in detections]
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        count = len(detections)
        # Confidence dominates; count gives a small boost for useful multi-person scenes.
        score = mean_conf * 85.0 + min(20.0, count * 6.0)
        return round(float(max(0.0, min(100.0, score))), 2)

    def process_pil_image(self, image: Image.Image) -> Dict:
        rgb = np.array(image.convert("RGB"))
        detector, detector_channel = self._resolve_detector_channel()
        results = self._predict_rgb(rgb, model=detector)
        names = self._names(detector)
        detections = self._extract_detections(results, names, detector_channel=detector_channel)
        detections.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)

        review_payload = None
        if self.review_pipeline.enabled and detections:
            try:
                review_payload = self.review_pipeline.review(rgb, detections, target_part=self.target_part)
                detections = list(review_payload.get("detections", detections))
            except Exception as exc:
                review_payload = {
                    "enabled": True,
                    "detections": detections,
                    "summary": {"warning": f"bbox_review_failed: {exc}"},
                }

        vis = self._visualize_detections(rgb, detections)

        score = self._compute_score(detections)
        count = len(detections)
        mean_conf = float(np.mean([d.get("confidence", 0.0) for d in detections])) if detections else 0.0
        max_conf = float(max([d.get("confidence", 0.0) for d in detections])) if detections else 0.0

        warning = self._warning_from_stats(count, mean_conf, self.target_part)
        quality_flag = self._quality_flag_from_score(score)
        review_summary = review_payload.get("summary", {}) if isinstance(review_payload, dict) else {}
        review_score = float(review_summary.get("aggregate_score", 0.0) or 0.0)
        fail_count = int(review_summary.get("fail_count", 0) or 0)
        review_count = int(review_summary.get("review_count", 0) or 0)

        if self.review_pipeline.enabled and self.review_pipeline.override_score:
            score = round(float(score * 0.55 + review_score * 100.0 * 0.45), 2)
            quality_flag = self._quality_flag_from_score(score)

        if fail_count > 0:
            warning = warning or "BBox center review found failed boxes"
            if quality_flag == "auto_accept":
                quality_flag = "user_review"
        elif review_count > 0:
            warning = warning or "BBox center review needs manual confirmation"
            if quality_flag == "auto_accept":
                quality_flag = "user_review"

        return {
            "visualized_image": Image.fromarray(vis),
            "detections": detections,
            "score": score,
            "warning": warning,
            "quality_flag": quality_flag,
            "target_part": self.target_part,
            "detector_channel": detector_channel,
            "metrics": {
                "count": count,
                "mean_confidence": mean_conf,
                "max_confidence": max_conf,
                "target_part": self.target_part,
                "detector_channel": detector_channel,
                "review_enabled": bool(self.review_pipeline.enabled),
                "review_score": float(review_score),
                "review_fail_count": int(fail_count),
                "review_need_check_count": int(review_count),
            },
            "review": review_payload,
        }

    def _predict_rgb(self, rgb: np.ndarray, model=None):
        detector = model or self.model
        return detector.predict(
            source=rgb,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False,
        )

    def _extract_detections(self, results, names: Dict[int, str], detector_channel: str = "main") -> List[Dict]:
        detections: List[Dict] = []
        if not results:
            return detections
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.empty((0, 4))
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.empty((0,))
        cls = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else np.zeros((len(xyxy),))

        for i in range(len(xyxy)):
            class_id = int(cls[i]) if i < len(cls) else -1
            label = names.get(class_id, str(class_id))
            is_person = self._is_person_label(class_id, label)
            if detector_channel == "main":
                if self.target_part != "person" and not is_person:
                    continue
                if self.target_part == "person" and self.person_only and not is_person:
                    continue
            elif self.target_part == "person" and self.person_only and not is_person:
                continue
            box = xyxy[i].tolist()
            if detector_channel == "specialized":
                target_box = self._normalize_box(box)
            else:
                target_box = self._derive_target_box(box)
            if target_box is None:
                continue
            detections.append(
                {
                    "class_id": class_id,
                    "label": self._target_part_label(label),
                    "confidence": float(conf[i]) if i < len(conf) else 0.0,
                    "xyxy": target_box,
                    "target_part": self.target_part,
                    "source_label": label,
                    "detector_channel": detector_channel,
                }
            )
        return detections

    def _target_part_label(self, source_label: str) -> str:
        if self.target_part == "person":
            return source_label
        if self.target_part == "head":
            return "head_part"
        if self.target_part == "upper_body":
            return "upper_body_part"
        if self.target_part == "lower_body":
            return "lower_body_part"
        return source_label

    def _normalize_box(self, xyxy: List[float]) -> Optional[List[float]]:
        if len(xyxy) < 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        width = x2 - x1
        height = y2 - y1
        if width <= 1.0 or height <= 1.0:
            return None
        return [x1, y1, x2, y2]

    def _derive_target_box(self, xyxy: List[float]) -> Optional[List[float]]:
        base_box = self._normalize_box(xyxy)
        if base_box is None:
            return None
        x1, y1, x2, y2 = base_box
        width = x2 - x1
        height = y2 - y1

        if self.target_part == "person":
            return [x1, y1, x2, y2]
        if self.target_part == "head":
            return self._build_head_box(x1, y1, width, height)
        if self.target_part == "upper_body":
            return self._build_upper_body_box(x1, y1, width, height)
        if self.target_part == "lower_body":
            return self._build_lower_body_box(x1, y1, width, height)
        return [x1, y1, x2, y2]

    @staticmethod
    def _build_head_box(x1: float, y1: float, width: float, height: float) -> List[float]:
        return [
            x1 + width * 0.18,
            y1 + height * 0.00,
            x1 + width * 0.82,
            y1 + height * 0.36,
        ]

    @staticmethod
    def _build_upper_body_box(x1: float, y1: float, width: float, height: float) -> List[float]:
        return [
            x1 + width * 0.08,
            y1 + height * 0.10,
            x1 + width * 0.92,
            y1 + height * 0.62,
        ]

    @staticmethod
    def _build_lower_body_box(x1: float, y1: float, width: float, height: float) -> List[float]:
        return [
            x1 + width * 0.08,
            y1 + height * 0.45,
            x1 + width * 0.92,
            y1 + height * 1.00,
        ]

    @staticmethod
    def _visualize_detections(rgb: np.ndarray, detections: List[Dict]) -> np.ndarray:
        vis = rgb.copy()
        height, width = vis.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["xyxy"]
            x1 = max(0, min(width - 1, int(round(x1))))
            y1 = max(0, min(height - 1, int(round(y1))))
            x2 = max(0, min(width - 1, int(round(x2))))
            y2 = max(0, min(height - 1, int(round(y2))))
            color = (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            text = f"{det.get('label', 'obj')} {float(det.get('confidence', 0.0)):.2f}"
            cv2.putText(vis, text, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return vis

    @staticmethod
    def _warning_from_stats(count: int, mean_conf: float, target_part: str = "person") -> Optional[str]:
        if count == 0:
            if target_part == "person":
                return "No bounding boxes detected"
            return f"No {target_part} bounding boxes detected"
        if mean_conf < 0.35:
            return "Low confidence detections"
        return None

    @staticmethod
    def _quality_flag_from_score(score: float) -> str:
        if score >= 80:
            return "auto_accept"
        if score <= 40:
            return "auto_reject"
        return "user_review"
