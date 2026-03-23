"""
BBox Processor based on Ultralytics YOLO.
Generates a visualized bbox map and structured detection metadata.
"""
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


class BBoxProcessor:
    """Simple wrapper around YOLO object detection for dataset curation."""

    def __init__(
        self,
        model_source: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 20,
        person_only: bool = True,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for BBox detection. Install with: pip install ultralytics") from exc

        self.model = YOLO(model_source)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.person_only = bool(person_only)

    def _names(self) -> Dict[int, str]:
        names = getattr(self.model, "names", {}) or {}
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
        results = self.model.predict(
            source=rgb,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False,
        )

        names = self._names()
        detections: List[Dict] = []

        if results:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.empty((0, 4))
                conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.empty((0,))
                cls = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else np.zeros((len(xyxy),))

                for i in range(len(xyxy)):
                    class_id = int(cls[i]) if i < len(cls) else -1
                    label = names.get(class_id, str(class_id))
                    if self.person_only and not self._is_person_label(class_id, label):
                        continue
                    box = xyxy[i].tolist()
                    detections.append(
                        {
                            "class_id": class_id,
                            "label": label,
                            "confidence": float(conf[i]) if i < len(conf) else 0.0,
                            "xyxy": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        }
                    )

        detections.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)

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

        score = self._compute_score(detections)
        count = len(detections)
        mean_conf = float(np.mean([d.get("confidence", 0.0) for d in detections])) if detections else 0.0
        max_conf = float(max([d.get("confidence", 0.0) for d in detections])) if detections else 0.0

        warning: Optional[str] = None
        if count == 0:
            warning = "No bounding boxes detected"
        elif mean_conf < 0.35:
            warning = "Low confidence detections"

        if score >= 80:
            quality_flag = "auto_accept"
        elif score <= 40:
            quality_flag = "auto_reject"
        else:
            quality_flag = "user_review"

        return {
            "visualized_image": Image.fromarray(vis),
            "detections": detections,
            "score": score,
            "warning": warning,
            "quality_flag": quality_flag,
            "metrics": {
                "count": count,
                "mean_confidence": mean_conf,
                "max_confidence": max_conf,
            },
        }
