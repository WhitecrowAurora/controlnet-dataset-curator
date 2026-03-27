"""
ViTPose Processor - YOLO + ViTPose ONNX implementation
Provides pose estimation without controlnet_aux dependency
"""

import cv2
import numpy as np
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image


class VitPoseProcessor:
    """ViTPose processor using YOLO for detection and ViTPose ONNX for pose estimation"""

    # COCO 17 keypoint indices
    COCO_KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # OpenPose 18 keypoint order
    OPENPOSE_KEYPOINTS = [
        'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
        'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip',
        'right_knee', 'right_ankle', 'left_hip', 'left_knee',
        'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear'
    ]

    def __init__(self, yolo_model_path: str, vitpose_model_path: str, device: str = 'cuda'):
        """
        Initialize ViTPose processor

        Args:
            yolo_model_path: Path to YOLO model (e.g., 'yolov8n.pt')
            vitpose_model_path: Path to ViTPose ONNX model
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.yolo_model_path = yolo_model_path
        self.vitpose_model_path = vitpose_model_path

        # Initialize models
        self._init_yolo()
        self._init_vitpose()

    def _init_yolo(self):
        """Initialize YOLO detector"""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.yolo_model_path)
            print(f"[INFO] Loaded YOLO model: {self.yolo_model_path}")
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def _init_vitpose(self):
        """Initialize ViTPose ONNX session"""
        try:
            import onnxruntime as ort

            # Check if model file exists
            if not Path(self.vitpose_model_path).exists():
                raise FileNotFoundError(f"ViTPose model not found: {self.vitpose_model_path}")

            # Configure ONNX Runtime providers
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']

            self.vitpose_session = ort.InferenceSession(
                self.vitpose_model_path,
                providers=providers
            )
            print(f"[INFO] Loaded ViTPose model: {self.vitpose_model_path}")
        except ImportError:
            raise ImportError("onnxruntime-gpu not installed. Run: pip install onnxruntime-gpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load ViTPose model: {e}")

    def detect_persons(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect persons in image using YOLO

        Args:
            image: BGR image (OpenCV format)

        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        # Lower confidence threshold for better detection
        results = self.yolo_model(image, classes=[0], conf=0.2, verbose=False)  # class 0 = person, conf=0.2

        boxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes.xyxy.cpu().numpy():
                    boxes.append(box)

        return boxes

    def estimate_pose(self, cropped_image: np.ndarray, bbox_shape: Tuple[int, int]) -> np.ndarray:
        """
        Estimate pose for a cropped person image

        Args:
            cropped_image: Cropped person image (BGR)
            bbox_shape: Original bounding box (width, height) for dynamic thickness

        Returns:
            COCO 17 keypoints array (17, 3) with [x, y, confidence]
        """
        # ViTPose input size (height, width)
        input_size = (256, 192)

        # Resize crop
        resized = cv2.resize(cropped_image, (input_size[1], input_size[0]))

        # Preprocess: normalize and convert to NCHW
        blob = self._preprocess(resized)

        # Run inference
        ort_inputs = {self.vitpose_session.get_inputs()[0].name: blob}
        heatmaps = self.vitpose_session.run(None, ort_inputs)[0]

        # Extract keypoints from heatmaps
        keypoints = self._extract_keypoints(heatmaps, bbox_shape)

        return keypoints

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ViTPose"""
        # Normalize using ImageNet stats
        blob = image.astype(np.float32) / 255.0
        blob -= np.array([0.485, 0.456, 0.406])
        blob /= np.array([0.229, 0.224, 0.225])

        # HWC to CHW
        blob = np.transpose(blob, (2, 0, 1))

        # Add batch dimension
        blob = np.expand_dims(blob, axis=0)

        return blob

    def _extract_keypoints(self, heatmaps: np.ndarray, bbox_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract keypoints from heatmaps using argmax

        Args:
            heatmaps: Output heatmaps from ViTPose (1, 17, H, W)
            bbox_shape: Original bounding box (width, height)

        Returns:
            COCO 17 keypoints (17, 3)
        """
        # Remove batch dimension
        heatmaps = heatmaps[0]  # (17, H, W)

        keypoints = []
        for i in range(17):  # COCO has 17 keypoints
            heatmap = heatmaps[i]

            # Find max location
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            confidence = heatmap[y, x]

            # Scale back to original crop size
            scale_x = bbox_shape[0] / heatmap.shape[1]
            scale_y = bbox_shape[1] / heatmap.shape[0]

            keypoints.append([
                x * scale_x,
                y * scale_y,
                confidence
            ])

        return np.array(keypoints)

    def calculate_neck_coordinate(self, keypoints: np.ndarray, conf_thresh: float = 0.4) -> np.ndarray:
        """
        Calculate neck coordinate from COCO 17 keypoints
        Uses confidence-weighted interpolation

        Args:
            keypoints: COCO 17 keypoints (17, 3)
            conf_thresh: Minimum confidence threshold

        Returns:
            Neck keypoint [x, y, confidence]
        """
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]

        # Both shoulders visible
        if l_shoulder[2] > conf_thresh and r_shoulder[2] > conf_thresh:
            neck_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
            neck_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
            neck_conf = min(l_shoulder[2], r_shoulder[2])
            return np.array([neck_x, neck_y, neck_conf])

        # Single shoulder visible - use biomechanical estimation
        vis_shoulder = None
        vis_ear = None

        if l_shoulder[2] > conf_thresh:
            vis_shoulder = l_shoulder
            vis_ear = keypoints[3] if keypoints[3][2] > conf_thresh else keypoints[1]
        elif r_shoulder[2] > conf_thresh:
            vis_shoulder = r_shoulder
            vis_ear = keypoints[4] if keypoints[4][2] > conf_thresh else keypoints[2]

        if vis_shoulder is not None and vis_ear[2] > conf_thresh:
            # Spatial vector derivation
            dx = vis_ear[0] - vis_shoulder[0]
            neck_x = vis_shoulder[0] + dx * 0.3  # Horizontal offset coefficient
            neck_y = vis_shoulder[1]
            neck_conf = vis_shoulder[2] * 0.8  # Confidence penalty
            return np.array([neck_x, neck_y, neck_conf])

        # Fallback: return zero confidence
        return np.array([0.0, 0.0, 0.0])

    def coco_to_openpose(self, coco_keypoints: np.ndarray) -> List[np.ndarray]:
        """
        Convert COCO 17 to OpenPose 18 format

        Args:
            coco_keypoints: COCO 17 keypoints (17, 3)

        Returns:
            OpenPose 18 keypoints list
        """
        neck = self.calculate_neck_coordinate(coco_keypoints)

        # OpenPose order: nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist,
        #                 r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, r_eye, l_eye, r_ear, l_ear
        openpose_points = [
            coco_keypoints[0],   # nose
            neck,                # neck (interpolated)
            coco_keypoints[6],   # r_shoulder
            coco_keypoints[8],   # r_elbow
            coco_keypoints[10],  # r_wrist
            coco_keypoints[5],   # l_shoulder
            coco_keypoints[7],   # l_elbow
            coco_keypoints[9],   # l_wrist
            coco_keypoints[12],  # r_hip
            coco_keypoints[14],  # r_knee
            coco_keypoints[16],  # r_ankle
            coco_keypoints[11],  # l_hip
            coco_keypoints[13],  # l_knee
            coco_keypoints[15],  # l_ankle
            coco_keypoints[2],   # r_eye
            coco_keypoints[1],   # l_eye
            coco_keypoints[4],   # r_ear
            coco_keypoints[3],   # l_ear
        ]

        return openpose_points

    def calculate_dynamic_thickness(self, bbox_width: int, bbox_height: int, base_resolution: float = 512.0) -> Dict[str, int]:
        """
        Calculate dynamic line thickness based on bounding box size

        Args:
            bbox_width: Bounding box width
            bbox_height: Bounding box height
            base_resolution: Base resolution for scaling

        Returns:
            Dict with thickness values
        """
        max_dim = max(bbox_width, bbox_height)
        scale_factor = max_dim / base_resolution

        return {
            'body': max(1, int(math.ceil(4.0 * scale_factor))),
            'circle': max(1, int(math.ceil(4.0 * scale_factor)))
        }

    def draw_skeleton(self, canvas: np.ndarray, openpose_points: List[np.ndarray], thickness: int = 4) -> np.ndarray:
        """
        Draw OpenPose skeleton on canvas

        Args:
            canvas: Output canvas (BGR)
            openpose_points: OpenPose 18 keypoints
            thickness: Line thickness

        Returns:
            Canvas with skeleton drawn
        """
        # OpenPose limb connections
        limb_seq = [
            (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),  # body + arms
            (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),  # legs
            (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)  # head
        ]

        # Colors (BGR format)
        colors = [
            [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
            [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
            [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
            [255, 0, 255], [255, 0, 170], [255, 0, 85]
        ]

        # Confidence threshold for drawing. ViTPose heatmaps can contain low-confidence edge peaks
        # that produce "spider" skeletons, especially on anime/portraits.
        draw_conf_thresh = 0.2

        # Draw limbs
        for i, (p1_idx, p2_idx) in enumerate(limb_seq):
            pt1 = openpose_points[p1_idx]
            pt2 = openpose_points[p2_idx]

            # Skip if confidence too low
            if pt1[2] < draw_conf_thresh or pt2[2] < draw_conf_thresh:
                continue

            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])

            # Draw elliptical polygon (like original OpenPose)
            length = math.hypot(x2 - x1, y2 - y1)
            if length > 0:
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                polygon = cv2.ellipse2Poly(
                    (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    (int(length / 2), thickness),
                    int(angle), 0, 360, 1
                )
                cv2.fillConvexPoly(canvas, polygon, colors[i])

        # Draw joints
        for i, pt in enumerate(openpose_points):
            if pt[2] > draw_conf_thresh:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), thickness, colors[i], thickness=-1)

        return canvas

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process image and generate OpenPose control map

        Args:
            image: Input image (BGR or RGB)

        Returns:
            OpenPose control map (BGR)
        """
        image = self._ensure_bgr_image(image)
        canvas = np.zeros_like(image)

        boxes = self.detect_persons(image)
        print(f"[DEBUG] ViTPose detected {len(boxes)} person(s)")
        boxes = self._select_primary_person_boxes(boxes)

        # Heuristic guardrails to avoid drawing obviously invalid poses.
        # These thresholds are intentionally conservative.
        min_keypoint_conf = 0.3
        min_valid_keypoints = 6

        for box in boxes:
            canvas = self._draw_person_pose_on_canvas(
                canvas=canvas,
                image=image,
                box=box,
                min_keypoint_conf=min_keypoint_conf,
                min_valid_keypoints=min_valid_keypoints,
            )

        return canvas

    @staticmethod
    def _ensure_bgr_image(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    @staticmethod
    def _select_primary_person_boxes(boxes: List[List[float]]) -> List[List[float]]:
        if len(boxes) <= 1:
            return boxes
        return sorted(boxes, key=lambda b: float((b[2] - b[0]) * (b[3] - b[1])), reverse=True)[:1]

    def _draw_person_pose_on_canvas(
        self,
        *,
        canvas: np.ndarray,
        image: np.ndarray,
        box,
        min_keypoint_conf: float,
        min_valid_keypoints: int,
    ) -> np.ndarray:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return canvas

        bbox_shape = (x2 - x1, y2 - y1)
        coco_keypoints = self.estimate_pose(crop, bbox_shape)
        coco_keypoints = self._clamp_coco_keypoints(coco_keypoints, bbox_shape)

        if not self._is_plausible_coco_pose(coco_keypoints, min_keypoint_conf, min_valid_keypoints):
            return canvas

        openpose_points = self.coco_to_openpose(coco_keypoints)
        for pt in openpose_points:
            pt[0] += x1
            pt[1] += y1

        thickness = self.calculate_dynamic_thickness(bbox_shape[0], bbox_shape[1])
        return self.draw_skeleton(canvas, openpose_points, thickness['body'])

    @staticmethod
    def _clamp_coco_keypoints(coco_keypoints: np.ndarray, bbox_shape: tuple) -> np.ndarray:
        coco_keypoints[:, 0] = np.clip(coco_keypoints[:, 0], 0.0, float(bbox_shape[0] - 1))
        coco_keypoints[:, 1] = np.clip(coco_keypoints[:, 1], 0.0, float(bbox_shape[1] - 1))
        return coco_keypoints

    @staticmethod
    def _is_plausible_coco_pose(coco_keypoints: np.ndarray, min_keypoint_conf: float, min_valid_keypoints: int) -> bool:
        conf = coco_keypoints[:, 2]
        valid_keypoints = int(np.sum(conf > min_keypoint_conf))
        print(f"[DEBUG] ViTPose found {valid_keypoints}/17 valid keypoints (conf > {min_keypoint_conf})")
        has_shoulder = bool(conf[5] > min_keypoint_conf or conf[6] > min_keypoint_conf)
        has_hip = bool(conf[11] > min_keypoint_conf or conf[12] > min_keypoint_conf)
        return valid_keypoints >= min_valid_keypoints and has_shoulder and has_hip

    def process_pil_image(self, pil_image: Image.Image) -> Image.Image:
        """
        Process PIL image and return control map

        Args:
            pil_image: Input PIL image

        Returns:
            OpenPose control map as PIL image
        """
        # Convert PIL to OpenCV (RGB -> BGR)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Process
        control_map = self.process_image(image)

        # Convert back to PIL (BGR -> RGB)
        control_map_rgb = cv2.cvtColor(control_map, cv2.COLOR_BGR2RGB)

        return Image.fromarray(control_map_rgb)


class PoseProcessorManager:
    """Manager to switch between different pose processors (DWpose vs ViTPose)"""

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.processor_type = None
        self.processor = None

    def initialize(self, model_type: str, **kwargs):
        """
        Initialize pose processor

        Args:
            model_type: 'DWpose', 'SDPose-Wholebody', 'ViTPose', 'Openpose', or 'Custom Path'
            **kwargs: Additional arguments (custom_path, yolo_path, vitpose_path, etc.)
        """
        self.processor_type = model_type

        if model_type in ['SDPose-Wholebody', 'ViTPose']:
            # Use ViTPoseProcessor
            yolo_path = kwargs.get('yolo_path', 'yolov8n.pt')
            vitpose_path = kwargs.get('vitpose_path', 'vitpose_huge.onnx')

            self.processor = VitPoseProcessor(yolo_path, vitpose_path, self.device)
            print(f"[INFO] Initialized ViTPoseProcessor for {model_type}")

        elif model_type == 'Custom Path' and 'custom_path' in kwargs:
            # Custom DWpose model
            from controlnet_aux import DWposeDetector
            self.processor = DWposeDetector.from_pretrained(kwargs['custom_path'], device=self.device)
            print(f"[INFO] Loaded custom DWpose model from {kwargs['custom_path']}")

        elif model_type == 'Openpose':
            from controlnet_aux import OpenposeDetector
            self.processor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            if hasattr(self.processor, 'to'):
                self.processor.to(self.device)
            print(f"[INFO] Loaded Openpose model on {self.device}")

        else:
            # Default: DWpose
            from controlnet_aux import DWposeDetector
            self.processor = DWposeDetector(device=self.device)
            print("[INFO] Loaded DWpose (Default) model")

    def __call__(self, image):
        """
        Process image - unified interface

        Args:
            image: PIL Image or numpy array

        Returns:
            OpenPose control map
        """
        if self.processor is None:
            raise RuntimeError("Processor not initialized. Call initialize() first.")

        # Handle different processor types
        if isinstance(self.processor, VitPoseProcessor):
            # ViTPoseProcessor expects PIL
            from PIL import Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            return self.processor.process_pil_image(image)
        else:
            # controlnet_aux processors expect PIL
            from PIL import Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            return self.processor(image)

    def is_vitpose(self) -> bool:
        """Check if using ViTPose processor"""
        return isinstance(self.processor, VitPoseProcessor)
