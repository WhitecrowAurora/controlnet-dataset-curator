"""
ControlNet Processor with Smart Retry Strategy
Generates Canny/OpenPose/Depth variants and scores them
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import tempfile

from .jsona_backup import JsonaBackupManager

# Lazy imports for heavy dependencies
# torch and controlnet_aux will be imported when needed


class ControlNetProcessor:
    """Process images with ControlNet models and smart retry"""

    def __init__(self, config: dict):
        self.config = config

        # Load enabled models
        # Support both nested (processing.control_types) and flat (control_types) structures
        processing_config = config.get('processing', {})
        if 'control_types' in processing_config:
            self.enabled_types = processing_config['control_types']
        else:
            self.enabled_types = config.get('control_types', {})

        self.canny_detector = None
        self.dwpose_detector = None
        self.zoe_detector = None
        self.torch = None
        self.device = None

        # Get parallel processing threads from config
        self.parallel_threads = processing_config.get('parallel_threads', 3)
        self.jsona_backup_manager = JsonaBackupManager(
            backup_interval_entries=processing_config.get('jsona_backup_every_entries', 200),
            rolling_keep=processing_config.get('jsona_backup_keep', 10),
            backup_interval_seconds=processing_config.get('jsona_backup_every_seconds', 600),
        )

        need_canny = bool(self.enabled_types.get('canny', False))
        need_openpose = bool(self.enabled_types.get('openpose', False))
        need_depth = bool(self.enabled_types.get('depth', False))

        # Canny runs via OpenCV to avoid importing controlnet_aux (its package __init__
        # imports torch and other heavy deps). This keeps "GUI-only / Canny-only" usable.
        if need_canny:
            self.canny_detector = None

        # Initialize depth anything flag
        self.use_depth_anything = False

        # OpenPose/Depth require torch + controlnet_aux.
        if need_openpose or need_depth:
            self._ensure_temp_dir()

            # Fix torch DLL loading in GUI environment - AGGRESSIVE FIX
            import sys
            import ctypes
            import ctypes.util

            # Method 1: Add to PATH
            torch_lib_path = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'torch', 'lib')
            if os.path.exists(torch_lib_path):
                # Clear and re-add to ensure it's at the front
                path_parts = os.environ.get('PATH', '').split(os.pathsep)
                # Remove any existing torch lib paths
                path_parts = [p for p in path_parts if 'torch' not in p.lower() or 'lib' not in p.lower()]
                # Add torch lib at the very front
                os.environ['PATH'] = torch_lib_path + os.pathsep + os.pathsep.join(path_parts)

                # Method 2: Add DLL directory (Python 3.8+)
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(torch_lib_path)
                        print(f"[DEBUG] Added DLL directory: {torch_lib_path}")
                    except Exception as e:
                        print(f"[DEBUG] Failed to add DLL directory: {e}")

                # Method 3: Try to preload dependencies in correct order
                try:
                    # Load dependencies first
                    dll_order = [
                        'fbgemm.dll',
                        'asmjit.dll',
                        'cpuinfo.dll',
                        'uv.dll',
                        'c10.dll',
                        'torch_cpu.dll'
                    ]

                    for dll_name in dll_order:
                        dll_path = os.path.join(torch_lib_path, dll_name)
                        if os.path.exists(dll_path):
                            try:
                                ctypes.CDLL(dll_path)
                                print(f"[DEBUG] Preloaded {dll_name}")
                            except Exception as e:
                                print(f"[DEBUG] Failed to preload {dll_name}: {e}")
                                # Continue anyway, some DLLs might not exist
                except Exception as e:
                    print(f"[DEBUG] Error during DLL preloading: {e}")

            try:
                import torch
                from controlnet_aux import ZoeDetector
                print(f"[DEBUG] Successfully imported torch {torch.__version__}")
            except ImportError as e:
                raise ImportError(
                    f"Required dependencies not installed: {e}\n"
                    "Install with: install_dependencies.bat (or pip install torch controlnet-aux)"
                )

            self.torch = torch
            self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

            # Get model configuration
            model_config = processing_config.get('model_config', {})

            if need_openpose:
                from .vitpose_processor import PoseProcessorManager

                openpose_config = model_config.get('openpose', {})
                model_type = openpose_config.get('type', 'DWpose (Default)')
                custom_path = openpose_config.get('custom_path')

                # Initialize pose processor manager for seamless switching
                self.pose_manager = PoseProcessorManager(device=str(self.device))

                if model_type in ['SDPose-Wholebody', 'ViTPose']:
                    # Use YOLO + ViTPose pipeline
                    # Get YOLO version and model type
                    yolo_version = openpose_config.get('yolo_version', 'YOLO26 (推荐)')
                    yolo_model_type = openpose_config.get('yolo_model_type', '通用 (General)')

                    # Determine YOLO model path based on type
                    if '动漫专用' in yolo_model_type:
                        yolo_path = 'models/yolo/anime_person_detect_v1.3_s.pt'
                    elif 'YOLO26' in yolo_version:
                        yolo_path = 'models/yolo/yolo26n.pt'
                    elif 'YOLOv11' in yolo_version:
                        yolo_path = 'models/yolo/yolo11n.pt'
                    else:  # YOLOv8
                        yolo_path = 'models/yolo/yolov8n.pt'

                    # Different ViTPose models for different tasks
                    if model_type == 'SDPose-Wholebody':
                        vitpose_path = openpose_config.get('vitpose_path', 'models/vitpose/vitpose-l-wholebody.onnx')
                    else:  # ViTPose
                        vitpose_path = openpose_config.get('vitpose_path', 'models/vitpose/vitpose-l-coco.onnx')

                    self.pose_manager.initialize(
                        model_type,
                        yolo_path=yolo_path,
                        vitpose_path=vitpose_path
                    )

                    # Show which detector is being used
                    detector_name = "动漫专用检测器" if '动漫专用' in yolo_model_type else yolo_version
                    print(f"[INFO] Using {model_type} with {detector_name} + ViTPose pipeline")
                elif model_type == 'Custom Path' and custom_path:
                    self.pose_manager.initialize('Custom Path', custom_path=custom_path)
                elif model_type == 'Openpose':
                    self.pose_manager.initialize('Openpose')
                else:
                    # Default: DWpose
                    self.pose_manager.initialize('DWpose (Default)')
                    print("[INFO] Using DWpose (Default) model")

                # Keep reference for backward compatibility
                self.dwpose_detector = self.pose_manager

            if need_depth:
                depth_config = model_config.get('depth', {})
                model_type = depth_config.get('type', 'Depth Anything V2')
                custom_path = depth_config.get('custom_path')

                print(f"[DEBUG] Using device: {self.device}")

                # Only support Depth Anything V2 or Custom Path
                if model_type == 'Custom Path' and custom_path:
                    # Load custom Depth Anything V2 model from path
                    try:
                        from depth_anything_v2.dpt import DepthAnythingV2
                        import torch

                        # Try to load from custom path
                        # Assume custom path points to a checkpoint file
                        model_configs = {
                            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                        }

                        encoder = 'vitl'  # Default to large
                        self.depth_anything_model = DepthAnythingV2(**model_configs[encoder])
                        self.depth_anything_model.load_state_dict(torch.load(custom_path, map_location='cpu'))
                        self.depth_anything_model = self.depth_anything_model.to(self.device).eval()
                        self.zoe_detector = None
                        self.use_depth_anything = True
                        print(f"[INFO] Loaded custom Depth Anything V2 model from {custom_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to load custom model: {e}")
                        raise ImportError(f"Failed to load custom Depth Anything V2 model: {e}")
                else:
                    # Use Depth Anything V2 (default)
                    try:
                        from depth_anything_v2.dpt import DepthAnythingV2
                        import torch

                        # Initialize Depth Anything V2 model
                        model_configs = {
                            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                        }

                        encoder = 'vitl'  # Use large model for best quality
                        self.depth_anything_model = DepthAnythingV2(**model_configs[encoder])
                        self.depth_anything_model.load_state_dict(torch.hub.load_state_dict_from_url(
                            f'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_{encoder}.pth',
                            map_location='cpu'
                        ))
                        self.depth_anything_model = self.depth_anything_model.to(self.device).eval()
                        self.zoe_detector = None  # Use custom processing
                        self.use_depth_anything = True
                        print("[INFO] Using Depth Anything V2 (Large) model")
                    except ImportError as e:
                        print("[ERROR] depth-anything-v2 not installed!")
                        print("[INFO] Install with: pip install depth-anything-v2")
                        print("[INFO] Or use Tools -> Install Depth Anything V2 from menu")
                        raise ImportError(f"depth-anything-v2 is required for Depth processing. Install it first: {e}")

                print(f"[DEBUG] Depth detector loaded on device: {next(self.zoe_detector.model.parameters()).device if hasattr(self.zoe_detector, 'model') else 'unknown'}")

        # Retry configuration (nested under processing)
        if 'retry_strategy' in processing_config:
            self.retry_config = processing_config['retry_strategy']
        else:
            self.retry_config = config.get('retry_strategy', {})
        self.retry_enabled = self.retry_config.get('enabled', True)
        self.max_retries = self.retry_config.get('max_retries', 2)
        self.retry_threshold = self.retry_config.get('retry_threshold', 40)
        self.random_offset = self.retry_config.get('random_offset_range', 20)

        # Canny presets
        self.canny_presets = [
            {"name": "light", "low": 30, "high": 120},
            {"name": "medium", "low": 50, "high": 150},
            {"name": "clean", "low": 80, "high": 180},
            {"name": "strong", "low": 120, "high": 220},
        ]

        # OpenPose presets (different detection sensitivity)
        self.openpose_presets = [
            {"name": "minimal", "include_hands": False, "include_face": False},
            {"name": "body_only", "include_hands": False, "include_face": True},
            {"name": "standard", "include_hands": True, "include_face": False},
            {"name": "full", "include_hands": True, "include_face": True},
        ]

        # Depth presets (we'll use different post-processing or parameters if available)
        # Since ZoeDetector doesn't expose many parameters, we'll generate 4 variants
        # by applying different contrast/brightness adjustments
        self.depth_presets = [
            {"name": "soft", "gamma": 0.8, "contrast": 0.9},
            {"name": "normal", "gamma": 1.0, "contrast": 1.0},
            {"name": "enhanced", "gamma": 1.2, "contrast": 1.1},
            {"name": "strong", "gamma": 1.4, "contrast": 1.2},
        ]

    def _ensure_temp_dir(self) -> None:
        # Some environments have no usable temp dir, which can break torch import.
        try:
            tempfile.gettempdir()
            return
        except Exception:
            pass

        tmp_dir = Path.cwd() / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TEMP"] = str(tmp_dir)
        os.environ["TMP"] = str(tmp_dir)

        # Verify we now have a usable temp directory.
        tempfile.gettempdir()

    def process_image(self, image_path: str, output_dir: str, basename: str) -> Dict:
        """
        Process single image and generate all enabled control types (parallel processing)

        Returns:
            {
                'canny': {
                    'variants': [{'path': str, 'score': float, 'preset': str, 'thresholds': dict}, ...],
                    'best_score': float,
                    'retry_count': int,
                    'quality_flag': str  # 'auto_accept', 'auto_reject', 'user_review'
                },
                'openpose': {
                    'variants': [{'path': str, 'score': float, 'preset': str, 'warning': str, 'visibility_ratio': float}, ...],
                    'best_score': float,
                    'quality_flag': str
                },
                'depth': {
                    'variants': [{'path': str, 'score': float, 'preset': str, 'warning': str, 'metrics': dict}, ...],
                    'best_score': float,
                    'quality_flag': str
                }
            }
        """
        with Image.open(image_path) as _im:
            img = _im.convert("RGB")

        results = {}

        # Use ThreadPoolExecutor for parallel processing
        from concurrent.futures import ThreadPoolExecutor, as_completed

        tasks = {}  # future -> task_name mapping

        with ThreadPoolExecutor(max_workers=self.parallel_threads) as executor:
            # Submit tasks for enabled control types
            if bool(self.enabled_types.get('canny', False)):
                future = executor.submit(self._process_canny_with_retry, img, output_dir, basename)
                tasks[future] = 'canny'

            if self.dwpose_detector:
                future = executor.submit(self._process_openpose, img, output_dir, basename)
                tasks[future] = 'openpose'

            if hasattr(self, 'depth_anything_model') and self.depth_anything_model:
                future = executor.submit(self._process_depth, img, output_dir, basename)
                tasks[future] = 'depth'

            # Collect results as they complete
            for future in as_completed(tasks):
                task_name = tasks[future]
                try:
                    result = future.result()
                    results[task_name] = result
                except Exception as e:
                    print(f"[ERROR] Failed to process {task_name}: {e}")
                    import traceback
                    traceback.print_exc()

        # NOTE: JSON metadata is now written only when image is accepted
        # See main_window.py _save_accepted() -> _write_jsona_metadata()
        # self._generate_json_metadata(image_path, output_dir, basename, results)

        return results

    def _generate_json_metadata(self, image_path: str, output_dir: str, basename: str, results: Dict):
        """Generate JSON metadata for auto-accepted variants only"""
        import json
        import time

        # Get paths
        image_dir = os.path.dirname(image_path)
        extract_dir = os.path.dirname(image_dir)
        tag_file = os.path.join(extract_dir, 'tags', f"{basename}.txt")

        # Read tags if available
        hint_prompt = ""
        if os.path.exists(tag_file):
            with open(tag_file, 'r', encoding='utf-8') as f:
                hint_prompt = f.read().strip()

        # Map control type to folder name
        folder_map = {
            'canny': 'canny',
            'openpose': 'pose',
            'depth': 'depth'
        }

        # Check if using single JSONA file
        use_single_jsona = self.config.get('processing', {}).get('single_jsona', False)

        # Collect all entries if using single file
        all_entries = [] if use_single_jsona else None

        # Generate entries for all variants (auto-accepted and manually accepted)
        for control_type, control_result in results.items():
            if not control_result or not control_result.get('variants'):
                continue

            folder_name = folder_map.get(control_type, control_type)

            # Find the best variant (highest score)
            variants = control_result.get('variants', [])
            if not variants:
                continue

            best_variant = max(variants, key=lambda v: v.get('score', 0))
            control_image_path = best_variant.get('path', '')

            if not control_image_path:
                continue

            # Create metadata entry using stable absolute paths.
            entry = {
                "hint_image_path": os.path.abspath(image_path).replace('\\', '/'),
                "hint_prompt": hint_prompt,
                "control_hints_path": os.path.abspath(control_image_path).replace('\\', '/'),
                "task_id": folder_name
            }

            # Append to appropriate file
            if use_single_jsona:
                all_entries.append(entry)
            else:
                # Append to separate file for each control type
                self._append_to_json_file(output_dir, folder_name, [entry])

        # Write all entries to single file if enabled
        if use_single_jsona and all_entries:
            self._append_to_json_file(output_dir, 'metadata', all_entries)

    def _append_to_json_file(self, output_dir: str, folder_name: str, entries: list):
        """Append entries to JSONA using the shared backup manager."""
        json_file = os.path.join(output_dir, f"{folder_name}.jsona")
        added_entries = self.jsona_backup_manager.append_entries(json_file, entries)
        if added_entries == 0:
            print(f"[DEBUG] No new JSONA entries appended to {json_file}")

    def _generate_canny(self, img: Image.Image, low: int, high: int) -> Image.Image:
        if self.canny_detector is not None:
            return self.canny_detector(img, low_threshold=low, high_threshold=high)

        gray = np.array(img.convert("L"))
        edges = cv2.Canny(gray, low, high)
        return Image.fromarray(edges)

    def _process_canny_with_retry(self, img: Image.Image, output_dir: str, basename: str) -> Dict:
        """Generate Canny variants with smart retry strategy"""
        retry_count = 0
        best_score = 0
        variants = []

        while retry_count <= self.max_retries:
            # Generate 4 variants
            current_variants = []
            for preset in self.canny_presets:
                # Apply random offset on retry
                if retry_count > 0 and self.retry_enabled:
                    low = preset['low'] + np.random.randint(-self.random_offset, self.random_offset)
                    high = preset['high'] + np.random.randint(-self.random_offset, self.random_offset)
                    low = max(10, min(low, 200))
                    high = max(low + 50, min(high, 255))
                else:
                    low = preset['low']
                    high = preset['high']

                # Generate Canny
                canny_img = self._generate_canny(img, low=low, high=high)

                # Save variant
                if retry_count > 0:
                    filename = f"{basename}_{preset['name']}_retry{retry_count}.png"
                else:
                    filename = f"{basename}_{preset['name']}.png"

                variant_path = os.path.join(output_dir, 'canny', filename)
                os.makedirs(os.path.dirname(variant_path), exist_ok=True)
                canny_img.save(variant_path)

                # Score variant
                score = self._score_canny(variant_path)
                current_variants.append({
                    'path': variant_path,
                    'score': score,
                    'preset': preset['name'],
                    'thresholds': {'low': low, 'high': high}
                })

                best_score = max(best_score, score)

            variants.extend(current_variants)

            # Check if retry needed
            if not self.retry_enabled or best_score >= self.retry_threshold or retry_count >= self.max_retries:
                break

            retry_count += 1

        # Determine quality flag
        if best_score >= 80:
            quality_flag = 'auto_accept'
        elif best_score <= 40:
            quality_flag = 'auto_reject'
        else:
            quality_flag = 'user_review'

        return {
            'variants': variants,
            'best_score': best_score,
            'retry_count': retry_count,
            'quality_flag': quality_flag
        }

    def _process_openpose(self, img: Image.Image, output_dir: str, basename: str) -> Dict:
        """Generate OpenPose variants (4 presets)"""
        variants = []
        best_score = 0

        # Check if using ViTPoseProcessor (which has unified interface)
        is_vitpose = hasattr(self.pose_manager, 'is_vitpose') and self.pose_manager.is_vitpose()

        for preset in self.openpose_presets:
            filename = f"{basename}_{preset['name']}.png"
            pose_path = os.path.join(output_dir, 'pose', filename)
            os.makedirs(os.path.dirname(pose_path), exist_ok=True)

            # Generate pose with preset parameters
            if is_vitpose:
                # ViTPoseProcessor has unified interface
                pose_img = self.pose_manager(img)
            else:
                # controlnet_aux processors don't support include_hands/include_face params
                pose_img = self.dwpose_detector(img)
            pose_img.save(pose_path)

            # Score pose
            is_valid, warning, visibility_ratio = self._score_openpose(pose_path)

            print(f"[DEBUG] Pose scoring: is_valid={is_valid}, warning={warning}, vis_ratio={visibility_ratio:.4f}")

            # Calculate score (0-100)
            if is_valid:
                # Adjust thresholds for ViTPose (finer lines)
                is_vitpose = hasattr(self, 'pose_manager') and hasattr(self.pose_manager, 'is_vitpose') and self.pose_manager.is_vitpose()
                if is_vitpose:
                    # ViTPose optimal range: 0.01-0.35
                    min_optimal = 0.01
                    max_optimal = 0.35
                else:
                    # DWpose optimal range: 0.05-0.35
                    min_optimal = 0.05
                    max_optimal = 0.35

                # Map visibility ratio to score
                if visibility_ratio < min_optimal:
                    score = 0
                elif visibility_ratio > max_optimal:
                    # Over-detected, penalize
                    score = max(0, 100 - (visibility_ratio - max_optimal) * 200)
                else:
                    # Optimal range: map to 60-100
                    normalized = (visibility_ratio - min_optimal) / (max_optimal - min_optimal)
                    score = 60 + normalized * 40

                # Penalize if warning
                if warning:
                    score = max(0, score - 20)
            else:
                score = 0

            print(f"[DEBUG] Calculated score: {score}")

            # Check if using ViTPose
            is_vitpose = hasattr(self, 'pose_manager') and hasattr(self.pose_manager, 'is_vitpose') and self.pose_manager.is_vitpose()

            variants.append({
                'path': pose_path,
                'score': score,
                'preset': preset['name'],
                'warning': warning,
                'visibility_ratio': visibility_ratio,
                'is_vitpose': is_vitpose
            })

            best_score = max(best_score, score)

        # Determine quality flag based on best score
        if best_score >= 80:
            quality_flag = 'auto_accept'
        elif best_score <= 40:
            quality_flag = 'auto_reject'
        else:
            quality_flag = 'user_review'

        return {
            'variants': variants,
            'best_score': best_score,
            'quality_flag': quality_flag
        }

    def _process_depth(self, img: Image.Image, output_dir: str, basename: str) -> Dict:
        """Generate Depth map variants (4 presets with different post-processing)"""
        # Store original size
        original_size = img.size  # (width, height)

        # Generate base depth map using Depth Anything V2
        import torch
        import cv2

        # Convert PIL to numpy
        raw_img = np.array(img)

        # Depth Anything V2 expects RGB
        if raw_img.shape[2] == 4:  # RGBA
            raw_img = raw_img[:, :, :3]

        try:
            with torch.no_grad():
                depth = self.depth_anything_model.infer_image(raw_img)
        except Exception as e:
            print(f"[ERROR] Depth inference failed: {e}")
            # Return empty result on error
            return {
                'variants': [],
                'best_score': 0,
                'quality_flag': 'auto_reject'
            }

        # Normalize depth to 0-255
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)

        # Convert to PIL Image (grayscale)
        base_depth_img = Image.fromarray(depth_normalized)

        # Depth Anything V2 maintains original resolution
        print(f"[INFO] Depth Anything V2 output size: {base_depth_img.size}, original: {original_size}")

        base_depth_array = np.array(base_depth_img).astype(np.float32)

        variants = []
        best_score = 0

        for preset in self.depth_presets:
            filename = f"{basename}_{preset['name']}.png"
            depth_path = os.path.join(output_dir, 'depth', filename)
            os.makedirs(os.path.dirname(depth_path), exist_ok=True)

            # Apply gamma and contrast adjustment
            gamma = preset['gamma']
            contrast = preset['contrast']

            # Normalize to 0-1
            normalized = base_depth_array / 255.0

            # Apply gamma correction
            adjusted = np.power(normalized, gamma)

            # Apply contrast
            adjusted = (adjusted - 0.5) * contrast + 0.5
            adjusted = np.clip(adjusted, 0, 1)

            # Convert back to 0-255
            adjusted_img = (adjusted * 255).astype(np.uint8)
            depth_img = Image.fromarray(adjusted_img)
            depth_img.save(depth_path)

            # Score depth
            is_valid, warning, metrics = self._score_depth(depth_path)

            # Calculate score (0-100) based on metrics
            if is_valid:
                std = metrics.get('std', 0)
                dynamic_range = metrics.get('dynamic_range', 0)
                grad_mean = metrics.get('grad_mean', 0)

                # Adjusted thresholds for Depth Anything V2 (higher quality output)
                # Standard deviation: optimal > 70, minimum 40
                std_score = min(1.0, max(0.0, (std - 40.0) / 40.0))

                # Dynamic range: optimal > 200, minimum 120
                dr_score = min(1.0, max(0.0, (dynamic_range - 120.0) / 100.0))

                # Gradient score (optimal 2-6 for high-res depth maps)
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
                score = combined * 100

                # Penalize if warning
                if warning:
                    score = max(0, score - 15)
            else:
                score = 0

            variants.append({
                'path': depth_path,
                'score': score,
                'preset': preset['name'],
                'warning': warning,
                'metrics': metrics
            })

            best_score = max(best_score, score)

        # Determine quality flag based on best score
        if best_score >= 80:
            quality_flag = 'auto_accept'
        elif best_score <= 40:
            quality_flag = 'auto_reject'
        else:
            quality_flag = 'user_review'

        return {
            'variants': variants,
            'best_score': best_score,
            'quality_flag': quality_flag
        }

    def _score_canny(self, canny_path: str) -> float:
        """
        Score Canny edge map quality
        Based on canny_rating.py logic
        """
        img = cv2.imread(canny_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0

        # Calculate white pixel ratio
        total_pixels = img.shape[0] * img.shape[1]
        white_pixels = np.sum(img > 200)
        white_ratio = white_pixels / total_pixels

        # White ratio score (target: 15% ± 5%)
        white_score = 50 * max(0, 1 - abs(white_ratio - 0.15) / 0.20)

        # Connected components analysis
        _, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        num_components = len(stats) - 1  # Exclude background

        if num_components == 0:
            return white_score

        # Filter small noise (area < 10 pixels)
        areas = stats[1:, cv2.CC_STAT_AREA]
        small_noise = np.sum(areas < 10)
        valid_components = areas[areas >= 10]

        # Clean score (penalize small noise)
        clean_score = 30 * (1 / (1 + small_noise / 200))

        # Thickness score (average area of valid components)
        if len(valid_components) > 0:
            avg_area = np.mean(valid_components)
            thick_score = 20 * min(avg_area / 80, 1.0)
        else:
            thick_score = 0

        total_score = white_score + clean_score + thick_score
        return round(total_score, 2)

    def _get_active_profile(self) -> dict:
        scoring = self.config.get('scoring', {})
        active = scoring.get('active_profile', 'general')
        profiles = scoring.get('profiles', {})
        profile = profiles.get(active, {}) if isinstance(profiles, dict) else {}
        return profile if isinstance(profile, dict) else {}

    def _score_openpose(self, pose_path: str) -> Tuple[bool, Optional[str], float]:
        """
        Score OpenPose quality
        Based on rating.py audit_openpose_tensor logic

        Returns:
            (is_valid, warning_message)
        """
        # Load pose image and extract keypoints
        # Note: DWpose outputs RGB image, need to parse keypoints from metadata or re-detect
        # For now, we'll use a simplified check based on image content

        img = cv2.imread(pose_path)
        if img is None:
            return False, "Failed to load pose image"

        # Convert to grayscale and check if pose is visible
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use higher threshold to filter out noise and near-black pixels
        non_black_pixels = np.sum(gray > 30)
        total_pixels = gray.shape[0] * gray.shape[1]
        visibility_ratio = non_black_pixels / total_pixels

        print(f"[DEBUG] Pose image stats: mean={gray.mean():.2f}, max={gray.max()}, non_black_pixels={non_black_pixels}, total={total_pixels}")

        profile = self._get_active_profile()
        pose_cfg = profile.get('openpose', {}) if isinstance(profile, dict) else {}

        # Lower thresholds for ViTPose (finer lines)
        is_vitpose = hasattr(self, 'pose_manager') and hasattr(self.pose_manager, 'is_vitpose') and self.pose_manager.is_vitpose()
        if is_vitpose:
            min_vis = float(pose_cfg.get('min_visibility_ratio', 0.01))  # Lower for ViTPose
            max_vis = float(pose_cfg.get('max_visibility_ratio', 0.35))
        else:
            min_vis = float(pose_cfg.get('min_visibility_ratio', 0.05))
            max_vis = float(pose_cfg.get('max_visibility_ratio', 0.35))

        print(f"[DEBUG] Pose visibility_ratio: {visibility_ratio:.4f}, min: {min_vis}, max: {max_vis}")

        if visibility_ratio < min_vis:
            return False, "No pose detected", float(visibility_ratio)

        if visibility_ratio > max_vis:
            # Likely multi-person / over-detected; keep as warning but still usable.
            return True, "Pose too dense (possible multi-person)", float(visibility_ratio)

        # If more than 5% visible, assume pose is valid
        # (Full keypoint validation would require accessing DWpose output data)
        return True, None, float(visibility_ratio)

    def _score_depth(self, depth_path: str) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """
        Score Depth map quality
        Based on rating.py audit_depth_tensor logic

        Returns:
            (is_valid, warning_message)
        """
        depth_tensor = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_tensor is None:
            return False, "Failed to load depth map"

        depth_float = depth_tensor.astype(np.float32)

        profile = self._get_active_profile()
        depth_cfg = profile.get('depth', {}) if isinstance(profile, dict) else {}
        min_std = float(depth_cfg.get('min_std', 15.0))
        min_dyn = float(depth_cfg.get('min_dynamic_range', 50.0))
        min_grad_mean = float(depth_cfg.get('min_grad_mean', 1.5))
        max_grad_mean = float(depth_cfg.get('max_grad_mean', 25.0))

        # Check standard deviation
        std_dev = np.std(depth_float)
        if std_dev < min_std:
            return False, f"Low depth variation (std: {std_dev:.1f})", {
                'std': float(std_dev)
            }

        # Check dynamic range
        p95 = np.percentile(depth_float, 95)
        p05 = np.percentile(depth_float, 5)
        dynamic_range = p95 - p05

        if dynamic_range < min_dyn:
            return False, f"Low dynamic range ({dynamic_range:.1f})", {
                'std': float(std_dev),
                'dynamic_range': float(dynamic_range)
            }

        # Gradient energy check (reject dead/noisy maps)
        gx = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        grad_mean = float(np.mean(grad))

        if grad_mean < min_grad_mean:
            return False, f"Depth too flat (grad_mean: {grad_mean:.2f})", {
                'std': float(std_dev),
                'dynamic_range': float(dynamic_range),
                'grad_mean': grad_mean
            }

        warning = None
        if grad_mean > max_grad_mean:
            warning = f"Depth noisy (grad_mean: {grad_mean:.2f})"

        return True, warning, {
            'std': float(std_dev),
            'dynamic_range': float(dynamic_range),
            'grad_mean': grad_mean
        }
