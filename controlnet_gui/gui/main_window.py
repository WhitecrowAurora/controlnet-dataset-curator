"""
Main window integrating all GUI components.
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QMenu, QAction, QStatusBar, QMessageBox, QFileDialog, QActionGroup,
    QDialog, QTextEdit, QPushButton, QScrollArea, QLabel,
    QInputDialog,
    QLineEdit, QPlainTextEdit, QAbstractSpinBox, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker, QProcess
from PyQt5.QtGui import QIcon
import json
import os
from typing import Optional
from queue import Queue, Empty, Full

from .settings_panel import SettingsPanel
from .image_list import ImageListWidget
from .preview_panel import PreviewPanel
from .jsona_manager_dialog import JsonaManagerDialog
from ..core.data_source import LocalDataSource
from ..core.controlnet_processor import ControlNetProcessor
from ..core.image_prefilter import ImagePreFilter
from ..core.progress_manager import ProgressManager
from ..core.jsona_backup import JsonaBackupManager
from ..core.score_converter import (
    canny_to_10_scale, openpose_to_10_scale, depth_to_10_scale
)
from ..language import get_lang_manager, tr


def build_progress_key(image_path: str, control_type: str, basename: str = '') -> str:
    """Create a stable per-review-item progress key."""
    if image_path:
        normalized_path = os.path.abspath(image_path).replace('\\', '/').lower()
    else:
        normalized_path = basename
    return f"{normalized_path}::{control_type}"


class InstallProgressDialog(QDialog):
    """Dialog showing real-time pip installation progress"""

    def __init__(self, parent, title: str, packages: list, reinstall: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(700, 500)

        self.packages = packages
        self.reinstall = reinstall
        self.process = None
        self.success = False

        # Layout
        layout = QVBoxLayout(self)

        # Console output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace;")
        layout.addWidget(self.console)

        # Close button (initially disabled)
        self.close_button = QPushButton(tr('please_wait'))
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.accept)
        layout.addWidget(self.close_button)

        # Start installation
        self._start_installation()

    def _start_installation(self):
        """Start pip installation process"""
        import sys

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._handle_stdout)
        self.process.readyReadStandardError.connect(self._handle_stderr)
        self.process.finished.connect(self._handle_finished)

        # Build command
        cmd = [sys.executable, '-m', 'pip', 'install', '-v']
        if self.reinstall:
            cmd.extend(['--force-reinstall', '--no-cache-dir'])
        cmd.extend(self.packages)

        self.console.append(f"$ {' '.join(cmd)}\n")

        # Start process
        self.process.start(sys.executable, cmd[1:])

    def _handle_stdout(self):
        """Handle stdout from pip"""
        data = self.process.readAllStandardOutput()
        text = bytes(data).decode('utf-8', errors='ignore')
        self.console.append(text)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _handle_stderr(self):
        """Handle stderr from pip"""
        data = self.process.readAllStandardError()
        text = bytes(data).decode('utf-8', errors='ignore')
        self.console.append(text)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _handle_finished(self, exit_code, exit_status):
        """Handle process completion"""
        self.success = (exit_code == 0)

        if self.success:
            self.console.append(f"\n{'='*60}")
            self.console.append(f"[成功] {tr('installation_complete')}")
            self.console.append(f"{'='*60}\n")
            self.close_button.setText(tr('close'))
        else:
            self.console.append(f"\n{'='*60}")
            self.console.append(f"[失败] {tr('installation_failed')}")
            self.console.append(f"{'='*60}\n")
            self.close_button.setText(tr('close'))

        self.close_button.setEnabled(True)


class InstallProgressDialogWithIndex(QDialog):
    """Dialog showing real-time pip installation progress with custom index URL"""

    def __init__(self, parent, title: str, packages: list, index_url: str, uninstall_first: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(700, 500)

        self.packages = packages
        self.index_url = index_url
        self.uninstall_first = uninstall_first
        self.process = None
        self.success = False
        self.current_step = 0
        self.total_steps = 2 if uninstall_first else 1

        # Layout
        layout = QVBoxLayout(self)

        # Console output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace;")
        layout.addWidget(self.console)

        # Close button (initially disabled)
        self.close_button = QPushButton(tr('please_wait'))
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.accept)
        layout.addWidget(self.close_button)

        # Start installation
        self._start_next_step()

    def _start_next_step(self):
        """Start next installation step"""
        import sys

        self.current_step += 1

        if self.current_step == 1 and self.uninstall_first:
            # Step 1: Uninstall existing packages
            self.console.append(f"{'='*60}")
            self.console.append(f"Step {self.current_step}/{self.total_steps}: Uninstalling existing packages...")
            self.console.append(f"{'='*60}\n")

            self.process = QProcess(self)
            self.process.readyReadStandardOutput.connect(self._handle_stdout)
            self.process.readyReadStandardError.connect(self._handle_stderr)
            self.process.finished.connect(self._handle_step_finished)

            cmd = [sys.executable, '-m', 'pip', 'uninstall', '-y'] + self.packages
            self.console.append(f"$ {' '.join(cmd)}\n")
            self.process.start(sys.executable, cmd[1:])

        elif (self.current_step == 2 and self.uninstall_first) or (self.current_step == 1 and not self.uninstall_first):
            # Step 2 (or 1): Install with custom index
            self.console.append(f"\n{'='*60}")
            self.console.append(f"Step {self.current_step}/{self.total_steps}: Installing packages from {self.index_url}...")
            self.console.append(f"{'='*60}\n")

            self.process = QProcess(self)
            self.process.readyReadStandardOutput.connect(self._handle_stdout)
            self.process.readyReadStandardError.connect(self._handle_stderr)
            self.process.finished.connect(self._handle_step_finished)

            cmd = [sys.executable, '-m', 'pip', 'install', '-v', '--timeout', '1000', '--index-url', self.index_url] + self.packages
            self.console.append(f"$ {' '.join(cmd)}\n")
            self.process.start(sys.executable, cmd[1:])

    def _handle_stdout(self):
        """Handle stdout from pip"""
        data = self.process.readAllStandardOutput()
        text = bytes(data).decode('utf-8', errors='ignore')
        self.console.append(text)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _handle_stderr(self):
        """Handle stderr from pip"""
        data = self.process.readAllStandardError()
        text = bytes(data).decode('utf-8', errors='ignore')
        self.console.append(text)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _handle_step_finished(self, exit_code, exit_status):
        """Handle step completion"""
        if exit_code != 0 and self.current_step == 1 and self.uninstall_first:
            # Uninstall failed, but continue anyway (packages might not exist)
            self.console.append(f"\n[警告] 卸载完成但有警告 (如果之前未安装则正常)\n")
            self._start_next_step()
        elif exit_code != 0:
            # Installation failed
            self.success = False
            self.console.append(f"\n{'='*60}")
            self.console.append(f"[失败] {tr('installation_failed')}")
            self.console.append(f"{'='*60}\n")
            self.close_button.setText(tr('close'))
            self.close_button.setEnabled(True)
        elif self.current_step < self.total_steps:
            # Continue to next step
            self._start_next_step()
        else:
            # All steps completed successfully
            self.success = True
            self.console.append(f"\n{'='*60}")
            self.console.append(f"[成功] {tr('installation_complete')}")
            self.console.append(f"{'='*60}\n")
            self.close_button.setText(tr('close'))
            self.close_button.setEnabled(True)


class ProcessingThread(QThread):
    """Background thread for continuous image processing."""

    image_ready = pyqtSignal()  # Signal that new image is available in queue
    processing_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)
    stats_updated = pyqtSignal(dict)
    progress_updated = pyqtSignal(str)  # Signal for progress text updates

    def __init__(self, data_source, processor, prefilter, progress_manager, settings):
        super().__init__()
        self.data_source = data_source
        self.processor = processor
        self.prefilter = prefilter
        self.progress_manager = progress_manager
        self.settings = settings
        self.running = True
        self.paused = False
        self.mutex = QMutex()

        # Review queue
        self.review_queue = Queue(maxsize=settings.get('processing', {}).get('preload_count', 20))

        self.stats = {
            'total': 0,
            'auto_accept': 0,
            'auto_reject': 0,
            'need_review': 0,
            'in_queue': 0
        }

    def run(self):
        """Continuously process images from data source."""
        try:
            output_config = self.settings.get('output', {})
            base_dir = output_config.get('base_dir', './output')

            # Create output directories
            os.makedirs(os.path.join(base_dir, 'accepted'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'rejected'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'reviewed'), exist_ok=True)

            # Use the data_source passed in constructor (respects custom directory setting)
            image_dir = self.data_source.image_dir
            tag_dir = self.data_source.tag_dir

            # Get list of images
            image_files = sorted([f for f in os.listdir(image_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])

            total_images = len(image_files)
            self.progress_updated.emit(f"找到 {total_images} 张图片")

            processed_count = 0
            for filename in image_files:
                # Check if stopped
                if not self.running:
                    break

                # Check if paused
                while self.paused and self.running:
                    self.msleep(100)

                if not self.running:
                    break

                basename = os.path.splitext(filename)[0]

                # Per-control review items are skipped later using a stable progress key.

                processed_count += 1
                self.progress_updated.emit(f"正在处理: {basename} ({processed_count}/{total_images})")

                image_path = os.path.join(image_dir, filename)

                # Apply pre-filter (blur detection)
                from PIL import Image
                with Image.open(image_path) as _im:
                    img = _im.convert("RGB")
                prefilter_result = self.prefilter.evaluate(img)

                # Process image with ControlNet
                result = self.processor.process_image(image_path, base_dir, basename)

                # Add prefilter info
                result['prefilter'] = prefilter_result
                result['basename'] = basename
                result['image_path'] = image_path

                # Load tags if available
                tag_file = os.path.join(tag_dir, f"{basename}.txt")
                if os.path.exists(tag_file):
                    with open(tag_file, 'r', encoding='utf-8') as f:
                        result['tags'] = f.read().strip()

                # Append custom tags if enabled
                custom_tags_config = self.settings.get('custom_tags', {})
                if custom_tags_config.get('enabled', False):
                    custom_tags = custom_tags_config.get('tags', '').strip()
                    if custom_tags:
                        existing_tags = result.get('tags', '')
                        result['tags'] = f"{existing_tags}, {custom_tags}" if existing_tags else custom_tags

                # Generate multiple rows - one for each enabled control type
                # Extract control results from the result dict
                canny_result = result.get('canny', {})
                openpose_result = result.get('openpose', {})
                depth_result = result.get('depth', {})

                control_types_to_display = []
                if canny_result and canny_result.get('variants'):
                    control_types_to_display.append(('canny', canny_result))
                if openpose_result and openpose_result.get('variants'):
                    control_types_to_display.append(('openpose', openpose_result))
                if depth_result and depth_result.get('variants'):
                    control_types_to_display.append(('depth', depth_result))

                if not control_types_to_display:
                    # No valid control type, skip this image
                    continue

                # Create one row for each control type
                for control_type, control_result in control_types_to_display:
                    best_score = control_result.get('best_score', 0)
                    progress_key = build_progress_key(image_path, control_type, basename)

                    if self.progress_manager and self.progress_manager.is_processed(progress_key):
                        continue

                    # Thresholds from profile (per control type)
                    scoring = self.settings.get('scoring', {})
                    active_profile = scoring.get('active_profile', 'general')
                    profiles = scoring.get('profiles', {})
                    profile = profiles.get(active_profile, {}) if isinstance(profiles, dict) else {}

                    print(f"[DEBUG] scoring config: active_profile={active_profile}, profiles keys={list(profiles.keys()) if profiles else 'None'}")
                    print(f"[DEBUG] profile content: {profile}")

                    # Get thresholds based on control type
                    if control_type == 'canny':
                        auto_accept_th = float(profile.get('canny_auto_accept', profile.get('auto_accept', 55)))
                        auto_reject_th = float(profile.get('canny_auto_reject', profile.get('auto_reject', 40)))
                    elif control_type == 'openpose':
                        auto_accept_th = float(profile.get('pose_auto_accept', profile.get('auto_accept', 55)))
                        auto_reject_th = float(profile.get('pose_auto_reject', profile.get('auto_reject', 40)))
                    elif control_type == 'depth':
                        auto_accept_th = float(profile.get('depth_auto_accept', profile.get('auto_accept', 55)))
                        auto_reject_th = float(profile.get('depth_auto_reject', profile.get('auto_reject', 40)))
                    else:
                        auto_accept_th = float(profile.get('auto_accept', 55))
                        auto_reject_th = float(profile.get('auto_reject', 40))

                    # Calculate 1-10 score for the active control type
                    if control_type == 'canny':
                        control_10_score = canny_to_10_scale(best_score) if best_score > 0 else 1.0
                    elif control_type == 'openpose':
                        # Get best variant
                        best_variant = max(control_result.get('variants', []),
                                         key=lambda v: v.get('score', 0), default={})
                        # Check if using ViTPose
                        is_vitpose = best_variant.get('is_vitpose', False)
                        control_10_score = openpose_to_10_scale(
                            best_variant.get('visibility_ratio', 0),
                            best_variant.get('score', 0) >= 60,  # is_valid threshold
                            best_variant.get('warning'),
                            is_vitpose=is_vitpose
                        )
                    elif control_type == 'depth':
                        # Get best variant
                        best_variant = max(control_result.get('variants', []),
                                         key=lambda v: v.get('score', 0), default={})
                        control_10_score = depth_to_10_scale(
                            best_variant.get('metrics', {}),
                            best_variant.get('score', 0) >= 60,  # is_valid threshold
                            best_variant.get('warning')
                        )

                    gui_data = {
                        'original_image': img,
                        'original_path': image_path,
                        'image_path': image_path,  # For deletion
                        'basename': basename,
                        'tags': result.get('tags', ''),
                        'prefilter': prefilter_result,
                        'control_type': control_type,  # 'canny', 'openpose', or 'depth'
                        'progress_key': progress_key,
                        'variants': [],  # Control variants
                        'best_score': best_score,  # Raw score (0-100)
                        'control_10_score': control_10_score,  # 1-10 score
                        'profile': active_profile
                        }

                    # Convert variants to GUI format
                    from PIL import Image as PILImage
                    for variant in control_result.get('variants', []):
                        # Load and copy image to ensure it persists after file close
                        variant_img = PILImage.open(variant['path']).convert("RGB").copy()

                        # Calculate 1-10 score based on control type
                        if control_type == 'canny':
                            variant_10_score = canny_to_10_scale(variant['score'])
                            preset_info = variant.get('thresholds', {})
                            preset_name = variant.get('preset', 'unknown')
                        elif control_type == 'openpose':
                            is_vitpose = variant.get('is_vitpose', False)
                            variant_10_score = openpose_to_10_scale(
                                variant.get('visibility_ratio', 0),
                                variant.get('score', 0) >= 60,
                                variant.get('warning'),
                                is_vitpose=is_vitpose
                            )
                            preset_info = {'warning': variant.get('warning')}
                            preset_name = variant.get('preset', 'unknown')
                        elif control_type == 'depth':
                            variant_10_score = depth_to_10_scale(
                                variant.get('metrics', {}),
                                variant.get('score', 0) >= 60,
                                variant.get('warning')
                            )
                            preset_info = variant.get('metrics', {})
                            preset_name = variant.get('preset', 'unknown')

                        gui_data['variants'].append({
                            'image': variant_img,
                            'preset': preset_info,
                            'preset_name': preset_name,
                            'score': variant['score'],  # Raw score (0-100)
                            'score_10': variant_10_score  # 1-10 score
                        })

                    # Mark best variant for UI highlight
                    if gui_data['variants']:
                        # Check if all variants have the same score
                        scores = [v.get('score', 0) for v in gui_data['variants']]
                        all_same_score = len(set(scores)) == 1 and len(scores) > 1

                        if all_same_score:
                            # All variants have same score - auto-select first one
                            best_index = 0
                            gui_data['auto_selected'] = True
                        else:
                            # Find best variant by score
                            best_index = max(range(len(gui_data['variants'])),
                                             key=lambda i: gui_data['variants'][i].get('score', 0))
                            gui_data['auto_selected'] = False

                        for i, v in enumerate(gui_data['variants']):
                            v['is_best'] = (i == best_index)

                    # Update statistics
                    with QMutexLocker(self.mutex):
                        self.stats['total'] += 1

                        # Check if should auto-accept due to same scores
                        # Only auto-accept if scores are same AND > 0 (not all failed)
                        auto_accept_same_score = gui_data.get('auto_selected', False) and best_score > 0

                        print(f"[DEBUG] Checking auto-accept: control_type={control_type}, best_score={best_score}, auto_accept_th={auto_accept_th}, auto_reject_th={auto_reject_th}")

                        if best_score >= auto_accept_th or auto_accept_same_score:
                            # Auto accept (either high score or same scores)
                            print(f"[DEBUG] AUTO ACCEPT: {basename} - {control_type}")
                            self.stats['auto_accept'] += 1
                            gui_data['auto_action'] = 'accept'  # Mark for auto-accept
                            self._save_accepted(gui_data, base_dir, extra=result)
                            self.progress_manager.mark_processed(progress_key)
                            self.progress_manager.mark_accepted(progress_key, auto=True)

                        elif best_score <= auto_reject_th:
                            # Auto reject
                            print(f"[DEBUG] AUTO REJECT: {basename} - {control_type}")
                            self.stats['auto_reject'] += 1
                            gui_data['auto_action'] = 'reject'  # Mark for auto-reject
                            self._handle_rejected(gui_data, output_config)
                            self.progress_manager.mark_processed(progress_key)
                            self.progress_manager.mark_rejected(progress_key, auto=True)

                        else:
                            # Need review - don't mark as processed yet
                            print(f"[DEBUG] NEED REVIEW: {basename} - {control_type}")
                            self.stats['need_review'] += 1
                            gui_data['auto_action'] = None

                    # Send statistics update
                    self.stats_updated.emit(self.stats.copy())

                    # Check if auto-pass should skip review
                    auto_pass_no_review = self.settings.get('processing', {}).get('auto_pass_no_review', True)
                    enqueued = False

                    # Decide whether to add to review queue
                    if auto_reject_th < best_score < auto_accept_th:
                        # Needs review - retry with short timeouts so stop/exit can interrupt cleanly.
                        while self.running:
                            try:
                                self.review_queue.put(gui_data, timeout=0.1)
                                enqueued = True
                                break
                            except Full:
                                continue
                        if not enqueued and not self.running:
                            break
                    elif not auto_pass_no_review:
                        # Setting disabled - show all auto items for review
                        try:
                            self.review_queue.put(gui_data, timeout=0.1)
                            enqueued = True
                        except Full:
                            pass
                    # else: auto_pass_no_review enabled - skip adding auto items to queue displaying this auto item

                    with QMutexLocker(self.mutex):
                        self.stats['in_queue'] = self.review_queue.qsize()
                    self.stats_updated.emit(self.stats.copy())
                    if enqueued:
                        self.image_ready.emit()

            # Generate report
            report_config = self.settings.get('report', {})
            if report_config.get('enabled', True):
                report_file = os.path.join(base_dir, 'report.txt')
                self.progress_manager.generate_report(report_file)

            self.processing_complete.emit()

        except Exception as e:
            import traceback
            self.error_occurred.emit(f"{str(e)}\n{traceback.format_exc()}")

    def get_next_image(self) -> Optional[dict]:
        """Get next image from review queue (non-blocking)"""
        try:
            result = self.review_queue.get_nowait()
            with QMutexLocker(self.mutex):
                self.stats['in_queue'] = self.review_queue.qsize()
            self.stats_updated.emit(self.stats.copy())
            return result
        except Empty:
            return None

    def _save_accepted(self, result: dict, base_dir: str, extra: Optional[dict] = None):
        """Save auto-accepted image (original + tags + selected control maps)"""
        try:
            basename = result.get('basename', 'unknown')
            variants = result.get('variants', [])
            control_type = result.get('control_type', 'canny')  # Get current control type

            print(f"[DEBUG] _save_accepted called: basename={basename}, control_type={control_type}, variants={len(variants)}")

            if not variants:
                print(f"[DEBUG] No variants, skipping save")
                return

            # Find best variant
            best_variant = max(variants, key=lambda v: v['score'])
            preset_name = best_variant['preset_name']

            out_dir = os.path.join(base_dir, 'accepted')
            os.makedirs(out_dir, exist_ok=True)

            # Save original image (only once per basename)
            orig = result.get('original_image')
            orig_path = os.path.join(out_dir, f"{basename}.png")
            if orig is not None and not os.path.exists(orig_path):
                orig.save(orig_path)

            # Save tags (only once per basename)
            tags = result.get('tags', '')
            tag_path = os.path.join(out_dir, f"{basename}.txt")
            if tags and not os.path.exists(tag_path):
                with open(tag_path, 'w', encoding='utf-8') as f:
                    f.write(tags)

            # Save the control image based on control_type
            control_img = best_variant['image']
            print(f"[DEBUG] Saving control image for type: {control_type}")
            if control_type == 'canny':
                save_path = os.path.join(out_dir, f"{basename}_canny.png")
                control_img.save(save_path)
                print(f"[DEBUG] Saved canny to: {save_path}")
            elif control_type == 'openpose':
                save_path = os.path.join(out_dir, f"{basename}_openpose.png")
                control_img.save(save_path)
                print(f"[DEBUG] Saved openpose to: {save_path}")
            elif control_type == 'depth':
                save_path = os.path.join(out_dir, f"{basename}_depth.png")
                control_img.save(save_path)
                print(f"[DEBUG] Saved depth to: {save_path}")

            # Save metadata (append control type info)
            meta_path = os.path.join(out_dir, f"{basename}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            else:
                meta = {
                    'profile': result.get('profile', ''),
                    'basename': basename
                }

            # Add control-specific metadata
            if control_type == 'canny':
                meta['canny_preset'] = preset_name
                meta['canny_thresholds'] = best_variant.get('preset', {})
                meta['canny_score'] = result.get('best_score', 0)
            elif control_type == 'openpose':
                meta['openpose_preset'] = preset_name
                meta['openpose_score'] = result.get('best_score', 0)
            elif control_type == 'depth':
                meta['depth_preset'] = preset_name
                meta['depth_score'] = result.get('best_score', 0)

            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # Write to JSONA files for training
            self._write_jsona_metadata(result, base_dir, extra)

        except Exception as e:
            print(f"Failed to save auto-accepted image: {e}")

    def _write_jsona_metadata(self, result: dict, base_dir: str, extra: Optional[dict] = None):
        """Write JSONA metadata for accepted image (both auto and manual)"""
        try:
            print(f"[DEBUG] _write_jsona_metadata called for {result.get('basename', 'unknown')}")
            basename = result.get('basename', 'unknown')
            image_path = result.get('image_path', '')
            control_type = result.get('control_type', 'canny')

            if not image_path:
                print(f"[DEBUG] No image_path found, skipping jsona write")
                return

            print(f"[DEBUG] image_path: {image_path}")
            print(f"[DEBUG] base_dir: {base_dir}")
            print(f"[DEBUG] control_type: {control_type}")

            # Get tags
            image_dir = os.path.dirname(image_path)
            extract_dir = os.path.dirname(image_dir)
            tag_file = os.path.join(extract_dir, 'tags', f"{basename}.txt")

            hint_prompt = ""
            if os.path.exists(tag_file):
                with open(tag_file, 'r', encoding='utf-8') as f:
                    hint_prompt = f.read().strip()

            use_single_jsona = self.settings.get('processing', {}).get('single_jsona', False)
            control_map = {
                'canny': ('canny', '_canny.png'),
                'openpose': ('pose', '_openpose.png'),
                'depth': ('depth', '_depth.png')
            }

            if control_type not in control_map:
                print(f"[DEBUG] Unknown control_type: {control_type}")
                return

            folder_name, file_suffix = control_map[control_type]
            accepted_image_path = os.path.join(base_dir, 'accepted', f"{basename}.png")
            source_image_path = accepted_image_path if os.path.exists(accepted_image_path) else image_path
            control_path = os.path.join(base_dir, 'accepted', f"{basename}{file_suffix}")

            print(f"[DEBUG] Control path: {control_path}")

            if os.path.exists(control_path):
                entry = self._create_jsona_entry(source_image_path, hint_prompt, control_path, folder_name)
                print(f"[DEBUG] Created {control_type} entry: {entry}")

                if use_single_jsona:
                    self.processor._append_to_json_file(base_dir, 'metadata', [entry])
                else:
                    self.processor._append_to_json_file(base_dir, folder_name, [entry])
                print(f"[DEBUG] Successfully wrote to {folder_name}.jsona")

                if hasattr(self, 'settings_panel'):
                    self.settings_panel._update_jsona_statistics()
            else:
                print(f"[DEBUG] Control path doesn't exist: {control_path}")

        except Exception as e:
            print(f"[ERROR] Failed to write JSONA metadata: {e}")
            import traceback
            traceback.print_exc()

    def _create_jsona_entry(self, image_path: str, hint_prompt: str, control_path: str, task_id: str) -> dict:
        """Create a JSONA entry with stable absolute paths."""
        hint_image_path = os.path.abspath(image_path).replace('\\', '/')
        control_hints_path = os.path.abspath(control_path).replace('\\', '/')

        return {
            "hint_image_path": hint_image_path,
            "hint_prompt": hint_prompt,
            "control_hints_path": control_hints_path,
            "task_id": task_id
        }

    def _handle_rejected(self, result: dict, output_config: dict):
        """Handle auto-rejected image without removing the original source file."""
        import shutil
        from datetime import datetime

        basename = result.get('basename', 'unknown')
        control_type = result.get('control_type', 'unknown')
        base_dir = output_config.get('base_dir', './output')
        delete_dir = os.path.join(base_dir, '.delete')
        os.makedirs(delete_dir, exist_ok=True)

        try:
            image_path = result.get('image_path')
            if image_path and os.path.exists(image_path):
                # Keep the source image in place so other control types can still finish review.
                dest_image = os.path.join(delete_dir, os.path.basename(image_path))
                if not os.path.exists(dest_image):
                    shutil.copy2(image_path, dest_image)

                # Copy tag file if it exists.
                tag_path = image_path.replace('/images/', '/tags/').replace('\\images\\', '\\tags\\')
                tag_path = os.path.splitext(tag_path)[0] + '.txt'
                if os.path.exists(tag_path):
                    dest_tag = os.path.join(delete_dir, os.path.basename(tag_path))
                    if not os.path.exists(dest_tag):
                        shutil.copy2(tag_path, dest_tag)

            metadata = {
                'basename': basename,
                'deleted_at': datetime.now().isoformat(),
                'reason': f"Low quality score: {result.get('best_score', 0)}",
                'control_type': control_type,
                'original_path': image_path
            }

            prefilter = result.get('prefilter', {})
            if prefilter.get('quality_warning', False):
                metadata['prefilter_warning'] = prefilter.get('sharpness_level', 'unknown')

            metadata_file = os.path.join(delete_dir, f"{basename}_{control_type}_meta.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Failed to record rejected image metadata: {e}")

    def pause(self):
        """Pause processing"""
        self.paused = True

    def resume(self):
        """Resume processing"""
        self.paused = False

    def stop(self):
        """Stop processing"""
        self.running = False
        self.paused = False


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, config_path: str = "config.json"):
        super().__init__()
        self.config_path = config_path
        self.config = self._load_config()
        self.processing_thread: Optional[ProcessingThread] = None
        self.data_source = None
        self.processor = None
        self.prefilter = None
        self.progress_manager = None
        self.delete_behavior = '7days'  # Default: keep for 7 days
        self.current_stats = {
            'total': 0,
            'auto_accept': 0,
            'auto_reject': 0,
            'need_review': 0,
            'in_queue': 0
        }
        self.max_display_rows = 20  # Max rows displayed in GUI
        self._auto_expanding_window = False
        self.lang_manager = get_lang_manager()
        self.jsona_backup_interval = 200

        # Load language from config
        saved_lang = self.config.get('language', 'zh')  # Default to Chinese
        self.lang_manager.set_language(saved_lang)

        self._init_ui()
        self._connect_signals()

    def _load_config(self) -> dict:
        """Load configuration from file."""
        abs_config_path = os.path.abspath(self.config_path)
        print(f"[DEBUG] Loading config from: {abs_config_path}")
        print(f"[DEBUG] Config file exists: {os.path.exists(self.config_path)}")
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"[DEBUG] Config loaded, scoring.active_profile = {config.get('scoring', {}).get('active_profile')}")
                return config
        print(f"[DEBUG] Config file not found, returning empty dict")
        return {}

    def _save_config(self):
        """Save configuration to file."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("ControlNet 控制图筛选工具")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(800, 500)  # Allow window to be resized smaller

        # Create menu bar
        self._create_menu_bar()

        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create splitter for resizable panels
        self.main_splitter = QSplitter(Qt.Horizontal)

        # Left panel: Settings (wrapped in scroll area)
        self.settings_panel = SettingsPanel(self.config)
        # Keep settings panel readable but do not let it squeeze review area.
        self.settings_panel.setMinimumWidth(340)
        self.settings_panel.setMaximumWidth(460)

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidget(self.settings_panel)
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.settings_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.settings_scroll.setMinimumWidth(350)
        self.settings_scroll.setMaximumWidth(480)

        # Set scroll area background to match the dark theme
        self.settings_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #1e1e1e;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.main_splitter.addWidget(self.settings_scroll)

        # Middle panel: Image list
        self.image_list = ImageListWidget()
        self.image_list.setMinimumWidth(760)
        self.main_splitter.addWidget(self.image_list)

        # Right panel: Preview
        self.preview_panel = PreviewPanel()
        self.preview_panel.setMinimumWidth(340)
        self.main_splitter.addWidget(self.preview_panel)

        # Set initial sizes (settings / review / preview).
        self.main_splitter.setSizes([380, 820, 340])
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        self.main_splitter.setCollapsible(2, False)

        main_layout.addWidget(self.main_splitter)
        self._ensure_review_area_width()

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add device info label (permanent widget on the right)
        self.device_label = QLabel("Device: Checking...")
        self.device_label.setStyleSheet("color: #888; padding: 0 10px;")
        self.status_bar.addPermanentWidget(self.device_label)

        # Check device in background
        self._check_device_info()

        self.status_bar.showMessage("Ready")

        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d0d0d;
            }
            QStatusBar {
                background-color: #1a1a1a;
                color: #ffffff;
                border-top: 1px solid #444;
            }
        """)

    def _check_device_info(self):
        """Check and display device information"""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.device_label.setText(f"运行在: {device_name} (CUDA)")
                self.device_label.setStyleSheet("color: #4CAF50; padding: 0 10px;")  # Green for GPU
                self.device_label.setToolTip("")
            else:
                self.device_label.setText("运行在: CPU (可通过 工具→重装Torch 尝试使用GPU)")
                self.device_label.setStyleSheet("color: #FFA726; padding: 0 10px;")  # Orange for CPU
                self.device_label.setToolTip("")
        except Exception as e:
            self.device_label.setText("Device: Unknown")
            self.device_label.setStyleSheet("color: #888; padding: 0 10px;")

    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        self.file_menu = menubar.addMenu(tr('file'))

        self.load_config_action = QAction(tr('load_config'), self)
        self.load_config_action.triggered.connect(self._load_config_dialog)
        self.file_menu.addAction(self.load_config_action)

        self.save_config_action = QAction(tr('save_config'), self)
        self.save_config_action.setShortcut("Ctrl+S")
        self.save_config_action.triggered.connect(self._save_config)
        self.file_menu.addAction(self.save_config_action)

        self.file_menu.addSeparator()

        self.exit_action = QAction(tr('exit'), self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)

        # Edit menu
        self.edit_menu = menubar.addMenu(tr('edit'))

        self.pause_action = QAction(tr('pause_processing'), self)
        self.pause_action.setEnabled(False)
        self.pause_action.triggered.connect(self._toggle_pause)
        self.edit_menu.addAction(self.pause_action)

        self.stop_action = QAction(tr('stop_processing'), self)
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self._stop_processing)
        self.edit_menu.addAction(self.stop_action)

        self.edit_menu.addSeparator()

        # Delete behavior submenu
        self.delete_menu = self.edit_menu.addMenu(tr('delete_behavior'))

        delete_group = QActionGroup(self)
        delete_group.setExclusive(True)

        self.delete_permanent_action = QAction(tr('delete_permanent'), self)
        self.delete_permanent_action.setCheckable(True)
        self.delete_permanent_action.triggered.connect(lambda: self._set_delete_behavior('permanent'))
        delete_group.addAction(self.delete_permanent_action)
        self.delete_menu.addAction(self.delete_permanent_action)

        self.delete_7days_action = QAction(tr('delete_7days'), self)
        self.delete_7days_action.setCheckable(True)
        self.delete_7days_action.setChecked(True)  # Default
        self.delete_7days_action.triggered.connect(lambda: self._set_delete_behavior('7days'))
        delete_group.addAction(self.delete_7days_action)
        self.delete_menu.addAction(self.delete_7days_action)

        self.delete_30days_action = QAction(tr('delete_30days'), self)
        self.delete_30days_action.setCheckable(True)
        self.delete_30days_action.triggered.connect(lambda: self._set_delete_behavior('30days'))
        delete_group.addAction(self.delete_30days_action)
        self.delete_menu.addAction(self.delete_30days_action)

        self.delete_never_action = QAction(tr('delete_never'), self)
        self.delete_never_action.setCheckable(True)
        self.delete_never_action.triggered.connect(lambda: self._set_delete_behavior('never'))
        delete_group.addAction(self.delete_never_action)
        self.delete_menu.addAction(self.delete_never_action)

        self.edit_menu.addSeparator()

        self.clear_action = QAction(tr('clear_all_images'), self)
        self.clear_action.triggered.connect(self._clear_images)
        self.edit_menu.addAction(self.clear_action)

        self.cleanup_delete_action = QAction(tr('cleanup_delete_folder'), self)
        self.cleanup_delete_action.triggered.connect(self._cleanup_delete_folder)
        self.edit_menu.addAction(self.cleanup_delete_action)

        # Tools menu
        self.tools_menu = menubar.addMenu(tr('tools'))

        self.install_torch_action = QAction(tr('install_torch'), self)
        self.install_torch_action.triggered.connect(self._install_torch)
        self.tools_menu.addAction(self.install_torch_action)

        self.reinstall_torch_action = QAction(tr('reinstall_torch'), self)
        self.reinstall_torch_action.triggered.connect(self._reinstall_torch)
        self.tools_menu.addAction(self.reinstall_torch_action)

        self.tools_menu.addSeparator()

        self.install_datasets_action = QAction(tr('install_datasets'), self)
        self.install_datasets_action.triggered.connect(self._install_datasets)
        self.tools_menu.addAction(self.install_datasets_action)

        self.fix_datasets_action = QAction(tr('fix_datasets'), self)
        self.fix_datasets_action.triggered.connect(self._fix_datasets)
        self.tools_menu.addAction(self.fix_datasets_action)

        self.tools_menu.addSeparator()

        self.install_depth_anything_action = QAction("安装 Depth Anything V2", self)
        self.install_depth_anything_action.triggered.connect(self._install_depth_anything)
        self.tools_menu.addAction(self.install_depth_anything_action)

        self.tools_menu.addSeparator()

        self.jsona_manager_action = QAction('JSONA 管理', self)
        self.jsona_manager_action.triggered.connect(self._open_jsona_manager)
        self.tools_menu.addAction(self.jsona_manager_action)

        self.jsona_import_action = QAction('导入 JSONA', self)
        self.jsona_import_action.triggered.connect(self._open_jsona_importer)
        self.tools_menu.addAction(self.jsona_import_action)

        self.jsona_checker_action = QAction('JSONA 核对检查器', self)
        self.jsona_checker_action.triggered.connect(self._open_jsona_checker)
        self.tools_menu.addAction(self.jsona_checker_action)

        self.tools_menu.addSeparator()

        self.install_all_deps_action = QAction(tr('install_all_dependencies'), self)
        self.install_all_deps_action.triggered.connect(self._install_all_dependencies)
        self.tools_menu.addAction(self.install_all_deps_action)

        self.tools_menu.addSeparator()

        self.check_deps_action = QAction(tr('check_dependencies'), self)
        self.check_deps_action.triggered.connect(self._check_dependencies)
        self.tools_menu.addAction(self.check_deps_action)

        # Language menu
        self.lang_menu = menubar.addMenu(tr('language'))

        lang_group = QActionGroup(self)
        lang_group.setExclusive(True)

        self.lang_en_action = QAction("English", self)
        self.lang_en_action.setCheckable(True)
        self.lang_en_action.setChecked(self.lang_manager.language == 'en')
        self.lang_en_action.triggered.connect(lambda: self._change_language('en'))
        lang_group.addAction(self.lang_en_action)
        self.lang_menu.addAction(self.lang_en_action)

        self.lang_zh_action = QAction("中文", self)
        self.lang_zh_action.setCheckable(True)
        self.lang_zh_action.setChecked(self.lang_manager.language == 'zh')
        self.lang_zh_action.triggered.connect(lambda: self._change_language('zh'))
        lang_group.addAction(self.lang_zh_action)
        self.lang_menu.addAction(self.lang_zh_action)

        # Help menu
        self.help_menu = menubar.addMenu(tr('help'))

        self.about_action = QAction(tr('about'), self)
        self.about_action.triggered.connect(self._show_about)
        self.help_menu.addAction(self.about_action)

    def _connect_signals(self):
        """Connect signals and slots."""
        # Settings panel signals
        self.settings_panel.settings_changed.connect(self._on_settings_changed)
        self.settings_panel.start_processing.connect(self._start_processing)
        self.settings_panel.extraction_finished.connect(self._on_extraction_finished)
        self.settings_panel.btn_pause.clicked.connect(self._toggle_pause)
        self.settings_panel.btn_stop.clicked.connect(self._stop_processing)

        # Image list signals
        self.image_list.variant_selected.connect(self._on_variant_selected)
        self.image_list.variant_confirmed.connect(self._on_variant_confirmed)
        self.image_list.row_discarded.connect(self._on_row_discarded)

    def _is_text_input_focused(self) -> bool:
        """Avoid hijacking keyboard when user is typing in form inputs."""
        widget = self.focusWidget()
        return isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox))

    def _show_active_row_preview(self):
        """Show preview for current active review row."""
        active_row = self.image_list.get_active_row()
        if not active_row:
            self.preview_panel.clear()
            return

        data = active_row.get_data()
        if not data:
            return

        selected_variant = active_row.get_selected_variant()
        self.preview_panel.show_preview(data, selected_variant if selected_variant >= 0 else -1)

    def _handle_review_shortcuts(self, event) -> bool:
        """Keyboard shortcuts for manual review list."""
        if self.image_list.get_row_count() == 0:
            return False

        if self._is_text_input_focused():
            return False

        focused = self.focusWidget()
        if focused and self.settings_panel.isAncestorOf(focused):
            return False

        modifiers = event.modifiers()
        if modifiers not in (Qt.NoModifier, Qt.KeypadModifier):
            return False

        key = event.key()

        if key == Qt.Key_Up:
            handled = self.image_list.activate_next_row(-1)
            if handled:
                self._show_active_row_preview()
            return handled

        if key == Qt.Key_Down:
            handled = self.image_list.activate_next_row(1)
            if handled:
                self._show_active_row_preview()
            return handled

        if key == Qt.Key_Left:
            return self.image_list.select_next_variant_on_active(-1)

        if key == Qt.Key_Right:
            return self.image_list.select_next_variant_on_active(1)

        if key in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4):
            return self.image_list.select_variant_on_active(key - Qt.Key_1)

        if key in (Qt.Key_Return, Qt.Key_Enter):
            if self.image_list.confirm_active_selection():
                return True
            self.status_bar.showMessage("请先按 1-4 选择变体，再按 Enter 确认")
            return True

        if key in (Qt.Key_D, Qt.Key_Delete, Qt.Key_Backspace):
            return self.image_list.discard_active_row()

        return False

    def keyPressEvent(self, event):
        """Global keyboard entry point for review shortcuts."""
        if self._handle_review_shortcuts(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def _ensure_review_area_width(self):
        """Ensure review area width by expanding window to the right when needed."""
        if not hasattr(self, 'main_splitter'):
            return

        if self._auto_expanding_window:
            return

        sizes = self.main_splitter.sizes()
        if len(sizes) != 3:
            return

        _, middle, _ = sizes
        min_middle = 980

        if middle >= min_middle:
            return

        need = min_middle - middle
        geom = self.geometry()
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return

        available = screen.availableGeometry()
        current_right = geom.x() + geom.width()
        max_right = available.x() + available.width()
        expandable = max_right - current_right
        if expandable <= 0:
            return

        expand_by = min(need, expandable)
        if expand_by <= 0:
            return

        self._auto_expanding_window = True
        try:
            self.setGeometry(geom.x(), geom.y(), geom.width() + expand_by, geom.height())
        finally:
            self._auto_expanding_window = False

    def resizeEvent(self, event):
        """Keep review area usable on window resize."""
        super().resizeEvent(event)
        self._ensure_review_area_width()

    def _on_settings_changed(self, settings: dict):
        """Handle settings changes."""
        self.config = settings
        self.status_bar.showMessage("Settings updated")

    def _on_extraction_finished(self, count: int):
        """Handle data extraction completion"""
        QMessageBox.information(
            self,
            tr('extraction_complete'),
            f"{tr('extracted_count')}: {count}"
        )
        self.status_bar.showMessage(f"Extracted {count} samples")

    def _start_processing(self):
        """Start image processing."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.status_bar.showMessage("Processing already in progress")
            return

        # Get current settings
        settings = self.settings_panel.get_settings()

        # Determine input directory based on user selection
        try:
            processing_config = settings.get('processing', {})
            use_custom_dir = processing_config.get('use_custom_dir', False)

            print(f"[DEBUG] use_custom_dir: {use_custom_dir}")
            print(f"[DEBUG] processing_config: {processing_config}")

            if use_custom_dir:
                # Use custom directory
                custom_input_dir = processing_config.get('custom_input_dir', './images')
                image_dir = custom_input_dir
                print(f"[DEBUG] Using custom directory: {image_dir}")
                # Try to find tags directory (same parent, or sibling 'tags' folder)
                parent_dir = os.path.dirname(custom_input_dir)
                tag_dir = os.path.join(parent_dir, 'tags')
                if not os.path.exists(tag_dir):
                    # If no tags folder, use same directory
                    tag_dir = custom_input_dir
            else:
                # Use data source directory
                data_config = settings.get('data_source', {})
                source_type = data_config.get('type', 'local_parquet')
                print(f"[DEBUG] Using data source directory, type: {source_type}")

                if source_type == 'local_parquet':
                    extract_dir = data_config.get('local_parquet', {}).get('extract_dir', './extracted')
                else:
                    extract_dir = data_config.get('streaming', {}).get('extract_dir', './extracted')

                image_dir = os.path.join(extract_dir, 'images')
                tag_dir = os.path.join(extract_dir, 'tags')

            # Check if image directory exists
            if not os.path.exists(image_dir):
                QMessageBox.warning(
                    self,
                    "No Data",
                    f"Image directory not found: {image_dir}\n\nPlease check your input directory settings."
                )
                return

            self.data_source = LocalDataSource(
                image_dir=image_dir,
                tag_dir=tag_dir
            )

            # Validate that at least one control type is selected
            processing_config = settings.get('processing', {})
            control_types = processing_config.get('control_types', {})
            if not any(control_types.values()):
                QMessageBox.warning(
                    self,
                    "No Control Type Selected",
                    "Please select at least one control type (Canny, OpenPose, or Depth) in Processing Settings."
                )
                return

            # Fix torch DLL loading before creating processor
            import sys
            torch_lib_path = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'torch', 'lib')
            if os.path.exists(torch_lib_path):
                # Add to PATH if not already there
                if torch_lib_path not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
                # Add DLL directory for Python 3.8+
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(torch_lib_path)
                    except Exception:
                        pass

            # Check if depth is enabled and depth-anything-v2 is installed
            processing_config = settings.get('processing', {})
            control_types = processing_config.get('control_types', {})
            if control_types.get('depth', False):
                try:
                    import depth_anything_v2
                except ImportError:
                    # Depth Anything V2 not installed, ask user to install
                    reply = QMessageBox.question(
                        self,
                        "需要安装 Depth Anything V2",
                        "您启用了 Depth 处理，但未安装 depth-anything-v2 库。\n\n"
                        "这是高质量深度图生成所必需的。\n\n"
                        "安装过程中程序会自动关闭，安装完成后请手动重启。\n\n"
                        "是否现在安装？",
                        QMessageBox.Yes | QMessageBox.No
                    )

                    if reply == QMessageBox.Yes:
                        # Use external installation to avoid file lock issues
                        self._install_package_external(['depth-anything-v2'], '安装 Depth Anything V2')
                        return
                    else:
                        # User declined, don't start processing
                        QMessageBox.warning(
                            self,
                            "无法继续",
                            "未安装 Depth Anything V2，无法进行 Depth 处理。\n\n"
                            "请禁用 Depth 或从 Tools 菜单安装后再试。"
                        )
                        return

            # Create processor
            self.processor = ControlNetProcessor(settings)

            output_config = settings.get('output', {})
            base_dir = output_config.get('base_dir', './output')
            self.processor.jsona_backup_manager.prepare_output_backups(base_dir)

            # Create prefilter
            prefilter_config = settings.get('prefilter', {})
            self.prefilter = ImagePreFilter(prefilter_config)

            # Create progress manager
            progress_config = settings.get('progress', {})
            progress_file = progress_config.get('progress_file', '.progress.json')
            self.progress_manager = ProgressManager(progress_file)

            # Clear existing images
            self.image_list.clear()

            # Start processing thread
            self.processing_thread = ProcessingThread(
                self.data_source,
                self.processor,
                self.prefilter,
                self.progress_manager,
                settings
            )
            self.processing_thread.image_ready.connect(self._on_image_ready)
            self.processing_thread.processing_complete.connect(self._on_processing_complete)
            self.processing_thread.error_occurred.connect(self._on_processing_error)
            self.processing_thread.stats_updated.connect(self._on_stats_updated)
            self.processing_thread.progress_updated.connect(self._on_progress_updated)
            self.processing_thread.start()

            # Enable pause and stop buttons
            self.pause_action.setEnabled(True)
            self.pause_action.setText(tr('pause_processing'))
            self.stop_action.setEnabled(True)
            self.settings_panel.btn_pause.setEnabled(True)
            self.settings_panel.btn_stop.setEnabled(True)

            self.status_bar.showMessage(tr('processing_started'))

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Failed to start processing:\n{str(e)}\n\n{traceback.format_exc()}")

    def _on_image_ready(self):
        """Handle new image ready from processing thread."""
        # Pull from queue instead of using emitted data
        self._refill_display()

    def _on_stats_updated(self, stats: dict):
        """Update status bar with statistics."""
        self.current_stats = stats
        # Don't update status bar here, let progress_updated handle it

    def _on_progress_updated(self, progress_text: str):
        """Update status bar with current progress and statistics."""
        if hasattr(self, 'current_stats'):
            stats = self.current_stats
            msg = (f"{progress_text} | "
                   f"{tr('processed')}: {stats['total']} | "
                   f"{tr('auto_accepted')}: {stats['auto_accept']} | "
                   f"{tr('auto_rejected')}: {stats['auto_reject']} | "
                   f"{tr('need_review')}: {stats['need_review']} | "
                   f"{tr('in_queue')}: {stats.get('in_queue', 0)} | "
                   f"{tr('displaying')}: {len(self.image_list._rows)}")
        else:
            msg = progress_text
        self.status_bar.showMessage(msg)

    def _on_processing_complete(self):
        """Handle processing completion."""
        # Disable pause and stop buttons
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.settings_panel.btn_pause.setEnabled(False)
        self.settings_panel.btn_stop.setEnabled(False)

        msg = (f"{tr('processing_complete')} - {tr('total')}: {self.current_stats['total']}, "
               f"{tr('auto_accepted')}: {self.current_stats['auto_accept']}, "
               f"{tr('auto_rejected')}: {self.current_stats['auto_reject']}, "
               f"{tr('need_review')}: {self.current_stats['need_review']}")
        self.status_bar.showMessage(msg)

        # Show completion dialog
        QMessageBox.information(
            self,
            tr('processing_complete'),
            f"{tr('all_images_processed')}\n\n"
            f"{tr('total')}: {self.current_stats['total']}\n"
            f"{tr('auto_accepted')}: {self.current_stats['auto_accept']}\n"
            f"{tr('auto_rejected')}: {self.current_stats['auto_reject']}\n"
            f"{tr('need_review')}: {self.current_stats['need_review']}\n"
            f"{tr('displaying')}: {len(self.image_list._rows)}"
        )

    def _on_processing_error(self, error: str):
        """Handle processing error."""
        QMessageBox.critical(self, "Processing Error", error)
        self.status_bar.showMessage("Processing failed")

    def _on_variant_selected(self, row_index: int, variant_index: int):
        """Handle variant selection (preview only)."""
        row = self.image_list._rows[row_index]
        data = row._data
        self.preview_panel.show_preview(data, variant_index)

    def _on_variant_confirmed(self, row_index: int, variant_index: int):
        """Handle variant confirmation (save and remove row)."""
        row = self.image_list._rows[row_index]
        data = row._data

        # 保存选中的变体
        self._save_confirmed_variant(data, variant_index)

        # 删除其他变体（如果需要）
        self._delete_other_variants(data, variant_index)

        # 从列表中移除这一行
        self.image_list.remove_row(row_index)
        self.preview_panel.clear()

        # 尝试从队列补充新图片
        self._refill_display()

        # 更新状态栏
        count = len(self.image_list._rows)
        self.status_bar.showMessage(f"已保存，当前显示 {count} 张")

    def _refill_display(self):
        """Refill GUI display from queue"""
        if not self.processing_thread:
            return

        self._ensure_review_area_width()

        # Continuously get images from queue until max display count reached
        while len(self.image_list._rows) < self.max_display_rows:
            next_image = self.processing_thread.get_next_image()
            if next_image is None:
                break  # Queue empty
            self.image_list.add_row(next_image)

        if len(self.image_list._rows) > 0 and getattr(self.preview_panel, '_current_data', None) is None:
            self._show_active_row_preview()

    def _save_confirmed_variant(self, data: dict, variant_index: int):
        """Save user-confirmed variant and append to JSON"""
        try:
            print(f"[DEBUG] _save_confirmed_variant called: basename={data.get('basename')}, control_type={data.get('control_type')}, variant_index={variant_index}")

            settings = self.settings_panel.get_settings()
            output_config = settings.get('output', {})
            base_dir = output_config.get('base_dir', './output')
            review_dir = os.path.join(base_dir, 'reviewed')
            os.makedirs(review_dir, exist_ok=True)

            basename = data.get('basename', 'unknown')
            control_type = data.get('control_type', 'canny')  # canny, openpose, depth
            progress_key = data.get('progress_key', basename)
            variants = data.get('variants', [])

            if variant_index < len(variants):
                variant = variants[variant_index]

                orig_img = data.get('original_image')
                image_path = data.get('image_path', '')

                orig_save_path = os.path.join(review_dir, f"{basename}.png")
                if orig_img and not os.path.exists(orig_save_path):
                    orig_img.save(orig_save_path)
                    print(f"[DEBUG] Saved original to: {orig_save_path}")

                tags = data.get('tags', '')
                tag_path = os.path.join(review_dir, f"{basename}.txt")
                if tags and not os.path.exists(tag_path):
                    with open(tag_path, 'w', encoding='utf-8') as f:
                        f.write(tags)
                    print(f"[DEBUG] Saved tags to: {tag_path}")

                control_img = variant.get('image')
                if control_img:
                    suffix_map = {
                        'canny': '_canny.png',
                        'openpose': '_openpose.png',
                        'depth': '_depth.png'
                    }
                    suffix = suffix_map.get(control_type, f'_{control_type}.png')
                    control_save_path = os.path.join(review_dir, f"{basename}{suffix}")
                    control_img.save(control_save_path)
                    print(f"[DEBUG] Saved control image to: {control_save_path}")

                    source_image_path = orig_save_path if os.path.exists(orig_save_path) else image_path
                    if source_image_path:
                        folder_map = {
                            'canny': 'canny',
                            'openpose': 'pose',
                            'depth': 'depth'
                        }
                        folder_name = folder_map.get(control_type, control_type)
                        entry = self._create_jsona_entry(source_image_path, tags, control_save_path, folder_name)

                        print(f"[DEBUG] Created jsona entry: {entry}")

                        if self.processor:
                            use_single_jsona = settings.get('processing', {}).get('single_jsona', False)
                            target_file = 'metadata' if use_single_jsona else folder_name
                            self.processor._append_to_json_file(base_dir, target_file, [entry])
                            print(f"[DEBUG] Successfully wrote to {target_file}.jsona")
                            if hasattr(self, 'settings_panel'):
                                self.settings_panel._update_jsona_statistics()

                if self.progress_manager:
                    self.progress_manager.mark_processed(progress_key)
                    self.progress_manager.mark_accepted(progress_key, auto=False)

        except Exception as e:
            QMessageBox.warning(self, tr('save_failed'), f"{tr('cannot_save_image')}: {str(e)}")

    def _delete_other_variants(self, data: dict, keep_index: int):
        """Delete other unselected variants (logical deletion for in-memory data)"""
        # For extracted data, no need to actually delete files
        # Just don't save other variants
        pass

    def _on_row_discarded(self, row_index: int):
        """Handle row discard."""
        row = self.image_list._rows[row_index]
        data = row._data
        progress_key = data.get('progress_key', data.get('basename', 'unknown'))

        # Mark as processed and user rejected in progress manager
        if self.progress_manager:
            self.progress_manager.mark_processed(progress_key)
            self.progress_manager.mark_rejected(progress_key, auto=False)

        self.image_list.remove_row(row_index)
        self.preview_panel.clear()

        # Try to refill from queue
        self._refill_display()

        count = len(self.image_list._rows)
        self.status_bar.showMessage(f"{tr('discarded')}, {tr('displaying')}: {count}")

    def _toggle_pause(self):
        """Toggle pause/resume state"""
        if not self.processing_thread or not self.processing_thread.isRunning():
            return

        if self.processing_thread.paused:
            self.processing_thread.resume()
            self.pause_action.setText(tr('pause_processing'))
            self.settings_panel.btn_pause.setText(tr('pause_processing'))
            self.status_bar.showMessage(tr('processing_resumed'))
        else:
            self.processing_thread.pause()
            self.pause_action.setText(tr('resume_processing'))
            self.settings_panel.btn_pause.setText(tr('resume_processing'))
            self.status_bar.showMessage(tr('processing_paused'))

    def _stop_processing(self):
        """Stop processing thread safely"""
        if not self.processing_thread or not self.processing_thread.isRunning():
            return

        # Stop the thread
        self.processing_thread.stop()

        # Wait for thread to finish (with timeout)
        if not self.processing_thread.wait(5000):  # 5 second timeout
            print("[WARNING] Processing thread did not stop gracefully, forcing termination")
            self.processing_thread.terminate()
            self.processing_thread.wait()

        # Clear GPU memory if using CUDA
        try:
            if self.processor and hasattr(self.processor, 'torch'):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("[DEBUG] Cleared CUDA cache")
        except Exception as e:
            print(f"[DEBUG] Error clearing CUDA cache: {e}")

        # Update UI
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.settings_panel.btn_pause.setEnabled(False)
        self.settings_panel.btn_stop.setEnabled(False)
        self.status_bar.showMessage("处理已停止")

    def _change_language(self, language: str):
        """Change UI language and restart application"""
        # Save to config
        self.config['language'] = language
        self._save_config()

        # Restart application
        self._restart_application()

    def _restart_application(self):
        """Restart the application"""
        import sys
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QProcess

        # Get the current executable and arguments
        program = sys.executable
        arguments = sys.argv

        # Close current window
        self.close()

        # Start new instance
        QProcess.startDetached(program, arguments)

        # Quit current instance
        QApplication.quit()

    def _update_ui_language(self):
        """Update all UI text with current language"""
        # Update menu bar
        self.file_menu.setTitle(tr('file'))
        self.load_config_action.setText(tr('load_config'))
        self.save_config_action.setText(tr('save_config'))
        self.exit_action.setText(tr('exit'))

        self.edit_menu.setTitle(tr('edit'))
        if self.processing_thread and self.processing_thread.paused:
            self.pause_action.setText(tr('resume_processing'))
        else:
            self.pause_action.setText(tr('pause_processing'))
        self.clear_action.setText(tr('clear_all_images'))

        self.lang_menu.setTitle(tr('language'))

        self.help_menu.setTitle(tr('help'))
        self.about_action.setText(tr('about'))
        self.jsona_manager_action.setText('JSONA 管理')
        self.jsona_import_action.setText('导入 JSONA')
        self.jsona_checker_action.setText('JSONA 核对检查器')

        # Update image list
        self.image_list.update_language()

        # Recreate settings panel with new language
        old_settings = self.settings_panel.get_settings()
        self.settings_panel.deleteLater()
        self.settings_panel = SettingsPanel(self.config)
        self.settings_panel.settings_changed.connect(self._on_settings_changed)
        self.settings_panel.start_processing.connect(self._start_processing)

        # Replace in splitter
        splitter = self.centralWidget().layout().itemAt(0).widget()
        splitter.replaceWidget(0, self.settings_panel)

    def _clear_images(self):
        """Clear all images."""
        reply = QMessageBox.question(
            self,
            "Clear Images",
            "Are you sure you want to clear all images?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.image_list.clear()
            self.preview_panel.clear()
            self.status_bar.showMessage("Images cleared")

    def _set_delete_behavior(self, behavior: str):
        """Set delete behavior for rejected images"""
        self.delete_behavior = behavior
        behavior_text = {
            'permanent': tr('delete_permanent'),
            '7days': tr('delete_7days'),
            '30days': tr('delete_30days'),
            'never': tr('delete_never')
        }.get(behavior, behavior)
        self.status_bar.showMessage(f"{tr('delete_behavior')}: {behavior_text}")

    def _cleanup_delete_folder(self):
        """Clean up .delete folder based on retention policy"""
        from datetime import datetime, timedelta
        import shutil

        if not hasattr(self, 'delete_behavior'):
            self.delete_behavior = '7days'

        if self.delete_behavior == 'never':
            QMessageBox.information(
                self,
                tr('cleanup_delete_folder'),
                tr('delete_never_no_cleanup')
            )
            return

        # Get retention days
        retention_days = {
            'permanent': 0,
            '7days': 7,
            '30days': 30
        }.get(self.delete_behavior, 7)

        settings = self.settings_panel.get_settings()
        base_dir = settings.get('output', {}).get('base_dir', './output')
        delete_dir = os.path.join(base_dir, '.delete')

        if not os.path.exists(delete_dir):
            QMessageBox.information(
                self,
                tr('cleanup_delete_folder'),
                tr('delete_folder_empty')
            )
            return

        # Count files to delete
        now = datetime.now()
        files_to_delete = []

        for filename in os.listdir(delete_dir):
            if filename.endswith('_meta.json'):
                meta_path = os.path.join(delete_dir, filename)
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    deleted_at = datetime.fromisoformat(metadata.get('deleted_at', ''))
                    age_days = (now - deleted_at).days

                    if age_days >= retention_days:
                        basename = metadata.get('basename', '')
                        files_to_delete.append(basename)
                except Exception as e:
                    print(f"Error reading metadata {filename}: {e}")

        if not files_to_delete:
            QMessageBox.information(
                self,
                tr('cleanup_delete_folder'),
                tr('no_files_to_cleanup')
            )
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            tr('cleanup_delete_folder'),
            f"{tr('confirm_cleanup')}: {len(files_to_delete)} {tr('files')}\n"
            f"{tr('retention_policy')}: {retention_days} {tr('days')}",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            deleted_count = 0
            for basename in files_to_delete:
                try:
                    # Delete image, tag, and metadata files
                    for ext in ['.png', '.jpg', '.jpeg', '.webp', '.txt', '_meta.json']:
                        file_path = os.path.join(delete_dir, f"{basename}{ext}")
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {basename}: {e}")

            QMessageBox.information(
                self,
                tr('cleanup_delete_folder'),
                f"{tr('cleanup_complete')}: {deleted_count} {tr('files')}"
            )
            self.status_bar.showMessage(f"Cleaned up {deleted_count} files from .delete folder")

    def _load_config_dialog(self):
        """Show load config dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "JSON Files (*.json)"
        )
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)

                # Recreate settings panel with new config
                old_settings = self.settings_panel
                self.settings_panel = SettingsPanel(self.config)

                # Reconnect signals
                self.settings_panel.settings_changed.connect(self._on_settings_changed)
                self.settings_panel.start_processing.connect(self._start_processing)
                self.settings_panel.extraction_finished.connect(self._on_extraction_finished)

                # Replace in splitter
                splitter = self.centralWidget().layout().itemAt(0).widget()
                splitter.replaceWidget(0, self.settings_panel)
                old_settings.deleteLater()

                self.status_bar.showMessage(f"Loaded config from {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load config: {str(e)}")

    def _install_torch(self):
        """Install PyTorch with version selection"""
        # Create version selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("安装 PyTorch")
        dialog.setMinimumWidth(550)

        layout = QVBoxLayout(dialog)

        # Info label
        info_label = QLabel("选择要安装的 PyTorch 版本:")
        layout.addWidget(info_label)

        # Version selection
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QCheckBox
        version_group = QButtonGroup(dialog)

        cpu_radio = QRadioButton("仅 CPU 版本")
        cuda124_radio = QRadioButton("CUDA 12.4")
        cuda126_radio = QRadioButton("CUDA 12.6")
        cuda128_radio = QRadioButton("CUDA 12.8")

        version_group.addButton(cpu_radio, 0)
        version_group.addButton(cuda124_radio, 1)
        version_group.addButton(cuda126_radio, 2)
        version_group.addButton(cuda128_radio, 3)

        # Try to detect CUDA and pre-select
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'CUDA Version' in result.stdout:
                cuda128_radio.setChecked(True)  # Default to latest for GPU users
            else:
                cpu_radio.setChecked(True)
        except:
            cpu_radio.setChecked(True)

        layout.addWidget(cpu_radio)
        layout.addWidget(cuda124_radio)
        layout.addWidget(cuda126_radio)
        layout.addWidget(cuda128_radio)

        # Mirror selection
        layout.addWidget(QLabel("\n下载源选择:"))
        mirror_group = QButtonGroup(dialog)

        official_radio = QRadioButton("官方源")
        nju_radio = QRadioButton("南京大学镜像")

        mirror_group.addButton(official_radio, 0)
        mirror_group.addButton(nju_radio, 1)

        official_radio.setChecked(True)  # Default to official

        layout.addWidget(official_radio)
        layout.addWidget(nju_radio)

        # Warning label
        warning_label = QLabel("\n注意: 这将卸载现有的 PyTorch 并安装所选版本。\n下载大小: CUDA 版本约 2-3GB，CPU 版本约 200MB。")
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet("color: #FFA726;")
        layout.addWidget(warning_label)

        # Buttons
        from PyQt5.QtWidgets import QDialogButtonBox
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() == QDialog.Accepted:
            selected_id = version_group.checkedId()
            mirror_id = mirror_group.checkedId()

            # Determine CUDA version suffix
            if selected_id == 0:
                cuda_suffix = "cpu"
                version_name = "CPU"
            elif selected_id == 1:
                cuda_suffix = "cu124"
                version_name = "CUDA 12.4"
            elif selected_id == 2:
                cuda_suffix = "cu126"
                version_name = "CUDA 12.6"
            else:
                cuda_suffix = "cu128"
                version_name = "CUDA 12.8"

            # Determine index URL based on mirror
            if mirror_id == 0:
                # Official
                index_url = f"https://download.pytorch.org/whl/{cuda_suffix}"
            else:
                # NJU (Nanjing University)
                index_url = f"https://mirrors.nju.edu.cn/pytorch/whl/{cuda_suffix}"

            # Uninstall first, then install
            self._run_pip_install_with_index(['torch', 'torchvision'], index_url, f'安装 PyTorch ({version_name})', uninstall_first=True)

    def _reinstall_torch(self):
        """Reinstall PyTorch"""
        # Just call _install_torch which now has version selection
        self._install_torch()

    def _install_datasets(self):
        """Install datasets library"""
        reply = QMessageBox.question(
            self,
            tr('install_datasets'),
            tr('install_datasets_confirm'),
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self._run_pip_install(['datasets'], tr('installing_datasets'))

    def _fix_datasets(self):
        """Fix datasets library by downgrading to version that doesn't require torch"""
        reply = QMessageBox.question(
            self,
            tr('fix_datasets'),
            tr('fix_datasets_confirm'),
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Downgrade to datasets 2.x which supports HF_DATASETS_DISABLE_TORCH
            self._run_pip_install(['datasets<3.0.0'], tr('fixing_datasets'), reinstall=True)

    def _install_all_dependencies(self):
        """Install all dependencies"""
        reply = QMessageBox.question(
            self,
            tr('install_all_dependencies'),
            tr('install_all_dependencies_confirm'),
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            packages = ['torch', 'torchvision', 'datasets', 'controlnet-aux', 'opencv-python']
            self._run_pip_install(packages, tr('installing_all_dependencies'))

    def _run_pip_install(self, packages: list, title: str, reinstall: bool = False):
        """Run pip install in a separate process with real-time output"""
        # Show installation dialog
        dialog = InstallProgressDialog(self, title, packages, reinstall)
        dialog.exec_()

        # Check if installation was successful
        if dialog.success:
            # Ask if user wants to restart
            reply = QMessageBox.question(
                self,
                tr('installation_complete'),
                f"{tr('successfully_installed')}: {', '.join(packages)}\n\n{tr('restart_now')}",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self._restart_application()
        else:
            QMessageBox.critical(
                self,
                tr('installation_failed'),
                f"{tr('failed_to_install')}: {', '.join(packages)}\n\n{tr('check_console_output')}"
            )

    def _run_pip_install_with_index(self, packages: list, index_url: str, title: str, uninstall_first: bool = False):
        """Run pip install with custom index URL"""
        # Show installation dialog with index URL
        dialog = InstallProgressDialogWithIndex(self, title, packages, index_url, uninstall_first)
        dialog.exec_()

        # Check if installation was successful
        if dialog.success:
            # Update device info
            self._check_device_info()

            # Ask if user wants to restart
            reply = QMessageBox.question(
                self,
                tr('installation_complete'),
                f"{tr('successfully_installed')}: {', '.join(packages)}\n\n{tr('restart_now')}",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self._restart_application()
        else:
            QMessageBox.critical(
                self,
                tr('installation_failed'),
                f"{tr('failed_to_install')}: {', '.join(packages)}\n\n{tr('check_console_output')}"
            )

    def _restart_application(self):
        """Restart the application"""
        import sys
        import subprocess
        from PyQt5.QtWidgets import qApp

        # Get the current executable and arguments
        python = sys.executable
        script = sys.argv[0]

        # Close the application
        qApp.quit()

        # Start a new instance
        subprocess.Popen([python, script])

    def _install_depth_anything(self):
        """Install Depth Anything V2"""
        import sys

        reply = QMessageBox.question(
            self,
            "安装 Depth Anything V2",
            "这将安装 depth-anything-v2 库，用于高质量深度图生成。\n\n"
            "注意: 如果 torch 已加载,程序会自动关闭并在后台安装。\n"
            "安装完成后请手动重启程序。\n\n"
            "是否继续？",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Check if torch is already imported
            if 'torch' in sys.modules:
                # Torch is loaded, need to close app and install externally
                self._install_package_external(['depth-anything-v2'], '安装 Depth Anything V2')
            else:
                # Torch not loaded, can install normally
                self._run_pip_install(['depth-anything-v2'], '安装 Depth Anything V2')

    def _install_package_external(self, packages: list, title: str):
        """Install package externally (close app and run in separate process)"""
        import sys
        import subprocess

        # Save current config
        self._save_config()

        # Create a batch script to install
        batch_script = os.path.join(os.path.dirname(sys.executable), 'install_packages.bat')
        with open(batch_script, 'w', encoding='utf-8') as f:
            f.write('@echo off\n')
            f.write('chcp 65001 >nul\n')  # Set UTF-8 encoding
            f.write(f'title {title}\n')
            f.write('echo ========================================\n')
            f.write(f'echo {title}\n')
            f.write('echo ========================================\n')
            f.write('echo.\n')
            f.write('echo 正在安装，请稍候...\n')
            f.write('echo.\n')

            # Install packages
            packages_str = ' '.join(packages)
            f.write(f'"{sys.executable}" -m pip install {packages_str}\n')

            f.write('echo.\n')
            f.write('if %ERRORLEVEL% EQU 0 (\n')
            f.write('    echo ========================================\n')
            f.write('    echo 安装成功！\n')
            f.write('    echo ========================================\n')
            f.write('    echo.\n')
            f.write('    echo 请重新启动程序。\n')
            f.write(') else (\n')
            f.write('    echo ========================================\n')
            f.write('    echo 安装失败！\n')
            f.write('    echo ========================================\n')
            f.write('    echo.\n')
            f.write('    echo 请检查错误信息。\n')
            f.write(')\n')
            f.write('echo.\n')
            f.write('pause\n')
            f.write(f'del "{batch_script}"\n')

        # Start batch script
        subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', batch_script], shell=True)

        QMessageBox.information(
            self,
            "正在安装",
            "安装窗口已打开，程序即将关闭。\n\n"
            "请等待安装完成后重启程序。"
        )

        # Close application
        from PyQt5.QtWidgets import qApp
        qApp.quit()

    def _check_dependencies(self):
        """Check installed dependencies"""
        deps_status = []

        # Check torch
        try:
            import torch
            deps_status.append(f"[已安装] torch: {torch.__version__}")
        except ImportError as e:
            deps_status.append(f"[未安装] torch: {tr('not_installed')}")
        except Exception as e:
            deps_status.append(f"[错误] torch: {tr('error_loading')} ({str(e)})")

        # Check datasets
        try:
            import datasets
            deps_status.append(f"[已安装] datasets: {datasets.__version__}")
        except ImportError:
            deps_status.append(f"[未安装] datasets: {tr('not_installed')}")
        except Exception as e:
            deps_status.append(f"[错误] datasets: {tr('error_loading')} ({str(e)})")

        # Check controlnet_aux
        try:
            import controlnet_aux
            deps_status.append(f"[已安装] controlnet_aux: installed")
        except ImportError:
            deps_status.append(f"[未安装] controlnet_aux: {tr('not_installed')}")
        except Exception as e:
            deps_status.append(f"[错误] controlnet_aux: {tr('error_loading')} ({str(e)})")

        # Check opencv
        try:
            import cv2
            deps_status.append(f"[已安装] opencv-python: {cv2.__version__}")
        except ImportError:
            deps_status.append(f"[未安装] opencv-python: {tr('not_installed')}")
        except Exception as e:
            deps_status.append(f"[错误] opencv-python: {tr('error_loading')} ({str(e)})")

        # Show results
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(tr('check_dependencies'))
        msg.setText(tr('dependencies_status') + '\n\n' + '\n'.join(deps_status))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def _create_jsona_backup_manager(self):
        """Create or reuse the JSONA backup manager for the current settings."""
        settings = self.settings_panel.get_settings()
        processing_config = settings.get('processing', {})
        if self.processor and hasattr(self.processor, 'jsona_backup_manager'):
            return self.processor.jsona_backup_manager
        return JsonaBackupManager(
            backup_interval_entries=processing_config.get('jsona_backup_every_entries', 200),
            rolling_keep=processing_config.get('jsona_backup_keep', 10),
            backup_interval_seconds=processing_config.get('jsona_backup_every_seconds', 600),
        )

    def _show_jsona_dialog(self, initial_action: str = ''):
        """Open the JSONA dialog, optionally triggering an initial action."""
        settings = self.settings_panel.get_settings()
        output_config = settings.get('output', {})
        base_dir = output_config.get('base_dir', './output')
        os.makedirs(base_dir, exist_ok=True)

        backup_manager = self._create_jsona_backup_manager()
        allow_mutation = not (self.processing_thread and self.processing_thread.isRunning())
        if not allow_mutation:
            QMessageBox.information(
                self,
                'JSONA 管理',
                '当前任务正在运行。为了避免和写入线程冲突，JSONA 管理器将以只读模式打开。'
            )

        dialog = JsonaManagerDialog(base_dir, backup_manager, allow_mutation=allow_mutation, parent=self)
        if initial_action == 'import' and allow_mutation:
            dialog.import_jsona()
        elif initial_action == 'check':
            dialog.check_selected()
        dialog.exec_()

    def _open_jsona_manager(self):
        """Open the JSONA manager dialog."""
        self._show_jsona_dialog()

    def _open_jsona_importer(self):
        """Open the JSONA dialog and start import flow."""
        self._show_jsona_dialog(initial_action='import')

    def _open_jsona_checker(self):
        """Open the JSONA dialog and run checker for the selected file."""
        self._show_jsona_dialog(initial_action='check')

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About",
            "ControlNet Data Processing Tool\n\n"
            "A portable tool for processing and filtering ControlNet training data.\n\n"
            "Features:\n"
            "- Multi-threshold Canny edge detection\n"
            "- OpenPose and Depth map generation\n"
            "- Automatic quality scoring\n"
            "- Three-tier filtering system\n"
            "- Streaming and local file support"
        )

    def closeEvent(self, event):
        """Handle window close event."""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Processing is still running. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.processing_thread.stop()
                if not self.processing_thread.wait(5000):  # 5 second timeout
                    print("[WARNING] Processing thread did not stop, forcing termination")
                    self.processing_thread.terminate()
                    self.processing_thread.wait()

                # Clear GPU memory
                try:
                    if self.processor and hasattr(self.processor, 'torch'):
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            print("[DEBUG] Cleared CUDA cache on exit")
                except Exception as e:
                    print(f"[DEBUG] Error clearing CUDA cache on exit: {e}")

                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
