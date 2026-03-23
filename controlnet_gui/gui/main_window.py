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
import threading
from typing import Dict, List, Optional
from queue import Queue, Empty, Full

from .settings_panel import SettingsPanel
from .image_list import ImageListWidget
from .preview_panel import PreviewPanel
from .jsona_manager_dialog import JsonaManagerDialog
from .duplicate_resolution_dialog import DuplicateResolutionDialog
from .vlm_tag_dialog import VlmTagDialog
from ..core.data_source import LocalDataSource
from ..core.controlnet_processor import ControlNetProcessor
from ..core.image_prefilter import ImagePreFilter
from ..core.progress_manager import ProgressManager
from ..core.jsona_backup import JsonaBackupManager
from ..core.review_inbox import ReviewInbox, ReviewInboxPolicy
from ..core.tag_formats import build_nl_prompt, build_xml_fragment
from ..core.vlm_client import VlmConfig
from ..core.score_converter import (
    bbox_to_10_scale, canny_to_10_scale, openpose_to_10_scale, depth_to_10_scale
)
from ..language import get_lang_manager, tr


def build_progress_key(image_path: str, control_type: str, basename: str = '') -> str:
    """Create a stable per-review-item progress key."""
    if image_path:
        normalized_path = os.path.abspath(image_path).replace('\\', '/').lower()
    else:
        normalized_path = basename
    return f"{normalized_path}::{control_type}"


def score_filter_to_10_scale(aesthetic) -> float:
    try:
        return max(0.0, min(10.0, float(aesthetic or 0) * 2.0))
    except Exception:
        return 0.0


def score_filter_to_100_scale(aesthetic) -> float:
    try:
        return max(0.0, min(100.0, float(aesthetic or 0) * 20.0))
    except Exception:
        return 0.0


def save_score_mode_accept(
    base_dir: str,
    basename: str,
    original_image,
    image_path: str,
    tags: str = '',
    prefilter: Optional[dict] = None,
    profile: str = '',
    review_source: str = '',
) -> str:
    import shutil
    from PIL import Image as PILImage

    out_dir = os.path.join(base_dir, 'score_accepted')
    os.makedirs(out_dir, exist_ok=True)

    image_save_path = os.path.join(out_dir, f"{basename}.png")
    if original_image is not None:
        original_image.save(image_save_path)
    elif image_path and os.path.exists(image_path):
        try:
            with PILImage.open(image_path) as _im:
                _im.convert("RGB").save(image_save_path)
        except Exception:
            shutil.copy2(image_path, image_save_path)

    if tags:
        with open(os.path.join(out_dir, f"{basename}.txt"), 'w', encoding='utf-8') as f:
            f.write(tags)

    meta_path = os.path.join(out_dir, f"{basename}_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    else:
        meta = {}

    meta.update({
        'basename': basename,
        'profile': profile,
        'processing_mode': 'image_score',
        'original_path': image_path,
    })
    if review_source:
        meta['accepted_from'] = review_source

    prefilter = prefilter or {}
    if prefilter.get('quality_warning', False):
        meta['prefilter_warning'] = prefilter.get('sharpness_level', 'unknown')

    score_filter = prefilter.get('score_filter', {}) if isinstance(prefilter, dict) else {}
    if score_filter:
        meta['score_filter'] = {
            'aesthetic': score_filter.get('aesthetic'),
            'composition': score_filter.get('composition'),
            'color': score_filter.get('color'),
            'sexual': score_filter.get('sexual'),
            'in_domain_prob': score_filter.get('in_domain_prob'),
            'in_domain_pred': score_filter.get('in_domain_pred'),
            'reason': score_filter.get('reason', ''),
            'device': score_filter.get('device'),
        }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return image_save_path


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
    duplicate_resolution_needed = pyqtSignal(dict)

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
            'inbox_pending': 0,
            'in_queue': 0
        }
        self._duplicate_resolution_default = None
        self._duplicate_resolution_event = None
        self._duplicate_resolution_result = None

    def _request_duplicate_resolution(self, payload: Dict) -> str:
        if self._duplicate_resolution_default in ('overwrite', 'new_revision'):
            return self._duplicate_resolution_default
        self._duplicate_resolution_event = threading.Event()
        self._duplicate_resolution_result = {'action': 'new_revision', 'apply_all': False}
        self.duplicate_resolution_needed.emit(payload)
        while self.running:
            if self._duplicate_resolution_event.wait(0.2):
                break
        result = self._duplicate_resolution_result or {'action': 'new_revision', 'apply_all': False}
        action = str(result.get('action') or 'new_revision')
        if result.get('apply_all'):
            self._duplicate_resolution_default = action
        self._duplicate_resolution_event = None
        self._duplicate_resolution_result = None
        return action

    def resolve_duplicate_decision(self, action: str, apply_all: bool = False):
        self._duplicate_resolution_result = {
            'action': action,
            'apply_all': bool(apply_all),
        }
        if self._duplicate_resolution_event is not None:
            self._duplicate_resolution_event.set()

    @staticmethod
    def _short_text(value: str, limit: int = 800) -> str:
        text = (value or '').strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + '...'

    def _format_existing_review_summary(self, record: dict) -> str:
        stored = record.get('stored', {}) or {}
        variants = list(stored.get('variants', []) or [])
        lines = [
            f"revision: {record.get('revision', 0)}",
            f"time: {record.get('ts', '')}",
            f"type: {record.get('control_type', 'unknown')}",
            f"score: {record.get('best_score', 0)}",
            f"profile: {record.get('profile', '')}",
            f"候选图数量: {len(variants)}",
            "",
            "tags:",
            self._short_text(record.get('tags', '')),
        ]
        if record.get('prefilter'):
            lines.extend(["", "prefilter:", json.dumps(record.get('prefilter', {}), ensure_ascii=False, indent=2)])
        if variants:
            variant_lines = []
            for variant in variants[:4]:
                chunk = [f"- idx={variant.get('idx', 0)} score={variant.get('score', 0)} preset={variant.get('preset', 'unknown')}"]
                if variant.get('thresholds'):
                    chunk.append(f"  thresholds={variant.get('thresholds')}")
                if variant.get('warning'):
                    chunk.append(f"  warning={variant.get('warning')}")
                if variant.get('metrics'):
                    chunk.append(f"  metrics={variant.get('metrics')}")
                variant_lines.extend(chunk)
            lines.extend(["", "variants:", "\n".join(variant_lines)])
        return "\n".join(lines).strip()

    def _format_live_review_summary(
        self,
        *,
        basename: str,
        control_type: str,
        tags: str,
        variants: list,
        best_score: float,
        profile: str,
        prefilter: Optional[dict],
        next_revision: int,
    ) -> str:
        lines = [
            f"revision: {next_revision}",
            "time: 当前这次",
            f"type: {control_type}",
            f"score: {best_score}",
            f"profile: {profile}",
            f"候选图数量: {len(variants or [])}",
            "",
            "tags:",
            self._short_text(tags),
        ]
        if prefilter:
            lines.extend(["", "prefilter:", json.dumps(prefilter, ensure_ascii=False, indent=2)])
        if variants:
            variant_lines = []
            for idx, variant in enumerate((variants or [])[:4]):
                chunk = [f"- idx={idx} score={variant.get('score', 0)} preset={variant.get('preset', 'unknown')}"]
                if variant.get('thresholds'):
                    chunk.append(f"  thresholds={variant.get('thresholds')}")
                if variant.get('warning'):
                    chunk.append(f"  warning={variant.get('warning')}")
                if variant.get('metrics'):
                    chunk.append(f"  metrics={variant.get('metrics')}")
                variant_lines.extend(chunk)
            lines.extend(["", "variants:", "\n".join(variant_lines)])
        return "\n".join(lines).strip()

    def _score_mode_accept_threshold(self) -> float:
        processing_cfg = self.settings.get('processing', {}) or {}
        try:
            value = float(processing_cfg.get('score_mode_auto_accept', 4.0))
        except Exception:
            value = 4.0
        return max(0.0, min(5.0, value))

    def _build_score_mode_gui_data(
        self,
        *,
        img,
        image_path: str,
        basename: str,
        tags: str,
        prefilter_result: dict,
        progress_key: str,
    ) -> dict:
        score_filter = prefilter_result.get('score_filter', {}) or {}
        score_filter_cfg = ((self.settings.get('prefilter', {}) or {}).get('score_filter', {}) or {})
        aesthetic = score_filter.get('aesthetic')
        score_10 = score_filter_to_10_scale(aesthetic)
        raw_score = score_filter_to_100_scale(aesthetic)
        try:
            min_aesthetic_score = float(score_filter_cfg.get('min_aesthetic_score', 2.5))
        except Exception:
            min_aesthetic_score = 2.5
        try:
            min_in_domain_prob = float(score_filter_cfg.get('min_in_domain_prob', 0.5))
        except Exception:
            min_in_domain_prob = 0.5
        auto_accept_aesthetic = self._score_mode_accept_threshold()
        variant = {
            'image': img.copy(),
            'path': image_path,
            'preset': {
                'aesthetic': score_filter.get('aesthetic'),
                'composition': score_filter.get('composition'),
                'color': score_filter.get('color'),
                'sexual': score_filter.get('sexual'),
                'in_domain_prob': score_filter.get('in_domain_prob'),
                'in_domain_pred': score_filter.get('in_domain_pred'),
                'reason': score_filter.get('reason', ''),
                'device': score_filter.get('device'),
                'passed': score_filter.get('passed', True),
                'auto_accept_aesthetic': auto_accept_aesthetic,
                'min_aesthetic_score': min_aesthetic_score,
                'require_in_domain': bool(score_filter_cfg.get('require_in_domain', True)),
                'min_in_domain_prob': min_in_domain_prob,
            },
            'preset_name': '原图评分',
            'score': raw_score,
            'score_10': score_10,
            'is_best': True,
        }
        return {
            'original_image': img,
            'original_path': image_path,
            'image_path': image_path,
            'basename': basename,
            'tags': tags,
            'prefilter': prefilter_result,
            'control_type': 'prefilter_score',
            'progress_key': progress_key,
            'variants': [variant],
            'best_score': raw_score,
            'control_10_score': score_10,
            'profile': 'image_score',
            'processing_mode': 'image_score',
        }

    def run(self):
        """Continuously process images from data source."""
        try:
            output_config = self.settings.get('output', {})
            base_dir = output_config.get('base_dir', './output')
            processing_cfg = self.settings.get('processing', {}) or {}
            processing_mode = str(processing_cfg.get('mode', 'controlnet') or 'controlnet').strip().lower()
            if processing_mode not in {'controlnet', 'image_score'}:
                processing_mode = 'controlnet'

            # Create output directories
            os.makedirs(os.path.join(base_dir, 'accepted'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'rejected'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'reviewed'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'score_accepted'), exist_ok=True)

            # Unattended mode: save review-needed items to disk inbox, do not enqueue images into GUI.
            self.unattended_mode = bool(processing_cfg.get('unattended_mode', False))
            self.review_inbox = None
            if self.unattended_mode:
                policy = ReviewInboxPolicy(
                    max_mb=int(processing_cfg.get('unattended_inbox_max_mb', 2048)),
                    on_full=str(processing_cfg.get('unattended_inbox_full_action', 'pause')).lower(),
                )
                self.review_inbox = ReviewInbox(base_dir, policy=policy)
                with QMutexLocker(self.mutex):
                    self.stats['inbox_pending'] = len(self.review_inbox.iter_pending())
                self.stats_updated.emit(self.stats.copy())

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
                prefilter_progress_key = build_progress_key(image_path, 'prefilter_score', basename)
                if (
                    self.prefilter.has_active_score_filter()
                    and self.progress_manager
                    and (
                        (processing_mode == 'image_score' and self.progress_manager.is_processed(prefilter_progress_key))
                        or (processing_mode == 'controlnet' and self.progress_manager.is_rejected(prefilter_progress_key))
                    )
                ):
                    continue

                # Apply pre-filter (blur detection)
                from PIL import Image
                with Image.open(image_path) as _im:
                    img = _im.convert("RGB")
                prefilter_result = self.prefilter.evaluate(img)

                tag_file = os.path.join(tag_dir, f"{basename}.txt")
                tags = ''
                if os.path.exists(tag_file):
                    with open(tag_file, 'r', encoding='utf-8') as f:
                        tags = f.read().strip()

                # Append custom tags if enabled
                custom_tags_config = self.settings.get('custom_tags', {})
                if custom_tags_config.get('enabled', False):
                    custom_tags = custom_tags_config.get('tags', '').strip()
                    if custom_tags:
                        tags = f"{tags}, {custom_tags}" if tags else custom_tags

                if processing_mode == 'image_score':
                    score_filter = prefilter_result.get('score_filter', {}) or {}
                    gui_data = self._build_score_mode_gui_data(
                        img=img,
                        image_path=image_path,
                        basename=basename,
                        tags=tags,
                        prefilter_result=prefilter_result,
                        progress_key=prefilter_progress_key,
                    )
                    best_score = float(gui_data.get('best_score', 0) or 0)
                    auto_accept_th = score_filter_to_100_scale(self._score_mode_accept_threshold())
                    gui_preset = (((gui_data.get('variants') or [{}])[0]).get('preset', {}) or {})
                    variants_src = [{
                        'path': image_path,
                        'score': best_score,
                        'preset': '原图评分',
                        'metrics': {
                            'aesthetic': score_filter.get('aesthetic'),
                            'composition': score_filter.get('composition'),
                            'color': score_filter.get('color'),
                            'sexual': score_filter.get('sexual'),
                            'in_domain_prob': score_filter.get('in_domain_prob'),
                            'in_domain_pred': score_filter.get('in_domain_pred'),
                            'device': score_filter.get('device'),
                            'reason': score_filter.get('reason', ''),
                            'auto_accept_aesthetic': gui_preset.get('auto_accept_aesthetic'),
                            'min_aesthetic_score': gui_preset.get('min_aesthetic_score'),
                            'require_in_domain': gui_preset.get('require_in_domain'),
                            'min_in_domain_prob': gui_preset.get('min_in_domain_prob'),
                        },
                    }]

                    with QMutexLocker(self.mutex):
                        self.stats['total'] += 1

                        if prefilter_result.get('skip_processing') or score_filter.get('should_reject', False):
                            self.stats['auto_reject'] += 1
                            gui_data['auto_action'] = 'reject'
                            reject_result = {
                                'basename': basename,
                                'image_path': image_path,
                                'control_type': 'prefilter_score',
                                'best_score': best_score,
                                'prefilter': prefilter_result,
                                'reject_reason': score_filter.get('reason', ''),
                            }
                            self._handle_rejected(reject_result, output_config)
                            if self.progress_manager:
                                self.progress_manager.mark_processed(prefilter_progress_key)
                                self.progress_manager.mark_rejected(prefilter_progress_key, auto=True)
                        elif best_score >= auto_accept_th:
                            self.stats['auto_accept'] += 1
                            gui_data['auto_action'] = 'accept'
                            self._save_accepted(gui_data, base_dir)
                            if self.progress_manager:
                                self.progress_manager.mark_processed(prefilter_progress_key)
                                self.progress_manager.mark_accepted(prefilter_progress_key, auto=True)
                        else:
                            gui_data['auto_action'] = None

                    self.stats_updated.emit(self.stats.copy())

                    enqueued = False
                    if gui_data.get('auto_action') is None:
                        if getattr(self, 'unattended_mode', False) and self.review_inbox is not None:
                            duplicate_mode = 'reuse'
                            duplicate_info = self.review_inbox.inspect_duplicate(
                                progress_key=prefilter_progress_key,
                                basename=basename,
                                control_type='prefilter_score',
                                original_path=image_path,
                                tag_text=tags,
                                variants=variants_src,
                                best_score=best_score,
                                profile='image_score',
                                prefilter=prefilter_result,
                            )
                            if duplicate_info.get('status') == 'duplicate':
                                duplicate_mode = self._request_duplicate_resolution({
                                    'kind': 'image_score_review',
                                    'basename': basename,
                                    'control_type': 'prefilter_score',
                                    'existing_revision': int(duplicate_info.get('record', {}).get('revision', 0) or 0),
                                    'next_revision': int(duplicate_info.get('next_revision', 0) or 0),
                                    'record_count': int(duplicate_info.get('record_count', 0) or 0),
                                    'existing_summary': self._format_existing_review_summary(duplicate_info.get('record', {}) or {}),
                                    'new_summary': self._format_live_review_summary(
                                        basename=basename,
                                        control_type='prefilter_score',
                                        tags=tags,
                                        variants=variants_src,
                                        best_score=best_score,
                                        profile='image_score',
                                        prefilter=prefilter_result,
                                        next_revision=int(duplicate_info.get('next_revision', 0) or 0),
                                    ),
                                })
                            stored, info = self.review_inbox.add_item(
                                progress_key=prefilter_progress_key,
                                basename=basename,
                                control_type='prefilter_score',
                                original_path=image_path,
                                tag_text=tags,
                                variants=variants_src,
                                best_score=best_score,
                                profile='image_score',
                                prefilter=prefilter_result,
                                duplicate_mode=duplicate_mode,
                            )
                            if stored:
                                pending_delta = int(info.get('pending_delta', 0) or 0)
                                if pending_delta > 0:
                                    with QMutexLocker(self.mutex):
                                        self.stats['need_review'] += pending_delta
                                        self.stats['inbox_pending'] = int(self.stats.get('inbox_pending', 0)) + pending_delta
                                        self.stats['in_queue'] = self.review_queue.qsize()
                                    self.stats_updated.emit(self.stats.copy())
                                if self.progress_manager:
                                    self.progress_manager.mark_processed(prefilter_progress_key)
                            else:
                                error_code = str(info.get('error', '') or '').lower()
                                if error_code == 'inbox_full':
                                    current_mb = info.get('current_mb', 0)
                                    max_mb = info.get('max_mb', 0)
                                    self.progress_updated.emit(
                                        f"审核箱已满 ({current_mb:.0f}/{max_mb:.0f} MB)，已自动暂停。请处理审核箱或调大上限后继续。"
                                    )
                                    on_full = str(info.get('on_full', 'pause')).lower()
                                    if on_full == 'stop':
                                        self.running = False
                                        break
                                    self.paused = True
                                else:
                                    self.progress_updated.emit(
                                        f"写入审核箱失败({error_code or 'unknown_error'})，已自动暂停。"
                                        "请检查输出目录权限、源文件可读性或磁盘空间后继续。"
                                    )
                                    self.paused = True
                        else:
                            while self.running:
                                try:
                                    self.review_queue.put(gui_data, timeout=0.1)
                                    enqueued = True
                                    break
                                except Full:
                                    continue
                            if enqueued:
                                with QMutexLocker(self.mutex):
                                    self.stats['need_review'] += 1
                            if not enqueued and not self.running:
                                break
                    elif not self.settings.get('processing', {}).get('auto_pass_no_review', True):
                        if not getattr(self, 'unattended_mode', False):
                            try:
                                self.review_queue.put(gui_data, timeout=0.1)
                                enqueued = True
                            except Full:
                                pass

                    with QMutexLocker(self.mutex):
                        self.stats['in_queue'] = self.review_queue.qsize()
                    self.stats_updated.emit(self.stats.copy())
                    if enqueued:
                        self.image_ready.emit()
                    continue

                if prefilter_result.get('skip_processing'):
                    score_filter = prefilter_result.get('score_filter', {}) or {}
                    reject_result = {
                        'basename': basename,
                        'image_path': image_path,
                        'control_type': 'prefilter_score',
                        'best_score': score_filter_to_100_scale(score_filter.get('aesthetic')),
                        'prefilter': prefilter_result,
                        'reject_reason': score_filter.get('reason', ''),
                    }
                    with QMutexLocker(self.mutex):
                        self.stats['total'] += 1
                        self.stats['auto_reject'] += 1
                    self._handle_rejected(reject_result, output_config)
                    if self.progress_manager:
                        self.progress_manager.mark_processed(prefilter_progress_key)
                        self.progress_manager.mark_rejected(prefilter_progress_key, auto=True)
                    self.stats_updated.emit(self.stats.copy())
                    continue

                # Process image with ControlNet
                result = self.processor.process_image(image_path, base_dir, basename)

                # Add prefilter info
                result['prefilter'] = prefilter_result
                result['basename'] = basename
                result['image_path'] = image_path

                result['tags'] = tags

                # Generate multiple rows - one for each enabled control type
                # Extract control results from the result dict
                canny_result = result.get('canny', {})
                openpose_result = result.get('openpose', {})
                depth_result = result.get('depth', {})
                bbox_result = result.get('bbox', {})

                control_types_to_display = []
                if canny_result and canny_result.get('variants'):
                    control_types_to_display.append(('canny', canny_result))
                if openpose_result and openpose_result.get('variants'):
                    control_types_to_display.append(('openpose', openpose_result))
                if depth_result and depth_result.get('variants'):
                    control_types_to_display.append(('depth', depth_result))
                if bbox_result and bbox_result.get('variants'):
                    control_types_to_display.append(('bbox', bbox_result))

                if not control_types_to_display:
                    # No valid control type, skip this image
                    continue

                def _variants_for_review_src(_control_type: str, _control_result: dict):
                    variants = list((_control_result or {}).get('variants', []) or [])
                    if _control_type == 'canny':
                        best_by_preset = {}
                        for v in variants:
                            preset = v.get('preset')
                            if preset is None:
                                continue
                            if preset not in best_by_preset or float(v.get('score', 0)) > float(best_by_preset[preset].get('score', 0)):
                                best_by_preset[preset] = v

                        order = {'light': 0, 'medium': 1, 'clean': 2, 'strong': 3}
                        selected = list(best_by_preset.values())
                        selected.sort(key=lambda v: order.get(v.get('preset'), 999))

                        # If we have fewer than 4 (unexpected), fill with next best remaining.
                        if len(selected) < 4 and variants:
                            used = set(id(x) for x in selected)
                            rest = [v for v in variants if id(v) not in used]
                            rest.sort(key=lambda v: float(v.get('score', 0)), reverse=True)
                            selected.extend(rest[: max(0, 4 - len(selected))])

                        return selected[:4]

                    return variants[:4]

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
                    elif control_type == 'bbox':
                        auto_accept_th = float(profile.get('bbox_auto_accept', profile.get('auto_accept', 55)))
                        auto_reject_th = float(profile.get('bbox_auto_reject', profile.get('auto_reject', 40)))
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
                    elif control_type == 'bbox':
                        control_10_score = bbox_to_10_scale(best_score) if best_score > 0 else 1.0

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
                    variants_src = _variants_for_review_src(control_type, control_result)
                    for variant in variants_src:
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
                        elif control_type == 'bbox':
                            variant_10_score = bbox_to_10_scale(variant.get('score', 0))
                            preset_info = {
                                **(variant.get('metrics', {}) or {}),
                                'warning': variant.get('warning'),
                                'detections': variant.get('detections', []),
                            }
                            preset_name = variant.get('preset', 'bbox')

                        gui_data['variants'].append({
                            'image': variant_img,
                            'path': variant.get('path'),
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
                            gui_data['auto_action'] = None

                    # Send statistics update
                    self.stats_updated.emit(self.stats.copy())

                    # Check if auto-pass should skip review
                    auto_pass_no_review = self.settings.get('processing', {}).get('auto_pass_no_review', True)
                    enqueued = False

                    # Decide whether to add to review queue
                    if auto_reject_th < best_score < auto_accept_th:
                        if getattr(self, 'unattended_mode', False) and self.review_inbox is not None:
                            duplicate_mode = 'reuse'
                            duplicate_info = self.review_inbox.inspect_duplicate(
                                progress_key=progress_key,
                                basename=basename,
                                control_type=control_type,
                                original_path=image_path,
                                tag_text=result.get('tags', ''),
                                variants=variants_src,
                                best_score=best_score,
                                profile=active_profile,
                                prefilter=prefilter_result,
                            )
                            if duplicate_info.get('status') == 'duplicate':
                                duplicate_mode = self._request_duplicate_resolution({
                                    'kind': 'controlnet_review',
                                    'basename': basename,
                                    'control_type': control_type,
                                    'existing_revision': int(duplicate_info.get('record', {}).get('revision', 0) or 0),
                                    'next_revision': int(duplicate_info.get('next_revision', 0) or 0),
                                    'record_count': int(duplicate_info.get('record_count', 0) or 0),
                                    'existing_summary': self._format_existing_review_summary(duplicate_info.get('record', {}) or {}),
                                    'new_summary': self._format_live_review_summary(
                                        basename=basename,
                                        control_type=control_type,
                                        tags=result.get('tags', ''),
                                        variants=variants_src,
                                        best_score=best_score,
                                        profile=active_profile,
                                        prefilter=prefilter_result,
                                        next_revision=int(duplicate_info.get('next_revision', 0) or 0),
                                    ),
                                })
                            stored, info = self.review_inbox.add_item(
                                progress_key=progress_key,
                                basename=basename,
                                control_type=control_type,
                                original_path=image_path,
                                tag_text=result.get('tags', ''),
                                variants=variants_src,
                                best_score=best_score,
                                profile=active_profile,
                                prefilter=prefilter_result,
                                duplicate_mode=duplicate_mode,
                            )

                            if stored:
                                pending_delta = int(info.get('pending_delta', 0) or 0)
                                if pending_delta > 0:
                                    with QMutexLocker(self.mutex):
                                        self.stats['need_review'] += pending_delta
                                        self.stats['inbox_pending'] = int(self.stats.get('inbox_pending', 0)) + pending_delta
                                        self.stats['in_queue'] = self.review_queue.qsize()
                                    self.stats_updated.emit(self.stats.copy())

                                # Mark as processed so restarts don't re-add duplicates; user can import inbox later.
                                if self.progress_manager:
                                    self.progress_manager.mark_processed(progress_key)
                            else:
                                error_code = str(info.get('error', '') or '').lower()
                                if error_code == 'inbox_full':
                                    # Inbox full: auto pause/stop.
                                    current_mb = info.get('current_mb', 0)
                                    max_mb = info.get('max_mb', 0)
                                    self.progress_updated.emit(
                                        f"审核箱已满 ({current_mb:.0f}/{max_mb:.0f} MB)，已自动暂停。请处理审核箱或调大上限后继续。"
                                    )
                                    on_full = str(info.get('on_full', 'pause')).lower()
                                    if on_full == 'stop':
                                        self.running = False
                                        break
                                    self.paused = True
                                else:
                                    self.progress_updated.emit(
                                        f"写入审核箱失败({error_code or 'unknown_error'})，已自动暂停。"
                                        "请检查输出目录权限、源文件可读性或磁盘空间后继续。"
                                    )
                                    self.paused = True
                        else:
                            # Needs review - retry with short timeouts so stop/exit can interrupt cleanly.
                            while self.running:
                                try:
                                    self.review_queue.put(gui_data, timeout=0.1)
                                    enqueued = True
                                    break
                                except Full:
                                    continue
                            if enqueued:
                                with QMutexLocker(self.mutex):
                                    self.stats['need_review'] += 1
                            if not enqueued and not self.running:
                                break
                    elif not auto_pass_no_review:
                        # Setting disabled - show all auto items for review
                        if not getattr(self, 'unattended_mode', False):
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

            if control_type == 'prefilter_score':
                save_score_mode_accept(
                    base_dir=base_dir,
                    basename=basename,
                    original_image=result.get('original_image'),
                    image_path=result.get('image_path', ''),
                    tags=result.get('tags', ''),
                    prefilter=result.get('prefilter', {}),
                    profile=result.get('profile', 'image_score'),
                    review_source='auto_accept',
                )
                return

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
            elif control_type == 'bbox':
                save_path = os.path.join(out_dir, f"{basename}_bbox.png")
                control_img.save(save_path)
                print(f"[DEBUG] Saved bbox to: {save_path}")

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
            elif control_type == 'bbox':
                meta['bbox_preset'] = preset_name
                meta['bbox_score'] = result.get('best_score', 0)

            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # Write to JSONA files for training
            self._write_jsona_metadata(result, base_dir, extra)

        except Exception as e:
            print(f"Failed to save auto-accepted image: {e}")

    def _create_jsona_backup_manager(self):
        processing_config = self.settings.get('processing', {}) or {}
        if self.processor and hasattr(self.processor, 'jsona_backup_manager'):
            return self.processor.jsona_backup_manager
        return JsonaBackupManager(
            backup_interval_entries=processing_config.get('jsona_backup_every_entries', 200),
            rolling_keep=processing_config.get('jsona_backup_keep', 10),
            backup_interval_seconds=processing_config.get('jsona_backup_every_seconds', 600),
        )

    def _write_jsona_entries(self, base_dir: str, target_name: str, entries: List[dict]) -> Dict[str, int]:
        if not entries:
            return {'added': 0, 'updated': 0}
        os.makedirs(base_dir, exist_ok=True)
        jsona_path = os.path.join(base_dir, f"{target_name}.jsona")
        backup_manager = self._create_jsona_backup_manager()
        if target_name in {'tag', 'nl', 'xml'}:
            return backup_manager.upsert_entries(jsona_path, entries)
        added = backup_manager.append_entries(jsona_path, entries)
        return {'added': added, 'updated': 0}

    def _write_text_jsona_entries(self, base_dir: str, source_image_path: str, tag_text: str) -> Dict[str, Dict[str, int]]:
        prompt = (tag_text or '').strip()
        if not prompt or not source_image_path:
            return {}

        xml_config = (self.settings.get('processing', {}) or {}).get('xml_mapping', {})
        results = {
            'tag': self._write_jsona_entries(
                base_dir,
                'tag',
                [self._create_jsona_entry(source_image_path, prompt, source_image_path, 'tag')],
            ),
            'nl': self._write_jsona_entries(
                base_dir,
                'nl',
                [self._create_jsona_entry(source_image_path, build_nl_prompt(prompt), source_image_path, 'nl')],
            ),
        }
        xml_fragment = build_xml_fragment(prompt, xml_config)
        if xml_fragment:
            results['xml'] = self._write_jsona_entries(
                base_dir,
                'xml',
                [self._create_jsona_entry(source_image_path, xml_fragment, source_image_path, 'xml')],
            )
        return results

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

            # Prefer already-loaded tags (includes custom tags) and fall back to sidecar tag file.
            hint_prompt = (result.get('tags') or '').strip()
            if not hint_prompt:
                try:
                    tag_file = os.path.join(os.path.dirname(image_path), f"{basename}.txt")
                    if os.path.exists(tag_file):
                        with open(tag_file, 'r', encoding='utf-8') as f:
                            hint_prompt = f.read().strip()
                except Exception:
                    pass

            use_single_jsona = self.settings.get('processing', {}).get('single_jsona', False)
            control_map = {
                'canny': ('canny', '_canny.png'),
                'openpose': ('pose', '_openpose.png'),
                'depth': ('depth', '_depth.png'),
                'bbox': ('bbox', '_bbox.png'),
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

                target_name = 'metadata' if use_single_jsona else folder_name
                self._write_jsona_entries(base_dir, target_name, [entry])
                print(f"[DEBUG] Successfully wrote to {target_name}.jsona")

                if hasattr(self, 'settings_panel'):
                    self.settings_panel._update_jsona_statistics()
            else:
                print(f"[DEBUG] Control path doesn't exist: {control_path}")

            # Always write tag/nl/xml JSONA as separate files (never merged into metadata.jsona).
            if hint_prompt:
                try:
                    self._write_text_jsona_entries(base_dir, source_image_path, hint_prompt)
                except Exception as e:
                    print(f"[WARNING] Failed to write tag/nl/xml jsona: {e}")

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
            deleted_at = datetime.now()
            safe_basename = ''.join('_' if ch in '<>:"/\\|?*' else ch for ch in str(basename or 'unknown')).strip() or 'unknown'
            record_id = f"{safe_basename}_{control_type}_{deleted_at.strftime('%Y%m%d_%H%M%S_%f')}"
            image_path = result.get('image_path')
            copied_files = []
            if image_path and os.path.exists(image_path):
                # Keep the source image in place so other control types can still finish review.
                image_ext = os.path.splitext(image_path)[1] or '.png'
                dest_image_name = f"{record_id}{image_ext}"
                dest_image = os.path.join(delete_dir, dest_image_name)
                shutil.copy2(image_path, dest_image)
                copied_files.append(dest_image_name)

                # Copy tag file if it exists.
                tag_path = image_path.replace('/images/', '/tags/').replace('\\images\\', '\\tags\\')
                tag_path = os.path.splitext(tag_path)[0] + '.txt'
                if os.path.exists(tag_path):
                    dest_tag_name = f"{record_id}.txt"
                    dest_tag = os.path.join(delete_dir, dest_tag_name)
                    shutil.copy2(tag_path, dest_tag)
                    copied_files.append(dest_tag_name)

            metadata = {
                'record_id': record_id,
                'basename': basename,
                'deleted_at': deleted_at.isoformat(),
                'reason': str(result.get('reject_reason', '') or '').strip()
                or f"Low quality score: {result.get('best_score', 0)}",
                'control_type': control_type,
                'original_path': image_path,
                'copied_files': copied_files,
            }

            prefilter = result.get('prefilter', {})
            if prefilter.get('quality_warning', False):
                metadata['prefilter_warning'] = prefilter.get('sharpness_level', 'unknown')
            score_filter = prefilter.get('score_filter', {}) if isinstance(prefilter, dict) else {}
            if score_filter:
                metadata['score_filter'] = {
                    'aesthetic': score_filter.get('aesthetic'),
                    'composition': score_filter.get('composition'),
                    'color': score_filter.get('color'),
                    'sexual': score_filter.get('sexual'),
                    'in_domain_prob': score_filter.get('in_domain_prob'),
                    'in_domain_pred': score_filter.get('in_domain_pred'),
                    'reason': score_filter.get('reason', ''),
                    'device': score_filter.get('device'),
                }

            metadata_file = os.path.join(delete_dir, f"{record_id}_meta.json")
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
        if self._duplicate_resolution_event is not None:
            self.resolve_duplicate_decision('new_revision', apply_all=False)


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
        self._standalone_jsona_backup_manager = None
        self._standalone_jsona_backup_manager_key = None
        self.current_stats = {
            'total': 0,
            'auto_accept': 0,
            'auto_reject': 0,
            'need_review': 0,
            'inbox_pending': 0,
            'in_queue': 0
        }
        # Offline/manual review queue populated from review_inbox (when processing thread is not running).
        self._inbox_review_queue: Queue = Queue()
        self._review_inbox_manager: Optional[ReviewInbox] = None
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

        self.install_score_dependencies_action = QAction("安装评分依赖", self)
        self.install_score_dependencies_action.triggered.connect(self._install_score_dependencies)
        self.tools_menu.addAction(self.install_score_dependencies_action)

        self.score_model_manager_action = QAction("评分模型管理", self)
        self.score_model_manager_action.triggered.connect(self._open_score_model_manager)
        self.tools_menu.addAction(self.score_model_manager_action)

        self.open_score_model_dir_action = QAction("打开评分模型目录", self)
        self.open_score_model_dir_action.triggered.connect(self._open_score_model_directory)
        self.tools_menu.addAction(self.open_score_model_dir_action)

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

        self.review_inbox_import_action = QAction('导入审核箱 (需要人工审核)', self)
        self.review_inbox_import_action.triggered.connect(self._import_review_inbox)
        self.tools_menu.addAction(self.review_inbox_import_action)

        self.vlm_tag_action = QAction('VLM 核对 Tag 并输出 NL', self)
        self.vlm_tag_action.triggered.connect(self._open_vlm_tag_dialog)
        self.tools_menu.addAction(self.vlm_tag_action)

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
        self._bind_settings_panel_signals()

        # Image list signals
        self.image_list.variant_selected.connect(self._on_variant_selected)
        self.image_list.variant_confirmed.connect(self._on_variant_confirmed)
        self.image_list.row_discarded.connect(self._on_row_discarded)

    def _bind_settings_panel_signals(self):
        """Connect signals from the current settings panel instance."""
        self.settings_panel.settings_changed.connect(self._on_settings_changed)
        self.settings_panel.start_processing.connect(self._start_processing)
        self.settings_panel.extraction_finished.connect(self._on_extraction_finished)
        self.settings_panel.btn_pause.clicked.connect(self._toggle_pause)
        self.settings_panel.btn_stop.clicked.connect(self._stop_processing)

    def _sync_settings_panel_runtime_state(self):
        """Keep the recreated settings panel buttons in sync with current runtime state."""
        is_running = bool(self.processing_thread and self.processing_thread.isRunning())

        self.settings_panel.btn_pause.setEnabled(is_running)
        self.settings_panel.btn_stop.setEnabled(is_running)

        if is_running and self.processing_thread.paused:
            self.settings_panel.btn_pause.setText(tr('resume_processing'))
        else:
            self.settings_panel.btn_pause.setText(tr('pause_processing'))

    def _replace_settings_panel(self, new_panel):
        """Replace the settings panel while preserving the outer scroll container."""
        old_panel = getattr(self, 'settings_panel', None)
        self.settings_panel = new_panel
        self.settings_panel.setMinimumWidth(340)
        self.settings_panel.setMaximumWidth(460)
        self.settings_scroll.setWidget(self.settings_panel)
        self._bind_settings_panel_signals()
        self._sync_settings_panel_runtime_state()

        if old_panel is not None:
            old_panel.deleteLater()

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
            processing_mode = str(processing_config.get('mode', 'controlnet') or 'controlnet').strip().lower()
            if processing_mode not in {'controlnet', 'image_score'}:
                processing_mode = 'controlnet'
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

            # Create prefilter
            prefilter_config = settings.get('prefilter', {})
            self.prefilter = ImagePreFilter(prefilter_config)
            if processing_mode == 'image_score' and not self.prefilter.has_active_score_filter():
                QMessageBox.critical(
                    self,
                    '评分模式不可用',
                    '当前已切换到图片评分模式，但原图评分筛选没有启用。\n\n'
                    '请先启用“原图评分筛选”，再开始处理。'
                )
                return
            if self.prefilter.has_active_score_filter():
                score_filter_status = self.prefilter.get_score_filter_status(force_refresh=True)
                if not score_filter_status.get('available', False):
                    QMessageBox.critical(
                        self,
                        '评分筛选不可用',
                        '原图评分筛选已启用，但当前环境不可用。\n\n'
                        + str(score_filter_status.get('message', ''))
                    )
                    return
                ok, score_error = self.prefilter.prepare_score_filter()
                if not ok:
                    QMessageBox.critical(
                        self,
                        '评分筛选初始化失败',
                        '原图评分筛选初始化失败。\n\n' + str(score_error)
                    )
                    return

            output_config = settings.get('output', {})
            base_dir = output_config.get('base_dir', './output')

            if processing_mode == 'controlnet':
                control_types = processing_config.get('control_types', {})
                if not any(control_types.values()):
                    QMessageBox.warning(
                        self,
                        "No Control Type Selected",
                        "Please select at least one control type (Canny, OpenPose, Depth, or BBox) in Processing Settings."
                    )
                    return

                # Fix torch DLL loading before creating processor
                import sys
                torch_lib_path = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'torch', 'lib')
                if os.path.exists(torch_lib_path):
                    if torch_lib_path not in os.environ.get('PATH', ''):
                        os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
                    if hasattr(os, 'add_dll_directory'):
                        try:
                            os.add_dll_directory(torch_lib_path)
                        except Exception:
                            pass

                if control_types.get('depth', False):
                    try:
                        import depth_anything_v2
                    except ImportError:
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
                            self._install_package_external(['depth-anything-v2'], '安装 Depth Anything V2')
                            return

                        QMessageBox.warning(
                            self,
                            "无法继续",
                            "未安装 Depth Anything V2，无法进行 Depth 处理。\n\n"
                            "请禁用 Depth 或从 Tools 菜单安装后再试。"
                        )
                        return

                if control_types.get('bbox', False):
                    try:
                        import ultralytics
                    except ImportError:
                        reply = QMessageBox.question(
                            self,
                            "需要安装 BBox 检测依赖",
                            "您启用了 BBox 处理，但未安装 ultralytics 库。\n\n"
                            "是否现在安装？",
                            QMessageBox.Yes | QMessageBox.No
                        )
                        if reply == QMessageBox.Yes:
                            self._run_pip_install(['ultralytics'], '安装 BBox 检测依赖')
                            return
                        QMessageBox.warning(
                            self,
                            "无法继续",
                            "未安装 ultralytics，无法进行 BBox 处理。\n\n"
                            "请禁用 BBox 或安装依赖后再试。"
                        )
                        return

                self.processor = ControlNetProcessor(settings)
                self.processor.jsona_backup_manager.prepare_output_backups(base_dir)
            else:
                self.processor = None

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
            self.processing_thread.duplicate_resolution_needed.connect(self._on_review_inbox_duplicate_resolution)
            self.processing_thread.start()

            # Enable pause and stop buttons
            self.pause_action.setEnabled(True)
            self.pause_action.setText(tr('pause_processing'))
            self.stop_action.setEnabled(True)
            self.settings_panel.btn_pause.setEnabled(True)
            self.settings_panel.btn_stop.setEnabled(True)

            if processing_mode == 'image_score':
                self.status_bar.showMessage('图片评分模式已启动')
            else:
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
        # Keep pause button text in sync even when pause is triggered automatically (e.g. unattended inbox full).
        if self.processing_thread and self.processing_thread.isRunning():
            if self.processing_thread.paused:
                self.pause_action.setText(tr('resume_processing'))
                self.settings_panel.btn_pause.setText(tr('resume_processing'))
            else:
                self.pause_action.setText(tr('pause_processing'))
                self.settings_panel.btn_pause.setText(tr('pause_processing'))

        if hasattr(self, 'current_stats'):
            stats = self.current_stats
            msg = (f"{progress_text} | "
                   f"{tr('processed')}: {stats['total']} | "
                   f"{tr('auto_accepted')}: {stats['auto_accept']} | "
                   f"{tr('auto_rejected')}: {stats['auto_reject']} | "
                   f"{tr('need_review')}: {stats['need_review']} | "
                   f"{tr('inbox_pending')}: {stats.get('inbox_pending', 0)} | "
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

    def _on_review_inbox_duplicate_resolution(self, payload: dict):
        dialog = DuplicateResolutionDialog(
            title='重复审核项',
            intro=(
                "检测到同一张图已有历史审核记录。\n"
                "左侧是旧记录，右侧是这次准备写入的新记录，请选择处理方式。"
            ),
            old_title='旧记录',
            old_text=payload.get('existing_summary', ''),
            new_title='新记录',
            new_text=payload.get('new_summary', ''),
            overwrite_label='最新记录覆盖旧记录',
            separate_label='加入审查窗口单独审查',
            apply_all_label='对本次剩余重复项都使用这个选择',
            parent=self,
        )
        dialog.exec_()
        action, apply_all = dialog.result_data()

        if self.processing_thread:
            self.processing_thread.resolve_duplicate_decision(action, apply_all)

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
        self._ensure_review_area_width()

        def _next_item():
            if self.processing_thread and self.processing_thread.isRunning():
                return self.processing_thread.get_next_image()
            try:
                return self._inbox_review_queue.get_nowait()
            except Empty:
                return None

        # Continuously get images from queue until max display count reached
        while len(self.image_list._rows) < self.max_display_rows:
            next_image = _next_item()
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

                if control_type == 'prefilter_score':
                    save_score_mode_accept(
                        base_dir=base_dir,
                        basename=basename,
                        original_image=orig_img,
                        image_path=image_path,
                        tags=tags,
                        prefilter=data.get('prefilter', {}),
                        profile=data.get('profile', 'image_score'),
                        review_source='manual_review',
                    )
                else:
                    suffix_map = {
                        'canny': '_canny.png',
                        'openpose': '_openpose.png',
                        'depth': '_depth.png',
                        'bbox': '_bbox.png',
                    }
                    suffix = suffix_map.get(control_type, f'_{control_type}.png')
                    control_save_path = os.path.join(review_dir, f"{basename}{suffix}")
                    control_img = variant.get('image')
                    if control_img is not None:
                        control_img.save(control_save_path)
                        print(f"[DEBUG] Saved control image to: {control_save_path}")
                    else:
                        source_variant_path = str(variant.get('path', '') or '').strip()
                        if source_variant_path and os.path.exists(source_variant_path):
                            import shutil
                            shutil.copy2(source_variant_path, control_save_path)
                            print(f"[DEBUG] Copied control image from path: {control_save_path}")
                        else:
                            raise ValueError('选中的控制图不存在或已失效，无法保存。')

                    source_image_path = orig_save_path if os.path.exists(orig_save_path) else image_path
                    if source_image_path and os.path.exists(control_save_path):
                        folder_map = {
                            'canny': 'canny',
                            'openpose': 'pose',
                            'depth': 'depth',
                            'bbox': 'bbox',
                        }
                        folder_name = folder_map.get(control_type, control_type)
                        entry = self._create_jsona_entry(source_image_path, tags, control_save_path, folder_name)

                        print(f"[DEBUG] Created jsona entry: {entry}")

                        use_single_jsona = settings.get('processing', {}).get('single_jsona', False)
                        target_file = 'metadata' if use_single_jsona else folder_name
                        self._write_jsona_entries(base_dir, target_file, [entry])
                        print(f"[DEBUG] Successfully wrote to {target_file}.jsona")

                        # Also write tag/nl/xml jsona for this accepted item.
                        if tags:
                            try:
                                self._write_text_jsona_entries(base_dir, source_image_path, tags, settings=settings)
                            except Exception as e:
                                print(f"[WARNING] Failed to write tag/nl/xml jsona (manual): {e}")

                        if hasattr(self, 'settings_panel'):
                            self.settings_panel._update_jsona_statistics()

                if self.progress_manager:
                    self.progress_manager.mark_processed(progress_key)
                    self.progress_manager.mark_accepted(progress_key, auto=False)
                    review_record_id = data.get('review_record_id', '')
                    if self._review_inbox_manager:
                        if review_record_id:
                            self._review_inbox_manager.mark_done_by_id(review_record_id)
                        elif progress_key:
                            self._review_inbox_manager.mark_done(progress_key)

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
            review_record_id = data.get('review_record_id', '')
            if self._review_inbox_manager:
                if review_record_id:
                    self._review_inbox_manager.mark_done_by_id(review_record_id)
                elif progress_key:
                    self._review_inbox_manager.mark_done(progress_key)

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
        self.review_inbox_import_action.setText('导入审核箱 (需要人工审核)')
        self.vlm_tag_action.setText('VLM 核对 Tag 并输出 NL')

        # Update image list
        self.image_list.update_language()

        # Recreate settings panel with new language
        self._replace_settings_panel(SettingsPanel(self.config))

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
        from datetime import datetime

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
        all_meta_records = []

        for filename in os.listdir(delete_dir):
            if filename.endswith('_meta.json'):
                meta_path = os.path.join(delete_dir, filename)
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    deleted_at = datetime.fromisoformat(metadata.get('deleted_at', ''))
                    age_days = (now - deleted_at).days
                    copied_files = metadata.get('copied_files', [])
                    if not isinstance(copied_files, list):
                        copied_files = []

                    record = {
                        'basename': str(metadata.get('basename', '') or ''),
                        'meta_path': meta_path,
                        'copied_files': [str(item) for item in copied_files if str(item or '').strip()],
                    }
                    all_meta_records.append(record)

                    if age_days >= retention_days:
                        files_to_delete.append(record)
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
            delete_meta_paths = {
                str(item.get('meta_path', '') or '')
                for item in files_to_delete
            }
            surviving_basenames = {}
            for record in all_meta_records:
                record_meta_path = str(record.get('meta_path', '') or '')
                basename = str(record.get('basename', '') or '')
                if record_meta_path and record_meta_path not in delete_meta_paths and basename:
                    surviving_basenames[basename] = surviving_basenames.get(basename, 0) + 1
            cleaned_fallback_basenames = set()

            for item in files_to_delete:
                basename = str(item.get('basename', '') or '')
                meta_path = item.get('meta_path')
                copied_files = [
                    str(name or '').strip()
                    for name in item.get('copied_files', []) or []
                    if str(name or '').strip()
                ]
                try:
                    if copied_files:
                        for copied_name in copied_files:
                            file_path = copied_name if os.path.isabs(copied_name) else os.path.join(delete_dir, copied_name)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                    elif basename and surviving_basenames.get(basename, 0) <= 0 and basename not in cleaned_fallback_basenames:
                        # Backward compatibility for older metadata records without copied_files.
                        for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.txt']:
                            file_path = os.path.join(delete_dir, f"{basename}{ext}")
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        cleaned_fallback_basenames.add(basename)
                    if meta_path and os.path.exists(meta_path):
                        os.remove(meta_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {meta_path or basename}: {e}")

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
                self._replace_settings_panel(SettingsPanel(self.config))

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

    def _get_missing_score_dependency_packages(self):
        import importlib.util

        torch_missing = (
            importlib.util.find_spec('torch') is None
            or importlib.util.find_spec('torchvision') is None
        )
        required = [
            ('transformers', 'transformers>=4.44.0'),
            ('huggingface_hub', 'huggingface-hub>=0.26.0'),
            ('open_clip', 'open-clip-torch>=2.26.1'),
            ('timm', 'timm>=1.0.12'),
            ('safetensors', 'safetensors>=0.4.5'),
            ('yaml', 'PyYAML>=6.0.2'),
            ('tqdm', 'tqdm>=4.66.5'),
            ('einops', 'einops>=0.7.0'),
        ]
        packages = []
        for module_name, package_name in required:
            if importlib.util.find_spec(module_name) is None:
                packages.append(package_name)
        return torch_missing, packages

    def _install_score_dependencies(self, skip_confirm: bool = False):
        """Install dependencies required by the original-image score filter."""
        import importlib.util
        import sys

        torch_missing, packages = self._get_missing_score_dependency_packages()
        if torch_missing:
            if not skip_confirm:
                reply = QMessageBox.question(
                    self,
                    '安装评分依赖',
                    '评分筛选依赖 PyTorch / torchvision。\n\n'
                    '这部分建议继续使用现有的 PyTorch 安装器选择 CPU / CUDA 版本，'
                    '避免直接装错版本。\n\n'
                    '是否现在打开 PyTorch 安装器？\n'
                    '安装完成后，再点一次“安装评分依赖”即可继续安装剩余依赖。',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply != QMessageBox.Yes:
                    return
            self._install_torch()
            return

        if not packages:
            QMessageBox.information(
                self,
                '安装评分依赖',
                '当前检测到评分依赖已经齐全，无需安装。'
            )
            return

        if not skip_confirm:
            package_preview = '\n'.join(f'  - {item}' for item in packages)
            reply = QMessageBox.question(
                self,
                '安装评分依赖',
                '将安装当前缺失的评分依赖:\n\n'
                f'{package_preview}\n\n'
                '是否继续？',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply != QMessageBox.Yes:
                return

        loaded_modules = {'torch', 'transformers', 'open_clip', 'timm'} & set(sys.modules.keys())
        if loaded_modules:
            self._install_package_external(packages, '安装评分依赖')
        else:
            self._run_pip_install(packages, '安装评分依赖')

    def _refresh_dependency_sensitive_ui(self):
        """Refresh UI elements that depend on newly installed packages."""
        try:
            self._check_device_info()
        except Exception:
            pass

        settings_panel = getattr(self, 'settings_panel', None)
        if settings_panel is None:
            return

        for method_name in ('_check_torch_availability', '_refresh_score_filter_status'):
            method = getattr(settings_panel, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

    def _run_pip_install(self, packages: list, title: str, reinstall: bool = False):
        """Run pip install in a separate process with real-time output"""
        # Show installation dialog
        dialog = InstallProgressDialog(self, title, packages, reinstall)
        dialog.exec_()

        # Check if installation was successful
        if dialog.success:
            self._refresh_dependency_sensitive_ui()

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
            self._refresh_dependency_sensitive_ui()

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
        manager_key = (
            processing_config.get('jsona_backup_every_entries', 200),
            processing_config.get('jsona_backup_keep', 10),
            processing_config.get('jsona_backup_every_seconds', 600),
        )
        if (
            self._standalone_jsona_backup_manager is None
            or self._standalone_jsona_backup_manager_key != manager_key
        ):
            self._standalone_jsona_backup_manager = JsonaBackupManager(
                backup_interval_entries=manager_key[0],
                rolling_keep=manager_key[1],
                backup_interval_seconds=manager_key[2],
            )
            self._standalone_jsona_backup_manager_key = manager_key
        return self._standalone_jsona_backup_manager

    def _write_jsona_entries(self, base_dir: str, target_name: str, entries: List[dict]) -> Dict[str, int]:
        if not entries:
            return {'added': 0, 'updated': 0}
        os.makedirs(base_dir, exist_ok=True)
        jsona_path = os.path.join(base_dir, f"{target_name}.jsona")
        backup_manager = self._create_jsona_backup_manager()
        if target_name in {'tag', 'nl', 'xml'}:
            return backup_manager.upsert_entries(jsona_path, entries)
        added = backup_manager.append_entries(jsona_path, entries)
        return {'added': added, 'updated': 0}

    def _write_text_jsona_entries(
        self,
        base_dir: str,
        source_image_path: str,
        tag_text: str,
        settings: Optional[dict] = None,
    ) -> Dict[str, Dict[str, int]]:
        prompt = (tag_text or '').strip()
        if not prompt or not source_image_path:
            return {}

        resolved_settings = settings or self.settings_panel.get_settings()
        xml_config = (resolved_settings.get('processing', {}) or {}).get('xml_mapping', {})
        results = {
            'tag': self._write_jsona_entries(
                base_dir,
                'tag',
                [self._create_jsona_entry(source_image_path, prompt, source_image_path, 'tag')],
            ),
            'nl': self._write_jsona_entries(
                base_dir,
                'nl',
                [self._create_jsona_entry(source_image_path, build_nl_prompt(prompt), source_image_path, 'nl')],
            ),
        }
        xml_fragment = build_xml_fragment(prompt, xml_config)
        if xml_fragment:
            results['xml'] = self._write_jsona_entries(
                base_dir,
                'xml',
                [self._create_jsona_entry(source_image_path, xml_fragment, source_image_path, 'xml')],
            )
        return results

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

    def _open_score_model_manager(self):
        settings_panel = getattr(self, 'settings_panel', None)
        if settings_panel is not None and hasattr(settings_panel, '_show_score_model_manager'):
            settings_panel._show_score_model_manager()

    def _open_score_model_directory(self):
        settings_panel = getattr(self, 'settings_panel', None)
        if settings_panel is not None and hasattr(settings_panel, '_open_score_filter_cache_root'):
            settings_panel._open_score_filter_cache_root()

    def _open_jsona_manager(self):
        """Open the JSONA manager dialog."""
        self._show_jsona_dialog()

    def _open_jsona_importer(self):
        """Open the JSONA dialog and start import flow."""
        self._show_jsona_dialog(initial_action='import')

    def _open_jsona_checker(self):
        """Open the JSONA dialog and run checker for the selected file."""
        self._show_jsona_dialog(initial_action='check')

    def _open_vlm_tag_dialog(self):
        """Open VLM dialog for tag verification and NL rewrite."""
        settings = self.settings_panel.get_settings()
        output_config = settings.get('output', {}) or {}
        base_dir = output_config.get('base_dir', './output')
        os.makedirs(base_dir, exist_ok=True)

        allow_mutation = not (self.processing_thread and self.processing_thread.isRunning())
        if not allow_mutation:
            QMessageBox.information(
                self,
                'VLM 推理',
                '当前任务正在运行。为了避免输出文件写入冲突，请先停止任务或等待任务结束后再运行 VLM 推理。'
            )
            return

        vlm_cfg = settings.get('vlm', {}) or {}
        initial = VlmConfig(
            backend=str(vlm_cfg.get('backend', 'openai') or 'openai'),
            base_url=str(vlm_cfg.get('base_url', 'http://127.0.0.1:1234') or 'http://127.0.0.1:1234'),
            api_key=str(vlm_cfg.get('api_key', '') or ''),
            model=str(vlm_cfg.get('model', '') or ''),
            timeout_seconds=int(vlm_cfg.get('timeout_seconds', 120) or 120),
        )
        dialog = VlmTagDialog(base_dir, self._create_jsona_backup_manager(), initial_cfg=initial, parent=self)
        dialog.exec_()

    def _import_review_inbox(self):
        """Import pending unattended-review items from output/review_inbox into the existing manual review UI."""
        try:
            settings = self.settings_panel.get_settings()
            output_config = settings.get('output', {}) or {}
            base_dir = output_config.get('base_dir', './output')

            processing_cfg = settings.get('processing', {}) or {}
            policy = ReviewInboxPolicy(
                max_mb=int(processing_cfg.get('unattended_inbox_max_mb', 2048)),
                on_full=str(processing_cfg.get('unattended_inbox_full_action', 'pause')).lower(),
            )

            inbox = ReviewInbox(base_dir, policy=policy)
            pending = inbox.iter_pending()
            if not pending:
                QMessageBox.information(self, '导入审核箱', '审核箱没有待处理条目。')
                return

            # Ensure required services exist for saving user decisions.
            if self.progress_manager is None:
                progress_config = settings.get('progress', {}) or {}
                progress_file = progress_config.get('progress_file', '.progress.json')
                self.progress_manager = ProgressManager(progress_file)

            self._review_inbox_manager = inbox

            # Reset current display and queue.
            self.image_list.clear()
            self.preview_panel.clear()
            try:
                while True:
                    self._inbox_review_queue.get_nowait()
            except Empty:
                pass

            from PIL import Image as PILImage

            loaded = 0
            skipped = 0
            for rec in pending:
                try:
                    stored = rec.get('stored', {}) or {}
                    orig_rel = stored.get('original')
                    if not orig_rel:
                        if rec.get('id'):
                            inbox.mark_done_by_id(rec.get('id'))
                        skipped += 1
                        continue

                    orig_path = os.path.join(inbox.root, orig_rel)
                    if not os.path.exists(orig_path):
                        if rec.get('id'):
                            inbox.mark_done_by_id(rec.get('id'))
                        skipped += 1
                        continue

                    control_type = rec.get('control_type', 'canny')
                    variants_rec = list(stored.get('variants', []) or [])
                    variants_rec.sort(key=lambda v: int(v.get('idx', 0)))
                    variants_rec = variants_rec[:4]

                    variants = []
                    for vrec in variants_rec:
                        v_rel = vrec.get('path')
                        if not v_rel:
                            continue
                        v_path = os.path.join(inbox.root, v_rel)
                        if not os.path.exists(v_path):
                            continue

                        with PILImage.open(v_path) as _im:
                            v_img = _im.convert("RGB").copy()

                        raw_score = float(vrec.get('score', 0))
                        preset_name = vrec.get('preset', 'unknown')

                        if control_type == 'canny':
                            preset_info = vrec.get('thresholds') or {}
                            score_10 = canny_to_10_scale(raw_score)
                        elif control_type == 'openpose':
                            preset_info = {'warning': vrec.get('warning')}
                            score_10 = openpose_to_10_scale(
                                float(vrec.get('visibility_ratio', 0) or 0),
                                raw_score >= 60,
                                vrec.get('warning'),
                                is_vitpose=bool(vrec.get('is_vitpose', False)),
                            )
                        elif control_type == 'depth':
                            preset_info = vrec.get('metrics') or {}
                            score_10 = depth_to_10_scale(
                                preset_info,
                                raw_score >= 60,
                                vrec.get('warning'),
                            )
                        elif control_type == 'bbox':
                            preset_info = vrec.get('metrics') or {}
                            if vrec.get('warning'):
                                preset_info = {**preset_info, 'warning': vrec.get('warning')}
                            if vrec.get('detections') is not None:
                                preset_info = {**preset_info, 'detections': vrec.get('detections')}
                            score_10 = bbox_to_10_scale(raw_score)
                        elif control_type == 'prefilter_score':
                            preset_info = vrec.get('metrics') or {}
                            aesthetic = preset_info.get('aesthetic')
                            score_10 = score_filter_to_10_scale(aesthetic if aesthetic is not None else (raw_score / 20.0))
                        else:
                            preset_info = {}
                            score_10 = 1.0

                        variants.append({
                            'image': v_img,
                            'path': v_path,
                            'preset': preset_info,
                            'preset_name': preset_name,
                            'score': raw_score,
                            'score_10': score_10,
                        })

                    if not variants:
                        skipped += 1
                        continue

                    best_i = max(range(len(variants)), key=lambda i: variants[i].get('score', 0))
                    for i in range(len(variants)):
                        variants[i]['is_best'] = (i == best_i)

                    with PILImage.open(orig_path) as _im:
                        orig_img = _im.convert("RGB").copy()

                    gui_data = {
                        'original_image': orig_img,
                        'original_path': rec.get('original_src') or orig_path,
                        'image_path': rec.get('original_src') or rec.get('original_path') or orig_path,
                        'basename': rec.get('basename', 'unknown'),
                        'tags': rec.get('tags', ''),
                        'prefilter': rec.get('prefilter', {}),
                        'control_type': control_type,
                        'progress_key': rec.get('progress_key'),
                        'variants': variants,
                        'best_score': float(rec.get('best_score', variants[best_i].get('score', 0))),
                        'control_10_score': float(variants[best_i].get('score_10', 1.0)),
                        'profile': rec.get('profile', 'general'),
                        'auto_action': None,
                        'review_record_id': rec.get('id', ''),
                    }

                    self._inbox_review_queue.put(gui_data)
                    loaded += 1
                except Exception:
                    skipped += 1
                    continue

            self._refill_display()
            self.status_bar.showMessage(f"已导入审核箱: {loaded} 条，跳过: {skipped} 条")
        except Exception as e:
            QMessageBox.warning(self, '导入审核箱', f"导入失败: {str(e)}")

    def _create_jsona_entry(self, image_path: str, hint_prompt: str, control_path: str, task_id: str) -> dict:
        """Create a JSONA entry with stable absolute paths (MainWindow helper)."""
        hint_image_path = os.path.abspath(image_path).replace('\\', '/')
        control_hints_path = os.path.abspath(control_path).replace('\\', '/')
        return {
            "hint_image_path": hint_image_path,
            "hint_prompt": hint_prompt or "",
            "control_hints_path": control_hints_path,
            "task_id": task_id,
        }

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
