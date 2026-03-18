"""
ControlNet GUI - Settings Panel
Data source selection, thread count, output path configuration
"""

import os
import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QSpinBox, QComboBox,
    QCheckBox, QFileDialog, QFormLayout, QListWidget, QTextEdit, QProgressBar,
    QDialog, QDialogButtonBox, QListWidgetItem, QMessageBox, QRadioButton, QApplication
)
from PyQt5.QtCore import pyqtSignal, QThread, pyqtSlot, Qt
from ..language import tr
from ..core.parquet_source import ParquetDataSource, StreamingDataSource
from ..core.jsona_backup import JsonaBackupManager
from ..core.tag_formats import build_xml_fragment


class SplitSelectionDialog(QDialog):
    """Dialog for selecting dataset splits"""

    def __init__(self, splits: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('select_splits'))
        self.setMinimumWidth(400)
        self.selected_splits = []

        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(tr('available_splits_info'))
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # List widget with checkboxes
        self.list_widget = QListWidget()
        for split in splits:
            item = QListWidgetItem(split)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # Check 'train' by default
            if split == 'train':
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_selected_splits(self) -> list:
        """Get list of selected splits"""
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected


class TextImportDialog(QDialog):
    """Simple text import dialog for pasted settings bundles."""

    def __init__(self, title: str, note: str, initial_text: str = '', parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(680, 420)

        layout = QVBoxLayout(self)

        label = QLabel(note)
        label.setWordWrap(True)
        layout.addWidget(label)

        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(initial_text)
        layout.addWidget(self.text_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_text(self) -> str:
        return self.text_edit.toPlainText()


class SettingsHistoryDialog(QDialog):
    """History viewer for settings changes."""

    def __init__(self, entries: list, field_labels: dict, parent=None):
        super().__init__(parent)
        self.entries = entries or []
        self.field_labels = field_labels or {}
        self._selected_entry = None

        self.setWindowTitle('参数修改历史')
        self.resize(860, 520)

        layout = QHBoxLayout(self)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(280)
        layout.addWidget(self.list_widget)

        right_layout = QVBoxLayout()
        layout.addLayout(right_layout, 1)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        right_layout.addWidget(self.detail_text, 1)

        button_row = QHBoxLayout()
        self.btn_copy_snapshot = QPushButton('复制该次历史快照')
        self.btn_copy_snapshot.clicked.connect(self._copy_snapshot)
        button_row.addWidget(self.btn_copy_snapshot)

        self.btn_restore = QPushButton('恢复选中记录')
        self.btn_restore.clicked.connect(self._restore_selected)
        button_row.addWidget(self.btn_restore)

        button_row.addStretch()

        btn_close = QPushButton('关闭')
        btn_close.clicked.connect(self.reject)
        button_row.addWidget(btn_close)
        right_layout.addLayout(button_row)

        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self._populate_entries()

    def _populate_entries(self):
        if not self.entries:
            self.list_widget.addItem('暂无历史记录')
            self.list_widget.setEnabled(False)
            self.btn_copy_snapshot.setEnabled(False)
            self.btn_restore.setEnabled(False)
            self.detail_text.setPlainText('还没有检测到设置变更历史。')
            return

        for entry in reversed(self.entries):
            timestamp = str(entry.get('timestamp', ''))
            changed_keys = entry.get('changed_keys', [])
            preview_names = [self.field_labels.get(key, key) for key in changed_keys[:3]]
            preview_text = '、'.join(preview_names) if preview_names else '未识别字段'
            if len(changed_keys) > 3:
                preview_text += f' 等 {len(changed_keys)} 项'
            reason = str(entry.get('reason_label', '') or '')
            label = f"{timestamp}  {reason}".strip()
            if preview_text:
                label = f"{label}\n{preview_text}"
            self.list_widget.addItem(label)

        self.list_widget.setCurrentRow(0)

    def _format_value(self, key: str, value):
        if key in {'hf_token', 'vlm_api_key'} and value:
            return '***'
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    def _on_row_changed(self, row: int):
        if row < 0 or not self.entries:
            self._selected_entry = None
            self.detail_text.clear()
            return

        entry = list(reversed(self.entries))[row]
        self._selected_entry = entry

        lines = [
            f"时间: {entry.get('timestamp', '')}",
            f"类型: {entry.get('reason_label', entry.get('reason', 'edit'))}",
        ]
        started_at = str(entry.get('started_at', '') or '').strip()
        if started_at:
            lines.append(f"开始时间: {started_at}")
        lines.append('')
        lines.append('变更字段:')

        changes = entry.get('changes', {})
        if not changes:
            lines.append('无具体差异，使用的是完整快照恢复或导入。')
        else:
            for key in entry.get('changed_keys', []):
                change = changes.get(key, {})
                label = self.field_labels.get(key, key)
                before_text = self._format_value(key, change.get('before'))
                after_text = self._format_value(key, change.get('after'))
                lines.append(f"[{label}]")
                lines.append(f"  之前: {before_text}")
                lines.append(f"  现在: {after_text}")
                lines.append('')

        self.detail_text.setPlainText('\n'.join(lines).strip())

    def _copy_snapshot(self):
        if not self._selected_entry:
            return
        snapshot = self._selected_entry.get('snapshot', {})
        QApplication.clipboard().setText(json.dumps(snapshot, ensure_ascii=False, indent=2))
        QMessageBox.information(self, '参数修改历史', '选中记录的历史快照已复制到剪贴板。\n\n敏感字段不会包含在内。')

    def _restore_selected(self):
        if not self._selected_entry:
            return
        self.accept()

    def get_selected_entry(self):
        return self._selected_entry


class ExtractionThread(QThread):
    """Background thread for data extraction"""
    progress_updated = pyqtSignal(int, int)  # current, total
    extraction_finished = pyqtSignal(int)  # count
    extraction_error = pyqtSignal(str)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source

    def run(self):
        try:
            count = self.source.extract(progress_callback=self._on_progress)
            self.extraction_finished.emit(count)
        except Exception as e:
            self.extraction_error.emit(str(e))

    def _on_progress(self, current: int, total: int):
        self.progress_updated.emit(current, total)


class SplitDetectionThread(QThread):
    """Background thread for detecting dataset splits"""
    splits_detected = pyqtSignal(list)  # List of splits
    detection_error = pyqtSignal(str)  # Error message

    def __init__(self, dataset_id, hf_token, parent=None):
        super().__init__(parent)
        self.dataset_id = dataset_id
        self.hf_token = hf_token

    def run(self):
        try:
            # Set environment variables
            os.environ['HF_DATASETS_DISABLE_TORCH'] = '1'
            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

            from datasets import get_dataset_split_names
            from ..core.data_source import parse_huggingface_url

            # Parse URL to get dataset ID
            parsed_id = parse_huggingface_url(self.dataset_id)

            # Get available splits
            splits = get_dataset_split_names(parsed_id, token=self.hf_token)

            if splits:
                self.splits_detected.emit(splits)
            else:
                self.detection_error.emit(tr('no_splits_found_message'))

        except (ImportError, OSError) as e:
            error_msg = str(e)
            if "torch" in error_msg.lower() or "dll" in error_msg.lower() or "c10.dll" in error_msg.lower():
                self.detection_error.emit("torch_dll_error")
            else:
                self.detection_error.emit(str(e))
        except Exception as e:
            self.detection_error.emit(str(e))


class StreamingExtractionThread(QThread):
    """Background thread for streaming data extraction"""
    progress_updated = pyqtSignal(int, int)  # current, total
    extraction_finished = pyqtSignal(int)  # count
    extraction_error = pyqtSignal(str)
    log_message = pyqtSignal(str)  # Log messages

    def __init__(self, dataset_id, split, extract_dir, num_samples, hf_token,
                 user_prefix, skip_count, num_threads, parent=None):
        super().__init__(parent)
        self.dataset_id = dataset_id
        self.split = split
        self.extract_dir = extract_dir
        self.num_samples = num_samples
        self.hf_token = hf_token
        self.user_prefix = user_prefix
        self.skip_count = skip_count
        self.num_threads = num_threads

    def run(self):
        try:
            # Ensure environment variable is set in this thread
            import os
            import sys
            os.environ['HF_DATASETS_DISABLE_TORCH'] = '1'
            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

            # Add torch DLL path to system PATH (fix for GUI environment)
            torch_lib_path = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'torch', 'lib')
            if os.path.exists(torch_lib_path):
                os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
                # Also try to add DLL directory for Python 3.8+
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(torch_lib_path)
                    except Exception:
                        pass

            self.log_message.emit(f"[INFO] Initializing extraction...")
            self.log_message.emit(f"[INFO] Dataset: {self.dataset_id}")
            self.log_message.emit(f"[INFO] Split: {self.split}")
            self.log_message.emit(f"[INFO] Skip: {self.skip_count} samples")
            self.log_message.emit(f"[INFO] Target samples: {self.num_samples}")
            if self.user_prefix:
                self.log_message.emit(f"[INFO] User prefix: {self.user_prefix}")
            self.log_message.emit("")

            source = StreamingDataSource(
                self.dataset_id,
                self.split,
                self.extract_dir,
                self.num_samples,
                self.hf_token,
                self.user_prefix,
                self.skip_count,
                self.num_threads
            )

            # Extract with log and progress callbacks
            count = source.extract(log_callback=self._log, progress_callback=self._progress)

            if count > 0:
                self.log_message.emit("")
                self.log_message.emit("=" * 50)
                self.log_message.emit(f"[SUCCESS] Extraction completed successfully!")
                self.log_message.emit(f"[SUCCESS] Total extracted: {count} samples")
                self.log_message.emit("=" * 50)
            self.extraction_finished.emit(count)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            self.log_message.emit("")
            self.log_message.emit("=" * 50)
            self.log_message.emit(f"[ERROR] Extraction failed!")
            self.log_message.emit(f"[ERROR] {str(e)}")
            self.log_message.emit("=" * 50)
            self.extraction_error.emit(f"{str(e)}\n{error_trace}")

    def _log(self, message: str):
        """Log callback for StreamingDataSource"""
        self.log_message.emit(message)

    def _progress(self, current: int, total: int):
        """Progress callback for StreamingDataSource"""
        self.progress_updated.emit(current, total)


class SettingsPanel(QWidget):
    """
    Settings Panel

    Signals:
        settings_changed: Settings changed
        start_processing: Start processing
    """

    settings_changed = pyqtSignal(dict)
    start_processing = pyqtSignal()
    extraction_started = pyqtSignal()
    extraction_finished = pyqtSignal(int)  # num_samples extracted

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.extraction_thread = None
        self.streaming_thread = None
        self._xml_custom_mapping_rows = []
        self._xml_field_options = ['artist', 'character_1', 'character_2']
        self._settings_tracking_suspended = True
        self._settings_history_limit = 500
        self._last_saved_flat_settings = None

        # Track if we've already checked dependencies/models this session
        self._vitpose_deps_checked = False
        self._vitpose_models_checked = {}  # {model_type: bool}

        self._setup_ui()
        self._load_config()

        # Load saved settings
        self.load_settings()

        # Connect signals to auto-save
        self._connect_autosave_signals()
        self._last_saved_flat_settings = deepcopy(self._collect_flat_settings())
        self._settings_tracking_suspended = False

        # Update JSONA statistics
        self._update_jsona_statistics()
        self._update_settings_history_status()

    def _setup_ui(self):
        """Setup UI layout"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Set panel style
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #ffffff;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #cccccc;
            }
            QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QLineEdit:focus {
                border: 1px solid #0078d4;
            }
            QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: #ffffff;
                selection-background-color: #0078d4;
            }
            QSpinBox {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #444;
                border-radius: 3px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            QPushButton {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
        """)

        # Data source settings
        data_source_group = self._create_data_source_group()
        main_layout.addWidget(data_source_group)

        # Retry and custom tags settings
        advanced_group = self._create_advanced_group()
        main_layout.addWidget(advanced_group)

        # Processing settings
        processing_group = self._create_processing_group()
        main_layout.addWidget(processing_group)

        # Output settings
        output_group = self._create_output_group()
        main_layout.addWidget(output_group)

        # Start button
        self.btn_start = QPushButton(tr('start_processing'))
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        self.btn_start.clicked.connect(self.start_processing.emit)
        main_layout.addWidget(self.btn_start)

        # Reset button
        self.btn_reset = QPushButton('重置为默认配置')
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #666666;
                color: white;
                font-size: 12px;
                padding: 8px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #777777;
            }
            QPushButton:pressed {
                background-color: #555555;
            }
        """)
        self.btn_reset.clicked.connect(self._reset_to_defaults)
        main_layout.addWidget(self.btn_reset)

        # Control buttons layout
        control_layout = QHBoxLayout()

        # Pause button
        self.btn_pause = QPushButton(tr('pause_processing'))
        self.btn_pause.setEnabled(False)
        self.btn_pause.setStyleSheet("""
            QPushButton {
                background-color: #ffa500;
                color: white;
                font-size: 12px;
                padding: 8px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #ff8c00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.btn_pause)

        # Stop button
        self.btn_stop = QPushButton(tr('stop_processing'))
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-size: 12px;
                padding: 8px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.btn_stop)

        main_layout.addLayout(control_layout)

        # Reset progress button
        self.btn_reset_progress = QPushButton("重置断点续传")
        self.btn_reset_progress.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                font-size: 11px;
                padding: 6px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        self.btn_reset_progress.clicked.connect(self._on_reset_progress)
        main_layout.addWidget(self.btn_reset_progress)

        main_layout.addStretch()

    def _create_data_source_group(self) -> QGroupBox:
        """Create data source settings group"""
        group = QGroupBox(tr('data_source'))
        layout = QVBoxLayout(group)

        # Data source type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel(tr('type') + ':'))
        self.combo_source_type = QComboBox()
        self.combo_source_type.addItems([tr('local_parquet'), tr('streaming_dataset')])
        self.combo_source_type.currentTextChanged.connect(self._on_source_type_changed)
        type_layout.addWidget(self.combo_source_type)
        type_layout.addStretch()
        layout.addLayout(type_layout)

        # Local parquet settings
        self.local_widget = QWidget()
        local_layout = QVBoxLayout(self.local_widget)

        # Parquet files list
        parquet_label = QLabel(tr('parquet_files') + ':')
        local_layout.addWidget(parquet_label)

        self.list_parquet_files = QListWidget()
        self.list_parquet_files.setMaximumHeight(100)
        local_layout.addWidget(self.list_parquet_files)

        # Add/Remove buttons
        btn_layout = QHBoxLayout()
        self.btn_add_parquet = QPushButton(tr('add_file'))
        self.btn_add_parquet.clicked.connect(self._add_parquet_file)
        btn_layout.addWidget(self.btn_add_parquet)

        self.btn_remove_parquet = QPushButton(tr('remove_file'))
        self.btn_remove_parquet.clicked.connect(self._remove_parquet_file)
        btn_layout.addWidget(self.btn_remove_parquet)
        btn_layout.addStretch()
        local_layout.addLayout(btn_layout)

        # Extract directory
        extract_layout = QHBoxLayout()
        extract_layout.addWidget(QLabel(tr('extract_dir') + ':'))
        self.edit_extract_dir = QLineEdit("./extracted")
        extract_layout.addWidget(self.edit_extract_dir)
        btn_browse_extract = QPushButton(tr('browse'))
        btn_browse_extract.clicked.connect(lambda: self._browse_directory(self.edit_extract_dir))
        extract_layout.addWidget(btn_browse_extract)
        local_layout.addLayout(extract_layout)

        # Num samples (0 = all)
        samples_layout = QHBoxLayout()
        samples_layout.addWidget(QLabel(tr('num_samples') + ':'))
        self.spin_local_samples = QSpinBox()
        self.spin_local_samples.setRange(0, 1000000)
        self.spin_local_samples.setValue(0)
        self.spin_local_samples.setSpecialValueText(tr('all'))
        samples_layout.addWidget(self.spin_local_samples)
        samples_layout.addStretch()
        local_layout.addLayout(samples_layout)

        # Extract button
        self.btn_extract = QPushButton(tr('extract_data'))
        self.btn_extract.clicked.connect(self._extract_parquet_data)
        local_layout.addWidget(self.btn_extract)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        local_layout.addWidget(self.progress_bar)

        layout.addWidget(self.local_widget)

        # Streaming dataset settings
        self.streaming_widget = QWidget()
        streaming_layout = QFormLayout(self.streaming_widget)

        self.edit_dataset_id = QLineEdit()
        self.edit_dataset_id.setPlaceholderText("username/dataset-name or https://huggingface.co/datasets/...")
        streaming_layout.addRow(tr('dataset_id') + ':', self.edit_dataset_id)

        self.edit_split = QLineEdit("")
        self.edit_split.setPlaceholderText(tr('split_placeholder'))
        streaming_layout.addRow(tr('split') + ':', self.edit_split)

        self.edit_hf_token = QLineEdit()
        self.edit_hf_token.setEchoMode(QLineEdit.Password)
        self.edit_hf_token.setPlaceholderText(tr('hf_token_placeholder'))
        streaming_layout.addRow(tr('hf_token') + ':', self.edit_hf_token)

        # User prefix for multi-user collaboration
        self.edit_user_prefix = QLineEdit()
        self.edit_user_prefix.setPlaceholderText(tr('user_prefix_placeholder'))
        streaming_layout.addRow(tr('user_prefix') + ':', self.edit_user_prefix)

        # Skip count (starting position)
        self.spin_skip_count = QSpinBox()
        self.spin_skip_count.setRange(0, 100000000)
        self.spin_skip_count.setValue(0)
        self.spin_skip_count.setSingleStep(10000)
        streaming_layout.addRow(tr('skip_count') + ':', self.spin_skip_count)

        self.spin_num_samples = QSpinBox()
        self.spin_num_samples.setRange(1, 1000000)
        self.spin_num_samples.setValue(10000)
        streaming_layout.addRow(tr('num_samples') + ':', self.spin_num_samples)

        # Multi-threading option
        multithread_layout = QHBoxLayout()
        self.check_multithread = QCheckBox(tr('enable_multithread'))
        self.check_multithread.setChecked(True)
        self.check_multithread.stateChanged.connect(self._on_multithread_changed)
        multithread_layout.addWidget(self.check_multithread)

        multithread_layout.addWidget(QLabel(tr('thread_count') + ':'))
        self.spin_thread_count = QSpinBox()
        self.spin_thread_count.setRange(1, 32)
        # Auto-detect CPU count and set default (leave 1 thread for system)
        import os
        cpu_count = os.cpu_count() or 4
        default_threads = max(1, cpu_count - 1)
        self.spin_thread_count.setValue(default_threads)
        multithread_layout.addWidget(self.spin_thread_count)

        # Auto button
        btn_auto_thread = QPushButton('自动')
        btn_auto_thread.setMaximumWidth(50)
        btn_auto_thread.clicked.connect(lambda: self.spin_thread_count.setValue(max(1, (os.cpu_count() or 4) - 1)))
        multithread_layout.addWidget(btn_auto_thread)

        multithread_layout.addStretch()
        streaming_layout.addRow('', multithread_layout)

        # Extract directory for streaming
        self.edit_streaming_extract_dir = QLineEdit("./extracted")
        btn_browse_streaming = QPushButton(tr('browse'))
        btn_browse_streaming.clicked.connect(lambda: self._browse_directory(self.edit_streaming_extract_dir))
        streaming_extract_layout = QHBoxLayout()
        streaming_extract_layout.addWidget(self.edit_streaming_extract_dir)
        streaming_extract_layout.addWidget(btn_browse_streaming)
        streaming_layout.addRow(tr('extract_dir') + ':', streaming_extract_layout)

        # Extract button for streaming
        self.btn_extract_streaming = QPushButton(tr('extract_data'))
        self.btn_extract_streaming.clicked.connect(self._extract_streaming_data)
        streaming_layout.addRow('', self.btn_extract_streaming)

        # Progress log for streaming
        self.streaming_log = QTextEdit()
        self.streaming_log.setReadOnly(True)
        self.streaming_log.setMaximumHeight(150)
        self.streaming_log.setVisible(False)
        self.streaming_log.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #444;
            }
        """)
        streaming_layout.addRow('', self.streaming_log)

        # Progress bar for extraction
        self.extraction_progress = QProgressBar()
        self.extraction_progress.setVisible(False)
        self.extraction_progress.setFormat('%v / %m images (%p%) - 0 it/s')
        self.extraction_progress.setMinimumWidth(240)  # Set minimum width
        # Set blue progress bar style
        self.extraction_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                background-color: #2d2d2d;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
        """)
        self.extraction_start_time = 0
        self.extraction_last_count = 0
        self.extraction_last_time = 0
        streaming_layout.addRow('', self.extraction_progress)

        layout.addWidget(self.streaming_widget)
        self.streaming_widget.hide()

        return group

    def _get_repo_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    def _get_settings_file_path(self) -> str:
        return os.path.join(self._get_repo_root(), 'settings.json')

    def _get_settings_history_file_path(self) -> str:
        return os.path.join(self._get_repo_root(), 'settings_history.jsonl')

    def _collect_flat_settings(self) -> dict:
        unattended_action = 'pause' if self.combo_unattended_inbox_full_action.currentIndex() == 0 else 'stop'
        local_parquet_files = [
            self.list_parquet_files.item(i).text()
            for i in range(self.list_parquet_files.count())
        ]
        return {
            'data_source_type': self.combo_source_type.currentIndex(),
            'local_parquet_files': local_parquet_files,
            'local_num_samples': self.spin_local_samples.value(),
            'dataset_id': self.edit_dataset_id.text(),
            'split': self.edit_split.text(),
            'hf_token': self.edit_hf_token.text(),
            'num_samples': self.spin_num_samples.value(),
            'user_prefix': self.edit_user_prefix.text(),
            'skip_count': self.spin_skip_count.value(),
            'enable_multithread': self.check_multithread.isChecked(),
            'thread_count': self.spin_thread_count.value(),
            'streaming_extract_dir': self.edit_streaming_extract_dir.text(),
            'local_extract_dir': self.edit_extract_dir.text(),
            'use_custom_dir': self.radio_use_custom_dir.isChecked(),
            'custom_input_dir': self.edit_custom_input_dir.text(),
            'processing_threads': self.spin_processing_threads.value(),
            'preload_count': self.spin_preload_count.value(),
            'canny_enabled': self.check_canny.isChecked(),
            'openpose_enabled': self.check_openpose.isChecked(),
            'depth_enabled': self.check_depth.isChecked(),
            'quality_profile': self.combo_quality_profile.currentText(),
            'canny_accept': self.spin_canny_accept.value(),
            'canny_reject': self.spin_canny_reject.value(),
            'pose_accept': self.spin_pose_accept.value(),
            'pose_reject': self.spin_pose_reject.value(),
            'depth_accept': self.spin_depth_accept.value(),
            'depth_reject': self.spin_depth_reject.value(),
            'openpose_model': self.combo_openpose_model.currentText(),
            'openpose_yolo_version': self.combo_yolo_version.currentText(),
            'openpose_yolo_model_type': self.combo_yolo_model_type.currentText(),
            'openpose_custom_path': self.edit_openpose_path.text(),
            'depth_model': self.combo_depth_model.currentText(),
            'depth_custom_path': self.edit_depth_path.text(),
            'parallel_threads': self.spin_parallel_threads.value(),
            'auto_pass_no_review': self.check_auto_pass_no_review.isChecked(),
            'single_jsona': self.check_single_jsona.isChecked(),
            'jsona_backup_every_entries': self.spin_jsona_backup_every_entries.value(),
            'jsona_backup_every_seconds': self.spin_jsona_backup_every_seconds.value(),
            'jsona_backup_keep': self.spin_jsona_backup_keep.value(),
            'unattended_mode': self.check_unattended_mode.isChecked(),
            'unattended_inbox_max_mb': self.spin_unattended_inbox_max_mb.value(),
            'unattended_inbox_full_action': unattended_action,
            'vlm_backend': self.combo_vlm_backend.currentText(),
            'vlm_base_url': self.edit_vlm_base_url.text(),
            'vlm_model': self.edit_vlm_model.text(),
            'vlm_api_key': self.edit_vlm_api_key.text(),
            'vlm_timeout_seconds': self.spin_vlm_timeout.value(),
            'xml_template_path': self.edit_xml_template_path.text(),
            'xml_artist_field_path': self.combo_xml_artist_field.currentText().strip(),
            'xml_artist_tag_index': self.spin_xml_artist_tag_index.value(),
            'xml_character1_field_path': self.combo_xml_character1_field.currentText().strip(),
            'xml_character1_tag_index': self.spin_xml_character1_tag_index.value(),
            'xml_character2_field_path': self.combo_xml_character2_field.currentText().strip(),
            'xml_character2_tag_index': self.spin_xml_character2_tag_index.value(),
            'xml_custom_mappings': self._serialize_xml_custom_mappings(),
            'enable_retry': self.check_enable_retry.isChecked(),
            'max_retries': self.spin_max_retries.value(),
            'append_tags': self.check_append_tags.isChecked(),
            'custom_tags': self.edit_custom_tags.toPlainText(),
            'output_dir': self.edit_output_dir.text(),
            'discard_action': self.combo_discard_action.currentText(),
        }

    def _get_settings_field_labels(self) -> dict:
        return {
            'data_source_type': '数据源类型',
            'local_parquet_files': 'Parquet 文件列表',
            'local_num_samples': '本地样本数',
            'dataset_id': '数据集 ID',
            'split': '数据集 Split',
            'hf_token': 'HF Token',
            'num_samples': '拉取样本数',
            'user_prefix': '用户前缀',
            'skip_count': '跳过数量',
            'enable_multithread': '启用多线程提取',
            'thread_count': '提取线程数',
            'streaming_extract_dir': '流式提取目录',
            'local_extract_dir': '本地提取目录',
            'use_custom_dir': '使用自定义输入目录',
            'custom_input_dir': '自定义输入目录',
            'processing_threads': '处理线程数',
            'preload_count': '预加载数量',
            'canny_enabled': '启用 Canny',
            'openpose_enabled': '启用 Pose',
            'depth_enabled': '启用 Depth',
            'quality_profile': '质量预设',
            'canny_accept': 'Canny 自动接受',
            'canny_reject': 'Canny 自动拒绝',
            'pose_accept': 'Pose 自动接受',
            'pose_reject': 'Pose 自动拒绝',
            'depth_accept': 'Depth 自动接受',
            'depth_reject': 'Depth 自动拒绝',
            'openpose_model': 'Pose 模型',
            'openpose_yolo_version': 'YOLO 版本',
            'openpose_yolo_model_type': 'Pose 检测器',
            'openpose_custom_path': 'Pose 自定义路径',
            'depth_model': 'Depth 模型',
            'depth_custom_path': 'Depth 自定义路径',
            'parallel_threads': '并行处理线程',
            'auto_pass_no_review': '自动通过无需审核',
            'single_jsona': '单个 JSONA',
            'jsona_backup_every_entries': 'JSONA 条数备份',
            'jsona_backup_every_seconds': 'JSONA 时间备份',
            'jsona_backup_keep': 'JSONA 保留份数',
            'unattended_mode': '无人值守模式',
            'unattended_inbox_max_mb': '审核箱最大大小',
            'unattended_inbox_full_action': '审核箱满额动作',
            'vlm_backend': 'VLM 后端',
            'vlm_base_url': 'VLM Base URL',
            'vlm_model': 'VLM 模型',
            'vlm_api_key': 'VLM API Key',
            'vlm_timeout_seconds': 'VLM 超时',
            'xml_template_path': 'XML 模板路径',
            'xml_artist_field_path': 'XML artist 字段',
            'xml_artist_tag_index': 'XML artist tag 序号',
            'xml_character1_field_path': 'XML character_1 字段',
            'xml_character1_tag_index': 'XML character_1 tag 序号',
            'xml_character2_field_path': 'XML character_2 字段',
            'xml_character2_tag_index': 'XML character_2 tag 序号',
            'xml_custom_mappings': 'XML 自定义映射',
            'enable_retry': '启用重试',
            'max_retries': '最大重试次数',
            'append_tags': '追加自定义标签',
            'custom_tags': '自定义标签',
            'output_dir': '输出目录',
            'discard_action': '丢弃动作',
        }

    def _get_history_reason_label(self, reason: str) -> str:
        mapping = {
            'edit': '手动修改',
            'import_key_params': '导入关键参数',
            'restore_history': '从历史恢复',
            'reset_defaults': '恢复默认设置',
        }
        return mapping.get(reason, reason or '手动修改')

    def _get_sensitive_history_keys(self) -> set:
        return {'hf_token', 'vlm_api_key'}

    def _sanitize_history_value(self, key: str, value):
        if key in self._get_sensitive_history_keys() and value:
            return '***'
        return deepcopy(value)

    def _build_history_snapshot(self, current: dict) -> dict:
        snapshot = {}
        for key, value in (current or {}).items():
            if key in self._get_sensitive_history_keys():
                continue
            snapshot[key] = deepcopy(value)
        return snapshot

    def _sanitize_history_entry(self, entry: dict) -> dict:
        if not isinstance(entry, dict):
            return {}

        sanitized = deepcopy(entry)
        changes = sanitized.get('changes', {})
        if isinstance(changes, dict):
            for key, change in list(changes.items()):
                if not isinstance(change, dict):
                    continue
                if key in self._get_sensitive_history_keys():
                    change['before'] = self._sanitize_history_value(key, change.get('before'))
                    change['after'] = self._sanitize_history_value(key, change.get('after'))

        snapshot = sanitized.get('snapshot', {})
        if isinstance(snapshot, dict):
            for key in self._get_sensitive_history_keys():
                snapshot.pop(key, None)
        return sanitized

    def _read_settings_history_entries(self) -> list:
        history_file = self._get_settings_history_file_path()
        if not os.path.exists(history_file):
            return []

        entries = []
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        entries.append(json.loads(raw))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Failed to read settings history: {e}")
        return entries

    def _update_settings_history_status(self):
        if not hasattr(self, 'label_settings_history_status'):
            return
        count = len(self._read_settings_history_entries())
        self.label_settings_history_status.setText(f'历史记录: {count} 条')

    def _append_settings_history_entry(self, entry: dict):
        if not isinstance(entry, dict):
            return

        entries = [self._sanitize_history_entry(item) for item in self._read_settings_history_entries()]
        entry_time = None
        try:
            entry_time = datetime.fromisoformat(str(entry.get('timestamp', '')))
        except Exception:
            pass

        if entries and entry.get('reason') == 'edit':
            last = entries[-1]
            last_time = None
            try:
                last_time = datetime.fromisoformat(str(last.get('timestamp', '')))
            except Exception:
                pass

            if last.get('reason') == 'edit' and last_time and entry_time and abs((entry_time - last_time).total_seconds()) <= 10:
                merged = deepcopy(last)
                merged['started_at'] = str(last.get('started_at') or last.get('timestamp') or '')
                merged['timestamp'] = entry.get('timestamp', merged.get('timestamp'))
                merged['snapshot'] = deepcopy(entry.get('snapshot', merged.get('snapshot', {})))

                merged_changes = deepcopy(last.get('changes', {}))
                for key, change in entry.get('changes', {}).items():
                    if key in merged_changes:
                        merged_changes[key]['after'] = change.get('after')
                    else:
                        merged_changes[key] = change
                merged['changes'] = merged_changes
                merged['changed_keys'] = sorted(merged_changes.keys())
                merged['reason_label'] = self._get_history_reason_label('edit')
                entries[-1] = self._sanitize_history_entry(merged)
            else:
                entries.append(self._sanitize_history_entry(entry))
        else:
            entries.append(self._sanitize_history_entry(entry))

        entries = entries[-self._settings_history_limit:]
        history_file = self._get_settings_history_file_path()
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                for item in entries:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Failed to write settings history: {e}")
            return

        self._update_settings_history_status()

    def _record_settings_history(self, previous: dict, current: dict, reason: str = 'edit', force_history: bool = False):
        previous = previous or {}
        current = current or {}

        changed = {}
        for key in sorted(set(previous.keys()) | set(current.keys())):
            before = previous.get(key)
            after = current.get(key)
            if before != after:
                changed[key] = {
                    'before': self._sanitize_history_value(key, before),
                    'after': self._sanitize_history_value(key, after),
                }

        if not changed and not force_history:
            return

        entry = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'reason': reason,
            'reason_label': self._get_history_reason_label(reason),
            'changed_keys': sorted(changed.keys()),
            'changes': changed,
            'snapshot': self._build_history_snapshot(current),
        }
        self._append_settings_history_entry(entry)

    def save_settings(self, history_reason: str = 'edit', force_history: bool = False):
        """Save current settings to config file"""
        if self._settings_tracking_suspended:
            return

        config = self._collect_flat_settings()
        previous = deepcopy(self._last_saved_flat_settings or {})
        config_file = self._get_settings_file_path()
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save settings: {e}")
            return

        self._record_settings_history(previous, config, reason=history_reason, force_history=force_history)
        self._last_saved_flat_settings = deepcopy(config)
        self.settings_changed.emit(self.get_settings())

    def load_settings(self):
        """Load settings from config file"""
        config_file = self._get_settings_file_path()

        if not os.path.exists(config_file):
            return

        previous_state = self._settings_tracking_suspended
        self._settings_tracking_suspended = True
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Restore settings
            if 'data_source_type' in config:
                self.combo_source_type.setCurrentIndex(config['data_source_type'])
            if 'local_parquet_files' in config:
                self.list_parquet_files.clear()
                for file_path in config.get('local_parquet_files') or []:
                    self.list_parquet_files.addItem(str(file_path))
            if 'local_num_samples' in config:
                self.spin_local_samples.setValue(int(config['local_num_samples']))
            if 'dataset_id' in config:
                self.edit_dataset_id.setText(config['dataset_id'])
            if 'split' in config:
                self.edit_split.setText(config['split'])
            if 'hf_token' in config:
                self.edit_hf_token.setText(config['hf_token'])
            if 'num_samples' in config:
                self.spin_num_samples.setValue(config['num_samples'])
            if 'user_prefix' in config:
                self.edit_user_prefix.setText(config['user_prefix'])
            if 'skip_count' in config:
                self.spin_skip_count.setValue(config['skip_count'])
            if 'enable_multithread' in config:
                self.check_multithread.setChecked(config['enable_multithread'])
            else:
                # Default to True if not in config
                self.check_multithread.setChecked(True)
            if 'thread_count' in config:
                # Use saved value, but ensure it's at least 2 if multi-threading is enabled
                saved_threads = config['thread_count']
                if saved_threads < 2 and config.get('enable_multithread', False):
                    # If saved value is too low, use auto-detected value
                    cpu_count = os.cpu_count() or 4
                    saved_threads = max(2, cpu_count - 1)
                self.spin_thread_count.setValue(saved_threads)
            if 'streaming_extract_dir' in config:
                self.edit_streaming_extract_dir.setText(config['streaming_extract_dir'])
            if 'local_extract_dir' in config:
                self.edit_extract_dir.setText(config['local_extract_dir'])

            # Processing settings
            if 'use_custom_dir' in config:
                self.radio_use_custom_dir.setChecked(config['use_custom_dir'])
                self.radio_use_data_source.setChecked(not config['use_custom_dir'])
            if 'custom_input_dir' in config:
                self.edit_custom_input_dir.setText(config['custom_input_dir'])
            if 'processing_threads' in config:
                self.spin_processing_threads.setValue(config['processing_threads'])
            if 'preload_count' in config:
                self.spin_preload_count.setValue(config['preload_count'])

            # Control types
            if 'canny_enabled' in config:
                self.check_canny.setChecked(config['canny_enabled'])
            if 'openpose_enabled' in config:
                self.check_openpose.setChecked(config['openpose_enabled'])
            if 'depth_enabled' in config:
                self.check_depth.setChecked(config['depth_enabled'])

            # Model settings
            if 'openpose_model' in config:
                self.combo_openpose_model.setCurrentText(config['openpose_model'])
            if 'openpose_yolo_version' in config:
                self.combo_yolo_version.setCurrentText(config['openpose_yolo_version'])
            if 'openpose_yolo_model_type' in config:
                self.combo_yolo_model_type.setCurrentText(config['openpose_yolo_model_type'])
            if 'openpose_custom_path' in config:
                self.edit_openpose_path.setText(config['openpose_custom_path'])
            if 'depth_model' in config:
                self.combo_depth_model.setCurrentText(config['depth_model'])
            if 'depth_custom_path' in config:
                self.edit_depth_path.setText(config['depth_custom_path'])

            # Quality profile and thresholds
            if 'quality_profile' in config:
                self.combo_quality_profile.setCurrentText(config['quality_profile'])
            if 'canny_accept' in config:
                self.spin_canny_accept.setValue(config['canny_accept'])
            if 'canny_reject' in config:
                self.spin_canny_reject.setValue(config['canny_reject'])
            if 'pose_accept' in config:
                self.spin_pose_accept.setValue(config['pose_accept'])
            if 'pose_reject' in config:
                self.spin_pose_reject.setValue(config['pose_reject'])
            if 'depth_accept' in config:
                self.spin_depth_accept.setValue(config['depth_accept'])
            if 'depth_reject' in config:
                self.spin_depth_reject.setValue(config['depth_reject'])

            # Advanced settings
            if 'parallel_threads' in config:
                self.spin_parallel_threads.setValue(config['parallel_threads'])
            if 'auto_pass_no_review' in config:
                self.check_auto_pass_no_review.setChecked(config['auto_pass_no_review'])
            if 'single_jsona' in config:
                self.check_single_jsona.setChecked(config['single_jsona'])
            if 'jsona_backup_every_entries' in config:
                self.spin_jsona_backup_every_entries.setValue(config['jsona_backup_every_entries'])
            if 'jsona_backup_every_seconds' in config:
                self.spin_jsona_backup_every_seconds.setValue(config['jsona_backup_every_seconds'])
            if 'jsona_backup_keep' in config:
                self.spin_jsona_backup_keep.setValue(config['jsona_backup_keep'])
            if 'unattended_mode' in config:
                self.check_unattended_mode.setChecked(bool(config['unattended_mode']))
            if 'unattended_inbox_max_mb' in config:
                try:
                    self.spin_unattended_inbox_max_mb.setValue(int(config['unattended_inbox_max_mb']))
                except Exception:
                    pass
            if 'unattended_inbox_full_action' in config:
                act = str(config['unattended_inbox_full_action']).lower()
                self.combo_unattended_inbox_full_action.setCurrentIndex(0 if act == 'pause' else 1)
            if 'vlm_backend' in config:
                self.combo_vlm_backend.setCurrentText(str(config['vlm_backend']))
            if 'vlm_base_url' in config:
                self.edit_vlm_base_url.setText(str(config['vlm_base_url']))
            if 'vlm_model' in config:
                self.edit_vlm_model.setText(str(config['vlm_model']))
            if 'vlm_api_key' in config:
                self.edit_vlm_api_key.setText(str(config['vlm_api_key']))
            if 'vlm_timeout_seconds' in config:
                try:
                    self.spin_vlm_timeout.setValue(int(config['vlm_timeout_seconds']))
                except Exception:
                    pass
            if 'xml_template_path' in config:
                self.edit_xml_template_path.setText(str(config['xml_template_path']))
                try:
                    self._analyze_xml_template()
                except Exception:
                    pass
            if 'xml_artist_field_path' in config:
                self.combo_xml_artist_field.setCurrentText(str(config['xml_artist_field_path']))
            if 'xml_artist_tag_index' in config:
                self.spin_xml_artist_tag_index.setValue(int(config['xml_artist_tag_index']))
            if 'xml_character1_field_path' in config:
                self.combo_xml_character1_field.setCurrentText(str(config['xml_character1_field_path']))
            if 'xml_character1_tag_index' in config:
                self.spin_xml_character1_tag_index.setValue(int(config['xml_character1_tag_index']))
            if 'xml_character2_field_path' in config:
                self.combo_xml_character2_field.setCurrentText(str(config['xml_character2_field_path']))
            if 'xml_character2_tag_index' in config:
                self.spin_xml_character2_tag_index.setValue(int(config['xml_character2_tag_index']))
            if 'xml_custom_mappings' in config:
                self._load_xml_custom_mappings(config['xml_custom_mappings'])
            if 'enable_retry' in config:
                self.check_enable_retry.setChecked(config['enable_retry'])
            if 'max_retries' in config:
                self.spin_max_retries.setValue(config['max_retries'])

            # Custom tags
            if 'append_tags' in config:
                self.check_append_tags.setChecked(config['append_tags'])
            if 'custom_tags' in config:
                self.edit_custom_tags.setPlainText(config['custom_tags'])

            # Output
            if 'output_dir' in config:
                self.edit_output_dir.setText(config['output_dir'])
            if 'discard_action' in config:
                self.combo_discard_action.setCurrentText(config['discard_action'])

        except Exception as e:
            print(f"Failed to load settings: {e}")
        finally:
            self._settings_tracking_suspended = previous_state

    def _apply_flat_settings(self, config: dict):
        if not isinstance(config, dict):
            return

        previous_state = self._settings_tracking_suspended
        self._settings_tracking_suspended = True
        try:
            if 'data_source_type' in config:
                self.combo_source_type.setCurrentIndex(int(config['data_source_type']))
            if 'local_parquet_files' in config:
                self.list_parquet_files.clear()
                for file_path in config.get('local_parquet_files') or []:
                    self.list_parquet_files.addItem(str(file_path))
            if 'local_num_samples' in config:
                self.spin_local_samples.setValue(int(config['local_num_samples']))
            if 'dataset_id' in config:
                self.edit_dataset_id.setText(str(config['dataset_id']))
            if 'split' in config:
                self.edit_split.setText(str(config['split']))
            if 'hf_token' in config:
                self.edit_hf_token.setText(str(config['hf_token']))
            if 'num_samples' in config:
                self.spin_num_samples.setValue(int(config['num_samples']))
            if 'user_prefix' in config:
                self.edit_user_prefix.setText(str(config['user_prefix']))
            if 'skip_count' in config:
                self.spin_skip_count.setValue(int(config['skip_count']))
            if 'enable_multithread' in config:
                self.check_multithread.setChecked(bool(config['enable_multithread']))
            if 'thread_count' in config:
                saved_threads = int(config['thread_count'])
                if saved_threads < 2 and bool(config.get('enable_multithread', self.check_multithread.isChecked())):
                    cpu_count = os.cpu_count() or 4
                    saved_threads = max(2, cpu_count - 1)
                self.spin_thread_count.setValue(saved_threads)
            if 'streaming_extract_dir' in config:
                self.edit_streaming_extract_dir.setText(str(config['streaming_extract_dir']))
            if 'local_extract_dir' in config:
                self.edit_extract_dir.setText(str(config['local_extract_dir']))
            if 'use_custom_dir' in config:
                use_custom_dir = bool(config['use_custom_dir'])
                self.radio_use_custom_dir.setChecked(use_custom_dir)
                self.radio_use_data_source.setChecked(not use_custom_dir)
            if 'custom_input_dir' in config:
                self.edit_custom_input_dir.setText(str(config['custom_input_dir']))
            if 'processing_threads' in config:
                self.spin_processing_threads.setValue(int(config['processing_threads']))
            if 'preload_count' in config:
                self.spin_preload_count.setValue(int(config['preload_count']))
            if 'canny_enabled' in config:
                self.check_canny.setChecked(bool(config['canny_enabled']))
            if 'openpose_enabled' in config:
                self.check_openpose.setChecked(bool(config['openpose_enabled']))
            if 'depth_enabled' in config:
                self.check_depth.setChecked(bool(config['depth_enabled']))
            if 'openpose_model' in config:
                self.combo_openpose_model.setCurrentText(str(config['openpose_model']))
            if 'openpose_yolo_version' in config:
                self.combo_yolo_version.setCurrentText(str(config['openpose_yolo_version']))
            if 'openpose_yolo_model_type' in config:
                self.combo_yolo_model_type.setCurrentText(str(config['openpose_yolo_model_type']))
            if 'openpose_custom_path' in config:
                self.edit_openpose_path.setText(str(config['openpose_custom_path']))
            if 'depth_model' in config:
                self.combo_depth_model.setCurrentText(str(config['depth_model']))
            if 'depth_custom_path' in config:
                self.edit_depth_path.setText(str(config['depth_custom_path']))
            if 'quality_profile' in config:
                self.combo_quality_profile.setCurrentText(str(config['quality_profile']))
            if 'canny_accept' in config:
                self.spin_canny_accept.setValue(int(config['canny_accept']))
            if 'canny_reject' in config:
                self.spin_canny_reject.setValue(int(config['canny_reject']))
            if 'pose_accept' in config:
                self.spin_pose_accept.setValue(int(config['pose_accept']))
            if 'pose_reject' in config:
                self.spin_pose_reject.setValue(int(config['pose_reject']))
            if 'depth_accept' in config:
                self.spin_depth_accept.setValue(int(config['depth_accept']))
            if 'depth_reject' in config:
                self.spin_depth_reject.setValue(int(config['depth_reject']))
            if 'parallel_threads' in config:
                self.spin_parallel_threads.setValue(int(config['parallel_threads']))
            if 'auto_pass_no_review' in config:
                self.check_auto_pass_no_review.setChecked(bool(config['auto_pass_no_review']))
            if 'single_jsona' in config:
                self.check_single_jsona.setChecked(bool(config['single_jsona']))
            if 'jsona_backup_every_entries' in config:
                self.spin_jsona_backup_every_entries.setValue(int(config['jsona_backup_every_entries']))
            if 'jsona_backup_every_seconds' in config:
                self.spin_jsona_backup_every_seconds.setValue(int(config['jsona_backup_every_seconds']))
            if 'jsona_backup_keep' in config:
                self.spin_jsona_backup_keep.setValue(int(config['jsona_backup_keep']))
            if 'unattended_mode' in config:
                self.check_unattended_mode.setChecked(bool(config['unattended_mode']))
            if 'unattended_inbox_max_mb' in config:
                self.spin_unattended_inbox_max_mb.setValue(int(config['unattended_inbox_max_mb']))
            if 'unattended_inbox_full_action' in config:
                action = str(config['unattended_inbox_full_action']).lower()
                self.combo_unattended_inbox_full_action.setCurrentIndex(0 if action == 'pause' else 1)
            if 'vlm_backend' in config:
                self.combo_vlm_backend.setCurrentText(str(config['vlm_backend']))
            if 'vlm_base_url' in config:
                self.edit_vlm_base_url.setText(str(config['vlm_base_url']))
            if 'vlm_model' in config:
                self.edit_vlm_model.setText(str(config['vlm_model']))
            if 'vlm_api_key' in config:
                self.edit_vlm_api_key.setText(str(config['vlm_api_key']))
            if 'vlm_timeout_seconds' in config:
                self.spin_vlm_timeout.setValue(int(config['vlm_timeout_seconds']))
            if 'xml_template_path' in config:
                self.edit_xml_template_path.setText(str(config['xml_template_path']))
                if self.edit_xml_template_path.text().strip():
                    try:
                        self._analyze_xml_template()
                    except Exception:
                        pass
                else:
                    self._refresh_xml_field_options([])
                    self.label_xml_template_status.setText('未加载 XML 结构，当前使用默认字段。')
            if 'xml_artist_field_path' in config:
                self.combo_xml_artist_field.setCurrentText(str(config['xml_artist_field_path']))
            if 'xml_artist_tag_index' in config:
                self.spin_xml_artist_tag_index.setValue(int(config['xml_artist_tag_index']))
            if 'xml_character1_field_path' in config:
                self.combo_xml_character1_field.setCurrentText(str(config['xml_character1_field_path']))
            if 'xml_character1_tag_index' in config:
                self.spin_xml_character1_tag_index.setValue(int(config['xml_character1_tag_index']))
            if 'xml_character2_field_path' in config:
                self.combo_xml_character2_field.setCurrentText(str(config['xml_character2_field_path']))
            if 'xml_character2_tag_index' in config:
                self.spin_xml_character2_tag_index.setValue(int(config['xml_character2_tag_index']))
            if 'xml_custom_mappings' in config:
                self._load_xml_custom_mappings(config['xml_custom_mappings'])
            if 'enable_retry' in config:
                self.check_enable_retry.setChecked(bool(config['enable_retry']))
            if 'max_retries' in config:
                self.spin_max_retries.setValue(int(config['max_retries']))
            if 'append_tags' in config:
                self.check_append_tags.setChecked(bool(config['append_tags']))
            if 'custom_tags' in config:
                self.edit_custom_tags.setPlainText(str(config['custom_tags']))
            if 'output_dir' in config:
                self.edit_output_dir.setText(str(config['output_dir']))
            if 'discard_action' in config:
                self.combo_discard_action.setCurrentText(str(config['discard_action']))
        finally:
            self._settings_tracking_suspended = previous_state

        self._refresh_xml_preview()
        self._update_jsona_statistics()

    def _connect_autosave_signals(self):
        """Connect signals to auto-save settings when changed"""
        # Data source
        self.combo_source_type.currentIndexChanged.connect(lambda: self.save_settings())
        self.edit_dataset_id.textChanged.connect(lambda: self.save_settings())
        self.edit_split.textChanged.connect(lambda: self.save_settings())
        self.edit_hf_token.textChanged.connect(lambda: self.save_settings())
        self.spin_num_samples.valueChanged.connect(lambda: self.save_settings())
        self.spin_local_samples.valueChanged.connect(lambda: self.save_settings())
        self.edit_user_prefix.textChanged.connect(lambda: self.save_settings())
        self.spin_skip_count.valueChanged.connect(lambda: self.save_settings())
        self.check_multithread.stateChanged.connect(lambda: self.save_settings())
        self.spin_thread_count.valueChanged.connect(lambda: self.save_settings())
        self.edit_streaming_extract_dir.textChanged.connect(lambda: self.save_settings())
        self.edit_extract_dir.textChanged.connect(lambda: self.save_settings())
        self.edit_output_dir.textChanged.connect(lambda: self._update_jsona_statistics())

        # Processing and output settings
        self.radio_use_custom_dir.toggled.connect(lambda: self.save_settings())
        self.edit_custom_input_dir.textChanged.connect(lambda: self.save_settings())
        self.spin_processing_threads.valueChanged.connect(lambda: self.save_settings())
        self.spin_preload_count.valueChanged.connect(lambda: self.save_settings())
        self.check_canny.stateChanged.connect(lambda: self.save_settings())
        self.check_openpose.stateChanged.connect(lambda: self.save_settings())
        self.check_depth.stateChanged.connect(lambda: self.save_settings())
        self.combo_openpose_model.currentTextChanged.connect(lambda: self.save_settings())
        self.combo_yolo_version.currentTextChanged.connect(lambda: self.save_settings())
        self.combo_yolo_model_type.currentTextChanged.connect(lambda: self.save_settings())
        self.edit_openpose_path.textChanged.connect(lambda: self.save_settings())
        self.combo_depth_model.currentTextChanged.connect(lambda: self.save_settings())
        self.edit_depth_path.textChanged.connect(lambda: self.save_settings())
        self.check_append_tags.stateChanged.connect(lambda: self.save_settings())
        self.edit_custom_tags.textChanged.connect(lambda: self.save_settings())
        self.edit_output_dir.textChanged.connect(lambda: self.save_settings())
        self.combo_discard_action.currentIndexChanged.connect(lambda: self.save_settings())

        # Advanced settings
        self.spin_parallel_threads.valueChanged.connect(lambda: self.save_settings())
        self.check_auto_pass_no_review.stateChanged.connect(lambda: self.save_settings())
        self.check_single_jsona.stateChanged.connect(lambda: self.save_settings())
        self.check_single_jsona.stateChanged.connect(lambda: self._update_jsona_statistics())
        self.spin_jsona_backup_every_entries.valueChanged.connect(lambda: self.save_settings())
        self.spin_jsona_backup_every_seconds.valueChanged.connect(lambda: self.save_settings())
        self.spin_jsona_backup_keep.valueChanged.connect(lambda: self.save_settings())
        self.check_unattended_mode.stateChanged.connect(lambda: self.save_settings())
        self.spin_unattended_inbox_max_mb.valueChanged.connect(lambda: self.save_settings())
        self.combo_unattended_inbox_full_action.currentIndexChanged.connect(lambda: self.save_settings())
        self.combo_vlm_backend.currentIndexChanged.connect(lambda: self.save_settings())
        self.edit_vlm_base_url.textChanged.connect(lambda: self.save_settings())
        self.edit_vlm_model.textChanged.connect(lambda: self.save_settings())
        self.edit_vlm_api_key.textChanged.connect(lambda: self.save_settings())
        self.spin_vlm_timeout.valueChanged.connect(lambda: self.save_settings())
        self.edit_xml_template_path.textChanged.connect(lambda: self.save_settings())
        self.edit_xml_template_path.textChanged.connect(lambda: self._refresh_xml_preview())
        self.combo_xml_artist_field.currentTextChanged.connect(lambda: self.save_settings())
        self.combo_xml_artist_field.currentTextChanged.connect(lambda: self._refresh_xml_preview())
        self.spin_xml_artist_tag_index.valueChanged.connect(lambda: self.save_settings())
        self.spin_xml_artist_tag_index.valueChanged.connect(lambda: self._refresh_xml_preview())
        self.combo_xml_character1_field.currentTextChanged.connect(lambda: self.save_settings())
        self.combo_xml_character1_field.currentTextChanged.connect(lambda: self._refresh_xml_preview())
        self.spin_xml_character1_tag_index.valueChanged.connect(lambda: self.save_settings())
        self.spin_xml_character1_tag_index.valueChanged.connect(lambda: self._refresh_xml_preview())
        self.combo_xml_character2_field.currentTextChanged.connect(lambda: self.save_settings())
        self.combo_xml_character2_field.currentTextChanged.connect(lambda: self._refresh_xml_preview())
        self.spin_xml_character2_tag_index.valueChanged.connect(lambda: self.save_settings())
        self.spin_xml_character2_tag_index.valueChanged.connect(lambda: self._refresh_xml_preview())
        self.edit_xml_preview_tags.textChanged.connect(self._refresh_xml_preview)

        # Threshold spinboxes
        self.spin_canny_accept.valueChanged.connect(lambda: self.save_settings())
        self.spin_canny_reject.valueChanged.connect(lambda: self.save_settings())
        self.spin_pose_accept.valueChanged.connect(lambda: self.save_settings())
        self.spin_pose_reject.valueChanged.connect(lambda: self.save_settings())
        self.spin_depth_accept.valueChanged.connect(lambda: self.save_settings())
        self.spin_depth_reject.valueChanged.connect(lambda: self.save_settings())
        self.combo_quality_profile.currentIndexChanged.connect(self._on_profile_changed)

    def _on_profile_changed(self):
        """Load thresholds when profile changes"""
        scoring = self.config.get('scoring', {})
        profiles = scoring.get('profiles', {})

        # Map translated text back to English key
        profile_text = self.combo_quality_profile.currentText()
        if profile_text == tr('general') or '通用' in profile_text:
            profile_key = 'general'
        elif profile_text == tr('anime') or '动漫' in profile_text:
            profile_key = 'anime'
        else:
            profile_key = 'general'

        profile = profiles.get(profile_key, {})
        default_accept = int(profile.get('auto_accept', 55))
        default_reject = int(profile.get('auto_reject', 40))

        self.spin_canny_accept.setValue(int(profile.get('canny_auto_accept', default_accept)))
        self.spin_canny_reject.setValue(int(profile.get('canny_auto_reject', default_reject)))
        self.spin_pose_accept.setValue(int(profile.get('pose_auto_accept', default_accept)))
        self.spin_pose_reject.setValue(int(profile.get('pose_auto_reject', default_reject)))
        self.spin_depth_accept.setValue(int(profile.get('depth_auto_accept', default_accept)))
        self.spin_depth_reject.setValue(int(profile.get('depth_auto_reject', default_reject)))
        self.save_settings()

    def _create_advanced_group(self) -> QGroupBox:
        """Create advanced settings group (retry + custom tags + parallel processing)"""
        group = QGroupBox(tr('advanced'))
        main_layout = QHBoxLayout(group)

        # Left side: existing settings
        left_layout = QVBoxLayout()

        # Smart retry
        retry_layout = QHBoxLayout()
        self.check_enable_retry = QCheckBox(tr('enable_retry'))
        self.check_enable_retry.setChecked(True)
        retry_layout.addWidget(self.check_enable_retry)

        retry_layout.addWidget(QLabel(tr('max_retries') + ':'))
        self.spin_max_retries = QSpinBox()
        self.spin_max_retries.setRange(1, 5)
        self.spin_max_retries.setValue(2)
        retry_layout.addWidget(self.spin_max_retries)
        retry_layout.addStretch()
        left_layout.addLayout(retry_layout)

        # Parallel processing threads
        parallel_layout = QHBoxLayout()
        parallel_layout.addWidget(QLabel('并行处理线程数:'))
        self.spin_parallel_threads = QSpinBox()
        self.spin_parallel_threads.setRange(1, 8)
        self.spin_parallel_threads.setValue(3)
        self.spin_parallel_threads.setToolTip(
            "同时处理 Canny/OpenPose/Depth 的线程数\n"
            "• 1 线程: 顺序处理 (最省显存)\n"
            "• 2-3 线程: 推荐 (平衡速度和显存)\n"
            "• 4+ 线程: 高速处理 (需要更多显存)\n"
            "根据显卡显存调整: 8GB → 2-3, 12GB+ → 3-4"
        )
        parallel_layout.addWidget(self.spin_parallel_threads)

        # Auto button (set to 3 as recommended)
        btn_auto_parallel = QPushButton('自动')
        btn_auto_parallel.setMaximumWidth(50)
        btn_auto_parallel.clicked.connect(lambda: self.spin_parallel_threads.setValue(3))
        parallel_layout.addWidget(btn_auto_parallel)

        parallel_layout.addStretch()
        left_layout.addLayout(parallel_layout)

        # Auto-pass without review
        auto_pass_layout = QHBoxLayout()
        self.check_auto_pass_no_review = QCheckBox('自动通过无需审核')
        self.check_auto_pass_no_review.setChecked(True)
        self.check_auto_pass_no_review.setToolTip(
            "勾选后，自动通过的图片不会在预览区显示\n"
            "取消勾选后，所有图片都会显示在预览区供审核"
        )
        auto_pass_layout.addWidget(self.check_auto_pass_no_review)
        auto_pass_layout.addStretch()
        left_layout.addLayout(auto_pass_layout)

        # Single JSONA file option
        single_jsona_layout = QHBoxLayout()
        self.check_single_jsona = QCheckBox('合并为单个 JSONA 文件')
        self.check_single_jsona.setChecked(False)
        self.check_single_jsona.setToolTip(
            "勾选后，所有 control type 的 metadata 写入同一个 metadata.jsona 文件\n"
            "取消勾选后，分别写入 canny.jsona、pose.jsona、depth.jsona"
        )
        single_jsona_layout.addWidget(self.check_single_jsona)
        single_jsona_layout.addStretch()
        left_layout.addLayout(single_jsona_layout)

        # JSONA backup strategy
        jsona_backup_group = QGroupBox('JSONA 备份策略')
        jsona_backup_layout = QFormLayout(jsona_backup_group)

        self.spin_jsona_backup_every_entries = QSpinBox()
        self.spin_jsona_backup_every_entries.setRange(10, 5000)
        self.spin_jsona_backup_every_entries.setValue(200)
        self.spin_jsona_backup_every_entries.setSingleStep(10)
        self.spin_jsona_backup_every_entries.setToolTip(
            '每追加多少条 JSONA 记录后创建一次滚动备份。'
        )
        jsona_backup_layout.addRow('按条数备份:', self.spin_jsona_backup_every_entries)

        self.spin_jsona_backup_every_seconds = QSpinBox()
        self.spin_jsona_backup_every_seconds.setRange(30, 86400)
        self.spin_jsona_backup_every_seconds.setValue(600)
        self.spin_jsona_backup_every_seconds.setSingleStep(30)
        self.spin_jsona_backup_every_seconds.setSuffix(' 秒')
        self.spin_jsona_backup_every_seconds.setToolTip(
            '即使新增条数未达到阈值，也会按这个时间间隔补一次备份。'
        )
        jsona_backup_layout.addRow('按时间备份:', self.spin_jsona_backup_every_seconds)

        self.spin_jsona_backup_keep = QSpinBox()
        self.spin_jsona_backup_keep.setRange(1, 100)
        self.spin_jsona_backup_keep.setValue(10)
        self.spin_jsona_backup_keep.setToolTip(
            '每个 JSONA 文件保留的最新滚动备份数量。'
        )
        jsona_backup_layout.addRow('保留份数:', self.spin_jsona_backup_keep)

        left_layout.addWidget(jsona_backup_group)

        # Unattended mode (disk-backed review inbox)
        unattended_group = QGroupBox('无人值守 (审核箱)')
        unattended_layout = QFormLayout(unattended_group)

        self.check_unattended_mode = QCheckBox('无人值守模式 (不显示中间审核图片)')
        self.check_unattended_mode.setToolTip(
            '开启后：需要人工审核的条目会保存到 output/review_inbox，界面不再堆积图片，适合挂机跑一整晚。'
        )
        unattended_layout.addRow(self.check_unattended_mode)

        self.spin_unattended_inbox_max_mb = QSpinBox()
        self.spin_unattended_inbox_max_mb.setRange(100, 102400)
        self.spin_unattended_inbox_max_mb.setValue(2048)
        self.spin_unattended_inbox_max_mb.setSingleStep(100)
        self.spin_unattended_inbox_max_mb.setSuffix(' MB')
        self.spin_unattended_inbox_max_mb.setToolTip('审核箱临时文件夹最大大小，超过后会按“满额动作”处理。')
        unattended_layout.addRow('最大大小:', self.spin_unattended_inbox_max_mb)

        self.combo_unattended_inbox_full_action = QComboBox()
        self.combo_unattended_inbox_full_action.addItems(['自动暂停', '自动停止'])
        self.combo_unattended_inbox_full_action.setToolTip('审核箱达到最大大小后的处理方式。')
        unattended_layout.addRow('满额动作:', self.combo_unattended_inbox_full_action)

        left_layout.addWidget(unattended_group)

        # VLM config (for tag verification + NL rewrite tool)
        vlm_group = QGroupBox('VLM 推理 (Tag 核对 / NL 重写)')
        vlm_layout = QFormLayout(vlm_group)

        self.combo_vlm_backend = QComboBox()
        self.combo_vlm_backend.addItems(['openai', 'ollama'])
        self.combo_vlm_backend.setToolTip(
            'openai: 兼容 OpenAI 的本地/在线接口 (例如 LM Studio / vLLM 等)\n'
            'ollama: 使用 Ollama 的本地接口 (/api/chat)\n'
            'CPU/GPU 由你选择的后端决定 (本工具只负责调用)。'
        )
        vlm_layout.addRow('后端:', self.combo_vlm_backend)

        self.edit_vlm_base_url = QLineEdit('http://127.0.0.1:1234')
        self.edit_vlm_base_url.setToolTip('例如: http://127.0.0.1:1234 (LM Studio) 或 http://127.0.0.1:11434 (Ollama)')
        vlm_layout.addRow('Base URL:', self.edit_vlm_base_url)

        self.edit_vlm_model = QLineEdit('')
        self.edit_vlm_model.setToolTip('模型名由后端决定，例如 qwen2.5-vl、llava 等')
        vlm_layout.addRow('Model:', self.edit_vlm_model)

        self.edit_vlm_api_key = QLineEdit('')
        self.edit_vlm_api_key.setEchoMode(QLineEdit.Password)
        self.edit_vlm_api_key.setToolTip('仅 openai 后端需要 (本地后端通常可留空)')
        vlm_layout.addRow('API Key:', self.edit_vlm_api_key)

        self.spin_vlm_timeout = QSpinBox()
        self.spin_vlm_timeout.setRange(10, 600)
        self.spin_vlm_timeout.setValue(120)
        self.spin_vlm_timeout.setSuffix(' 秒')
        vlm_layout.addRow('超时:', self.spin_vlm_timeout)

        left_layout.addWidget(vlm_group)

        # Custom tags
        tags_layout = QVBoxLayout()
        self.check_append_tags = QCheckBox(tr('append_tags'))
        tags_layout.addWidget(self.check_append_tags)

        self.edit_custom_tags = QTextEdit()
        self.edit_custom_tags.setMaximumHeight(60)
        self.edit_custom_tags.setPlaceholderText(tr('tag_placeholder'))
        self.edit_custom_tags.setEnabled(False)
        self.check_append_tags.toggled.connect(self.edit_custom_tags.setEnabled)
        tags_layout.addWidget(self.edit_custom_tags)
        left_layout.addLayout(tags_layout)

        xml_group = QGroupBox('XML 配置')
        xml_layout = QVBoxLayout(xml_group)

        xml_template_row = QHBoxLayout()
        xml_template_row.addStretch()
        btn_browse_xml_template = QPushButton('从XML文件加载结构')
        btn_browse_xml_template.clicked.connect(self._browse_xml_template)
        xml_template_row.addWidget(btn_browse_xml_template)
        xml_layout.addLayout(xml_template_row)

        self.edit_xml_template_path = QLineEdit('')
        self.edit_xml_template_path.setVisible(False)
        xml_layout.addWidget(self.edit_xml_template_path)

        self.label_xml_template_status = QLabel('未加载 XML 结构，当前使用默认字段。')
        self.label_xml_template_status.setWordWrap(True)
        self.label_xml_template_status.setStyleSheet('color: #999;')
        xml_layout.addWidget(self.label_xml_template_status)

        self.label_xml_fields_hint = QLabel('可选字段: artist, character_1, character_2')
        self.label_xml_fields_hint.setWordWrap(True)
        self.label_xml_fields_hint.setStyleSheet('color: #999;')
        xml_layout.addWidget(self.label_xml_fields_hint)

        standard_form = QFormLayout()
        self.combo_xml_artist_field = self._create_xml_field_combo('artist')
        self.spin_xml_artist_tag_index = self._create_xml_tag_index_spin(1)
        standard_form.addRow('artist:', self._create_xml_mapping_widget(self.combo_xml_artist_field, self.spin_xml_artist_tag_index))

        self.combo_xml_character1_field = self._create_xml_field_combo('character_1')
        self.spin_xml_character1_tag_index = self._create_xml_tag_index_spin(2)
        standard_form.addRow('character_1:', self._create_xml_mapping_widget(self.combo_xml_character1_field, self.spin_xml_character1_tag_index))

        self.combo_xml_character2_field = self._create_xml_field_combo('character_2')
        self.spin_xml_character2_tag_index = self._create_xml_tag_index_spin(3)
        standard_form.addRow('character_2:', self._create_xml_mapping_widget(self.combo_xml_character2_field, self.spin_xml_character2_tag_index))
        xml_layout.addLayout(standard_form)

        xml_custom_group = QGroupBox('自定义 XML 字段映射')
        xml_custom_layout = QVBoxLayout(xml_custom_group)
        xml_custom_note = QLabel('从上面的可选字段里选择，或直接手动输入字段路径。索引从 1 开始。')
        xml_custom_note.setWordWrap(True)
        xml_custom_note.setStyleSheet('color: #999;')
        xml_custom_layout.addWidget(xml_custom_note)

        self.xml_custom_map_layout = QVBoxLayout()
        xml_custom_layout.addLayout(self.xml_custom_map_layout)

        xml_custom_btn_row = QHBoxLayout()
        btn_add_xml_mapping = QPushButton('新增映射')
        btn_add_xml_mapping.clicked.connect(lambda: self._add_xml_custom_mapping_row())
        xml_custom_btn_row.addWidget(btn_add_xml_mapping)
        xml_custom_btn_row.addStretch()
        xml_custom_layout.addLayout(xml_custom_btn_row)
        xml_layout.addWidget(xml_custom_group)

        xml_preview_group = QGroupBox('XML 预览')
        xml_preview_layout = QVBoxLayout(xml_preview_group)
        xml_preview_note = QLabel('在这里输入一串 tag，下面会按当前 XML 映射配置实时生成预览。')
        xml_preview_note.setWordWrap(True)
        xml_preview_note.setStyleSheet('color: #999;')
        xml_preview_layout.addWidget(xml_preview_note)

        self.edit_xml_preview_tags = QTextEdit()
        self.edit_xml_preview_tags.setMaximumHeight(72)
        self.edit_xml_preview_tags.setPlaceholderText('例如: artist_name, character_a, character_b, copyright_name')
        xml_preview_layout.addWidget(self.edit_xml_preview_tags)

        self.edit_xml_preview_output = QTextEdit()
        self.edit_xml_preview_output.setReadOnly(True)
        self.edit_xml_preview_output.setMinimumHeight(120)
        self.edit_xml_preview_output.setPlaceholderText('这里会显示 XML 片段预览')
        xml_preview_layout.addWidget(self.edit_xml_preview_output)

        xml_preview_btn_row = QHBoxLayout()
        xml_preview_btn_row.addStretch()
        btn_copy_xml_preview = QPushButton('复制结果')
        btn_copy_xml_preview.clicked.connect(self._copy_xml_preview_output)
        xml_preview_btn_row.addWidget(btn_copy_xml_preview)
        xml_preview_layout.addLayout(xml_preview_btn_row)
        xml_layout.addWidget(xml_preview_group)

        left_layout.addWidget(xml_group)

        main_layout.addLayout(left_layout)

        # Right side: JSONA statistics
        right_layout = QVBoxLayout()
        stats_group = QGroupBox('JSONA 文件统计')
        stats_layout = QVBoxLayout(stats_group)

        # Canny statistics
        canny_stat_layout = QHBoxLayout()
        self.label_canny_count = QLabel('Canny: 0 张图')
        canny_stat_layout.addWidget(self.label_canny_count)
        self.btn_reset_canny_jsona = QPushButton('重置')
        self.btn_reset_canny_jsona.setMaximumWidth(60)
        self.btn_reset_canny_jsona.clicked.connect(lambda: self._reset_jsona_file('canny'))
        canny_stat_layout.addWidget(self.btn_reset_canny_jsona)
        stats_layout.addLayout(canny_stat_layout)

        # Pose statistics
        pose_stat_layout = QHBoxLayout()
        self.label_pose_count = QLabel('Pose: 0 张图')
        pose_stat_layout.addWidget(self.label_pose_count)
        self.btn_reset_pose_jsona = QPushButton('重置')
        self.btn_reset_pose_jsona.setMaximumWidth(60)
        self.btn_reset_pose_jsona.clicked.connect(lambda: self._reset_jsona_file('pose'))
        pose_stat_layout.addWidget(self.btn_reset_pose_jsona)
        stats_layout.addLayout(pose_stat_layout)

        # Depth statistics
        depth_stat_layout = QHBoxLayout()
        self.label_depth_count = QLabel('Depth: 0 张图')
        depth_stat_layout.addWidget(self.label_depth_count)
        self.btn_reset_depth_jsona = QPushButton('重置')
        self.btn_reset_depth_jsona.setMaximumWidth(60)
        self.btn_reset_depth_jsona.clicked.connect(lambda: self._reset_jsona_file('depth'))
        depth_stat_layout.addWidget(self.btn_reset_depth_jsona)
        stats_layout.addLayout(depth_stat_layout)

        # Metadata statistics
        metadata_stat_layout = QHBoxLayout()
        self.label_metadata_count = QLabel('Metadata: 0 条')
        metadata_stat_layout.addWidget(self.label_metadata_count)
        self.btn_reset_metadata_jsona = QPushButton('重置')
        self.btn_reset_metadata_jsona.setMaximumWidth(60)
        self.btn_reset_metadata_jsona.clicked.connect(lambda: self._reset_jsona_file('metadata'))
        metadata_stat_layout.addWidget(self.btn_reset_metadata_jsona)
        stats_layout.addLayout(metadata_stat_layout)

        # Tag/NL/XML statistics
        tag_stat_layout = QHBoxLayout()
        self.label_tag_count = QLabel('Tag: 0 条')
        tag_stat_layout.addWidget(self.label_tag_count)
        self.btn_reset_tag_jsona = QPushButton('重置')
        self.btn_reset_tag_jsona.setMaximumWidth(60)
        self.btn_reset_tag_jsona.clicked.connect(lambda: self._reset_jsona_file('tag'))
        tag_stat_layout.addWidget(self.btn_reset_tag_jsona)
        stats_layout.addLayout(tag_stat_layout)

        nl_stat_layout = QHBoxLayout()
        self.label_nl_count = QLabel('NL: 0 条')
        nl_stat_layout.addWidget(self.label_nl_count)
        self.btn_reset_nl_jsona = QPushButton('重置')
        self.btn_reset_nl_jsona.setMaximumWidth(60)
        self.btn_reset_nl_jsona.clicked.connect(lambda: self._reset_jsona_file('nl'))
        nl_stat_layout.addWidget(self.btn_reset_nl_jsona)
        stats_layout.addLayout(nl_stat_layout)

        xml_stat_layout = QHBoxLayout()
        self.label_xml_count = QLabel('XML: 0 条')
        xml_stat_layout.addWidget(self.label_xml_count)
        self.btn_reset_xml_jsona = QPushButton('重置')
        self.btn_reset_xml_jsona.setMaximumWidth(60)
        self.btn_reset_xml_jsona.clicked.connect(lambda: self._reset_jsona_file('xml'))
        xml_stat_layout.addWidget(self.btn_reset_xml_jsona)
        stats_layout.addLayout(xml_stat_layout)

        self.label_jsona_mode = QLabel('当前模式: 分类型 JSONA')
        self.label_jsona_mode.setStyleSheet('color: #888; font-size: 11px;')
        stats_layout.addWidget(self.label_jsona_mode)

        # Refresh button
        btn_refresh_stats = QPushButton('刷新统计')
        btn_refresh_stats.clicked.connect(self._update_jsona_statistics)
        stats_layout.addWidget(btn_refresh_stats)

        share_group = QGroupBox('参数工具')
        share_layout = QVBoxLayout(share_group)

        share_note = QLabel('复制关键参数给别人粘贴导入，并保留带时间戳的设置修改历史。')
        share_note.setWordWrap(True)
        share_note.setStyleSheet('color: #888;')
        share_layout.addWidget(share_note)

        btn_copy_key_params = QPushButton('复制关键参数')
        btn_copy_key_params.clicked.connect(self._copy_key_parameters)
        share_layout.addWidget(btn_copy_key_params)

        btn_import_key_params = QPushButton('导入关键参数')
        btn_import_key_params.clicked.connect(self._import_key_parameters)
        share_layout.addWidget(btn_import_key_params)

        btn_view_history = QPushButton('查看参数历史')
        btn_view_history.clicked.connect(self._show_settings_history)
        share_layout.addWidget(btn_view_history)

        self.label_settings_history_status = QLabel('历史记录: 0 条')
        self.label_settings_history_status.setStyleSheet('color: #888; font-size: 11px;')
        share_layout.addWidget(self.label_settings_history_status)

        share_layout.addStretch()
        right_layout.addWidget(share_group)
        right_layout.addWidget(stats_group)
        right_layout.addStretch()

        main_layout.addLayout(right_layout)

        return group

    def _create_processing_group(self) -> QGroupBox:
        """Create processing settings group"""
        group = QGroupBox(tr('processing'))
        layout = QFormLayout(group)

        # Input directory source selection
        input_source_layout = QHBoxLayout()
        self.radio_use_data_source = QRadioButton(tr('use_data_source_dir'))
        self.radio_use_data_source.setChecked(True)
        self.radio_use_custom_dir = QRadioButton(tr('use_custom_dir'))
        input_source_layout.addWidget(self.radio_use_data_source)
        input_source_layout.addWidget(self.radio_use_custom_dir)
        input_source_layout.addStretch()
        layout.addRow(tr('input_source') + ':', input_source_layout)

        # Custom input directory
        custom_input_layout = QHBoxLayout()
        self.edit_custom_input_dir = QLineEdit("./images")
        self.edit_custom_input_dir.setEnabled(False)
        custom_input_layout.addWidget(self.edit_custom_input_dir)
        btn_browse_custom_input = QPushButton(tr('browse'))
        btn_browse_custom_input.setEnabled(False)
        btn_browse_custom_input.clicked.connect(lambda: self._browse_directory(self.edit_custom_input_dir))
        custom_input_layout.addWidget(btn_browse_custom_input)
        layout.addRow('', custom_input_layout)

        # Connect radio buttons to enable/disable custom input
        self.radio_use_custom_dir.toggled.connect(lambda checked: self.edit_custom_input_dir.setEnabled(checked))
        self.radio_use_custom_dir.toggled.connect(lambda checked: btn_browse_custom_input.setEnabled(checked))

        # Quality profile (general/anime)
        self.combo_quality_profile = QComboBox()
        self.combo_quality_profile.addItems([tr('general'), tr('anime')])
        layout.addRow(tr('quality_profile') + ':', self.combo_quality_profile)

        # Canny thresholds
        canny_threshold_layout = QHBoxLayout()
        canny_threshold_layout.addWidget(QLabel('Canny 自动接受:'))
        self.spin_canny_accept = QSpinBox()
        self.spin_canny_accept.setRange(0, 100)
        self.spin_canny_accept.setValue(55)
        self.spin_canny_accept.setSuffix(' 分')
        canny_threshold_layout.addWidget(self.spin_canny_accept)
        canny_threshold_layout.addWidget(QLabel('  拒绝:'))
        self.spin_canny_reject = QSpinBox()
        self.spin_canny_reject.setRange(0, 100)
        self.spin_canny_reject.setValue(40)
        self.spin_canny_reject.setSuffix(' 分')
        canny_threshold_layout.addWidget(self.spin_canny_reject)
        canny_threshold_layout.addStretch()
        layout.addRow('', canny_threshold_layout)

        # OpenPose thresholds
        pose_threshold_layout = QHBoxLayout()
        pose_threshold_layout.addWidget(QLabel('Pose 自动接受:'))
        self.spin_pose_accept = QSpinBox()
        self.spin_pose_accept.setRange(0, 100)
        self.spin_pose_accept.setValue(60)
        self.spin_pose_accept.setSuffix(' 分')
        pose_threshold_layout.addWidget(self.spin_pose_accept)
        pose_threshold_layout.addWidget(QLabel('  拒绝:'))
        self.spin_pose_reject = QSpinBox()
        self.spin_pose_reject.setRange(0, 100)
        self.spin_pose_reject.setValue(40)
        self.spin_pose_reject.setSuffix(' 分')
        pose_threshold_layout.addWidget(self.spin_pose_reject)
        pose_threshold_layout.addStretch()
        layout.addRow('', pose_threshold_layout)

        # Depth thresholds
        depth_threshold_layout = QHBoxLayout()
        depth_threshold_layout.addWidget(QLabel('Depth 自动接受:'))
        self.spin_depth_accept = QSpinBox()
        self.spin_depth_accept.setRange(0, 100)
        self.spin_depth_accept.setValue(80)
        self.spin_depth_accept.setSuffix(' 分')
        depth_threshold_layout.addWidget(self.spin_depth_accept)
        depth_threshold_layout.addWidget(QLabel('  拒绝:'))
        self.spin_depth_reject = QSpinBox()
        self.spin_depth_reject.setRange(0, 100)
        self.spin_depth_reject.setValue(40)
        self.spin_depth_reject.setSuffix(' 分')
        depth_threshold_layout.addWidget(self.spin_depth_reject)
        depth_threshold_layout.addStretch()
        layout.addRow('', depth_threshold_layout)

        # Thread count
        thread_layout = QHBoxLayout()
        self.spin_processing_threads = QSpinBox()
        self.spin_processing_threads.setRange(1, 16)
        self.spin_processing_threads.setValue(1)
        thread_layout.addWidget(self.spin_processing_threads)

        # Auto button
        btn_auto_processing = QPushButton('自动')
        btn_auto_processing.setMaximumWidth(50)
        btn_auto_processing.clicked.connect(lambda: self.spin_processing_threads.setValue(max(1, (os.cpu_count() or 4) - 1)))
        thread_layout.addWidget(btn_auto_processing)
        thread_layout.addStretch()

        layout.addRow(tr('threads') + ':', thread_layout)

        # Preload count
        self.spin_preload_count = QSpinBox()
        self.spin_preload_count.setRange(5, 50)
        self.spin_preload_count.setValue(15)
        layout.addRow(tr('preload_count') + ':', self.spin_preload_count)

        # Control types
        control_layout = QVBoxLayout()

        # Canny
        canny_layout = QHBoxLayout()
        self.check_canny = QCheckBox("Canny")
        self.check_canny.setChecked(True)
        canny_layout.addWidget(self.check_canny)
        canny_layout.addWidget(QLabel("(OpenCV - No model needed)"))
        canny_layout.addStretch()
        control_layout.addLayout(canny_layout)

        # OpenPose
        openpose_layout = QHBoxLayout()
        self.check_openpose = QCheckBox("OpenPose")
        openpose_layout.addWidget(self.check_openpose)

        # Install torch button for OpenPose (shown when torch not available)
        self.btn_install_torch_openpose = QPushButton("安装 PyTorch")
        self.btn_install_torch_openpose.clicked.connect(self._on_install_torch_clicked)
        self.btn_install_torch_openpose.setVisible(False)
        openpose_layout.addWidget(self.btn_install_torch_openpose)

        openpose_layout.addWidget(QLabel(tr('model') + ':'))
        self.combo_openpose_model = QComboBox()
        self.combo_openpose_model.addItems([
            'DWpose (Default) (推荐10系20系老卡使用)',
            'SDPose-Wholebody (30,40,50系新卡推荐使用)',
            'ViTPose (30,40,50系新卡推荐使用)',
            'Openpose (推荐10系20系老卡使用)',
            'Custom Path'
        ])
        self.combo_openpose_model.setEnabled(False)
        openpose_layout.addWidget(self.combo_openpose_model)

        # YOLO version selection (only for ViTPose/SDPose)
        openpose_layout.addWidget(QLabel('YOLO:'))
        self.combo_yolo_version = QComboBox()
        self.combo_yolo_version.addItems(['YOLO26 (推荐)', 'YOLOv11', 'YOLOv8'])
        self.combo_yolo_version.setEnabled(False)
        self.combo_yolo_version.setVisible(False)
        openpose_layout.addWidget(self.combo_yolo_version)

        # YOLO model type selection (only for ViTPose/SDPose)
        openpose_layout.addWidget(QLabel('检测器:'))
        self.combo_yolo_model_type = QComboBox()
        self.combo_yolo_model_type.addItems(['通用 (General)', '动漫专用 (Anime)'])
        self.combo_yolo_model_type.setEnabled(False)
        self.combo_yolo_model_type.setVisible(False)
        openpose_layout.addWidget(self.combo_yolo_model_type)

        self.edit_openpose_path = QLineEdit()
        self.edit_openpose_path.setPlaceholderText(tr('model_path_placeholder'))
        self.edit_openpose_path.setEnabled(False)
        self.edit_openpose_path.setVisible(False)  # Hidden by default
        openpose_layout.addWidget(self.edit_openpose_path)
        self.btn_browse_openpose = QPushButton(tr('browse'))
        self.btn_browse_openpose.setEnabled(False)
        self.btn_browse_openpose.setVisible(False)  # Hidden by default
        self.btn_browse_openpose.clicked.connect(lambda: self._browse_directory(self.edit_openpose_path))
        openpose_layout.addWidget(self.btn_browse_openpose)
        openpose_layout.addStretch()
        control_layout.addLayout(openpose_layout)

        # Depth
        depth_layout = QHBoxLayout()
        self.check_depth = QCheckBox("Depth")
        depth_layout.addWidget(self.check_depth)

        # Install torch button for Depth (shown when torch not available)
        self.btn_install_torch_depth = QPushButton("安装 PyTorch")
        self.btn_install_torch_depth.clicked.connect(self._on_install_torch_clicked)
        self.btn_install_torch_depth.setVisible(False)
        depth_layout.addWidget(self.btn_install_torch_depth)

        depth_layout.addWidget(QLabel(tr('model') + ':'))
        self.combo_depth_model = QComboBox()
        self.combo_depth_model.addItems(['Depth Anything V2', 'Custom Path'])
        self.combo_depth_model.setEnabled(False)
        depth_layout.addWidget(self.combo_depth_model)
        self.edit_depth_path = QLineEdit()
        self.edit_depth_path.setPlaceholderText(tr('model_path_placeholder'))
        self.edit_depth_path.setEnabled(False)
        self.edit_depth_path.setVisible(False)  # Hidden by default
        depth_layout.addWidget(self.edit_depth_path)
        self.btn_browse_depth = QPushButton(tr('browse'))
        self.btn_browse_depth.setEnabled(False)
        self.btn_browse_depth.setVisible(False)  # Hidden by default
        self.btn_browse_depth.clicked.connect(lambda: self._browse_directory(self.edit_depth_path))
        depth_layout.addWidget(self.btn_browse_depth)
        depth_layout.addStretch()
        control_layout.addLayout(depth_layout)

        # Connect signals to enable/disable model selection
        self.check_openpose.toggled.connect(self._on_openpose_toggled)
        self.combo_openpose_model.currentTextChanged.connect(self._on_openpose_model_changed)
        self.combo_yolo_model_type.currentTextChanged.connect(self._on_yolo_model_type_changed)

        self.check_depth.toggled.connect(self._on_depth_toggled)
        self.combo_depth_model.currentTextChanged.connect(self._on_depth_model_changed)

        # Check torch availability on init
        self._check_torch_availability()

        layout.addRow(tr('control_types') + ':', control_layout)

        return group

    def _create_xml_field_combo(self, default_text: str = '') -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(self._xml_field_options)
        if default_text:
            combo.setCurrentText(default_text)
        return combo

    def _create_xml_tag_index_spin(self, default_value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(1, 999)
        spin.setValue(default_value)
        spin.setToolTip('索引从 1 开始，例如 1 表示第一个 tag。')
        return spin

    def _create_xml_mapping_widget(self, field_combo: QComboBox, index_spin: QSpinBox) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel('字段:'))
        layout.addWidget(field_combo, 1)
        layout.addWidget(QLabel('第'))
        layout.addWidget(index_spin)
        layout.addWidget(QLabel('个 tag'))
        return container

    def _browse_xml_template(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择 XML 文件', self.edit_output_dir.text() or '.', 'XML Files (*.xml);;All Files (*.*)')
        if path:
            self.edit_xml_template_path.setText(path)
            self._analyze_xml_template()

    def _collect_xml_field_paths(self, element, prefix: str = '') -> list:
        children = [child for child in list(element) if isinstance(child.tag, str)]
        current = f"{prefix}/{element.tag}" if prefix else str(element.tag)
        if not children:
            return [current]
        results = []
        for child in children:
            results.extend(self._collect_xml_field_paths(child, current))
        return results

    def _parse_xml_field_paths(self, file_path: str) -> list:
        if not file_path or not os.path.exists(file_path):
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = f.read().strip()
        if not raw:
            return []
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            root = ET.fromstring(f"<root>{raw}</root>")
        raw_paths = self._collect_xml_field_paths(root)
        cleaned = []
        for path in raw_paths:
            parts = [p for p in str(path).split('/') if p]
            if parts and parts[0] == 'root':
                parts = parts[1:]
            cleaned_path = '/'.join(parts)
            if cleaned_path and cleaned_path not in cleaned:
                cleaned.append(cleaned_path)
        return cleaned

    def _refresh_xml_field_options(self, field_paths: list = None):
        base_defaults = ['artist', 'character_1', 'character_2']
        options = []
        for item in base_defaults + list(field_paths or []):
            token = str(item or '').strip()
            if token and token not in options:
                options.append(token)
        self._xml_field_options = options or base_defaults

        combos = [
            getattr(self, 'combo_xml_artist_field', None),
            getattr(self, 'combo_xml_character1_field', None),
            getattr(self, 'combo_xml_character2_field', None),
        ]
        for row in self._xml_custom_mapping_rows:
            combos.append(row.get('field_combo'))

        for combo in combos:
            if combo is None:
                continue
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self._xml_field_options)
            combo.setCurrentText(current)
            combo.blockSignals(False)

        self.label_xml_fields_hint.setText('可选字段: ' + ', '.join(self._xml_field_options))
        self._refresh_xml_preview()

    def _analyze_xml_template(self):
        path = self.edit_xml_template_path.text().strip()
        if not path:
            self._refresh_xml_field_options([])
            self.label_xml_template_status.setText('未加载 XML 结构，当前使用默认字段。')
            return
        if not os.path.exists(path):
            self._refresh_xml_field_options([])
            self.label_xml_template_status.setText(f'XML 文件不存在: {path}')
            self.label_xml_fields_hint.setText('可选字段: artist, character_1, character_2')
            return
        try:
            field_paths = self._parse_xml_field_paths(path)
            self._refresh_xml_field_options(field_paths)
            if field_paths:
                self.label_xml_template_status.setText(f'已加载结构: {os.path.basename(path)}')
                self.label_xml_fields_hint.setText('可选字段: ' + ', '.join(field_paths))
            else:
                self.label_xml_template_status.setText(f'已加载结构: {os.path.basename(path)}')
                self.label_xml_fields_hint.setText('没有解析到可用字段，将继续使用默认字段。')
            self.save_settings()
        except Exception as e:
            QMessageBox.warning(self, 'XML 配置', f'XML 字段分析失败:\n{e}')

    def _add_xml_custom_mapping_row(self, field_path: str = '', tag_index: int = 1):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        field_combo = self._create_xml_field_combo(field_path)
        row_layout.addWidget(QLabel('字段:'))
        row_layout.addWidget(field_combo, 1)

        index_spin = self._create_xml_tag_index_spin(tag_index)
        row_layout.addWidget(QLabel('第'))
        row_layout.addWidget(index_spin)
        row_layout.addWidget(QLabel('个 tag'))

        btn_remove = QPushButton('删除')
        btn_remove.setMaximumWidth(60)
        row_layout.addWidget(btn_remove)

        row = {
            'widget': row_widget,
            'field_combo': field_combo,
            'index_spin': index_spin,
        }
        btn_remove.clicked.connect(lambda: self._remove_xml_custom_mapping_row(row))
        field_combo.currentTextChanged.connect(lambda _=None: self.save_settings())
        field_combo.currentTextChanged.connect(lambda _=None: self._refresh_xml_preview())
        index_spin.valueChanged.connect(lambda _=None: self.save_settings())
        index_spin.valueChanged.connect(lambda _=None: self._refresh_xml_preview())

        self._xml_custom_mapping_rows.append(row)
        self.xml_custom_map_layout.addWidget(row_widget)

    def _remove_xml_custom_mapping_row(self, row: dict):
        if row not in self._xml_custom_mapping_rows:
            return
        self._xml_custom_mapping_rows.remove(row)
        widget = row.get('widget')
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        self.save_settings()
        self._refresh_xml_preview()

    def _serialize_xml_custom_mappings(self) -> list:
        results = []
        for row in self._xml_custom_mapping_rows:
            field_combo = row.get('field_combo')
            index_spin = row.get('index_spin')
            if field_combo is None or index_spin is None:
                continue
            field_path = field_combo.currentText().strip()
            if not field_path:
                continue
            results.append({
                'field_path': field_path,
                'tag_index': int(index_spin.value()),
            })
        return results

    def _clear_xml_custom_mapping_rows(self):
        for row in list(self._xml_custom_mapping_rows):
            widget = row.get('widget')
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._xml_custom_mapping_rows = []

    def _load_xml_custom_mappings(self, rows: list):
        self._clear_xml_custom_mapping_rows()
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            self._add_xml_custom_mapping_row(
                field_path=str(row.get('field_path', '') or ''),
                tag_index=int(row.get('tag_index', 1) or 1),
            )
        self._refresh_xml_preview()

    def _build_xml_mapping_config(self) -> dict:
        return {
            'template_path': self.edit_xml_template_path.text().strip(),
            'artist_field_path': self.combo_xml_artist_field.currentText().strip() or 'artist',
            'artist_tag_index': int(self.spin_xml_artist_tag_index.value()),
            'character_1_field_path': self.combo_xml_character1_field.currentText().strip() or 'character_1',
            'character_1_tag_index': int(self.spin_xml_character1_tag_index.value()),
            'character_2_field_path': self.combo_xml_character2_field.currentText().strip() or 'character_2',
            'character_2_tag_index': int(self.spin_xml_character2_tag_index.value()),
            'custom_mappings': self._serialize_xml_custom_mappings(),
        }

    def _refresh_xml_preview(self):
        if not hasattr(self, 'edit_xml_preview_output'):
            return
        tag_text = self.edit_xml_preview_tags.toPlainText().strip()
        if not tag_text:
            self.edit_xml_preview_output.clear()
            return
        try:
            xml_fragment = build_xml_fragment(tag_text, self._build_xml_mapping_config())
            self.edit_xml_preview_output.setPlainText(xml_fragment or '(当前配置下没有生成任何 XML 字段)')
        except Exception as e:
            self.edit_xml_preview_output.setPlainText(f'XML 预览生成失败:\n{e}')

    def _copy_xml_preview_output(self):
        text = self.edit_xml_preview_output.toPlainText().strip()
        if not text:
            QMessageBox.information(self, 'XML 预览', '当前没有可复制的 XML 预览内容。')
            return
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, 'XML 预览', 'XML 预览结果已复制到剪贴板。')

    def _get_shareable_setting_keys(self) -> list:
        return [
            'processing_threads',
            'preload_count',
            'canny_enabled',
            'openpose_enabled',
            'depth_enabled',
            'quality_profile',
            'canny_accept',
            'canny_reject',
            'pose_accept',
            'pose_reject',
            'depth_accept',
            'depth_reject',
            'openpose_model',
            'openpose_yolo_version',
            'openpose_yolo_model_type',
            'depth_model',
            'parallel_threads',
            'auto_pass_no_review',
            'single_jsona',
            'jsona_backup_every_entries',
            'jsona_backup_every_seconds',
            'jsona_backup_keep',
            'unattended_mode',
            'unattended_inbox_max_mb',
            'unattended_inbox_full_action',
            'vlm_backend',
            'vlm_model',
            'vlm_timeout_seconds',
            'xml_artist_field_path',
            'xml_artist_tag_index',
            'xml_character1_field_path',
            'xml_character1_tag_index',
            'xml_character2_field_path',
            'xml_character2_tag_index',
            'xml_custom_mappings',
            'enable_retry',
            'max_retries',
            'append_tags',
            'custom_tags',
            'discard_action',
        ]

    def _copy_key_parameters(self):
        flat_settings = self._collect_flat_settings()
        shareable = {
            key: deepcopy(flat_settings.get(key))
            for key in self._get_shareable_setting_keys()
            if key in flat_settings
        }
        payload = {
            'format': 'controlnet_gui_key_params',
            'version': 1,
            'exported_at': datetime.now().isoformat(timespec='seconds'),
            'params': shareable,
        }
        QApplication.clipboard().setText(json.dumps(payload, ensure_ascii=False, indent=2))
        QMessageBox.information(
            self,
            '关键参数',
            '关键参数已复制到剪贴板。\n\n已自动排除路径、Token、API Key 等机器相关信息。'
        )

    def _import_key_parameters(self):
        clipboard_text = QApplication.clipboard().text().strip()
        initial_text = clipboard_text if clipboard_text.startswith('{') else ''
        dialog = TextImportDialog(
            '导入关键参数',
            '把别人发来的关键参数 JSON 粘贴到这里。支持直接粘贴“复制关键参数”生成的内容。',
            initial_text=initial_text,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        raw_text = dialog.get_text().strip()
        if not raw_text:
            QMessageBox.warning(self, '导入关键参数', '没有检测到可导入的内容。')
            return

        try:
            data = json.loads(raw_text)
        except Exception as e:
            QMessageBox.warning(self, '导入关键参数', f'JSON 解析失败:\n{e}')
            return

        if not isinstance(data, dict):
            QMessageBox.warning(self, '导入关键参数', '导入内容必须是 JSON 对象。')
            return

        if 'params' in data and isinstance(data.get('params'), dict):
            config = data.get('params', {})
        else:
            config = data

        allowed_keys = set(self._get_shareable_setting_keys())
        filtered_config = {key: value for key, value in config.items() if key in allowed_keys}
        if not filtered_config:
            QMessageBox.warning(self, '导入关键参数', '未找到可识别的关键参数字段。')
            return

        skipped_labels = []
        openpose_model = str(filtered_config.get('openpose_model', '') or '')
        if openpose_model == 'Custom Path':
            filtered_config.pop('openpose_model', None)
            skipped_labels.append('Pose 模型(Custom Path)')

        depth_model = str(filtered_config.get('depth_model', '') or '')
        if depth_model == 'Custom Path':
            filtered_config.pop('depth_model', None)
            skipped_labels.append('Depth 模型(Custom Path)')

        if not filtered_config:
            QMessageBox.warning(
                self,
                '导入关键参数',
                '导入内容只包含 Custom Path 模型选择，但关键参数不会携带机器路径。\n\n为避免导入后模型状态与路径不一致，本次未应用这些字段。'
            )
            return

        self._apply_flat_settings(filtered_config)
        self.save_settings(history_reason='import_key_params', force_history=True)

        field_labels = self._get_settings_field_labels()
        changed_labels = [field_labels.get(key, key) for key in filtered_config.keys()]
        preview = '、'.join(changed_labels[:6])
        if len(changed_labels) > 6:
            preview += f' 等 {len(changed_labels)} 项'
        message = f'关键参数已导入。\n\n涉及字段: {preview}'
        if skipped_labels:
            message += (
                '\n\n以下字段未导入: ' + '、'.join(skipped_labels) +
                '\n原因: 关键参数不会携带本机自定义模型路径，已自动跳过以避免状态不一致。'
            )
        QMessageBox.information(self, '导入关键参数', message)

    def _show_settings_history(self):
        entries = self._read_settings_history_entries()
        dialog = SettingsHistoryDialog(entries, self._get_settings_field_labels(), parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        entry = dialog.get_selected_entry()
        if not entry:
            return

        snapshot = entry.get('snapshot')
        if not isinstance(snapshot, dict):
            QMessageBox.warning(self, '参数修改历史', '该条历史记录没有可恢复的快照。')
            return

        reply = QMessageBox.question(
            self,
            '参数修改历史',
            '确定要恢复到这条历史记录对应的参数吗？\n\n当前面板参数会被覆盖。',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self._apply_flat_settings(snapshot)
        self.save_settings(history_reason='restore_history', force_history=True)
        QMessageBox.information(self, '参数修改历史', '已恢复选中的参数快照。')

    def _create_output_group(self) -> QGroupBox:
        """Create output settings group"""
        group = QGroupBox(tr('output'))
        layout = QFormLayout(group)

        # Output directory
        self.edit_output_dir = QLineEdit("./output")
        btn_browse_output = QPushButton(tr('browse'))
        btn_browse_output.clicked.connect(lambda: self._browse_directory(self.edit_output_dir))
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.edit_output_dir)
        output_layout.addWidget(btn_browse_output)
        layout.addRow(tr('output_dir') + ':', output_layout)

        # Discard action
        self.combo_discard_action = QComboBox()
        self.combo_discard_action.addItems([tr('move_to_trash'), tr('delete_permanently')])
        layout.addRow(tr('discard_action') + ':', self.combo_discard_action)

        return group

    def _on_source_type_changed(self, text: str):
        """Handle data source type change"""
        if text == tr('local_parquet'):
            self.local_widget.show()
            self.streaming_widget.hide()
        else:
            self.local_widget.hide()
            self.streaming_widget.show()

    def _add_parquet_file(self):
        """Add parquet file to list"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            tr('select_files'),
            "",
            "Parquet Files (*.parquet);;All Files (*.*)"
        )
        if files:
            for file in files:
                self.list_parquet_files.addItem(file)
            self.save_settings()

    def _remove_parquet_file(self):
        """Remove selected parquet file from list"""
        current_row = self.list_parquet_files.currentRow()
        if current_row >= 0:
            self.list_parquet_files.takeItem(current_row)
            self.save_settings()

    def _extract_parquet_data(self):
        """Extract data from parquet files with multi-threading"""
        parquet_files = [self.list_parquet_files.item(i).text()
                        for i in range(self.list_parquet_files.count())]

        if not parquet_files:
            return

        extract_dir = self.edit_extract_dir.text()
        num_samples = self.spin_local_samples.value()
        num_threads = self.config.get('processing', {}).get('thread_count', 4)

        self.btn_extract.setEnabled(False)
        self.btn_extract.setText(tr('extracting_data'))
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        try:
            source = ParquetDataSource(parquet_files, extract_dir, num_samples, num_threads)
            self.extraction_thread = ExtractionThread(source)
            self.extraction_thread.progress_updated.connect(self._on_extraction_progress)
            self.extraction_thread.extraction_finished.connect(self._on_extraction_finished)
            self.extraction_thread.extraction_error.connect(self._on_extraction_error)
            self.extraction_thread.start()

        except Exception as e:
            print(f"Extraction error: {e}")
            self.btn_extract.setText(tr('extract_data'))
            self.btn_extract.setEnabled(True)
            self.progress_bar.setVisible(False)

    def _on_extraction_progress(self, current: int, total: int):
        """Update progress bar"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_extraction_finished(self, count: int):
        """Handle extraction completion"""
        self.btn_extract.setText(tr('extract_data'))
        self.btn_extract.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.extraction_finished.emit(count)

    def _on_extraction_error(self, error: str):
        """Handle extraction error"""
        print(f"Extraction error: {error}")
        self.btn_extract.setText(tr('extract_data'))
        self.btn_extract.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _on_multithread_changed(self, state):
        """Handle multithread checkbox state change"""
        self.spin_thread_count.setEnabled(state == 2)  # 2 = Qt.Checked

    def _on_openpose_toggled(self, checked):
        """Handle OpenPose checkbox toggle"""
        self.combo_openpose_model.setEnabled(checked)
        if checked:
            self._on_openpose_model_changed()
        else:
            self.edit_openpose_path.setEnabled(False)
            self.btn_browse_openpose.setEnabled(False)

    def _get_openpose_model_type(self):
        """Extract actual model type from display text"""
        display_text = self.combo_openpose_model.currentText()
        if 'DWpose' in display_text:
            return 'DWpose (Default)'
        elif 'SDPose-Wholebody' in display_text:
            return 'SDPose-Wholebody'
        elif 'ViTPose' in display_text:
            return 'ViTPose'
        elif 'Openpose' in display_text:
            return 'Openpose'
        elif 'Custom Path' in display_text:
            return 'Custom Path'
        return display_text

    def _on_openpose_model_changed(self):
        """Handle OpenPose model selection change"""
        model_type = self._get_openpose_model_type()
        is_custom = model_type == 'Custom Path'
        self.edit_openpose_path.setEnabled(is_custom)
        self.btn_browse_openpose.setEnabled(is_custom)
        self.edit_openpose_path.setVisible(is_custom)
        self.btn_browse_openpose.setVisible(is_custom)

        # Show/hide YOLO version selector
        is_vitpose = model_type in ['ViTPose', 'SDPose-Wholebody']
        self.combo_yolo_version.setVisible(is_vitpose)
        self.combo_yolo_version.setEnabled(is_vitpose)
        self.combo_yolo_model_type.setVisible(is_vitpose)
        self.combo_yolo_model_type.setEnabled(is_vitpose)

        # Check dependencies for ViTPose/SDPose-Wholebody (only once per session)
        if is_vitpose and not self._vitpose_deps_checked:
            missing_deps = []
            try:
                import ultralytics
            except ImportError:
                missing_deps.append('ultralytics')

            try:
                import onnxruntime
            except ImportError:
                missing_deps.append('onnxruntime-gpu')

            if missing_deps:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("缺少依赖")
                msg.setText(f"使用 {model_type} 需要安装以下依赖:\n\n" + "\n".join(f"  • {dep}" for dep in missing_deps))
                msg.setInformativeText("是否现在安装?")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.Yes)

                if msg.exec_() == QMessageBox.Yes:
                    self._install_vitpose_dependencies()
                else:
                    # Revert to DWpose
                    self.combo_openpose_model.blockSignals(True)
                    self.combo_openpose_model.setCurrentText('DWpose (Default) (推荐10系20系老卡使用)')
                    self.combo_openpose_model.blockSignals(False)
                return
            else:
                # Dependencies are installed, mark as checked
                self._vitpose_deps_checked = True

        # Check model files (only once per model type per session)
        if is_vitpose and model_type not in self._vitpose_models_checked:
            self._check_vitpose_model_files(model_type)
            # Mark as checked regardless of result (user can manually download if needed)
            self._vitpose_models_checked[model_type] = True

    def _on_yolo_model_type_changed(self):
        """Handle YOLO model type change (General vs Anime)"""
        model_type = self._get_openpose_model_type()
        if model_type in ['ViTPose', 'SDPose-Wholebody']:
            # Re-check model files when YOLO type changes
            self._check_vitpose_model_files(model_type)

    def _install_vitpose_dependencies(self):
        """Install ViTPose dependencies (ultralytics, onnxruntime-gpu)"""
        import subprocess
        import sys

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("安装依赖")
        msg.setText("正在安装 ViTPose 依赖...\n\n这可能需要几分钟时间。")
        msg.setStandardButtons(QMessageBox.NoButton)
        msg.show()

        try:
            # Install ultralytics and onnxruntime-gpu
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics', 'onnxruntime-gpu'])

            msg.setIcon(QMessageBox.Information)
            msg.setText("依赖安装成功！\n\n请重启程序以使用 ViTPose。")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

            # Mark as checked
            self._vitpose_deps_checked = True
        except subprocess.CalledProcessError as e:
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f"依赖安装失败:\n\n{str(e)}")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

            # Revert to DWpose
            self.combo_openpose_model.blockSignals(True)
            self.combo_openpose_model.setCurrentText('DWpose (Default)')
            self.combo_openpose_model.blockSignals(False)

    def _on_depth_toggled(self, checked):
        """Handle Depth checkbox toggle"""
        self.combo_depth_model.setEnabled(checked)
        if checked:
            self._on_depth_model_changed()
        else:
            self.edit_depth_path.setEnabled(False)
            self.btn_browse_depth.setEnabled(False)
            self.edit_depth_path.setVisible(False)
            self.btn_browse_depth.setVisible(False)

    def _on_depth_model_changed(self):
        """Handle Depth model selection change"""
        is_custom = self.combo_depth_model.currentText() == 'Custom Path'
        self.edit_depth_path.setEnabled(is_custom)
        self.btn_browse_depth.setEnabled(is_custom)
        self.edit_depth_path.setVisible(is_custom)
        self.btn_browse_depth.setVisible(is_custom)

    def _check_torch_availability(self):
        """Check if torch is available and update UI accordingly"""
        try:
            import torch
            torch_available = True
        except ImportError:
            torch_available = False

        if not torch_available:
            # Show install buttons, hide and disable other controls
            self.btn_install_torch_openpose.setVisible(True)
            self.btn_install_torch_depth.setVisible(True)

            # Disable and blur OpenPose controls
            self.check_openpose.setEnabled(False)
            self.combo_openpose_model.setEnabled(False)
            self.edit_openpose_path.setEnabled(False)
            self.btn_browse_openpose.setEnabled(False)
            self._apply_blur_effect(self.combo_openpose_model)
            self._apply_blur_effect(self.edit_openpose_path)
            self._apply_blur_effect(self.btn_browse_openpose)

            # Disable and blur Depth controls
            self.check_depth.setEnabled(False)
            self.combo_depth_model.setEnabled(False)
            self.edit_depth_path.setEnabled(False)
            self.btn_browse_depth.setEnabled(False)
            self._apply_blur_effect(self.combo_depth_model)
            self._apply_blur_effect(self.edit_depth_path)
            self._apply_blur_effect(self.btn_browse_depth)
        else:
            # Hide install buttons
            self.btn_install_torch_openpose.setVisible(False)
            self.btn_install_torch_depth.setVisible(False)

    def _apply_blur_effect(self, widget):
        """Apply visual blur/disabled effect to widget"""
        widget.setStyleSheet("opacity: 0.3;")

    def _on_install_torch_clicked(self):
        """Handle install torch button click - emit signal to main window"""
        # Get main window and call its install torch method
        main_window = self.window()
        if hasattr(main_window, '_install_torch'):
            main_window._install_torch()

    def _download_model_files(self, download_urls, model_type):
        """Download model files with progress dialog"""
        from PyQt5.QtWidgets import QProgressDialog, QMessageBox
        from PyQt5.QtCore import QThread, pyqtSignal
        import urllib.request
        import os

        class DownloadThread(QThread):
            progress = pyqtSignal(str, int, int)  # filename, downloaded, total
            finished = pyqtSignal(bool, str)  # success, error_message

            def __init__(self, download_urls):
                super().__init__()
                self.download_urls = download_urls

            def run(self):
                try:
                    for filename, url in self.download_urls:
                        self.progress.emit(filename, 0, 0)

                        def reporthook(block_num, block_size, total_size):
                            downloaded = block_num * block_size
                            self.progress.emit(filename, downloaded, total_size)

                        urllib.request.urlretrieve(url, filename, reporthook)

                    self.finished.emit(True, "")
                except Exception as e:
                    self.finished.emit(False, str(e))

        # Create progress dialog
        progress_dialog = QProgressDialog(self)
        progress_dialog.setWindowTitle("下载模型文件")
        progress_dialog.setLabelText("准备下载...")
        progress_dialog.setMinimum(0)
        progress_dialog.setMaximum(100)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setCancelButton(None)
        progress_dialog.setAutoClose(False)

        # Create and start download thread
        download_thread = DownloadThread(download_urls)

        # Store reference to self for nested functions
        parent_widget = self

        def update_progress(filename, downloaded, total):
            if total > 0:
                percent = int((downloaded / total) * 100)
                progress_dialog.setValue(percent)
                size_mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                progress_dialog.setLabelText(f"下载 {filename}\n{size_mb:.1f} MB / {total_mb:.1f} MB ({percent}%)")
            else:
                progress_dialog.setLabelText(f"下载 {filename}\n正在连接...")

        def download_finished(success, error_msg):
            progress_dialog.close()
            if success:
                QMessageBox.information(
                    parent_widget,
                    "下载完成",
                    f"模型文件下载成功！\n\n现在可以使用 {model_type} 模型了。"
                )
            else:
                QMessageBox.critical(
                    parent_widget,
                    "下载失败",
                    f"模型文件下载失败:\n\n{error_msg}\n\n请检查网络连接或手动下载。"
                )
                # Revert to DWpose
                parent_widget.combo_openpose_model.blockSignals(True)
                parent_widget.combo_openpose_model.setCurrentText('DWpose (Default) (推荐10系20系老卡使用)')
                parent_widget.combo_openpose_model.blockSignals(False)

        download_thread.progress.connect(update_progress)
        download_thread.finished.connect(download_finished)
        download_thread.start()

        progress_dialog.exec_()
        """Install ViTPose/SDPose-Wholebody dependencies"""
        # Get main window to use its InstallProgressDialog
        main_window = self.window()
        if not hasattr(main_window, '_run_pip_install'):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "错误",
                "无法访问安装功能。请重启程序后重试。"
            )
            return

        # Use main window's pip install method
        packages = ['ultralytics', 'onnxruntime-gpu']
        main_window._run_pip_install(packages, '安装 ViTPose/SDPose 依赖')

    def _check_vitpose_model_files(self, model_type):
        """Check if ViTPose model files exist"""
        import os

        # Create models directory structure
        models_dir = 'models'
        yolo_dir = os.path.join(models_dir, 'yolo')
        vitpose_dir = os.path.join(models_dir, 'vitpose')
        os.makedirs(yolo_dir, exist_ok=True)
        os.makedirs(vitpose_dir, exist_ok=True)

        # Get YOLO model filename based on version and type
        yolo_model_type = self.combo_yolo_model_type.currentText()
        yolo_version = self.combo_yolo_version.currentText()

        if '动漫专用' in yolo_model_type:
            # Use anime person detection model
            yolo_file = os.path.join(yolo_dir, 'anime_person_detect_v1.3_s.pt')
            yolo_url = 'https://huggingface.co/deepghs/anime_person_detection/resolve/main/person_detect_v1.3_s/model.pt'
        elif 'YOLO26' in yolo_version:
            yolo_file = os.path.join(yolo_dir, 'yolo26n.pt')
            yolo_url = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt'
        elif 'YOLOv11' in yolo_version:
            yolo_file = os.path.join(yolo_dir, 'yolo11n.pt')
            yolo_url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
        else:  # YOLOv8
            yolo_file = os.path.join(yolo_dir, 'yolov8n.pt')
            yolo_url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'

        # Different ViTPose models for different tasks
        if model_type == 'SDPose-Wholebody':
            vitpose_file = os.path.join(vitpose_dir, 'vitpose-l-wholebody.onnx')
            vitpose_url = 'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx'
        else:  # ViTPose
            vitpose_file = os.path.join(vitpose_dir, 'vitpose-l-coco.onnx')
            vitpose_url = 'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-l-coco.onnx'

        yolo_exists = os.path.exists(yolo_file)
        vitpose_exists = os.path.exists(vitpose_file)

        if not yolo_exists or not vitpose_exists:
            missing_files = []
            download_urls = []

            if not yolo_exists:
                missing_files.append(yolo_file)
                download_urls.append((yolo_file, yolo_url))
            if not vitpose_exists:
                missing_files.append(vitpose_file)
                download_urls.append((vitpose_file, vitpose_url))

            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("缺少模型文件")
            msg.setText(f"使用 {model_type} 需要以下模型文件:\n\n" + "\n".join(f"  • {f}" for f in missing_files))

            file_sizes = ""
            if not yolo_exists:
                if '动漫专用' in yolo_model_type:
                    file_sizes += f"\n{yolo_file}: ~22.5 MB"
                else:
                    file_sizes += f"\n{yolo_file}: ~6 MB"
            if not vitpose_exists:
                file_sizes += f"\n{vitpose_file}: ~1.23 GB"

            msg.setInformativeText(f"是否自动下载这些文件?{file_sizes}\n\n下载完成后会保存到程序根目录。")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.Yes)

            if msg.exec_() == QMessageBox.Yes:
                self._download_model_files(download_urls, model_type)
            else:
                # Revert to DWpose
                self.combo_openpose_model.blockSignals(True)
                self.combo_openpose_model.setCurrentText('DWpose (Default)')
                self.combo_openpose_model.blockSignals(False)

    def _on_reset_progress(self):
        """Handle reset progress button click"""
        from PyQt5.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "确认重置",
            "确定要重置断点续传进度吗？\n\n"
            "这将清空所有已处理记录，下次处理时会从头开始。\n"
            "已生成的文件不会被删除。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Get progress file path from config
            config = self.get_settings()
            progress_config = config.get('progress', {})
            progress_file = progress_config.get('progress_file', '.progress.json')

            # Delete progress file
            import os
            if os.path.exists(progress_file):
                try:
                    os.remove(progress_file)
                    QMessageBox.information(
                        self,
                        "重置成功",
                        f"断点续传进度已重置。\n\n已删除: {progress_file}"
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "重置失败",
                        f"无法删除进度文件: {e}"
                    )
            else:
                QMessageBox.information(
                    self,
                    "无需重置",
                    "进度文件不存在，无需重置。"
                )

    def _on_splits_detected(self, splits: list, dataset_id: str, hf_token: str, num_samples: int, extract_dir: str):
        """Handle successful split detection"""
        # Re-enable button
        self.btn_extract_streaming.setEnabled(True)
        self.btn_extract_streaming.setText(tr('extract_data'))

        # Show selection dialog
        dialog = SplitSelectionDialog(splits, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_splits = dialog.get_selected_splits()
            if not selected_splits:
                QMessageBox.warning(
                    self,
                    tr('no_split_selected'),
                    tr('no_split_selected_message')
                )
                return

            # Use first selected split
            split = selected_splits[0]
            if len(selected_splits) > 1:
                QMessageBox.information(
                    self,
                    tr('multiple_splits_info'),
                    tr('multiple_splits_message').format(split)
                )

            # Continue with extraction
            self._start_streaming_extraction(dataset_id, split, hf_token, num_samples, extract_dir)
        else:
            # User cancelled
            return

    def _on_split_detection_error(self, error_msg: str):
        """Handle split detection error"""
        # Re-enable button
        self.btn_extract_streaming.setEnabled(True)
        self.btn_extract_streaming.setText(tr('extract_data'))

        if error_msg == "torch_dll_error":
            # Torch DLL error - offer to use default split
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle(tr('auto_detect_failed'))
            msg.setText(tr('torch_dll_error_title'))
            msg.setInformativeText(tr('torch_dll_error_message'))
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.Yes)

            yes_button = msg.button(QMessageBox.Yes)
            yes_button.setText(tr('use_train_split'))
            no_button = msg.button(QMessageBox.No)
            no_button.setText(tr('cancel'))

            reply = msg.exec_()
            if reply == QMessageBox.Yes:
                # Continue with default 'train' split
                dataset_id = self.edit_dataset_id.text()
                hf_token = self.edit_hf_token.text() or None
                num_samples = self.spin_num_samples.value()
                extract_dir = self.edit_streaming_extract_dir.text()
                self._start_streaming_extraction(dataset_id, 'train', hf_token, num_samples, extract_dir)
        else:
            # Other errors
            import traceback
            reply = QMessageBox.question(
                self,
                tr('error'),
                f"{tr('failed_to_detect_splits')}: {error_msg}\n\n{tr('use_default_split')}",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                dataset_id = self.edit_dataset_id.text()
                hf_token = self.edit_hf_token.text() or None
                num_samples = self.spin_num_samples.value()
                extract_dir = self.edit_streaming_extract_dir.text()
                self._start_streaming_extraction(dataset_id, 'train', hf_token, num_samples, extract_dir)

    def _start_streaming_extraction(self, dataset_id: str, split: str, hf_token: str, num_samples: int, extract_dir: str):
        """Start the actual streaming extraction with the given split"""
        # Get user prefix and skip count
        user_prefix = self.edit_user_prefix.text().strip()
        skip_count = self.spin_skip_count.value()

        # Get thread count
        is_multithread_checked = self.check_multithread.isChecked()
        thread_count_value = self.spin_thread_count.value()
        num_threads = thread_count_value if is_multithread_checked else 1

        # Start extraction in background thread
        self.btn_extract_streaming.setEnabled(False)
        self.btn_extract_streaming.setText(tr('extracting_data'))
        self.streaming_log.setVisible(True)
        self.streaming_log.clear()
        self.streaming_log.append(f"[INFO] Preparing to extract data...")

        # Debug log (after clear)
        self.streaming_log.append(f"[DEBUG] Multi-threading checkbox state: {is_multithread_checked}")
        self.streaming_log.append(f"[DEBUG] Thread count spinbox value: {thread_count_value}")
        self.streaming_log.append(f"[DEBUG] Final num_threads passed to extraction: {num_threads}")
        self.streaming_log.append("")

        # Create and start thread
        self.streaming_thread = StreamingExtractionThread(
            dataset_id, split, extract_dir, num_samples, hf_token,
            user_prefix, skip_count, num_threads, self
        )
        self.streaming_thread.log_message.connect(self._on_streaming_log)
        self.streaming_thread.progress_updated.connect(self._on_streaming_progress)
        self.streaming_thread.extraction_finished.connect(self._on_streaming_extraction_finished)
        self.streaming_thread.extraction_error.connect(self._on_streaming_extraction_error)

        # Show and setup progress bar
        self.extraction_progress.setVisible(True)
        self.extraction_progress.setMaximum(num_samples)
        self.extraction_progress.setValue(0)

        self.streaming_thread.start()

    def _extract_streaming_data(self):
        """Extract data from streaming dataset"""
        dataset_id = self.edit_dataset_id.text()
        if not dataset_id:
            return

        split = self.edit_split.text().strip()
        hf_token = self.edit_hf_token.text() or None
        num_samples = self.spin_num_samples.value()
        extract_dir = self.edit_streaming_extract_dir.text()

        # Check for incomplete extraction
        import os
        import json
        progress_file = os.path.join(extract_dir, ".progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    processed_count = progress_data.get('processed_count', 0)

                    reply = QMessageBox.question(
                        self,
                        tr('resume_extraction'),
                        tr('resume_extraction_message').format(processed_count),
                        QMessageBox.Yes | QMessageBox.No
                    )

                    if reply == QMessageBox.No:
                        # User wants to start fresh, delete progress file
                        os.remove(progress_file)
            except Exception:
                pass

        # If split is empty, auto-detect and show selection dialog
        if not split:
            # Disable button and show loading message
            self.btn_extract_streaming.setEnabled(False)
            self.btn_extract_streaming.setText(tr('detecting_splits'))

            # Start split detection in background thread
            self.split_detection_thread = SplitDetectionThread(dataset_id, hf_token, self)
            self.split_detection_thread.splits_detected.connect(
                lambda splits: self._on_splits_detected(splits, dataset_id, hf_token, num_samples, extract_dir)
            )
            self.split_detection_thread.detection_error.connect(self._on_split_detection_error)
            self.split_detection_thread.start()
            return

        # If split is provided, start extraction directly
        self._start_streaming_extraction(dataset_id, split, hf_token, num_samples, extract_dir)

    def _on_streaming_progress(self, current: int, total: int):
        """Handle streaming extraction progress update"""
        import time

        self.extraction_progress.setMaximum(total)
        self.extraction_progress.setValue(current)

        # Calculate speed
        current_time = time.time()
        if self.extraction_start_time == 0:
            self.extraction_start_time = current_time
            self.extraction_last_time = current_time
            self.extraction_last_count = 0

        # Update speed every second
        time_diff = current_time - self.extraction_last_time
        if time_diff >= 1.0:
            count_diff = current - self.extraction_last_count
            speed = count_diff / time_diff if time_diff > 0 else 0
            self.extraction_progress.setFormat(f'%v / %m images (%p%) - {speed:.1f} it/s')
            self.extraction_last_time = current_time
            self.extraction_last_count = current

    def _on_streaming_log(self, message: str):
        """Handle streaming extraction log message"""
        self.streaming_log.append(message)
        # Auto-scroll to bottom
        self.streaming_log.verticalScrollBar().setValue(
            self.streaming_log.verticalScrollBar().maximum()
        )

    def _on_streaming_extraction_finished(self, count: int):
        """Handle streaming extraction completion"""
        self.btn_extract_streaming.setEnabled(True)
        self.btn_extract_streaming.setText(tr('extract_data'))
        self.extraction_finished.emit(count)

    def _on_streaming_extraction_error(self, error: str):
        """Handle streaming extraction error"""
        print(f"Streaming extraction error: {error}")
        self.btn_extract_streaming.setEnabled(True)
        self.btn_extract_streaming.setText(tr('extract_data'))

        # Check if it's a torch DLL error (check for multiple indicators)
        is_torch_dll_error = (
            ("torch" in error.lower() or "c10.dll" in error.lower()) and
            ("dll" in error.lower() or "WinError 1114" in error or "动态链接库" in error)
        )

        if is_torch_dll_error:
            # Show custom dialog with fix button
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle(tr('torch_dll_error'))
            msg.setText(tr('torch_dll_error_datasets'))
            msg.setInformativeText(tr('torch_dll_error_datasets_info'))
            msg.setDetailedText(error)

            # Add custom buttons
            fix_button = msg.addButton(tr('fix_now'), QMessageBox.ActionRole)
            cancel_button = msg.addButton(tr('cancel'), QMessageBox.RejectRole)

            msg.exec_()

            # Check which button was clicked
            if msg.clickedButton() == fix_button:
                # Trigger fix datasets from main window
                main_window = self.window()
                if hasattr(main_window, '_fix_datasets'):
                    main_window._fix_datasets()
        else:
            # Show normal error dialog
            QMessageBox.critical(
                self,
                tr('error'),
                f"{tr('extraction_failed')}: {error}"
            )

    def _browse_directory(self, line_edit: QLineEdit):
        """Browse for directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            tr('select_directory'),
            line_edit.text() or ""
        )
        if directory:
            line_edit.setText(directory)

    def _load_config(self):
        """Load configuration"""
        # Data source
        data_source = self.config.get('data_source', {})
        source_type = data_source.get('type', 'local_parquet')

        if source_type == 'local_parquet':
            self.combo_source_type.setCurrentText(tr('local_parquet'))
            local_config = data_source.get('local_parquet', {})
            parquet_files = local_config.get('parquet_files', [])
            self.list_parquet_files.clear()
            for file in parquet_files:
                self.list_parquet_files.addItem(file)
            self.edit_extract_dir.setText(local_config.get('extract_dir', './extracted'))
            self.spin_local_samples.setValue(local_config.get('num_samples', 0))
        else:
            self.combo_source_type.setCurrentText(tr('streaming_dataset'))
            streaming_config = data_source.get('streaming', {})
            self.edit_dataset_id.setText(streaming_config.get('dataset_id', ''))
            self.edit_split.setText(streaming_config.get('split', 'train'))
            self.edit_hf_token.setText(streaming_config.get('hf_token', '') or '')
            self.spin_num_samples.setValue(streaming_config.get('num_samples', 10000))
            self.edit_streaming_extract_dir.setText(streaming_config.get('extract_dir', './extracted'))

        # Processing
        processing = self.config.get('processing', {})
        self.spin_processing_threads.setValue(processing.get('thread_count', 1))
        self.spin_preload_count.setValue(processing.get('preload_count', 15))

        # Quality profile
        scoring = self.config.get('scoring', {})
        active_profile = scoring.get('active_profile', 'general')
        if active_profile not in ['general', 'anime']:
            active_profile = 'general'
        self.combo_quality_profile.setCurrentText(active_profile)

        # Load thresholds from active profile
        profiles = scoring.get('profiles', {})
        profile = profiles.get(active_profile, {})
        default_accept = int(profile.get('auto_accept', 55))
        default_reject = int(profile.get('auto_reject', 40))

        # Load per-control-type thresholds, fallback to default if not set
        self.spin_canny_accept.setValue(int(profile.get('canny_auto_accept', default_accept)))
        self.spin_canny_reject.setValue(int(profile.get('canny_auto_reject', default_reject)))
        self.spin_pose_accept.setValue(int(profile.get('pose_auto_accept', default_accept)))
        self.spin_pose_reject.setValue(int(profile.get('pose_auto_reject', default_reject)))
        self.spin_depth_accept.setValue(int(profile.get('depth_auto_accept', default_accept)))
        self.spin_depth_reject.setValue(int(profile.get('depth_auto_reject', default_reject)))

        # Control types (nested under processing)
        control_types = processing.get('control_types', {})
        self.check_canny.setChecked(control_types.get('canny', True))
        self.check_openpose.setChecked(control_types.get('openpose', False))
        self.check_depth.setChecked(control_types.get('depth', False))

        # Retry strategy (nested under processing)
        retry_config = processing.get('retry_strategy', {})
        self.check_enable_retry.setChecked(retry_config.get('enabled', True))
        self.spin_max_retries.setValue(retry_config.get('max_retries', 2))

        # Parallel processing threads
        self.spin_parallel_threads.setValue(processing.get('parallel_threads', 3))
        self.spin_jsona_backup_every_entries.setValue(processing.get('jsona_backup_every_entries', 200))
        self.spin_jsona_backup_every_seconds.setValue(processing.get('jsona_backup_every_seconds', 600))
        self.spin_jsona_backup_keep.setValue(processing.get('jsona_backup_keep', 10))
        xml_mapping = processing.get('xml_mapping', {})
        self.edit_xml_template_path.setText(str(xml_mapping.get('template_path', '') or ''))
        if self.edit_xml_template_path.text().strip():
            try:
                self._analyze_xml_template()
            except Exception:
                pass
        self.combo_xml_artist_field.setCurrentText(str(xml_mapping.get('artist_field_path', 'artist') or 'artist'))
        self.spin_xml_artist_tag_index.setValue(int(xml_mapping.get('artist_tag_index', 1) or 1))
        self.combo_xml_character1_field.setCurrentText(str(xml_mapping.get('character_1_field_path', 'character_1') or 'character_1'))
        self.spin_xml_character1_tag_index.setValue(int(xml_mapping.get('character_1_tag_index', 2) or 2))
        self.combo_xml_character2_field.setCurrentText(str(xml_mapping.get('character_2_field_path', 'character_2') or 'character_2'))
        self.spin_xml_character2_tag_index.setValue(int(xml_mapping.get('character_2_tag_index', 3) or 3))
        self._load_xml_custom_mappings(xml_mapping.get('custom_mappings', []))

        # Custom tags
        custom_tags = self.config.get('custom_tags', {})
        self.check_append_tags.setChecked(custom_tags.get('enabled', False))
        self.edit_custom_tags.setPlainText(custom_tags.get('tags', ''))

        # Output
        output = self.config.get('output', {})
        self.edit_output_dir.setText(output.get('base_dir', './output'))
        discard_action = output.get('discard_action', 'trash')
        if discard_action == 'delete':
            self.combo_discard_action.setCurrentText(tr('delete_permanently'))
        else:
            self.combo_discard_action.setCurrentText(tr('move_to_trash'))

    def get_settings(self) -> dict:
        """Get current settings"""
        settings = {
            'data_source': {},
            'processing': {},
            'vlm': {},
            'custom_tags': {},
            'output': {},
            'prefilter': self.config.get('prefilter', {}),
            'progress': self.config.get('progress', {}),
            'report': self.config.get('report', {}),
            'scoring': self.config.get('scoring', {})
        }

        # Data source
        if self.combo_source_type.currentText() == tr('local_parquet'):
            parquet_files = [self.list_parquet_files.item(i).text()
                           for i in range(self.list_parquet_files.count())]
            settings['data_source'] = {
                'type': 'local_parquet',
                'local_parquet': {
                    'parquet_files': parquet_files,
                    'extract_dir': self.edit_extract_dir.text(),
                    'num_samples': self.spin_local_samples.value()
                }
            }
        else:
            settings['data_source'] = {
                'type': 'streaming',
                'streaming': {
                    'dataset_id': self.edit_dataset_id.text(),
                    'split': self.edit_split.text(),
                    'hf_token': self.edit_hf_token.text() or None,
                    'num_samples': self.spin_num_samples.value(),
                    'extract_dir': self.edit_streaming_extract_dir.text()
                }
            }

        # Processing
        settings['processing'] = {
            'use_custom_dir': self.radio_use_custom_dir.isChecked(),
            'custom_input_dir': self.edit_custom_input_dir.text(),
            'thread_count': self.spin_processing_threads.value(),
            'preload_count': self.spin_preload_count.value(),
            'control_types': {
                'canny': self.check_canny.isChecked(),
                'openpose': self.check_openpose.isChecked(),
                'depth': self.check_depth.isChecked()
            },
            'model_config': {
                'openpose': {
                    'type': self._get_openpose_model_type(),
                    'yolo_version': self.combo_yolo_version.currentText() if self._get_openpose_model_type() in ['ViTPose', 'SDPose-Wholebody'] else None,
                    'yolo_model_type': self.combo_yolo_model_type.currentText() if self._get_openpose_model_type() in ['ViTPose', 'SDPose-Wholebody'] else None,
                    'custom_path': self.edit_openpose_path.text() if self._get_openpose_model_type() == 'Custom Path' else None
                },
                'depth': {
                    'type': self.combo_depth_model.currentText(),
                    'custom_path': self.edit_depth_path.text() if self.combo_depth_model.currentText() == 'Custom Path' else None
                }
            },
            'retry_strategy': {
                'enabled': self.check_enable_retry.isChecked(),
                'max_retries': self.spin_max_retries.value(),
                'retry_threshold': 40,
                'random_offset_range': 20
            },
            'parallel_threads': self.spin_parallel_threads.value(),
            'auto_pass_no_review': self.check_auto_pass_no_review.isChecked(),
            'single_jsona': self.check_single_jsona.isChecked(),
            'jsona_backup_every_entries': self.spin_jsona_backup_every_entries.value(),
            'jsona_backup_every_seconds': self.spin_jsona_backup_every_seconds.value(),
            'jsona_backup_keep': self.spin_jsona_backup_keep.value(),
            'xml_mapping': self._build_xml_mapping_config(),
            'unattended_mode': self.check_unattended_mode.isChecked(),
            'unattended_inbox_max_mb': self.spin_unattended_inbox_max_mb.value(),
            'unattended_inbox_full_action': (
                'pause' if self.combo_unattended_inbox_full_action.currentIndex() == 0 else 'stop'
            ),
        }

        settings['vlm'] = {
            'backend': self.combo_vlm_backend.currentText(),
            'base_url': self.edit_vlm_base_url.text().strip(),
            'model': self.edit_vlm_model.text().strip(),
            'api_key': self.edit_vlm_api_key.text(),
            'timeout_seconds': int(self.spin_vlm_timeout.value()),
        }

        # Apply selected quality profile
        if 'scoring' not in settings or not isinstance(settings['scoring'], dict):
            settings['scoring'] = {}

        # Map translated text back to English key
        profile_text = self.combo_quality_profile.currentText()
        if profile_text == tr('general') or '通用' in profile_text:
            profile_key = 'general'
        elif profile_text == tr('anime') or '动漫' in profile_text:
            profile_key = 'anime'
        else:
            profile_key = 'general'  # Default

        settings['scoring']['active_profile'] = profile_key

        # Update the active profile's thresholds with UI values
        if 'profiles' not in settings['scoring']:
            settings['scoring']['profiles'] = self.config.get('scoring', {}).get('profiles', {})

        if profile_key not in settings['scoring']['profiles']:
            settings['scoring']['profiles'][profile_key] = {}

        # Save per-control-type thresholds
        settings['scoring']['profiles'][profile_key]['canny_auto_accept'] = self.spin_canny_accept.value()
        settings['scoring']['profiles'][profile_key]['canny_auto_reject'] = self.spin_canny_reject.value()
        settings['scoring']['profiles'][profile_key]['pose_auto_accept'] = self.spin_pose_accept.value()
        settings['scoring']['profiles'][profile_key]['pose_auto_reject'] = self.spin_pose_reject.value()
        settings['scoring']['profiles'][profile_key]['depth_auto_accept'] = self.spin_depth_accept.value()
        settings['scoring']['profiles'][profile_key]['depth_auto_reject'] = self.spin_depth_reject.value()

        # Also keep the general auto_accept/auto_reject for backward compatibility
        settings['scoring']['profiles'][profile_key]['auto_accept'] = self.spin_canny_accept.value()
        settings['scoring']['profiles'][profile_key]['auto_reject'] = self.spin_canny_reject.value()

        # Custom tags
        settings['custom_tags'] = {
            'enabled': self.check_append_tags.isChecked(),
            'tags': self.edit_custom_tags.toPlainText()
        }

        # Output
        settings['output'] = {
            'base_dir': self.edit_output_dir.text(),
            'discard_action': 'delete' if self.combo_discard_action.currentText() == tr('delete_permanently') else 'trash',
            'trash_dir_name': self.config.get('output', {}).get('trash_dir_name', '_trash'),
            'accepted_dir_name': self.config.get('output', {}).get('accepted_dir_name', 'accepted')
        }

        return settings

    def _reset_to_defaults(self):
        """Reset all settings to default values from config.json"""
        from PyQt5.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            '确认重置',
            '确定要重置所有设置为默认配置吗？\n\n这将清除所有自定义设置（包括路径、阈值等）。',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Delete settings.json to force reload from config.json
            settings_file = self._get_settings_file_path()
            if os.path.exists(settings_file):
                try:
                    os.remove(settings_file)
                except Exception as e:
                    print(f"Failed to delete settings file: {e}")

            # Reload config
            previous_state = self._settings_tracking_suspended
            self._settings_tracking_suspended = True
            try:
                self._load_config()
            finally:
                self._settings_tracking_suspended = previous_state

            self.save_settings(history_reason='reset_defaults', force_history=True)

            QMessageBox.information(
                self,
                '重置完成',
                '设置已重置为默认配置。\n\n请重新配置您需要的选项。'
            )

    def _update_jsona_statistics(self):
        """Update JSONA file statistics display"""
        settings = self.get_settings()
        output_dir = settings['output']['base_dir']
        use_single_jsona = settings.get('processing', {}).get('single_jsona', False)

        # Count entries in each JSONA file from the actual write location.
        canny_count = self._count_jsona_entries(os.path.join(output_dir, 'canny.jsona'), expected_name='canny')
        pose_count = self._count_jsona_entries(os.path.join(output_dir, 'pose.jsona'), expected_name='pose')
        depth_count = self._count_jsona_entries(os.path.join(output_dir, 'depth.jsona'), expected_name='depth')
        metadata_count = self._count_jsona_entries(os.path.join(output_dir, 'metadata.jsona'), expected_name='metadata')
        tag_count = self._count_jsona_entries(os.path.join(output_dir, 'tag.jsona'), expected_name='tag')
        nl_count = self._count_jsona_entries(os.path.join(output_dir, 'nl.jsona'), expected_name='nl')
        xml_count = self._count_jsona_entries(os.path.join(output_dir, 'xml.jsona'), expected_name='xml')

        # Update labels and active mode hint.
        self.label_canny_count.setText(f'Canny: {canny_count} 张图')
        self.label_pose_count.setText(f'Pose: {pose_count} 张图')
        self.label_depth_count.setText(f'Depth: {depth_count} 张图')
        self.label_metadata_count.setText(f'Metadata: {metadata_count} 条')
        self.label_tag_count.setText(f'Tag: {tag_count} 条')
        self.label_nl_count.setText(f'NL: {nl_count} 条')
        self.label_xml_count.setText(f'XML: {xml_count} 条')

        if use_single_jsona:
            self.label_jsona_mode.setText('当前模式: 合并写入 metadata.jsona')
            self.btn_reset_metadata_jsona.setEnabled(True)
            self.btn_reset_canny_jsona.setEnabled(False)
            self.btn_reset_pose_jsona.setEnabled(False)
            self.btn_reset_depth_jsona.setEnabled(False)
            self.btn_reset_tag_jsona.setEnabled(True)
            self.btn_reset_nl_jsona.setEnabled(True)
            self.btn_reset_xml_jsona.setEnabled(True)
        else:
            self.label_jsona_mode.setText('当前模式: 分类型 JSONA')
            self.btn_reset_metadata_jsona.setEnabled(False)
            self.btn_reset_canny_jsona.setEnabled(True)
            self.btn_reset_pose_jsona.setEnabled(True)
            self.btn_reset_depth_jsona.setEnabled(True)
            self.btn_reset_tag_jsona.setEnabled(True)
            self.btn_reset_nl_jsona.setEnabled(True)
            self.btn_reset_xml_jsona.setEnabled(True)

    def _count_jsona_entries(self, jsona_path: str, expected_name: str = None) -> int:
        """Count valid entries in a JSONA file."""
        if not os.path.exists(jsona_path):
            return 0

        try:
            report = JsonaBackupManager().inspect_jsona_file(jsona_path, expected_name=expected_name, check_files=False)
            return report.get('valid_entry_count', report.get('entry_count', 0))
        except Exception as e:
            print(f"Error counting jsona entries: {e}")
            return 0

    def _reset_jsona_file(self, control_type: str):
        """Reset (clear) a specific jsona file with confirmation"""
        reply = QMessageBox.question(
            self,
            '确认重置',
            f'确定要清空 {control_type.upper()} 的 JSONA 文件吗？\n\n此操作不可恢复！',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            settings = self.get_settings()
            output_dir = settings['output']['base_dir']

            # Determine jsona filename
            if control_type == 'canny':
                jsona_file = 'canny.jsona'
            elif control_type == 'pose':
                jsona_file = 'pose.jsona'
            elif control_type == 'depth':
                jsona_file = 'depth.jsona'
            elif control_type == 'metadata':
                jsona_file = 'metadata.jsona'
            elif control_type == 'tag':
                jsona_file = 'tag.jsona'
            elif control_type == 'nl':
                jsona_file = 'nl.jsona'
            elif control_type == 'xml':
                jsona_file = 'xml.jsona'
            else:
                return

            jsona_path = os.path.join(output_dir, jsona_file)

            try:
                if os.path.exists(jsona_path):
                    backup_manager = JsonaBackupManager()
                    restore_point = backup_manager.replace_entries(jsona_path, [], reason='manual_reset')

                    message = f'{control_type.upper()} 的 JSONA 文件已清空。'
                    if restore_point:
                        message += f'\n\n恢复点备份：\n{restore_point}'
                    QMessageBox.information(self, '重置完成', message)

                    # Update statistics
                    self._update_jsona_statistics()
                else:
                    QMessageBox.information(
                        self,
                        '文件不存在',
                        f'{control_type.upper()} 的 JSONA 文件不存在。'
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    '重置失败',
                    f'清空 JSONA 文件时出错：\n{str(e)}'
                )
