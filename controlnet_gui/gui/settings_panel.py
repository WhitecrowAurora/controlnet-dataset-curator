"""
ControlNet GUI - Settings Panel
Data source selection, thread count, output path configuration
"""

import os
import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QSpinBox, QComboBox,
    QCheckBox, QFileDialog, QFormLayout, QListWidget, QTextEdit, QProgressBar,
    QDialog, QDialogButtonBox, QListWidgetItem, QMessageBox, QScrollArea, QRadioButton
)
from PyQt5.QtCore import pyqtSignal, QThread, pyqtSlot, Qt
from ..language import tr
from ..core.parquet_source import ParquetDataSource, StreamingDataSource
from ..core.jsona_backup import JsonaBackupManager


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

        # Track if we've already checked dependencies/models this session
        self._vitpose_deps_checked = False
        self._vitpose_models_checked = {}  # {model_type: bool}

        self._setup_ui()
        self._load_config()

        # Load saved settings
        self.load_settings()

        # Connect signals to auto-save
        self._connect_autosave_signals()

        # Update JSONA statistics
        self._update_jsona_statistics()

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

    def save_settings(self):
        """Save current settings to config file"""
        config = {
            # Data source
            'data_source_type': self.combo_source_type.currentIndex(),
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

            # Processing settings
            'use_custom_dir': self.radio_use_custom_dir.isChecked(),
            'custom_input_dir': self.edit_custom_input_dir.text(),
            'processing_threads': self.spin_processing_threads.value(),
            'preload_count': self.spin_preload_count.value(),

            # Control types
            'canny_enabled': self.check_canny.isChecked(),
            'openpose_enabled': self.check_openpose.isChecked(),
            'depth_enabled': self.check_depth.isChecked(),

            # Quality profile and thresholds
            'quality_profile': self.combo_quality_profile.currentText(),
            'canny_accept': self.spin_canny_accept.value(),
            'canny_reject': self.spin_canny_reject.value(),
            'pose_accept': self.spin_pose_accept.value(),
            'pose_reject': self.spin_pose_reject.value(),
            'depth_accept': self.spin_depth_accept.value(),
            'depth_reject': self.spin_depth_reject.value(),

            # Model settings
            'openpose_model': self.combo_openpose_model.currentText(),
            'openpose_yolo_version': self.combo_yolo_version.currentText(),
            'openpose_yolo_model_type': self.combo_yolo_model_type.currentText(),
            'openpose_custom_path': self.edit_openpose_path.text(),
            'depth_model': self.combo_depth_model.currentText(),
            'depth_custom_path': self.edit_depth_path.text(),

            # Advanced settings
            'parallel_threads': self.spin_parallel_threads.value(),
            'auto_pass_no_review': self.check_auto_pass_no_review.isChecked(),
            'single_jsona': self.check_single_jsona.isChecked(),
            'jsona_backup_every_entries': self.spin_jsona_backup_every_entries.value(),
            'jsona_backup_every_seconds': self.spin_jsona_backup_every_seconds.value(),
            'jsona_backup_keep': self.spin_jsona_backup_keep.value(),
            'enable_retry': self.check_enable_retry.isChecked(),
            'max_retries': self.spin_max_retries.value(),

            # Custom tags
            'append_tags': self.check_append_tags.isChecked(),
            'custom_tags': self.edit_custom_tags.toPlainText(),

            # Output
            'output_dir': self.edit_output_dir.text(),
            'discard_action': self.combo_discard_action.currentText(),
        }

        config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'settings.json')
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def load_settings(self):
        """Load settings from config file"""
        config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'settings.json')

        if not os.path.exists(config_file):
            return

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Restore settings
            if 'data_source_type' in config:
                self.combo_source_type.setCurrentIndex(config['data_source_type'])
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

    def _connect_autosave_signals(self):
        """Connect signals to auto-save settings when changed"""
        # Data source
        self.combo_source_type.currentIndexChanged.connect(lambda: self.save_settings())
        self.edit_dataset_id.textChanged.connect(lambda: self.save_settings())
        self.edit_split.textChanged.connect(lambda: self.save_settings())
        self.edit_hf_token.textChanged.connect(lambda: self.save_settings())
        self.spin_num_samples.valueChanged.connect(lambda: self.save_settings())
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

        self.label_jsona_mode = QLabel('当前模式: 分类型 JSONA')
        self.label_jsona_mode.setStyleSheet('color: #888; font-size: 11px;')
        stats_layout.addWidget(self.label_jsona_mode)

        # Refresh button
        btn_refresh_stats = QPushButton('刷新统计')
        btn_refresh_stats.clicked.connect(self._update_jsona_statistics)
        stats_layout.addWidget(btn_refresh_stats)

        stats_layout.addStretch()
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

    def _remove_parquet_file(self):
        """Remove selected parquet file from list"""
        current_row = self.list_parquet_files.currentRow()
        if current_row >= 0:
            self.list_parquet_files.takeItem(current_row)

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
            'jsona_backup_keep': self.spin_jsona_backup_keep.value()
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
            import os
            settings_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'settings.json')
            if os.path.exists(settings_file):
                try:
                    os.remove(settings_file)
                except Exception as e:
                    print(f"Failed to delete settings file: {e}")

            # Reload config
            self._load_config()

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

        # Update labels and active mode hint.
        self.label_canny_count.setText(f'Canny: {canny_count} 张图')
        self.label_pose_count.setText(f'Pose: {pose_count} 张图')
        self.label_depth_count.setText(f'Depth: {depth_count} 张图')
        self.label_metadata_count.setText(f'Metadata: {metadata_count} 条')

        if use_single_jsona:
            self.label_jsona_mode.setText('当前模式: 合并写入 metadata.jsona')
            self.btn_reset_metadata_jsona.setEnabled(True)
            self.btn_reset_canny_jsona.setEnabled(False)
            self.btn_reset_pose_jsona.setEnabled(False)
            self.btn_reset_depth_jsona.setEnabled(False)
        else:
            self.label_jsona_mode.setText('当前模式: 分类型 JSONA')
            self.btn_reset_metadata_jsona.setEnabled(False)
            self.btn_reset_canny_jsona.setEnabled(True)
            self.btn_reset_pose_jsona.setEnabled(True)
            self.btn_reset_depth_jsona.setEnabled(True)

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
