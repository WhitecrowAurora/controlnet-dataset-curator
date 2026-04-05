"""
ControlNet GUI - 单行图片组件
每行显示：原图 + 候选图 + 确认/废弃按钮
"""

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QFrame, QPushButton, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QCursor

import numpy as np


class ImageLabel(QLabel):
    """可点击的图片标签"""

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #444;
                background-color: #1a1a1a;
                border-radius: 4px;
            }
            QLabel:hover {
                border: 2px solid #0078d4;
                background-color: #2a2a2a;
            }
        """)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self._pixmap = None
        self._source_image = None

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        self._pixmap = pixmap

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def set_highlight(self, highlight: bool, color: str = "#00ff00"):
        """设置高亮边框"""
        if highlight:
            self.setStyleSheet(f"""
                QLabel {{
                    border: 3px solid {color};
                    background-color: #1a1a1a;
                    border-radius: 4px;
                }}
                QLabel:hover {{
                    border: 3px solid {color};
                    background-color: #2a2a2a;
                }}
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #444;
                    background-color: #1a1a1a;
                    border-radius: 4px;
                }
                QLabel:hover {
                    border: 2px solid #0078d4;
                    background-color: #2a2a2a;
                }
            """)


class ScoreLabel(QLabel):
    """分数显示标签"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("color: #888; font-size: 11px; padding: 2px;")

    def set_score(self, score: float, is_best: bool = False):
        """设置分数显示"""
        if is_best:
            self.setText(f"★ {score:.1f}")
            self.setStyleSheet("color: #00ff00; font-size: 11px; font-weight: bold; padding: 2px;")
        else:
            self.setText(f"{score:.1f}")
            self.setStyleSheet("color: #888; font-size: 11px; padding: 2px;")


class ImageRowWidget(QWidget):
    """
    单条审核记录组件

    支持三种显示模式：
    - 固定单行
    - 自适应单行
    - 双行 2x2
    """

    variant_selected = pyqtSignal(int, int)
    variant_confirmed = pyqtSignal(int, int)
    row_discarded = pyqtSignal(int)

    DISPLAY_MODE_FIXED_SINGLE = 'fixed_single'
    DISPLAY_MODE_ADAPTIVE_SINGLE = 'adaptive_single'
    DISPLAY_MODE_TWO_ROW = 'two_row'

    def __init__(self, row_index: int, parent=None):
        super().__init__(parent)
        self.row_index = row_index
        self._data = None
        self._selected_index = -1
        self._best_index = -1
        self._is_active = False
        self._display_mode = self.DISPLAY_MODE_FIXED_SINGLE

        self._root_layout = QVBoxLayout(self)
        self._content_widget = None
        self._main_content_layout = None
        self._reset_layout_refs()
        self._setup_ui()

    @classmethod
    def normalize_display_mode(cls, mode: str) -> str:
        normalized = str(mode or cls.DISPLAY_MODE_FIXED_SINGLE).strip().lower()
        alias = {
            cls.DISPLAY_MODE_FIXED_SINGLE: cls.DISPLAY_MODE_FIXED_SINGLE,
            'fixed': cls.DISPLAY_MODE_FIXED_SINGLE,
            'single': cls.DISPLAY_MODE_FIXED_SINGLE,
            cls.DISPLAY_MODE_ADAPTIVE_SINGLE: cls.DISPLAY_MODE_ADAPTIVE_SINGLE,
            'adaptive': cls.DISPLAY_MODE_ADAPTIVE_SINGLE,
            'responsive': cls.DISPLAY_MODE_ADAPTIVE_SINGLE,
            cls.DISPLAY_MODE_TWO_ROW: cls.DISPLAY_MODE_TWO_ROW,
            'double_row': cls.DISPLAY_MODE_TWO_ROW,
            'grid': cls.DISPLAY_MODE_TWO_ROW,
        }
        return alias.get(normalized, cls.DISPLAY_MODE_FIXED_SINGLE)

    def set_display_mode(self, mode: str):
        normalized = self.normalize_display_mode(mode)
        if normalized == self._display_mode:
            return
        self._display_mode = normalized
        self._rebuild_content_layout()

    def get_display_mode(self) -> str:
        return self._display_mode

    def _setup_ui(self):
        """设置基础 UI 容器"""
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self._root_layout.setSpacing(0)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._rebuild_content_layout()

    def _reset_layout_refs(self):
        self.lbl_original = None
        self.lbl_original_title = None
        self.variant_labels = []
        self.score_labels = []
        self.preset_labels = []
        self.variant_containers = []
        self.btn_confirm = None
        self.btn_discard = None

    def _rebuild_content_layout(self):
        current_data = self._data
        previous_selected = self._selected_index
        was_active = self._is_active

        if self._content_widget is not None:
            self._root_layout.removeWidget(self._content_widget)
            self._content_widget.deleteLater()

        self._reset_layout_refs()
        self._content_widget = QWidget(self)
        self._content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self._root_layout.addWidget(self._content_widget)

        if self._display_mode == self.DISPLAY_MODE_TWO_ROW:
            self._build_two_row_layout()
        else:
            self._build_single_row_layout()

        self._apply_layout_mode_sizes()
        self._apply_row_style()

        if current_data is not None:
            self.load_data(current_data)
            if 0 <= previous_selected < self.get_variant_count():
                self.select_variant(previous_selected, emit_signal=False)
            self.set_active(was_active)

    def _build_single_row_layout(self):
        main_layout = QHBoxLayout(self._content_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        main_layout.addWidget(self._create_original_container(), 0, Qt.AlignTop)
        main_layout.addWidget(self._create_separator(), 0, Qt.AlignVCenter)

        for index in range(4):
            container, img_label, score_label, preset_label = self._create_variant_container(index)
            self.variant_labels.append(img_label)
            self.score_labels.append(score_label)
            self.preset_labels.append(preset_label)
            self.variant_containers.append(container)
            main_layout.addWidget(container, 0, Qt.AlignTop)

        main_layout.addWidget(self._create_button_container(), 0, Qt.AlignVCenter)
        main_layout.addStretch(1)
        self._main_content_layout = main_layout

    def _build_two_row_layout(self):
        main_layout = QHBoxLayout(self._content_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        main_layout.addWidget(self._create_original_container(), 0, Qt.AlignTop)
        main_layout.addWidget(self._create_separator(), 0, Qt.AlignVCenter)

        variant_grid_widget = QWidget()
        variant_grid_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        variant_grid = QGridLayout(variant_grid_widget)
        variant_grid.setSpacing(8)
        variant_grid.setContentsMargins(0, 0, 0, 0)
        variant_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        for index in range(4):
            container, img_label, score_label, preset_label = self._create_variant_container(index)
            self.variant_labels.append(img_label)
            self.score_labels.append(score_label)
            self.preset_labels.append(preset_label)
            self.variant_containers.append(container)
            variant_grid.addWidget(container, index // 2, index % 2)

        main_layout.addWidget(variant_grid_widget, 0, Qt.AlignTop)
        main_layout.addWidget(self._create_button_container(), 0, Qt.AlignVCenter)
        main_layout.addStretch(1)
        self._main_content_layout = main_layout

    def _create_original_container(self) -> QWidget:
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout = QVBoxLayout(container)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        self.lbl_original = ImageLabel()
        self.lbl_original.clicked.connect(self._on_original_clicked)

        self.lbl_original_title = QLabel("原图")
        self.lbl_original_title.setAlignment(Qt.AlignCenter)
        self.lbl_original_title.setStyleSheet("color: #aaa; font-size: 10px;")

        layout.addWidget(self.lbl_original)
        layout.addWidget(self.lbl_original_title)
        return container

    def _create_separator(self) -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("color: #444;")
        return separator

    def _create_variant_container(self, index: int):
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout = QVBoxLayout(container)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        img_label = ImageLabel()
        img_label.clicked.connect(lambda idx=index: self._on_variant_clicked(idx))

        score_label = ScoreLabel()

        preset_label = QLabel(f"变体{index + 1}")
        preset_label.setAlignment(Qt.AlignCenter)
        preset_label.setStyleSheet("color: #666; font-size: 9px;")

        layout.addWidget(img_label)
        layout.addWidget(score_label)
        layout.addWidget(preset_label)
        return container, img_label, score_label, preset_label

    def _create_button_container(self) -> QWidget:
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 0, 10, 0)

        self.btn_confirm = QPushButton("✓ 确认")
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
        """)
        self.btn_confirm.clicked.connect(self._on_confirm_clicked)

        self.btn_discard = QPushButton("✗ 废弃")
        self.btn_discard.setStyleSheet("""
            QPushButton {
                background-color: #d13438;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #a72b2e;
            }
        """)
        self.btn_discard.clicked.connect(self._on_discard_clicked)

        layout.addStretch()
        layout.addWidget(self.btn_confirm)
        layout.addWidget(self.btn_discard)
        layout.addStretch()
        return container

    def _fixed_single_sizes(self):
        return {
            'original_width': 180,
            'original_height': 180,
            'variant_size': 160,
            'button_width': 84,
            'button_height': 40,
        }

    def _adaptive_single_sizes(self):
        width = max(self.width(), 760)
        usable_width = max(width - 170, 680)
        variant_size = int((usable_width - 210) / 5.15)
        variant_size = max(96, min(variant_size, 170))
        original_size = max(110, min(int(variant_size * 1.12), 200))
        button_width = 84 if variant_size >= 120 else 80
        button_height = 40
        return {
            'original_width': original_size,
            'original_height': original_size,
            'variant_size': variant_size,
            'button_width': button_width,
            'button_height': button_height,
        }

    def _two_row_sizes(self):
        width = max(self.width(), 720)
        usable_width = max(width - 210, 520)
        variant_size = int((usable_width - 180) / 2.35)
        variant_size = max(104, min(variant_size, 172))
        original_width = max(120, min(int(variant_size * 1.18), 220))
        original_height = variant_size * 2 + 34
        button_width = 88 if variant_size >= 132 else 82
        button_height = 40
        return {
            'original_width': original_width,
            'original_height': original_height,
            'variant_size': variant_size,
            'button_width': button_width,
            'button_height': button_height,
        }

    def _apply_layout_mode_sizes(self):
        if self.lbl_original is None:
            return

        if self._display_mode == self.DISPLAY_MODE_ADAPTIVE_SINGLE:
            sizes = self._adaptive_single_sizes()
        elif self._display_mode == self.DISPLAY_MODE_TWO_ROW:
            sizes = self._two_row_sizes()
        else:
            sizes = self._fixed_single_sizes()

        self.lbl_original.setFixedSize(sizes['original_width'], sizes['original_height'])
        for label in self.variant_labels:
            label.setFixedSize(sizes['variant_size'], sizes['variant_size'])

        if self.btn_confirm is not None:
            self.btn_confirm.setFixedSize(sizes['button_width'], sizes['button_height'])
        if self.btn_discard is not None:
            self.btn_discard.setFixedSize(sizes['button_width'], sizes['button_height'])

        self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._display_mode in {self.DISPLAY_MODE_ADAPTIVE_SINGLE, self.DISPLAY_MODE_TWO_ROW}:
            self._apply_layout_mode_sizes()
            self._refresh_loaded_images()

    def _apply_row_style(self):
        """根据当前状态更新行样式"""
        quality_color = self._get_quality_color()

        if self._is_active:
            self.setStyleSheet(f"""
                ImageRowWidget {{
                    background-color: #1a1a1a;
                    border-bottom: 1px solid #333;
                    border-left: 3px solid #0078d4;
                    border-top: 2px solid {quality_color};
                }}
                ImageRowWidget:hover {{
                    background-color: #1f1f1f;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                ImageRowWidget {{
                    background-color: #0d0d0d;
                    border-bottom: 1px solid #333;
                    border-left: 3px solid transparent;
                    border-top: 2px solid {quality_color};
                }}
                ImageRowWidget:hover {{
                    background-color: #151515;
                }}
            """)

    def _get_quality_color(self) -> str:
        """根据最佳候选图分数返回颜色"""
        if not self._data or 'variants' not in self._data:
            return "#333"

        variants = self._data.get('variants', [])
        if not variants:
            return "#333"

        scores = []
        for variant in variants:
            score = variant.get('score_10', 0)
            if score > 0:
                scores.append(score)

        if not scores:
            return "#333"

        best_score = max(scores)
        if best_score >= 8.0:
            return "#4CAF50"
        if best_score >= 6.0:
            return "#FFC107"
        if best_score >= 4.0:
            return "#FF9800"
        return "#F44336"

    def load_data(self, data: dict):
        """
        加载一条审核记录的数据

        Args:
            data: {
                'original_image': PIL.Image,
                'original_path': str,
                'basename': str,
                'control_type': str,
                'variants': [...]
            }
        """
        self._reset_loaded_data(data)
        self._show_original_if_available(data)
        variants = self._visible_variants(data)
        self._show_all_variant_containers()
        self._populate_variant_slots(variants)
        self._hide_unused_variant_slots(len(variants))
        self._resolve_best_variant_index(variants)
        self._preselect_single_variant_if_needed()
        self._refresh_variant_highlights()
        self._apply_row_style()

    def _reset_loaded_data(self, data: dict):
        self._data = data
        self._selected_index = -1
        self._best_index = -1

    def _show_original_if_available(self, data: dict):
        image = data.get('original_image')
        if image is not None:
            self._set_image(self.lbl_original, image)
            return
        self._clear_image_label(self.lbl_original)

    def _visible_variants(self, data: dict):
        return list(data.get('variants', [])[:4])

    def _show_all_variant_containers(self):
        for container in self.variant_containers:
            container.show()

    def _populate_variant_slots(self, variants):
        for index, variant in enumerate(variants):
            self._populate_variant_slot(index, variant)

    def _populate_variant_slot(self, index: int, variant: dict):
        image = variant.get('image')
        if image is not None:
            self._set_image(self.variant_labels[index], image)
        else:
            self._clear_image_label(self.variant_labels[index])

        score_10 = variant.get('score_10', 1.0)
        is_best = variant.get('is_best', False)
        self.score_labels[index].set_score(score_10, is_best)

        preset_name = variant.get('preset_name', f"变体{index + 1}")
        self.preset_labels[index].setText(preset_name)

        if is_best:
            self._best_index = index

    def _hide_unused_variant_slots(self, used_count: int):
        for index in range(used_count, len(self.variant_containers)):
            self._clear_image_label(self.variant_labels[index])
            self.score_labels[index].clear()
            self.preset_labels[index].clear()
            self.variant_containers[index].hide()

    def _resolve_best_variant_index(self, variants):
        if self._best_index != -1 or not variants:
            return
        scores = [variant.get('score_10', 1.0) for variant in variants]
        self._best_index = scores.index(max(scores)) if scores else -1

    def _preselect_single_variant_if_needed(self):
        if self.get_variant_count() == 1:
            self._selected_index = 0

    def _set_image(self, label: ImageLabel, pil_image):
        """将 PIL 图像缩放后显示到标签"""
        if label is None or pil_image is None:
            return

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        label._source_image = pil_image
        arr = np.array(pil_image).copy()
        h, w, ch = arr.shape
        bytes_per_line = ch * w
        q_image = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_image).scaled(
            max(1, label.width() - 4),
            max(1, label.height() - 4),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)

    def _clear_image_label(self, label: ImageLabel):
        if label is None:
            return
        label.clear()
        label._source_image = None

    def _refresh_loaded_images(self):
        if not self._data:
            return

        self._show_original_if_available(self._data)
        variants = self._visible_variants(self._data)
        for index, label in enumerate(self.variant_labels):
            if index < len(variants) and variants[index].get('image') is not None:
                self._set_image(label, variants[index]['image'])
            else:
                self._clear_image_label(label)

    def _on_original_clicked(self):
        """原图点击预留入口"""

    def _on_variant_clicked(self, variant_index: int):
        self.select_variant(variant_index, emit_signal=True)

    def _on_confirm_clicked(self):
        self.confirm_selected()

    def _on_discard_clicked(self):
        self.discard_row()

    def get_selected_variant(self) -> int:
        """获取当前选中的候选图索引"""
        return self._selected_index

    def get_data(self) -> dict:
        """获取当前行数据"""
        return self._data

    def set_row_index(self, index: int):
        """更新当前行索引"""
        self.row_index = index

    def set_active(self, active: bool):
        """设置当前行是否为键盘焦点行"""
        self._is_active = active
        self._apply_row_style()

    def get_variant_count(self) -> int:
        """获取可用候选图数量"""
        if not self._data:
            return 0
        return min(len(self._data.get('variants', [])), 4)

    def _refresh_variant_highlights(self):
        """刷新候选图高亮状态"""
        variant_count = self.get_variant_count()
        self.btn_confirm.setEnabled(self._selected_index >= 0)

        for index, label in enumerate(self.variant_labels):
            if index >= variant_count:
                label.set_highlight(False)
                continue

            if index == self._selected_index:
                label.set_highlight(True, "#0078d4")
            elif index == self._best_index:
                label.set_highlight(True, "#00ff00")
            else:
                label.set_highlight(False)

    def select_variant(self, variant_index: int, emit_signal: bool = True) -> bool:
        """程序化选择某个候选图"""
        if variant_index < 0 or variant_index >= self.get_variant_count():
            return False

        self._selected_index = variant_index
        self._refresh_variant_highlights()

        if emit_signal:
            self.variant_selected.emit(self.row_index, variant_index)
        return True

    def select_next_variant(self, step: int) -> bool:
        """按步进循环选择前后候选图"""
        variant_count = self.get_variant_count()
        if variant_count == 0:
            return False

        if self._selected_index < 0:
            next_index = 0 if step >= 0 else variant_count - 1
        else:
            next_index = (self._selected_index + step) % variant_count
        return self.select_variant(next_index, emit_signal=True)

    def confirm_selected(self) -> bool:
        """确认当前选中的候选图"""
        if self._selected_index < 0:
            return False
        self.variant_confirmed.emit(self.row_index, self._selected_index)
        return True

    def discard_row(self) -> bool:
        """废弃当前行"""
        self.row_discarded.emit(self.row_index)
        return True

    def get_best_variant_index(self) -> int:
        """获取最佳候选图索引"""
        return self._best_index

    def clear(self):
        """清空当前显示"""
        self._clear_image_label(self.lbl_original)
        for label in self.variant_labels:
            self._clear_image_label(label)
        for score_label in self.score_labels:
            score_label.clear()
        for preset_label in self.preset_labels:
            preset_label.clear()
        self._data = None
        self._selected_index = -1
        self._best_index = -1

