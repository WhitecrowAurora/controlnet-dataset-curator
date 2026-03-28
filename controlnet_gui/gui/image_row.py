"""
ControlNet GUI - 单行图片组件
每行显示：原图 + 4张不同阈值的Canny图 + 确认/废弃按钮
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QCursor

import numpy as np
from PIL import Image


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
    单行图片组件

    布局: [原图] [Canny1+分数] [Canny2+分数] [Canny3+分数] [Canny4+分数] [确认按钮] [废弃按钮]

    信号:
        variant_selected: (row_index, variant_index) 用户选择了某个变体
        variant_confirmed: (row_index, variant_index) 用户确认了某个变体
        row_discarded: (row_index) 用户废弃了整行
    """

    variant_selected = pyqtSignal(int, int)  # row_index, variant_index
    variant_confirmed = pyqtSignal(int, int)  # row_index, variant_index
    row_discarded = pyqtSignal(int)  # row_index
    
    def __init__(self, row_index: int, parent=None):
        super().__init__(parent)
        self.row_index = row_index
        self._data = None
        self._selected_index = -1
        self._best_index = -1
        self._is_active = False
        
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI布局"""
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(self._create_original_container())
        main_layout.addWidget(self._create_separator())

        self.variant_labels = []
        self.score_labels = []
        self.preset_labels = []
        self.variant_containers = []
        self._add_variant_containers(main_layout)
        main_layout.addWidget(self._create_button_container())
        self._apply_row_style()

    def _create_original_container(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_original = ImageLabel()
        self.lbl_original.setFixedSize(180, 180)
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

    def _add_variant_containers(self, main_layout: QHBoxLayout):
        for i in range(4):
            container, img_label, score_label, preset_label = self._create_variant_container(i)
            self.variant_labels.append(img_label)
            self.score_labels.append(score_label)
            self.preset_labels.append(preset_label)
            self.variant_containers.append(container)
            main_layout.addWidget(container)

    def _create_variant_container(self, index: int):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        img_label = ImageLabel()
        img_label.setFixedSize(160, 160)
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
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 0, 10, 0)

        self.btn_confirm = QPushButton("✓ 确认")
        self.btn_confirm.setFixedSize(80, 40)
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
        self.btn_discard.setFixedSize(80, 40)
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

    def _apply_row_style(self):
        """Update row style based on active state."""
        # 获取质量颜色
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
        """根据最佳变体分数获取质量颜色"""
        if not self._data or 'variants' not in self._data:
            return "#333"  # 默认灰色

        variants = self._data.get('variants', [])
        if not variants:
            return "#333"

        # 获取最佳分数
        scores = []
        for v in variants:
            score = v.get('score_10', 0)
            if score > 0:
                scores.append(score)

        if not scores:
            return "#333"

        best_score = max(scores)

        # 根据分数返回颜色（10分制）
        if best_score >= 8.0:
            return "#4CAF50"  # 绿色 - 高质量
        elif best_score >= 6.0:
            return "#FFC107"  # 黄色 - 中等质量
        elif best_score >= 4.0:
            return "#FF9800"  # 橙色 - 较低质量
        else:
            return "#F44336"  # 红色 - 低质量


    def load_data(self, data: dict):
        """
        加载数据

        Args:
            data: {
                'original_image': PIL.Image,
                'original_path': str,
                'basename': str,
                'control_type': str,  # 'canny', 'openpose', or 'depth'
                'variants': [
                    {
                        'image': PIL.Image,
                        'score': float,  # Raw score (0-100)
                        'score_10': float,  # 1-10 score
                        'is_best': bool,
                        'preset': dict,
                        'preset_name': str
                    },
                    ...
                ]
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

    def _reset_loaded_data(self, data: dict):
        self._data = data
        self._selected_index = -1
        self._best_index = -1

    def _show_original_if_available(self, data: dict):
        if data.get('original_image'):
            self._set_image(self.lbl_original, data['original_image'])

    def _visible_variants(self, data: dict):
        return list(data.get('variants', [])[:4])

    def _show_all_variant_containers(self):
        for container in self.variant_containers:
            container.show()

    def _populate_variant_slots(self, variants):
        for index, variant in enumerate(variants):
            self._populate_variant_slot(index, variant)

    def _populate_variant_slot(self, index: int, variant: dict):
        if variant.get('image'):
            self._set_image(self.variant_labels[index], variant['image'])

        score_10 = variant.get('score_10', 1.0)
        is_best = variant.get('is_best', False)
        self.score_labels[index].set_score(score_10, is_best)

        preset_name = variant.get('preset_name', f"变体{index + 1}")
        self.preset_labels[index].setText(preset_name)

        if is_best:
            self._best_index = index

    def _hide_unused_variant_slots(self, used_count: int):
        for index in range(used_count, len(self.variant_containers)):
            self.variant_labels[index].clear()
            self.score_labels[index].clear()
            self.preset_labels[index].clear()
            self.variant_containers[index].hide()

    def _resolve_best_variant_index(self, variants):
        if self._best_index != -1 or not variants:
            return
        scores = [variant.get('score_10', 1.0) for variant in variants]
        self._best_index = scores.index(max(scores)) if scores else -1

    def _preselect_single_variant_if_needed(self):
        # Only one candidate means no real choice; preselect it for faster review.
        if self.get_variant_count() == 1:
            self._selected_index = 0
    
    def _set_image(self, label: ImageLabel, pil_image):
        """将PIL图像设置到标签上"""
        if pil_image is None:
            return

        # 确保是RGB模式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # 转换为numpy数组并复制以确保数据独立
        arr = np.array(pil_image).copy()
        h, w, ch = arr.shape
        bytes_per_line = ch * w

        # 创建QImage（使用copy确保数据独立）
        q_image = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

        # 缩放并设置
        pixmap = QPixmap.fromImage(q_image).scaled(
            label.width() - 4, label.height() - 4,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)
    
    def _on_original_clicked(self):
        """原图被点击"""
        # 可以用于预览大图
        pass
    
    def _on_variant_clicked(self, variant_index: int):
        # Handle variant click.
        self.select_variant(variant_index, emit_signal=True)

    def _on_confirm_clicked(self):
        """确认按钮被点击"""
        self.confirm_selected()

    def _on_discard_clicked(self):
        """废弃按钮被点击"""
        self.discard_row()
    
    def get_selected_variant(self) -> int:
        """获取选中的变体索引"""
        return self._selected_index
    
    def get_data(self) -> dict:
        """获取数据"""
        return self._data
    
    def set_row_index(self, index: int):
        """设置行索引"""
        self.row_index = index

    def set_active(self, active: bool):
        """Set row active state for keyboard navigation."""
        self._is_active = active
        self._apply_row_style()

    def get_variant_count(self) -> int:
        """Get available variant count (max 4)."""
        if not self._data:
            return 0
        return min(len(self._data.get('variants', [])), 4)

    def _refresh_variant_highlights(self):
        """Refresh variant border highlights for selected/best states."""
        variant_count = self.get_variant_count()
        self.btn_confirm.setEnabled(self._selected_index >= 0)

        for i, label in enumerate(self.variant_labels):
            if i >= variant_count:
                label.set_highlight(False)
                continue

            if i == self._selected_index:
                label.set_highlight(True, "#0078d4")
            elif i == self._best_index:
                label.set_highlight(True, "#00ff00")
            else:
                label.set_highlight(False)

    def select_variant(self, variant_index: int, emit_signal: bool = True) -> bool:
        """Select a variant programmatically."""
        if variant_index < 0 or variant_index >= self.get_variant_count():
            return False

        self._selected_index = variant_index
        self._refresh_variant_highlights()

        if emit_signal:
            self.variant_selected.emit(self.row_index, variant_index)
        return True

    def select_next_variant(self, step: int) -> bool:
        """Select next/previous variant with wrap-around."""
        variant_count = self.get_variant_count()
        if variant_count == 0:
            return False

        if self._selected_index < 0:
            next_index = 0 if step >= 0 else variant_count - 1
        else:
            next_index = (self._selected_index + step) % variant_count

        return self.select_variant(next_index, emit_signal=True)

    def confirm_selected(self) -> bool:
        """Trigger confirm action for selected variant."""
        if self._selected_index < 0:
            return False
        self.variant_confirmed.emit(self.row_index, self._selected_index)
        return True

    def discard_row(self) -> bool:
        """Trigger discard action for this row."""
        self.row_discarded.emit(self.row_index)
        return True

    def get_best_variant_index(self) -> int:
        """获取最佳变体的索引"""
        return self._best_index

    def clear(self):
        """清空显示"""
        self.lbl_original.clear()
        for label in self.variant_labels:
            label.clear()
        for score_label in self.score_labels:
            score_label.clear()
        self._data = None
        self._selected_index = -1
        self._best_index = -1
