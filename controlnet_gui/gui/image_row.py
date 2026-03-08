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
        
        # 原图容器
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_layout.setSpacing(2)
        original_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_original = ImageLabel()
        self.lbl_original.setFixedSize(180, 180)
        self.lbl_original.clicked.connect(self._on_original_clicked)
        
        self.lbl_original_title = QLabel("原图")
        self.lbl_original_title.setAlignment(Qt.AlignCenter)
        self.lbl_original_title.setStyleSheet("color: #aaa; font-size: 10px;")
        
        original_layout.addWidget(self.lbl_original)
        original_layout.addWidget(self.lbl_original_title)
        main_layout.addWidget(original_container)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("color: #444;")
        main_layout.addWidget(separator)
        
        # 4个变体（Canny/OpenPose/Depth）
        self.variant_labels = []
        self.score_labels = []
        self.preset_labels = []  # 添加预设名称标签列表
        self.variant_containers = []

        for i in range(4):
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setSpacing(2)
            layout.setContentsMargins(0, 0, 0, 0)

            # 图片标签
            img_label = ImageLabel()
            img_label.setFixedSize(160, 160)
            img_label.clicked.connect(lambda idx=i: self._on_variant_clicked(idx))

            # 分数标签
            score_label = ScoreLabel()

            # 预设名称标签
            preset_label = QLabel(f"变体{i+1}")
            preset_label.setAlignment(Qt.AlignCenter)
            preset_label.setStyleSheet("color: #666; font-size: 9px;")

            layout.addWidget(img_label)
            layout.addWidget(score_label)
            layout.addWidget(preset_label)

            self.variant_labels.append(img_label)
            self.score_labels.append(score_label)
            self.preset_labels.append(preset_label)  # 保存引用
            self.variant_containers.append(container)

            main_layout.addWidget(container)

        # 添加按钮容器
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setSpacing(10)
        button_layout.setContentsMargins(10, 0, 10, 0)

        # 确认按钮
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

        # 废弃按钮
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

        button_layout.addStretch()
        button_layout.addWidget(self.btn_confirm)
        button_layout.addWidget(self.btn_discard)
        button_layout.addStretch()

        main_layout.addWidget(button_container)

        # 设置行背景
        self._apply_row_style()

    def _apply_row_style(self):
        """Update row style based on active state."""
        if self._is_active:
            self.setStyleSheet("""
                ImageRowWidget {
                    background-color: #1a1a1a;
                    border-bottom: 1px solid #333;
                    border-left: 3px solid #0078d4;
                }
                ImageRowWidget:hover {
                    background-color: #1f1f1f;
                }
            """)
        else:
            self.setStyleSheet("""
                ImageRowWidget {
                    background-color: #0d0d0d;
                    border-bottom: 1px solid #333;
                    border-left: 3px solid transparent;
                }
                ImageRowWidget:hover {
                    background-color: #151515;
                }
            """)
    

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
        self._data = data
        self._selected_index = -1
        self._best_index = -1

        # 显示原图
        if data.get('original_image'):
            self._set_image(self.lbl_original, data['original_image'])

        # 显示变体
        variants = data.get('variants', [])

        for i, variant in enumerate(variants[:4]):
            if variant.get('image'):
                self._set_image(self.variant_labels[i], variant['image'])

            # 使用 1-10 分数显示
            score_10 = variant.get('score_10', 1.0)
            is_best = variant.get('is_best', False)

            self.score_labels[i].set_score(score_10, is_best)

            # 更新预设名称标签
            preset_name = variant.get('preset_name', f"变体{i+1}")
            self.preset_labels[i].setText(preset_name)

            if is_best:
                self._best_index = i

        # 如果没有变体被标记为最佳，找出分数最高的
        if self._best_index == -1 and variants:
            scores = [v.get('score_10', 1.0) for v in variants[:4]]
            self._best_index = scores.index(max(scores)) if scores else -1

        self._refresh_variant_highlights()
    
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
