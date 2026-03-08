"""
ControlNet GUI - 预览面板
显示大图预览和详细评分信息
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

import numpy as np
from PIL import Image


class PreviewPanel(QWidget):
    """
    预览面板

    显示选中图片的大图预览和详细评分信息
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_data = None
        self._current_variant_index = -1
        self._setup_ui()

    def _setup_ui(self):
        """设置UI布局"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.setMinimumWidth(340)

        # Set panel background
        self.setStyleSheet("""
            QWidget {
                background-color: #0d0d0d;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #ffffff;
                font-weight: bold;
                background-color: #1a1a1a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        # 图片预览区域
        preview_group = QGroupBox("图片预览")
        preview_group.setMinimumWidth(320)
        preview_layout = QVBoxLayout(preview_group)

        self.lbl_preview = QLabel()
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumSize(300, 300)  # Reduce minimum size
        self.lbl_preview.setStyleSheet("""
            QLabel {
                border: 2px solid #444;
                background-color: #2d2d2d;
                border-radius: 4px;
            }
        """)
        preview_layout.addWidget(self.lbl_preview)

        main_layout.addWidget(preview_group)

        # 评分信息区域
        score_group = QGroupBox("评分详情")
        score_group.setMinimumWidth(320)
        score_layout = QVBoxLayout(score_group)

        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1a1a1a;
            }
        """)

        score_container = QWidget()
        score_container.setStyleSheet("background-color: #1a1a1a;")
        self.score_container_layout = QVBoxLayout(score_container)
        self.score_container_layout.setSpacing(5)

        # 评分标签
        self.lbl_canny_score = QLabel("Canny评分: --")
        self.lbl_pose_score = QLabel("OpenPose评分: --")
        self.lbl_depth_score = QLabel("Depth评分: --")
        self.lbl_total_score = QLabel("综合评分: --")
        self.lbl_quality_level = QLabel("质量等级: --")

        for label in [self.lbl_canny_score, self.lbl_pose_score,
                      self.lbl_depth_score, self.lbl_total_score,
                      self.lbl_quality_level]:
            label.setStyleSheet("color: #ffffff; font-size: 13px; padding: 3px; background-color: transparent;")
            self.score_container_layout.addWidget(label)

        # Canny详细信息
        self.lbl_canny_details = QLabel()
        self.lbl_canny_details.setWordWrap(True)
        self.lbl_canny_details.setStyleSheet("color: #cccccc; font-size: 11px; padding: 5px; background-color: transparent;")
        self.score_container_layout.addWidget(self.lbl_canny_details)

        self.score_container_layout.addStretch()

        scroll_area.setWidget(score_container)
        score_layout.addWidget(scroll_area)

        main_layout.addWidget(score_group)

        # 空状态
        self._show_empty_state()

    def _show_empty_state(self):
        """显示空状态"""
        self.lbl_preview.setText("点击图片查看预览")
        self.lbl_preview.setStyleSheet("""
            QLabel {
                border: 2px solid #444;
                background-color: #1a1a1a;
                border-radius: 4px;
                color: #666;
                font-size: 14px;
            }
        """)
        self.lbl_canny_score.setText("Canny评分: --")
        self.lbl_pose_score.setText("OpenPose评分: --")
        self.lbl_depth_score.setText("Depth评分: --")
        self.lbl_total_score.setText("综合评分: --")
        self.lbl_quality_level.setText("质量等级: --")
        self.lbl_canny_details.setText("")

    def show_preview(self, data: dict, variant_index: int = -1):
        """
        显示预览

        Args:
            data: 图片数据
            variant_index: 变体索引，-1表示显示原图
        """
        self._current_data = data
        self._current_variant_index = variant_index

        # 显示图片
        if variant_index == -1:
            # 显示原图
            image = data.get('original_image')
            if image:
                self._set_preview_image(image)
        else:
            # 显示变体
            variants = data.get('variants', [])
            if 0 <= variant_index < len(variants):
                variant = variants[variant_index]
                image = variant.get('image')
                if image:
                    self._set_preview_image(image)

                # 显示评分信息
                self._show_score_info(variant, data)

    def _set_preview_image(self, pil_image):
        """设置预览图片"""
        if pil_image is None:
            return

        # 确保是RGB模式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # 转换为QPixmap
        arr = np.array(pil_image).copy()
        h, w, ch = arr.shape
        bytes_per_line = ch * w

        q_image = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.lbl_preview.width() - 10,
            self.lbl_preview.height() - 10,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.lbl_preview.setPixmap(pixmap)
        self.lbl_preview.setStyleSheet("""
            QLabel {
                border: 2px solid #444;
                background-color: #1a1a1a;
                border-radius: 4px;
            }
        """)

    def _show_score_info(self, variant: dict, data: dict):
        """显示评分信息（1-10分制）"""
        control_type = data.get('control_type', 'canny')
        variant_10_score = variant.get('score_10', 1.0)
        is_best = variant.get('is_best', False)

        # 显示控制类型和评分
        control_type_name = {
            'canny': 'Canny',
            'openpose': 'OpenPose',
            'depth': 'Depth'
        }.get(control_type, control_type)

        self.lbl_canny_score.setText(
            f"{control_type_name}: {variant_10_score:.1f}/10" + (" ★ 最佳" if is_best else "")
        )

        # 显示详细信息
        preset = variant.get('preset', {})
        preset_name = variant.get('preset_name', 'unknown')
        raw_score = variant.get('score', 0)

        if control_type == 'canny':
            details = f"""
预设: {preset_name}
阈值: low={preset.get('low', 'N/A')}, high={preset.get('high', 'N/A')}
评分: {variant_10_score:.1f}/10 (原始: {raw_score:.1f}/100)
            """.strip()
        elif control_type == 'openpose':
            warning = preset.get('warning', '')
            warning_text = f"\n警告: {warning}" if warning else ""
            details = f"""
预设: {preset_name}
评分: {variant_10_score:.1f}/10 (原始: {raw_score:.1f}/100){warning_text}
            """.strip()
        elif control_type == 'depth':
            std = preset.get('std', 0)
            dynamic_range = preset.get('dynamic_range', 0)
            grad_mean = preset.get('grad_mean', 0)
            details = f"""
预设: {preset_name}
评分: {variant_10_score:.1f}/10 (原始: {raw_score:.1f}/100)
标准差: {std:.1f}
动态范围: {dynamic_range:.1f}
梯度均值: {grad_mean:.2f}
            """.strip()
        else:
            details = f"预设: {preset_name}\n评分: {variant_10_score:.1f}/10"

        self.lbl_canny_details.setText(details)

        # 隐藏不相关的评分
        self.lbl_pose_score.setText("")
        self.lbl_depth_score.setText("")

        # 综合评分就是当前控制类型的评分
        self.lbl_total_score.setText(f"评分: {variant_10_score:.1f}/10")

        # 质量等级
        if variant_10_score >= 8.0:
            quality_level = 'auto_accept'
            reason = '高质量，自动通过'
        elif variant_10_score <= 4.0:
            quality_level = 'auto_reject'
            reason = '低质量，自动废弃'
        else:
            quality_level = 'need_review'
            reason = '中等质量，需要人工审核'

        level_text = {
            'auto_accept': '自动通过 ✓',
            'auto_reject': '自动废弃 ✗',
            'need_review': '需要审核 ?'
        }.get(quality_level, quality_level)

        self.lbl_quality_level.setText(f"质量等级: {level_text}")

    def clear(self):
        """清空预览"""
        self._current_data = None
        self._current_variant_index = -1
        self._show_empty_state()
