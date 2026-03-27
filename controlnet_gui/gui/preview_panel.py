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
        self.setStyleSheet(self._preview_panel_stylesheet())
        main_layout.addWidget(self._create_preview_group())
        main_layout.addWidget(self._create_score_group())
        self._show_empty_state()

    def _preview_panel_stylesheet(self) -> str:
        return """
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
        """

    def _create_preview_group(self) -> QGroupBox:
        group = QGroupBox("图片预览")
        group.setMinimumWidth(320)
        layout = QVBoxLayout(group)

        self.lbl_preview = QLabel()
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumSize(300, 300)
        self.lbl_preview.setStyleSheet("""
            QLabel {
                border: 2px solid #444;
                background-color: #2d2d2d;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.lbl_preview)
        return group

    def _create_score_group(self) -> QGroupBox:
        group = QGroupBox("评分详情")
        group.setMinimumWidth(320)
        score_layout = QVBoxLayout(group)

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
        self._init_score_labels()

        scroll_area.setWidget(score_container)
        score_layout.addWidget(scroll_area)
        return group

    def _init_score_labels(self):
        self.lbl_canny_score = QLabel("Canny评分: --")
        self.lbl_pose_score = QLabel("OpenPose评分: --")
        self.lbl_depth_score = QLabel("Depth评分: --")
        self.lbl_total_score = QLabel("综合评分: --")
        self.lbl_quality_level = QLabel("质量等级: --")

        for label in [
            self.lbl_canny_score,
            self.lbl_pose_score,
            self.lbl_depth_score,
            self.lbl_total_score,
            self.lbl_quality_level,
        ]:
            label.setStyleSheet("color: #ffffff; font-size: 13px; padding: 3px; background-color: transparent;")
            self.score_container_layout.addWidget(label)

        self.lbl_canny_details = QLabel()
        self.lbl_canny_details.setWordWrap(True)
        self.lbl_canny_details.setStyleSheet(
            "color: #cccccc; font-size: 11px; padding: 5px; background-color: transparent;"
        )
        self.score_container_layout.addWidget(self.lbl_canny_details)
        self.score_container_layout.addStretch()

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

        control_type_name = self._control_type_display_name(control_type)
        self.lbl_canny_score.setText(
            f"{control_type_name}: {variant_10_score:.1f}/10" + (" ★ 最佳" if is_best else "")
        )

        preset = variant.get('preset', {})
        preset_name = variant.get('preset_name', 'unknown')
        raw_score = variant.get('score', 0)
        details = self._build_variant_details(control_type, preset, preset_name, variant_10_score, raw_score)
        self.lbl_canny_details.setText(details)

        self.lbl_pose_score.setText("")
        self.lbl_depth_score.setText("")

        self.lbl_total_score.setText(f"评分: {variant_10_score:.1f}/10")
        quality_level = self._resolve_quality_level(control_type, variant_10_score, preset)
        level_text = {
            'auto_accept': '自动通过 ✓',
            'auto_reject': '自动废弃 ✗',
            'need_review': '需要审核 ?',
        }.get(quality_level, quality_level)
        self.lbl_quality_level.setText(f"质量等级: {level_text}")

    def _control_type_display_name(self, control_type: str) -> str:
        return {
            'canny': 'Canny',
            'openpose': 'OpenPose',
            'depth': 'Depth',
            'bbox': 'BBox',
            'prefilter_score': '原图评分',
        }.get(control_type, control_type)

    def _build_variant_details(
        self,
        control_type: str,
        preset: dict,
        preset_name: str,
        variant_10_score: float,
        raw_score: float,
    ) -> str:
        if control_type == 'canny':
            return self._build_canny_details(preset_name, preset, variant_10_score, raw_score)
        if control_type == 'openpose':
            return self._build_openpose_details(preset_name, preset, variant_10_score, raw_score)
        if control_type == 'depth':
            return self._build_depth_details(preset_name, preset, variant_10_score, raw_score)
        if control_type == 'prefilter_score':
            return self._build_prefilter_details(preset)
        if control_type == 'bbox':
            return self._build_bbox_details(preset_name, preset, variant_10_score, raw_score)
        return f"预设: {preset_name}\n评分: {variant_10_score:.1f}/10"

    def _build_canny_details(
        self, preset_name: str, preset: dict, variant_10_score: float, raw_score: float
    ) -> str:
        return f"""
预设: {preset_name}
阈值: low={preset.get('low', 'N/A')}, high={preset.get('high', 'N/A')}
评分: {variant_10_score:.1f}/10 (原始: {raw_score:.1f}/100)
        """.strip()

    def _build_openpose_details(
        self, preset_name: str, preset: dict, variant_10_score: float, raw_score: float
    ) -> str:
        warning = preset.get('warning', '')
        warning_text = f"\n警告: {warning}" if warning else ""
        return f"""
预设: {preset_name}
评分: {variant_10_score:.1f}/10 (原始: {raw_score:.1f}/100){warning_text}
        """.strip()

    def _build_depth_details(
        self, preset_name: str, preset: dict, variant_10_score: float, raw_score: float
    ) -> str:
        std = preset.get('std', 0)
        dynamic_range = preset.get('dynamic_range', 0)
        grad_mean = preset.get('grad_mean', 0)
        return f"""
预设: {preset_name}
评分: {variant_10_score:.1f}/10 (原始: {raw_score:.1f}/100)
标准差: {std:.1f}
动态范围: {dynamic_range:.1f}
梯度均值: {grad_mean:.2f}
        """.strip()

    def _build_prefilter_details(self, preset: dict) -> str:
        auto_accept_aesthetic = preset.get('auto_accept_aesthetic', 'N/A')
        min_aesthetic_score = preset.get('min_aesthetic_score', 'N/A')
        require_in_domain = bool(preset.get('require_in_domain', False))
        min_in_domain_prob = preset.get('min_in_domain_prob', 'N/A')
        domain_rule = f">= {min_in_domain_prob}" if require_in_domain else '未启用'
        return f"""
美学分: {preset.get('aesthetic', 'N/A')}
构图分: {preset.get('composition', 'N/A')}
色彩分: {preset.get('color', 'N/A')}
性感分: {preset.get('sexual', 'N/A')}
目标域概率: {preset.get('in_domain_prob', 'N/A')}
目标域判定: {preset.get('in_domain_pred', 'N/A')}
最低美学分: {min_aesthetic_score}
自动通过阈值: {auto_accept_aesthetic}
目标域要求: {domain_rule}
设备: {preset.get('device', 'N/A')}
说明: {preset.get('reason', '') or '无'}
        """.strip()

    def _build_bbox_details(
        self, preset_name: str, preset: dict, variant_10_score: float, raw_score: float
    ) -> str:
        count = int(preset.get('count', 0) or 0)
        mean_conf = float(preset.get('mean_confidence', 0.0) or 0.0)
        max_conf = float(preset.get('max_confidence', 0.0) or 0.0)
        warning = str(preset.get('warning', '') or '').strip()
        labels_line = self._format_bbox_detection_labels(preset.get('detections') or [])
        warning_line = f"\n警告: {warning}" if warning else ""
        return f"""
预设: {preset_name}
评分: {variant_10_score:.1f}/10 (原始: {raw_score:.1f}/100)
检测框数量: {count}
平均置信度: {mean_conf:.3f}
最高置信度: {max_conf:.3f}
检测明细: {labels_line}{warning_line}
        """.strip()

    def _format_bbox_detection_labels(self, detections) -> str:
        labels = []
        for det in detections[:6]:
            if not isinstance(det, dict):
                continue
            label = str(det.get('label', 'obj') or 'obj')
            conf = float(det.get('confidence', 0.0) or 0.0)
            labels.append(f"{label}({conf:.2f})")
        return ", ".join(labels) if labels else '无'

    def _resolve_quality_level(self, control_type: str, variant_10_score: float, preset: dict) -> str:
        if control_type == 'prefilter_score':
            try:
                auto_accept_aesthetic = float(preset.get('auto_accept_aesthetic', 4.0) or 4.0)
            except Exception:
                auto_accept_aesthetic = 4.0
            try:
                aesthetic_value = float(preset.get('aesthetic'))
            except Exception:
                aesthetic_value = None
            if not preset.get('passed', True):
                return 'auto_reject'
            if aesthetic_value is not None and aesthetic_value >= auto_accept_aesthetic:
                return 'auto_accept'
            return 'need_review'

        if variant_10_score >= 8.0:
            return 'auto_accept'
        if variant_10_score <= 4.0:
            return 'auto_reject'
        return 'need_review'

    def clear(self):
        """清空预览"""
        self._current_data = None
        self._current_variant_index = -1
        self._show_empty_state()
