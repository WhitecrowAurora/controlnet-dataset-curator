"""
ControlNet GUI - 图片列表容器
可滚动的图片行列表
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel
from PyQt5.QtCore import Qt, pyqtSignal

from .image_row import ImageRowWidget
from ..language import tr


class ImageListWidget(QWidget):
    """
    图片列表容器

    管理多个ImageRowWidget，支持滚动显示

    信号:
        variant_selected: (row_index, variant_index) 用户选择了某个变体
        variant_confirmed: (row_index, variant_index) 用户确认了某个变体
        row_discarded: (row_index) 用户废弃了整行
    """

    variant_selected = pyqtSignal(int, int)
    variant_confirmed = pyqtSignal(int, int)
    row_discarded = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []
        self._active_row_index = -1
        self._display_mode = 'fixed_single'
        self._setup_ui()

    def _setup_ui(self):
        """设置UI布局"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #0d0d0d;
            }
            QScrollBar:vertical {
                background-color: #1a1a1a;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #444;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #555;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.scroll_area.viewport().setStyleSheet("background-color: #0d0d0d;")

        # 容器widget
        self.container = QWidget()
        self.container.setStyleSheet("background-color: #0d0d0d;")
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(0)
        self.container_layout.setAlignment(Qt.AlignTop)
        self.container_layout.addStretch()

        self.scroll_area.setWidget(self.container)
        main_layout.addWidget(self.scroll_area)

        # Empty state label
        self.empty_label = QLabel(tr('no_data'))
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 16px;
                padding: 50px;
            }
        """)
        self.container_layout.insertWidget(0, self.empty_label)

    def add_row(self, data: dict) -> ImageRowWidget:
        """
        添加一行

        Args:
            data: 图片数据

        Returns:
            创建的ImageRowWidget
        """
        # 隐藏空状态提示
        if self.empty_label.isVisible():
            self.empty_label.hide()

        row_index = len(self._rows)
        row_widget = ImageRowWidget(row_index)
        row_widget.set_display_mode(self._display_mode)
        row_widget.load_data(data)

        # 连接信号
        row_widget.variant_selected.connect(self._on_variant_selected)
        row_widget.variant_confirmed.connect(self._on_variant_confirmed)
        row_widget.row_discarded.connect(self._on_row_discarded)

        # 插入到stretch之前
        self.container_layout.insertWidget(row_index, row_widget)
        self._rows.append(row_widget)

        if self._active_row_index == -1:
            self.set_active_row(0)

        return row_widget

    def set_display_mode(self, mode: str):
        normalized = ImageRowWidget.normalize_display_mode(mode)
        if normalized == self._display_mode:
            return
        self._display_mode = normalized
        for row_widget in self._rows:
            row_widget.set_display_mode(normalized)

    def get_display_mode(self) -> str:
        return self._display_mode

    def remove_row(self, row_index: int):
        """移除指定行"""
        if 0 <= row_index < len(self._rows):
            was_active = (row_index == self._active_row_index)
            row_widget = self._rows[row_index]
            self.container_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            self._rows.pop(row_index)

            # 更新后续行的索引
            for i in range(row_index, len(self._rows)):
                self._rows[i].set_row_index(i)

            # 如果没有行了，显示空状态
            if len(self._rows) == 0:
                self._active_row_index = -1
                self.empty_label.show()
                return

            if row_index < self._active_row_index:
                self._active_row_index -= 1

            if was_active:
                self.set_active_row(min(row_index, len(self._rows) - 1))
            else:
                self.set_active_row(self._active_row_index)

    def clear(self):
        """清空所有行"""
        for row_widget in self._rows:
            self.container_layout.removeWidget(row_widget)
            row_widget.deleteLater()
        self._rows.clear()
        self._active_row_index = -1
        self.empty_label.show()

    def update_language(self):
        """Update UI language"""
        self.empty_label.setText(tr('no_data'))

    def get_row(self, row_index: int) -> ImageRowWidget:
        """获取指定行"""
        if 0 <= row_index < len(self._rows):
            return self._rows[row_index]
        return None

    def get_row_count(self) -> int:
        """获取行数"""
        return len(self._rows)

    def set_active_row(self, row_index: int) -> bool:
        """Set active row by index."""
        if row_index < 0 or row_index >= len(self._rows):
            return False

        if 0 <= self._active_row_index < len(self._rows):
            self._rows[self._active_row_index].set_active(False)

        self._active_row_index = row_index
        active_row = self._rows[row_index]
        active_row.set_active(True)
        self.scroll_area.ensureWidgetVisible(active_row, 0, 20)
        return True

    def get_active_row_index(self) -> int:
        """Get current active row index."""
        return self._active_row_index

    def get_active_row(self):
        """Get current active row widget."""
        if 0 <= self._active_row_index < len(self._rows):
            return self._rows[self._active_row_index]
        return None

    def activate_next_row(self, step: int) -> bool:
        """Move active row by step (+1/-1) with wrap-around."""
        if not self._rows:
            return False

        if self._active_row_index < 0:
            return self.set_active_row(0)

        next_index = (self._active_row_index + step) % len(self._rows)
        return self.set_active_row(next_index)

    def select_variant_on_active(self, variant_index: int) -> bool:
        """Select variant on active row."""
        row = self.get_active_row()
        if not row:
            return False
        return row.select_variant(variant_index, emit_signal=True)

    def select_next_variant_on_active(self, step: int) -> bool:
        """Select next/previous variant on active row."""
        row = self.get_active_row()
        if not row:
            return False
        return row.select_next_variant(step)

    def confirm_active_selection(self) -> bool:
        """Confirm selected variant on active row."""
        row = self.get_active_row()
        if not row:
            return False
        return row.confirm_selected()

    def discard_active_row(self) -> bool:
        """Discard active row."""
        row = self.get_active_row()
        if not row:
            return False
        return row.discard_row()

    def _on_variant_selected(self, row_index: int, variant_index: int):
        """变体被选中"""
        self.set_active_row(row_index)
        self.variant_selected.emit(row_index, variant_index)

    def _on_variant_confirmed(self, row_index: int, variant_index: int):
        """变体被确认"""
        self.variant_confirmed.emit(row_index, variant_index)

    def _on_row_discarded(self, row_index: int):
        """行被废弃"""
        self.row_discarded.emit(row_index)

    def get_all_rows(self):
        """获取所有行"""
        return self._rows

    def get_selected_rows(self):
        """获取所有被选中的行（暂时返回当前激活的行）"""
        # TODO: 实现多选功能
        active = self.get_active_row()
        return [active] if active else []

    def batch_accept_rows(self, rows):
        """批量接受行"""
        for row in rows:
            if row and row.get_selected_variant() >= 0:
                row_index = self._rows.index(row)
                variant_index = row.get_selected_variant()
                self.variant_confirmed.emit(row_index, variant_index)

    def batch_discard_rows(self, rows):
        """批量拒绝行"""
        for row in rows:
            if row:
                row_index = self._rows.index(row)
                self.row_discarded.emit(row_index)
