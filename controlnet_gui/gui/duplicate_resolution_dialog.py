"""Dialog for duplicate review-item resolution with side-by-side summaries."""

from __future__ import annotations

from typing import Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class DuplicateResolutionDialog(QDialog):
    def __init__(
        self,
        *,
        title: str,
        intro: str,
        old_title: str,
        old_text: str,
        new_title: str,
        new_text: str,
        overwrite_label: str,
        separate_label: str,
        apply_all_label: str,
        parent=None,
    ):
        super().__init__(parent)
        self._action = "new_revision"
        self.setWindowTitle(title)
        self.resize(980, 700)

        layout = QVBoxLayout(self)

        intro_label = QLabel(intro)
        intro_label.setWordWrap(True)
        intro_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(intro_label)

        compare_layout = QHBoxLayout()
        compare_layout.addWidget(self._build_panel(old_title, old_text), 1)
        compare_layout.addWidget(self._build_panel(new_title, new_text), 1)
        layout.addLayout(compare_layout, 1)

        bottom_row = QHBoxLayout()
        self.apply_all_box = QCheckBox(apply_all_label)
        bottom_row.addWidget(self.apply_all_box)
        bottom_row.addStretch(1)

        overwrite_btn = QPushButton(overwrite_label)
        overwrite_btn.clicked.connect(lambda: self._choose("overwrite"))
        bottom_row.addWidget(overwrite_btn)

        separate_btn = QPushButton(separate_label)
        separate_btn.clicked.connect(lambda: self._choose("new_revision"))
        separate_btn.setDefault(True)
        bottom_row.addWidget(separate_btn)

        layout.addLayout(bottom_row)

    def _build_panel(self, title: str, text: str):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)

        editor = QTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(text)
        layout.addWidget(editor)
        return group

    def _choose(self, action: str):
        self._action = action
        self.accept()

    def result_data(self) -> Tuple[str, bool]:
        return self._action, self.apply_all_box.isChecked()
