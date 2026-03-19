"""Dialog for inspecting, importing, and restoring managed JSONA files."""
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class JsonaManagerDialog(QDialog):
    """Manager for JSONA files, backups, and validation."""

    def __init__(self, output_dir: str, backup_manager, allow_mutation: bool = True, parent=None):
        super().__init__(parent)
        self.output_dir = output_dir
        self.backup_manager = backup_manager
        self.allow_mutation = allow_mutation
        self._file_rows = []
        self._reports = {}
        self._setup_ui()
        self.refresh_table()

    def _setup_ui(self):
        self.setWindowTitle('JSONA 管理')
        self.resize(1180, 620)

        layout = QVBoxLayout(self)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['类型', '文件', '条目数', '状态', '最后修改', '备份数'])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        self.result_label = QLabel('尚未执行 JSONA 核对。')
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        top_row = QHBoxLayout()
        self.refresh_button = QPushButton('刷新')
        self.refresh_button.clicked.connect(self.refresh_table)
        top_row.addWidget(self.refresh_button)

        self.import_button = QPushButton('导入 JSONA')
        self.import_button.clicked.connect(self.import_jsona)
        top_row.addWidget(self.import_button)

        self.merge_button = QPushButton('合并多个 JSONA')
        self.merge_button.clicked.connect(self.merge_jsona)
        top_row.addWidget(self.merge_button)

        self.check_button = QPushButton('核对选中项')
        self.check_button.clicked.connect(self.check_selected)
        top_row.addWidget(self.check_button)

        self.export_button = QPushButton('导出问题项')
        self.export_button.clicked.connect(self.export_selected_report)
        top_row.addWidget(self.export_button)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        bottom_row = QHBoxLayout()
        self.backup_button = QPushButton('备份选中项')
        self.backup_button.clicked.connect(self.backup_selected)
        bottom_row.addWidget(self.backup_button)

        self.restore_button = QPushButton('恢复选中项')
        self.restore_button.clicked.connect(self.restore_selected)
        bottom_row.addWidget(self.restore_button)

        self.remove_missing_button = QPushButton('移除不存在条目')
        self.remove_missing_button.clicked.connect(self.remove_missing_selected)
        bottom_row.addWidget(self.remove_missing_button)

        self.repair_button = QPushButton('修复结构')
        self.repair_button.clicked.connect(self.repair_selected)
        bottom_row.addWidget(self.repair_button)

        bottom_row.addStretch(1)
        self.close_button = QPushButton('关闭')
        self.close_button.clicked.connect(self.accept)
        bottom_row.addWidget(self.close_button)
        layout.addLayout(bottom_row)

        if not self.allow_mutation:
            for button in [
                self.import_button,
                self.merge_button,
                self.backup_button,
                self.restore_button,
                self.remove_missing_button,
                self.repair_button,
            ]:
                button.setEnabled(False)

    def refresh_table(self):
        self._file_rows = self.backup_manager.list_managed_files(self.output_dir)
        self.table.setRowCount(len(self._file_rows))

        existing_count = 0
        for row, info in enumerate(self._file_rows):
            if info['exists']:
                existing_count += 1

            modified_text = ''
            if isinstance(info.get('modified_at'), datetime):
                modified_text = info['modified_at'].strftime('%Y-%m-%d %H:%M:%S')

            if info['status'] == 'ok':
                status_text = '正常'
            elif info['status'] == 'missing':
                status_text = '缺失'
            elif info['status'] == 'issues':
                status_text = '需修复'
            else:
                status_text = f"异常: {info['error']}"

            values = [
                info['name'],
                info['path'],
                str(info.get('valid_entry_count', info.get('entry_count', 0))),
                status_text,
                modified_text,
                str(info.get('backup_count', 0)),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.table.setItem(row, col, item)

        self.summary_label.setText(
            f"输出目录: {self.output_dir}\n当前检测到 {existing_count} 个 JSONA 文件。"
        )
        if self._file_rows:
            self.table.selectRow(0)

    def _selected_info(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self._file_rows):
            return None
        return self._file_rows[row]

    def _format_report_summary(self, report: dict) -> str:
        return (
            f"总条目: {report.get('entry_count', 0)} | "
            f"结构有效: {report.get('valid_entry_count', 0)} | "
            f"文件存在: {report.get('existing_entry_count', 0)} | "
            f"结构异常: {report.get('invalid_entry_count', 0)} | "
            f"重复: {report.get('duplicate_count', 0)} | "
            f"缺文件: {report.get('missing_file_count', 0)} | "
            f"类型冲突: {report.get('type_mismatch_count', 0)} | "
            f"文本冲突: {report.get('prompt_conflict_count', 0)}"
        )

    def _remember_report(self, key: str, report: dict):
        self._reports[key] = report
        self.result_label.setText(self._format_report_summary(report))

    def import_jsona(self):
        source_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择要导入的 JSONA 文件',
            self.output_dir,
            'JSONA Files (*.jsona *.json);;All Files (*.*)',
        )
        if not source_path:
            return

        verify = QMessageBox.question(
            self,
            '导入 JSONA',
            '是否核对 JSONA 文件中的路径是否存在?',
            QMessageBox.Yes | QMessageBox.No,
        ) == QMessageBox.Yes

        try:
            report = self.backup_manager.import_jsona(source_path, self.output_dir, verify_files=verify)
        except Exception as exc:
            QMessageBox.warning(self, '导入 JSONA', f'导入失败:\n{exc}')
            return

        target_file = report['target_file']
        self._remember_report(target_file, report)
        self.refresh_table()
        for row, info in enumerate(self._file_rows):
            if info['path'] == target_file:
                self.table.selectRow(row)
                break

        message = (
            f"导入完成。\n\n"
            f"目标文件: {target_file}\n"
            f"新增条目: {report.get('imported_count', 0)}\n"
            f"合并重复: {report.get('merge_duplicate_count', 0)}\n"
            f"合并文本冲突: {report.get('merge_prompt_conflict_count', 0)}\n"
            f"目标类型: {report.get('target_name', 'unknown')}\n\n"
            f"{self._format_report_summary(report)}"
        )
        QMessageBox.information(self, '导入 JSONA', message)

        if (
            report.get('missing_file_count', 0) > 0
            or report.get('invalid_entry_count', 0) > 0
            or report.get('duplicate_count', 0) > 0
            or report.get('prompt_conflict_count', 0) > 0
            or report.get('merge_prompt_conflict_count', 0) > 0
        ):
            reply = QMessageBox.question(
                self,
                '导出问题项',
                '导入中发现异常条目。是否现在导出检查报告和问题条目?',
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                exported = self.backup_manager.export_report(report, self.output_dir)
                QMessageBox.information(
                    self,
                    '导出问题项',
                    f"已导出报告:\n{exported['summary']}"
                    + (f"\n\n问题条目:\n{exported['issues']}" if exported['issues'] else ''),
                )

    def merge_jsona(self):
        source_paths, _ = QFileDialog.getOpenFileNames(
            self,
            '选择要合并的 JSONA 文件',
            self.output_dir,
            'JSONA Files (*.jsona *.json);;All Files (*.*)',
        )
        if not source_paths:
            return

        verify = QMessageBox.question(
            self,
            '合并 JSONA',
            '是否在合并前核对 JSONA 中的路径是否存在?',
            QMessageBox.Yes | QMessageBox.No,
        ) == QMessageBox.Yes

        try:
            report = self.backup_manager.merge_jsona_files(source_paths, self.output_dir, verify_files=verify)
        except Exception as exc:
            QMessageBox.warning(self, '合并 JSONA', f'合并失败:\n{exc}')
            return

        self.refresh_table()

        for group in report.get('groups', []):
            self._remember_report(group['target_file'], group)

        groups = report.get('groups', [])
        if groups:
            first_target = groups[0]['target_file']
            for row, info in enumerate(self._file_rows):
                if info['path'] == first_target:
                    self.table.selectRow(row)
                    break

        lines = [
            f"已处理文件: {report.get('file_count', 0)}",
            f"合并目标数: {report.get('group_count', 0)}",
            "",
        ]
        for group in groups:
            lines.append(
                f"{group.get('target_name', 'unknown')} -> {group.get('target_file', '')}\n"
                f"来源文件: {len(group.get('source_files', []))} | "
                f"结构有效: {group.get('valid_entry_count', 0)} | "
                f"新增写入: {group.get('imported_count', 0)} | "
                f"合并重复: {group.get('merge_duplicate_count', 0)} | "
                f"合并文本冲突: {group.get('merge_prompt_conflict_count', 0)} | "
                f"结构异常: {group.get('invalid_entry_count', 0)} | "
                f"重复: {group.get('duplicate_count', 0)} | "
                f"缺文件: {group.get('missing_file_count', 0)}"
            )
            lines.append("")

        QMessageBox.information(self, '合并 JSONA', '\n'.join(lines).strip())

    def check_selected(self):
        info = self._selected_info()
        if not info:
            QMessageBox.warning(self, 'JSONA 管理', '请先选择一个 JSONA 文件。')
            return
        if not info['exists']:
            QMessageBox.information(self, 'JSONA 管理', '当前文件不存在，无法核对。')
            return

        report = self.backup_manager.inspect_jsona_file(info['path'], expected_name=info['name'], check_files=True)
        self._remember_report(info['path'], report)
        QMessageBox.information(self, 'JSONA 核对检查器', self._format_report_summary(report))

    def export_selected_report(self):
        info = self._selected_info()
        if not info:
            QMessageBox.warning(self, 'JSONA 管理', '请先选择一个 JSONA 文件。')
            return

        report = self._reports.get(info['path'])
        if (
            report is None
            or 'problem_entries' not in report
            or 'missing_entries' not in report
            or 'duplicate_entries' not in report
            or 'prompt_conflicts' not in report
        ):
            report = self.backup_manager.inspect_jsona_file(info['path'], expected_name=info['name'], check_files=True)
            self._remember_report(info['path'], report)

        exported = self.backup_manager.export_report(report, self.output_dir)
        QMessageBox.information(
            self,
            '导出问题项',
            f"已导出报告:\n{exported['summary']}"
            + (f"\n\n问题条目:\n{exported['issues']}" if exported['issues'] else ''),
        )

    def backup_selected(self):
        info = self._selected_info()
        if not info:
            QMessageBox.warning(self, 'JSONA 管理', '请先选择一个 JSONA 文件。')
            return
        if not info['exists']:
            QMessageBox.warning(self, 'JSONA 管理', '当前文件不存在，无法备份。')
            return

        backup_path = self.backup_manager.create_backup(info['path'], reason='manual')
        if backup_path:
            QMessageBox.information(self, 'JSONA 管理', f"已创建备份:\n{backup_path}")
            self.refresh_table()
        else:
            QMessageBox.warning(self, 'JSONA 管理', '备份失败。')

    def restore_selected(self):
        info = self._selected_info()
        if not info:
            QMessageBox.warning(self, 'JSONA 管理', '请先选择一个 JSONA 文件。')
            return

        backups = self.backup_manager.list_backups(info['path'])
        if not backups:
            QMessageBox.information(self, 'JSONA 管理', '这个 JSONA 文件当前没有可恢复的备份。')
            return

        labels = [
            f"{item['name']} | {item['modified_at'].strftime('%Y-%m-%d %H:%M:%S')} | {item['size']} bytes"
            for item in backups
        ]
        choice, ok = QInputDialog.getItem(self, '选择备份', '请选择要恢复的备份:', labels, 0, False)
        if not ok or not choice:
            return

        backup = backups[labels.index(choice)]
        reply = QMessageBox.question(
            self,
            '确认恢复',
            (
                f"将使用以下备份覆盖当前文件:\n\n{backup['path']}\n\n"
                '恢复前会自动备份当前 JSONA。是否继续?'
            ),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        restore_point = self.backup_manager.restore_backup(info['path'], backup['path'])
        message = f"恢复完成:\n{backup['path']}"
        if restore_point:
            message += f"\n\n恢复前备份:\n{restore_point}"
        QMessageBox.information(self, 'JSONA 管理', message)
        self.refresh_table()

    def remove_missing_selected(self):
        info = self._selected_info()
        if not info:
            QMessageBox.warning(self, 'JSONA 管理', '请先选择一个 JSONA 文件。')
            return
        if not info['exists']:
            QMessageBox.information(self, 'JSONA 管理', '当前文件不存在。')
            return

        report = self.backup_manager.inspect_jsona_file(info['path'], expected_name=info['name'], check_files=True)
        self._remember_report(info['path'], report)
        if report.get('missing_file_count', 0) == 0:
            QMessageBox.information(self, '移除不存在条目', '当前 JSONA 中没有检测到不存在的文件条目。')
            return

        reply = QMessageBox.question(
            self,
            '移除不存在条目',
            f"检测到 {report['missing_file_count']} 个不存在文件条目。是否从 JSONA 中移除这些条目?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        result = self.backup_manager.remove_missing_entries(info['path'], expected_name=info['name'])
        self._remember_report(info['path'], result)
        self.refresh_table()
        QMessageBox.information(
            self,
            '移除不存在条目',
            f"处理完成。\n保留条目: {result.get('written_entry_count', 0)}"
            + (f"\n\n恢复点:\n{result['restore_point']}" if result.get('restore_point') else ''),
        )

    def repair_selected(self):
        info = self._selected_info()
        if not info:
            QMessageBox.warning(self, 'JSONA 管理', '请先选择一个 JSONA 文件。')
            return
        if not info['exists']:
            QMessageBox.information(self, 'JSONA 管理', '当前文件不存在。')
            return

        report = self.backup_manager.inspect_jsona_file(info['path'], expected_name=info['name'], check_files=False)
        self._remember_report(info['path'], report)
        needs_repair = (
            report.get('invalid_entry_count', 0) > 0
            or report.get('duplicate_count', 0) > 0
            or report.get('type_mismatch_count', 0) > 0
            or report.get('fixed_field_count', 0) > 0
            or report.get('top_level_fixed', False)
        )
        if not needs_repair:
            QMessageBox.information(self, '修复结构', '当前 JSONA 未发现可自动修复的结构问题。')
            return

        reply = QMessageBox.question(
            self,
            '修复结构',
            '将移除无效/重复条目，并修正可确定的字段。修复前会自动备份当前 JSONA。是否继续?',
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            result = self.backup_manager.repair_jsona_file(info['path'], expected_name=info['name'])
        except Exception as exc:
            QMessageBox.warning(self, '修复结构', f'修复失败:\n{exc}')
            return

        self._remember_report(info['path'], result)
        self.refresh_table()
        QMessageBox.information(
            self,
            '修复结构',
            f"修复完成。\n保留条目: {result.get('written_entry_count', 0)}"
            + (f"\n\n恢复点:\n{result['restore_point']}" if result.get('restore_point') else ''),
        )
