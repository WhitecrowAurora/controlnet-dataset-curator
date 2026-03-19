"""Dialog: VLM tag verification and NL prompt rewrite (batch + manual review)."""

from __future__ import annotations

import os
import json
import threading
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
)

from .duplicate_resolution_dialog import DuplicateResolutionDialog
from ..core.jsona_backup import JsonaBackupManager
from ..core.vlm_client import VlmClient, VlmConfig
from ..core.vlm_review_inbox import VlmReviewInbox, VlmInboxPolicy


def _load_jsona(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # tolerate wrapped shapes
    if isinstance(data, dict):
        for k in ("entries", "items", "data", "records"):
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []


class _BatchWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    finished_err = pyqtSignal(str)
    paused = pyqtSignal(str)
    duplicate_resolution_needed = pyqtSignal(dict)

    def __init__(self, tag_jsona: str, out_dir: str, cfg: VlmConfig, inbox_policy: VlmInboxPolicy, backup_manager: JsonaBackupManager):
        super().__init__()
        self.tag_jsona = tag_jsona
        self.out_dir = out_dir
        self.cfg = cfg
        self.inbox_policy = inbox_policy
        self.backup_manager = backup_manager
        self._stop = False
        self._duplicate_resolution_default = None
        self._duplicate_resolution_event = None
        self._duplicate_resolution_result = None

    def stop(self):
        self._stop = True

    def _request_duplicate_resolution(self, payload: Dict) -> str:
        if self._duplicate_resolution_default in ("overwrite", "new_revision"):
            return self._duplicate_resolution_default
        self._duplicate_resolution_event = threading.Event()
        self._duplicate_resolution_result = {"action": "new_revision", "apply_all": False}
        self.duplicate_resolution_needed.emit(payload)
        self._duplicate_resolution_event.wait()
        result = self._duplicate_resolution_result or {"action": "new_revision", "apply_all": False}
        action = str(result.get("action") or "new_revision")
        if result.get("apply_all"):
            self._duplicate_resolution_default = action
        self._duplicate_resolution_event = None
        self._duplicate_resolution_result = None
        return action

    def resolve_duplicate_decision(self, action: str, apply_all: bool = False):
        self._duplicate_resolution_result = {
            "action": action,
            "apply_all": bool(apply_all),
        }
        if self._duplicate_resolution_event is not None:
            self._duplicate_resolution_event.set()

    @staticmethod
    def _short_text(value: str, limit: int = 900) -> str:
        text = (value or "").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _format_existing_summary(self, record: Dict) -> str:
        lines = [
            f"revision: {record.get('revision', 0)}",
            f"time: {record.get('ts', '')}",
            "",
            "tags:",
            self._short_text(record.get("tags", "")),
            "",
            "nl_prompt:",
            self._short_text(record.get("nl_prompt", "")),
            "",
            "reason:",
            self._short_text(record.get("reason", "")),
        ]
        return "\n".join(lines).strip()

    def _format_new_summary(self, *, next_revision: int, tags: str, nl_prompt: str, reason: str) -> str:
        lines = [
            f"revision: {next_revision}",
            "time: 当前这次",
            "",
            "tags:",
            self._short_text(tags),
            "",
            "nl_prompt:",
            self._short_text(nl_prompt),
            "",
            "reason:",
            self._short_text(reason),
        ]
        return "\n".join(lines).strip()

    def run(self):
        try:
            tag_entries = _load_jsona(self.tag_jsona)
            client = VlmClient(self.cfg)
            inbox = VlmReviewInbox(self.out_dir, policy=self.inbox_policy)

            nl_file = os.path.join(self.out_dir, "nl.jsona")
            written = 0
            updated = 0
            needs_review = 0
            processed = 0

            for item in tag_entries:
                if self._stop:
                    break

                if not isinstance(item, dict):
                    continue
                image_path = (item.get("hint_image_path") or "").strip()
                tags = (item.get("hint_prompt") or "").strip()
                if not image_path or not os.path.exists(image_path):
                    continue

                processed += 1
                self.progress.emit(f"VLM处理中: {processed}/{len(tag_entries)}")

                try:
                    result = client.rewrite(image_path, tags)
                    nl_prompt = (result.get("nl_prompt") or "").strip()
                    needs = bool(result.get("needs_review", False)) or (not nl_prompt)
                    reason = (result.get("reason") or "").strip()
                except Exception as exc:
                    nl_prompt = ""
                    needs = True
                    reason = f"inference_error: {exc}"

                if needs:
                    duplicate_mode = "reuse"
                    duplicate_info = inbox.inspect_duplicate(
                        key=image_path,
                        image_path=image_path,
                        tags=tags,
                        nl_prompt=nl_prompt,
                        reason=reason or "needs_review",
                    )
                    if duplicate_info.get("status") == "duplicate":
                        duplicate_mode = self._request_duplicate_resolution({
                            "image_path": image_path,
                            "existing_revision": int(duplicate_info.get("record", {}).get("revision", 0) or 0),
                            "next_revision": int(duplicate_info.get("next_revision", 0) or 0),
                            "record_count": int(duplicate_info.get("record_count", 0) or 0),
                            "existing_summary": self._format_existing_summary(duplicate_info.get("record", {}) or {}),
                            "new_summary": self._format_new_summary(
                                next_revision=int(duplicate_info.get("next_revision", 0) or 0),
                                tags=tags,
                                nl_prompt=nl_prompt,
                                reason=reason or "needs_review",
                            ),
                        })
                    ok, info = inbox.add_item(
                        key=image_path,
                        image_path=image_path,
                        tags=tags,
                        nl_prompt=nl_prompt,
                        reason=reason or "needs_review",
                        duplicate_mode=duplicate_mode,
                    )
                    if not ok:
                        on_full = str(info.get("on_full", "pause")).lower()
                        msg = f"VLM 审核箱已满，{on_full}。"
                        self.paused.emit(msg)
                        break
                    needs_review += int(info.get("pending_delta", 0) or 0)
                    continue

                entry = {
                    "hint_image_path": os.path.abspath(image_path).replace("\\", "/"),
                    "hint_prompt": nl_prompt,
                    "control_hints_path": os.path.abspath(image_path).replace("\\", "/"),
                    "task_id": "nl",
                }
                res = self.backup_manager.upsert_entries(nl_file, [entry])
                written += int(res.get("added", 0))
                updated += int(res.get("updated", 0))

            self.finished_ok.emit({
                "processed": processed,
                "written": written,
                "updated": updated,
                "needs_review": needs_review,
                "nl_file": nl_file,
                "inbox_dir": inbox.root,
            })
        except Exception as e:
            self.finished_err.emit(str(e))


class VlmTagDialog(QDialog):
    def __init__(self, output_dir: str, backup_manager: JsonaBackupManager, initial_cfg: Optional[VlmConfig] = None, parent=None):
        super().__init__(parent)
        self.output_dir = output_dir
        self.backup_manager = backup_manager
        self._initial_cfg = initial_cfg
        self._worker: Optional[_BatchWorker] = None
        self._inbox = VlmReviewInbox(self.output_dir)
        self._pending: List[Dict] = []
        self._setup_ui()
        self._refresh_manual_list()

    def _setup_ui(self):
        self.setWindowTitle("Tag 核对与 NL 重写 (VLM)")
        self.resize(980, 720)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._tab_batch = self._build_batch_tab()
        self._tab_manual = self._build_manual_tab()
        self.tabs.addTab(self._tab_batch, "批量推理")
        self.tabs.addTab(self._tab_manual, "人工核对")

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        bottom.addWidget(btn_close)
        layout.addLayout(bottom)

    def _build_batch_tab(self):
        w = QDialog()
        v = QVBoxLayout(w)

        intro = QLabel(
            "这个工具只负责调用你已经准备好的 VLM 服务，不负责安装模型。\n"
            "你只需要填写 backend / base_url / model / api_key；CPU 或 GPU 由后端自己决定。\n"
            "批量推理会读取 tag.jsona，稳定写入或更新 nl.jsona；不确定的结果会进入 review_inbox_vlm 供第二天人工核对。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #bbb; padding: 4px 0 8px 0;")
        v.addWidget(intro)

        cfg_group = QGroupBox("推理后端")
        cfg_form = QFormLayout(cfg_group)

        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["openai", "ollama"])
        cfg_form.addRow("backend:", self.combo_backend)

        self.edit_base_url = QLineEdit("http://127.0.0.1:1234")
        cfg_form.addRow("base_url:", self.edit_base_url)

        self.edit_api_key = QLineEdit("")
        self.edit_api_key.setEchoMode(QLineEdit.Password)
        cfg_form.addRow("api_key:", self.edit_api_key)

        self.edit_model = QLineEdit("")
        cfg_form.addRow("model:", self.edit_model)

        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(10, 600)
        self.spin_timeout.setValue(120)
        self.spin_timeout.setSuffix(" s")
        cfg_form.addRow("timeout:", self.spin_timeout)

        v.addWidget(cfg_group)

        # Apply initial values from settings.
        if self._initial_cfg:
            try:
                self.combo_backend.setCurrentText(self._initial_cfg.backend or "openai")
            except Exception:
                pass
            if self._initial_cfg.base_url:
                self.edit_base_url.setText(self._initial_cfg.base_url)
            if self._initial_cfg.api_key is not None:
                self.edit_api_key.setText(self._initial_cfg.api_key)
            if self._initial_cfg.model:
                self.edit_model.setText(self._initial_cfg.model)
            try:
                self.spin_timeout.setValue(int(self._initial_cfg.timeout_seconds))
            except Exception:
                pass

        io_group = QGroupBox("输入/输出")
        io_form = QFormLayout(io_group)

        self.edit_tag_jsona = QLineEdit(os.path.join(self.output_dir, "tag.jsona"))
        btn_pick_tag = QPushButton("选择…")
        btn_pick_tag.clicked.connect(self._pick_tag_jsona)
        row = QHBoxLayout()
        row.addWidget(self.edit_tag_jsona)
        row.addWidget(btn_pick_tag)
        io_form.addRow("tag.jsona:", row)
        io_note = QLabel("输出目标固定为 output/nl.jsona；同一张图重复写入时会按图片路径更新 nl，而不是无限追加重复项。")
        io_note.setWordWrap(True)
        io_note.setStyleSheet("color: #999;")
        io_form.addRow(io_note)

        self.label_batch_status = QLabel("未开始。")
        self.label_batch_status.setWordWrap(True)
        v.addWidget(io_group)
        v.addWidget(self.label_batch_status)

        inbox_group = QGroupBox("无人值守 (VLM 审核箱)")
        inbox_form = QFormLayout(inbox_group)

        self.spin_inbox_max_mb = QSpinBox()
        self.spin_inbox_max_mb.setRange(100, 102400)
        self.spin_inbox_max_mb.setValue(2048)
        self.spin_inbox_max_mb.setSuffix(" MB")
        inbox_form.addRow("最大大小:", self.spin_inbox_max_mb)

        self.combo_inbox_full = QComboBox()
        self.combo_inbox_full.addItems(["pause", "stop"])
        inbox_form.addRow("满额动作:", self.combo_inbox_full)
        inbox_note = QLabel("pause: 暂停当前批量推理，保留现场。\nstop: 直接结束当前批量推理。")
        inbox_note.setWordWrap(True)
        inbox_note.setStyleSheet("color: #999;")
        inbox_form.addRow(inbox_note)
        v.addWidget(inbox_group)

        btns = QHBoxLayout()
        self.btn_start = QPushButton("开始批量推理")
        self.btn_start.clicked.connect(self._start_batch)
        btns.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_batch)
        btns.addWidget(self.btn_stop)

        self.btn_refresh_manual = QPushButton("刷新人工核对列表")
        self.btn_refresh_manual.clicked.connect(self._refresh_manual_list)
        btns.addWidget(self.btn_refresh_manual)

        btns.addStretch(1)
        v.addLayout(btns)

        v.addStretch(1)
        return w

    def _build_manual_tab(self):
        w = QDialog()
        v = QVBoxLayout(w)

        top = QHBoxLayout()
        self.combo_pending = QComboBox()
        self.combo_pending.currentIndexChanged.connect(self._load_selected_pending)
        top.addWidget(QLabel("待处理:"))
        top.addWidget(self.combo_pending, 1)

        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self._refresh_manual_list)
        top.addWidget(btn_refresh)
        v.addLayout(top)

        self.lbl_image = QLabel("无")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMinimumHeight(320)
        self.lbl_image.setStyleSheet("border: 1px solid #444; background: #111; color: #888;")
        v.addWidget(self.lbl_image)

        self.edit_tags = QTextEdit()
        self.edit_tags.setPlaceholderText("tags")
        self.edit_tags.setFixedHeight(90)
        v.addWidget(self.edit_tags)

        self.edit_nl = QTextEdit()
        self.edit_nl.setPlaceholderText("nl_prompt")
        self.edit_nl.setFixedHeight(110)
        v.addWidget(self.edit_nl)

        self.lbl_reason = QLabel("")
        self.lbl_reason.setWordWrap(True)
        self.lbl_reason.setStyleSheet("color: #aaa;")
        v.addWidget(self.lbl_reason)

        btns = QHBoxLayout()
        self.btn_write = QPushButton("确认写入 NL")
        self.btn_write.clicked.connect(self._write_manual)
        btns.addWidget(self.btn_write)

        self.btn_skip = QPushButton("跳过 (标记已处理)")
        self.btn_skip.clicked.connect(self._skip_manual)
        btns.addWidget(self.btn_skip)

        btns.addStretch(1)
        v.addLayout(btns)
        return w

    def _pick_tag_jsona(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 tag.jsona", self.output_dir, "JSONA Files (*.jsona *.json)")
        if path:
            self.edit_tag_jsona.setText(path)

    def _make_cfg(self) -> VlmConfig:
        return VlmConfig(
            backend=self.combo_backend.currentText(),
            base_url=self.edit_base_url.text().strip(),
            api_key=self.edit_api_key.text(),
            model=self.edit_model.text().strip(),
            timeout_seconds=int(self.spin_timeout.value()),
        )

    def _make_inbox_policy(self) -> VlmInboxPolicy:
        return VlmInboxPolicy(
            max_mb=int(self.spin_inbox_max_mb.value()),
            on_full=self.combo_inbox_full.currentText(),
        )

    def _start_batch(self):
        tag_jsona = self.edit_tag_jsona.text().strip()
        if not tag_jsona:
            QMessageBox.warning(self, "批量推理", "请先选择 tag.jsona")
            return
        if not os.path.exists(tag_jsona):
            QMessageBox.warning(self, "批量推理", f"文件不存在:\n{tag_jsona}")
            return
        if os.path.basename(tag_jsona).lower() != "tag.jsona":
            reply = QMessageBox.question(
                self,
                "批量推理",
                "当前选择的文件名不是 tag.jsona。\n\n如果这不是标准 tag 输出，模型结果可能不符合预期。是否继续？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        if self._worker and self._worker.isRunning():
            return

        self.label_batch_status.setText("启动中…")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._worker = _BatchWorker(
            tag_jsona=tag_jsona,
            out_dir=self.output_dir,
            cfg=self._make_cfg(),
            inbox_policy=self._make_inbox_policy(),
            backup_manager=self.backup_manager,
        )
        self._worker.progress.connect(self.label_batch_status.setText)
        self._worker.paused.connect(self._on_batch_paused)
        self._worker.finished_ok.connect(self._on_batch_ok)
        self._worker.finished_err.connect(self._on_batch_err)
        self._worker.duplicate_resolution_needed.connect(self._on_duplicate_resolution_needed)
        self._worker.start()

    def _stop_batch(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self.label_batch_status.setText("正在停止…")

    def _on_batch_paused(self, msg: str):
        self.label_batch_status.setText(f"{msg}\n你可以先处理 review_inbox_vlm，再继续下一轮。")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._refresh_manual_list()

    def _on_batch_ok(self, report: dict):
        self.label_batch_status.setText(
            f"完成。processed={report.get('processed')} written={report.get('written')} updated={report.get('updated')} needs_review={report.get('needs_review')}\n"
            f"nl_file={report.get('nl_file')}\n"
            f"inbox={report.get('inbox_dir')}"
        )
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._refresh_manual_list()

    def _on_batch_err(self, err: str):
        self.label_batch_status.setText(f"失败: {err}")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._refresh_manual_list()

    def _on_duplicate_resolution_needed(self, payload: dict):
        dialog = DuplicateResolutionDialog(
            title="重复审核项",
            intro=(
                "检测到同一张图已有 VLM 历史审核记录。\n"
                "左侧是旧记录，右侧是这次准备写入的新记录，请选择处理方式。"
            ),
            old_title="旧记录",
            old_text=payload.get("existing_summary", ""),
            new_title="新记录",
            new_text=payload.get("new_summary", ""),
            overwrite_label="最新记录覆盖旧记录",
            separate_label="加入审查窗口单独审查",
            apply_all_label="对本次剩余重复项都使用这个选择",
            parent=self,
        )
        dialog.exec_()
        action, apply_all = dialog.result_data()
        if self._worker:
            self._worker.resolve_duplicate_decision(action, apply_all)

    def _refresh_manual_list(self):
        self._pending = self._inbox.iter_pending()
        self.combo_pending.blockSignals(True)
        try:
            self.combo_pending.clear()
            for r in self._pending:
                rid = r.get("id", "")
                src = os.path.basename(r.get("image_src", "") or "")
                self.combo_pending.addItem(f"{rid} | {src}", userData=rid)
        finally:
            self.combo_pending.blockSignals(False)
        self._load_selected_pending()

    def _load_selected_pending(self):
        idx = self.combo_pending.currentIndex()
        if idx < 0 or idx >= len(self._pending):
            self.lbl_image.setText("无")
            self.lbl_image.setPixmap(QPixmap())
            self.edit_tags.setPlainText("")
            self.edit_nl.setPlainText("")
            self.lbl_reason.setText("")
            return

        rec = self._pending[idx]
        stored = rec.get("stored", {}) or {}
        rel = stored.get("image")
        img_path = os.path.join(self._inbox.root, rel) if rel else rec.get("image_src")

        if img_path and os.path.exists(img_path):
            pix = QPixmap(img_path)
            if not pix.isNull():
                self.lbl_image.setPixmap(pix.scaled(self.lbl_image.width(), self.lbl_image.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.lbl_image.setText("")
            else:
                self.lbl_image.setText("无法预览")
        else:
            self.lbl_image.setText("文件不存在")

        self.edit_tags.setPlainText(rec.get("tags", "") or "")
        self.edit_nl.setPlainText(rec.get("nl_prompt", "") or "")
        self.lbl_reason.setText(f"reason: {rec.get('reason', '')}")

    def _write_manual(self):
        idx = self.combo_pending.currentIndex()
        if idx < 0 or idx >= len(self._pending):
            return
        rec = self._pending[idx]
        nl_prompt = (self.edit_nl.toPlainText() or "").strip()
        if not nl_prompt:
            QMessageBox.warning(self, "人工核对", "nl_prompt 不能为空")
            return

        image_path = (rec.get("image_src") or "").strip()
        if not image_path:
            stored = rec.get("stored", {}) or {}
            rel = stored.get("image")
            image_path = os.path.join(self._inbox.root, rel) if rel else ""
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "人工核对", "图片文件不存在，无法写入")
            return

        nl_file = os.path.join(self.output_dir, "nl.jsona")
        entry = {
            "hint_image_path": os.path.abspath(image_path).replace("\\", "/"),
            "hint_prompt": nl_prompt,
            "control_hints_path": os.path.abspath(image_path).replace("\\", "/"),
            "task_id": "nl",
        }
        try:
            self.backup_manager.upsert_entries(nl_file, [entry])
            self._inbox.mark_done(rec.get("id", ""))
            self._refresh_manual_list()
        except Exception as e:
            QMessageBox.warning(self, "人工核对", f"写入失败: {e}")

    def _skip_manual(self):
        idx = self.combo_pending.currentIndex()
        if idx < 0 or idx >= len(self._pending):
            return
        rec = self._pending[idx]
        self._inbox.mark_done(rec.get("id", ""))
        self._refresh_manual_list()
