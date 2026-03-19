"""Inbox for VLM review-needed items (text-centric).

Used by VLM batch rewrite when the model output is uncertain or invalid.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List, Optional, Tuple


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _dir_size_bytes(path: str) -> int:
    total = 0
    if not os.path.exists(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                continue
    return total


@dataclass
class VlmInboxPolicy:
    max_mb: int = 2048
    on_full: str = "pause"  # pause | stop


class VlmReviewInbox:
    def __init__(self, base_dir: str, policy: Optional[VlmInboxPolicy] = None, session_id: Optional[str] = None):
        self.base_dir = os.path.abspath(base_dir)
        self.root = os.path.join(self.base_dir, "review_inbox_vlm")
        self.items_root = os.path.join(self.root, "items")
        self.index_file = os.path.join(self.root, "inbox.json")
        self.policy = policy or VlmInboxPolicy()
        self.session_id = session_id or _now_ts()
        _safe_mkdir(self.items_root)

    def current_size_mb(self) -> float:
        return _dir_size_bytes(self.root) / (1024 * 1024)

    def would_exceed(self, extra_bytes: int) -> bool:
        max_bytes = int(self.policy.max_mb) * 1024 * 1024
        return _dir_size_bytes(self.root) + int(extra_bytes) > max_bytes

    @staticmethod
    def make_base_id(key: str) -> str:
        h = hashlib.sha1((key or "").encode("utf-8")).hexdigest()
        return h[:16]

    @classmethod
    def make_item_id(cls, key: str, revision: int = 0) -> str:
        base_id = cls.make_base_id(key)
        return base_id if revision <= 0 else f"{base_id}-r{revision}"

    @staticmethod
    def _record_base_id(record: Dict) -> str:
        root_id = str(record.get("root_id") or "").strip()
        if root_id:
            return root_id
        item_id = str(record.get("id") or "").strip()
        if "-r" in item_id:
            return item_id.split("-r", 1)[0]
        return item_id

    @staticmethod
    def _record_revision(record: Dict) -> int:
        raw = record.get("revision")
        try:
            return max(0, int(raw))
        except Exception:
            item_id = str(record.get("id") or "")
            if "-r" in item_id:
                try:
                    return max(0, int(item_id.split("-r", 1)[1]))
                except Exception:
                    return 0
        return 0

    def _item_dir(self, item_id: str) -> str:
        return os.path.join(self.items_root, self.session_id, item_id)

    def _estimate_bytes(self, paths: List[str]) -> int:
        total = 0
        for p in paths:
            if not p:
                continue
            try:
                total += os.path.getsize(p)
            except OSError:
                continue
        return total + 16 * 1024

    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path:
            return ""
        return os.path.abspath(path).replace("\\", "/")

    def _content_signature(self, *, key: str, image_path: str, tags: str, nl_prompt: str, reason: str) -> str:
        payload = {
            "key": key or "",
            "image_src": self._normalize_path(image_path),
            "tags": tags or "",
            "nl_prompt": nl_prompt or "",
            "reason": reason or "",
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def inspect_duplicate(
        self,
        *,
        key: str,
        image_path: str,
        tags: str,
        nl_prompt: str,
        reason: str,
        records: Optional[List[Dict]] = None,
    ) -> Dict:
        records = records if records is not None else self.load_index()
        base_id = self.make_base_id(key)
        signature = self._content_signature(
            key=key,
            image_path=image_path,
            tags=tags,
            nl_prompt=nl_prompt,
            reason=reason,
        )

        matches: List[Tuple[int, Dict]] = []
        same_match: Optional[Tuple[int, Dict]] = None
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            if self._record_base_id(record) != base_id:
                continue
            matches.append((idx, record))
            if record.get("content_signature") == signature:
                same_match = (idx, record)

        if same_match is not None:
            return {
                "status": "same",
                "base_id": base_id,
                "next_revision": self._record_revision(same_match[1]),
                "content_signature": signature,
                "existing_index": same_match[0],
                "record": same_match[1],
                "record_count": len(matches),
            }

        if not matches:
            return {
                "status": "new",
                "base_id": base_id,
                "next_revision": 0,
                "content_signature": signature,
                "existing_index": -1,
                "record": None,
                "record_count": 0,
            }

        matches.sort(key=lambda item: (self._record_revision(item[1]), item[1].get("ts", "")))
        latest_index, latest_record = matches[-1]
        return {
            "status": "duplicate",
            "base_id": base_id,
            "next_revision": self._record_revision(latest_record) + 1,
            "content_signature": signature,
            "existing_index": latest_index,
            "record": latest_record,
            "record_count": len(matches),
        }

    def load_index(self) -> List[Dict]:
        if not os.path.exists(self.index_file):
            return []
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def save_index(self, records: List[Dict]) -> None:
        _safe_mkdir(self.root)
        tmp = self.index_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records or [], f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.index_file)

    def iter_pending(self) -> List[Dict]:
        return [r for r in self.load_index() if isinstance(r, dict) and r.get("status") == "pending"]

    def mark_done(self, item_id: str) -> bool:
        records = self.load_index()
        changed = False
        for r in records:
            if not isinstance(r, dict):
                continue
            if r.get("id") == item_id and r.get("status") != "done":
                r["status"] = "done"
                r["done_ts"] = _now_ts()
                changed = True
        if changed:
            self.save_index(records)
        return changed

    def add_item(
        self,
        *,
        key: str,
        image_path: str,
        tags: str,
        nl_prompt: str,
        reason: str,
        duplicate_mode: str = "reuse",
    ) -> Tuple[bool, Dict]:
        existing_index = self.load_index()
        duplicate_info = self.inspect_duplicate(
            key=key,
            image_path=image_path,
            tags=tags,
            nl_prompt=nl_prompt,
            reason=reason,
            records=existing_index,
        )
        if duplicate_info["status"] == "same":
            existing = deepcopy(duplicate_info["record"])
            existing["already_exists"] = True
            existing["same_content"] = True
            existing["pending_delta"] = 0
            return True, existing
        if duplicate_info["status"] == "duplicate" and duplicate_mode == "reuse":
            existing = deepcopy(duplicate_info["record"])
            existing["already_exists"] = True
            existing["duplicate_conflict"] = True
            existing["pending_delta"] = 0
            return True, existing

        item_id = self.make_item_id(key, duplicate_info["next_revision"])

        item_dir = self._item_dir(item_id)
        _safe_mkdir(item_dir)

        extra_bytes = self._estimate_bytes([image_path])
        if self.would_exceed(extra_bytes):
            return False, {
                "error": "inbox_full",
                "max_mb": self.policy.max_mb,
                "current_mb": self.current_size_mb(),
                "need_mb": extra_bytes / (1024 * 1024),
                "on_full": self.policy.on_full,
            }

        img_dst = os.path.join(item_dir, "image" + os.path.splitext(image_path)[1].lower())
        try:
            shutil.copy2(image_path, img_dst)
            img_rel = os.path.relpath(img_dst, self.root)
        except Exception:
            img_rel = None

        record = {
            "id": item_id,
            "root_id": duplicate_info["base_id"],
            "revision": duplicate_info["next_revision"],
            "status": "pending",
            "ts": _now_ts(),
            "session": self.session_id,
            "content_signature": duplicate_info["content_signature"],
            "key": key,
            "image_src": image_path,
            "tags": tags,
            "nl_prompt": nl_prompt,
            "reason": reason,
            "stored": {
                "root": os.path.relpath(item_dir, self.root),
                "image": img_rel,
            },
        }
        pending_delta = 1

        with open(os.path.join(item_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        if duplicate_info["status"] == "duplicate" and duplicate_mode == "overwrite":
            existing_record = existing_index[duplicate_info["existing_index"]]
            if isinstance(existing_record, dict):
                previous_status = existing_record.get("status", "pending")
                existing_record["archived_status"] = existing_record.get("status", "pending")
                existing_record["status"] = "superseded"
                existing_record["superseded_ts"] = _now_ts()
                existing_record["superseded_by"] = item_id
                existing_record["superseded_mode"] = "overwrite"
                if previous_status == "pending":
                    pending_delta = 0
        existing_index.append(record)
        self.save_index(existing_index)
        record["pending_delta"] = pending_delta
        return True, record
