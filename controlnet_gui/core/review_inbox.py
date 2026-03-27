"""
Review inbox for unattended mode.

Stores "need review" items on disk so the GUI does not keep images in memory.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
import hashlib
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
class ReviewInboxPolicy:
    max_mb: int = 2048
    on_full: str = "pause"  # pause | stop


class ReviewInbox:
    """
    Disk-backed storage for review-needed items.

    Layout:
      <base_dir>/review_inbox/
        items/<session>/<id>/
          original.png (copied or linked)
          tags.txt (optional)
          v0.png..v3.png (variant images)
          meta.json (record)
        inbox.json (append-only list, safe to rewrite atomically)
    """

    def __init__(self, base_dir: str, policy: Optional[ReviewInboxPolicy] = None, session_id: Optional[str] = None):
        self.base_dir = os.path.abspath(base_dir)
        self.root = os.path.join(self.base_dir, "review_inbox")
        self.items_root = os.path.join(self.root, "items")
        self.index_file = os.path.join(self.root, "inbox.json")
        self.policy = policy or ReviewInboxPolicy()
        self.session_id = session_id or _now_ts()

        _safe_mkdir(self.items_root)

    @staticmethod
    def make_base_id(progress_key: str) -> str:
        h = hashlib.sha1((progress_key or "").encode("utf-8")).hexdigest()
        return h[:16]

    @classmethod
    def make_item_id(cls, progress_key: str, revision: int = 0) -> str:
        """Stable id safe for filesystem paths."""
        base_id = cls.make_base_id(progress_key)
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

    def current_size_mb(self) -> float:
        return _dir_size_bytes(self.root) / (1024 * 1024)

    def would_exceed(self, extra_bytes: int) -> bool:
        max_bytes = int(self.policy.max_mb) * 1024 * 1024
        return _dir_size_bytes(self.root) + int(extra_bytes) > max_bytes

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
        # JSON record + a bit of overhead
        return total + 32 * 1024

    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path:
            return ""
        return os.path.abspath(path).replace("\\", "/")

    def _content_signature(
        self,
        *,
        progress_key: str,
        basename: str,
        control_type: str,
        original_path: str,
        tag_text: str,
        variants: List[Dict],
        best_score: float,
        profile: str,
        prefilter: Optional[Dict] = None,
    ) -> str:
        variant_summary = []
        for idx, variant in enumerate((variants or [])[:4]):
            if not isinstance(variant, dict):
                continue
            variant_summary.append({
                "idx": idx,
                "score": round(float(variant.get("score", 0) or 0), 4),
                "preset": variant.get("preset", "unknown"),
                "thresholds": variant.get("thresholds"),
                "warning": variant.get("warning"),
                "metrics": variant.get("metrics"),
                "detections": variant.get("detections"),
                "visibility_ratio": variant.get("visibility_ratio"),
                "is_vitpose": bool(variant.get("is_vitpose", False)),
            })
        payload = {
            "progress_key": progress_key or "",
            "basename": basename or "",
            "control_type": control_type or "",
            "original_src": self._normalize_path(original_path),
            "tags": tag_text or "",
            "best_score": round(float(best_score or 0), 4),
            "profile": profile or "",
            "prefilter": prefilter or {},
            "variants": variant_summary,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def inspect_duplicate(
        self,
        *,
        progress_key: str,
        basename: str,
        control_type: str,
        original_path: str,
        tag_text: str,
        variants: List[Dict],
        best_score: float,
        profile: str,
        prefilter: Optional[Dict] = None,
        records: Optional[List[Dict]] = None,
    ) -> Dict:
        records = records if records is not None else self.load_index()
        base_id = self.make_base_id(progress_key)
        signature = self._content_signature(
            progress_key=progress_key,
            basename=basename,
            control_type=control_type,
            original_path=original_path,
            tag_text=tag_text,
            variants=variants,
            best_score=best_score,
            profile=profile,
            prefilter=prefilter,
        )

        matches, same_match = self._collect_duplicate_matches(records, base_id, signature)
        return self._build_duplicate_inspection_result(base_id, signature, matches, same_match)

    def _collect_duplicate_matches(
        self,
        records: List[Dict],
        base_id: str,
        signature: str,
    ) -> Tuple[List[Tuple[int, Dict]], Optional[Tuple[int, Dict]]]:
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
        return matches, same_match

    def _build_duplicate_inspection_result(
        self,
        base_id: str,
        signature: str,
        matches: List[Tuple[int, Dict]],
        same_match: Optional[Tuple[int, Dict]],
    ) -> Dict:
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

    def add_item(
        self,
        *,
        progress_key: str,
        basename: str,
        control_type: str,
        original_path: str,
        tag_text: str,
        variants: List[Dict],
        best_score: float,
        profile: str,
        prefilter: Optional[Dict] = None,
        duplicate_mode: str = "reuse",
    ) -> Tuple[bool, Dict]:
        """
        Returns (stored, record). If stored is False, caller should apply on_full policy.
        """
        existing_index, duplicate_info = self._prepare_add_item_duplicate_info(
            progress_key=progress_key,
            basename=basename,
            control_type=control_type,
            original_path=original_path,
            tag_text=tag_text,
            variants=variants,
            best_score=best_score,
            profile=profile,
            prefilter=prefilter,
        )
        early_result = self._resolve_add_item_duplicate_short_circuit(duplicate_info, duplicate_mode)
        if early_result is not None:
            return early_result

        item_id, item_dir = self._create_add_item_storage_dir(progress_key, duplicate_info)

        storage_result, storage_error = self._prepare_add_item_storage(
            item_dir=item_dir,
            original_path=original_path,
            variants=variants,
            tag_text=tag_text,
        )
        if storage_error is not None:
            return False, storage_error

        return self._finalize_add_item_record(
            existing_index=existing_index,
            duplicate_info=duplicate_info,
            duplicate_mode=duplicate_mode,
            item_id=item_id,
            item_dir=item_dir,
            progress_key=progress_key,
            basename=basename,
            control_type=control_type,
            original_path=original_path,
            tag_text=tag_text,
            profile=profile,
            best_score=best_score,
            prefilter=prefilter,
            storage_result=storage_result,
        )

    def _resolve_add_item_duplicate_short_circuit(
        self, duplicate_info: Dict, duplicate_mode: str
    ) -> Optional[Tuple[bool, Dict]]:
        handled, stored, resolved = self._resolve_duplicate_add_result(duplicate_info, duplicate_mode)
        if handled:
            return stored, resolved
        return None

    def _create_add_item_storage_dir(self, progress_key: str, duplicate_info: Dict) -> Tuple[str, str]:
        item_id = self.make_item_id(progress_key, duplicate_info["next_revision"])
        item_dir = self._item_dir(item_id)
        _safe_mkdir(item_dir)
        return item_id, item_dir

    def _prepare_add_item_duplicate_info(
        self,
        *,
        progress_key: str,
        basename: str,
        control_type: str,
        original_path: str,
        tag_text: str,
        variants: List[Dict],
        best_score: float,
        profile: str,
        prefilter: Optional[Dict],
    ) -> Tuple[List[Dict], Dict]:
        existing_index = self.load_index()
        duplicate_info = self.inspect_duplicate(
            progress_key=progress_key,
            basename=basename,
            control_type=control_type,
            original_path=original_path,
            tag_text=tag_text,
            variants=variants,
            best_score=best_score,
            profile=profile,
            prefilter=prefilter,
            records=existing_index,
        )
        return existing_index, duplicate_info

    def _prepare_add_item_storage(
        self,
        *,
        item_dir: str,
        original_path: str,
        variants: List[Dict],
        tag_text: str,
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        variant_paths, to_copy = self._prepare_variant_copy_inputs(original_path, variants)
        extra_bytes = self._estimate_bytes(to_copy)
        if self.would_exceed(extra_bytes):
            return None, self._build_inbox_full_error(extra_bytes)

        copied_original, original_copy_error = self._copy_inbox_original(item_dir, original_path)
        if copied_original is None:
            return None, self._build_original_copy_failed_error(original_copy_error)

        copied_variants, variant_copy_failures = self._copy_inbox_variants(item_dir, variant_paths, variants or [])
        if variants and not copied_variants:
            shutil.rmtree(item_dir, ignore_errors=True)
            return None, self._build_variant_copy_failed_error(variant_copy_failures, len(variants))

        return {
            "copied_original": copied_original,
            "copied_tags": self._write_inbox_tags(item_dir, tag_text),
            "copied_variants": copied_variants,
            "variant_copy_failures": variant_copy_failures,
        }, None

    def _finalize_add_item_record(
        self,
        *,
        existing_index: List[Dict],
        duplicate_info: Dict,
        duplicate_mode: str,
        item_id: str,
        item_dir: str,
        progress_key: str,
        basename: str,
        control_type: str,
        original_path: str,
        tag_text: str,
        profile: str,
        best_score: float,
        prefilter: Optional[Dict],
        storage_result: Dict,
    ) -> Tuple[bool, Dict]:
        record = self._build_inbox_record(
            item_id=item_id,
            item_dir=item_dir,
            duplicate_info=duplicate_info,
            progress_key=progress_key,
            basename=basename,
            control_type=control_type,
            original_path=original_path,
            tag_text=tag_text,
            profile=profile,
            best_score=best_score,
            prefilter=prefilter,
            copied_original=storage_result["copied_original"],
            copied_tags=storage_result["copied_tags"],
            copied_variants=storage_result["copied_variants"],
            variant_copy_failures=storage_result.get("variant_copy_failures", []),
        )
        return self._persist_added_record(
            existing_index, record, item_dir, duplicate_info, duplicate_mode, item_id
        )

    @staticmethod
    def _prepare_variant_copy_inputs(original_path: str, variants: List[Dict]) -> Tuple[List[str], List[str]]:
        variant_paths = [v.get("path") for v in (variants or [])][:4]
        to_copy = [original_path] + [p for p in variant_paths if p]
        return variant_paths, to_copy

    def _build_inbox_full_error(self, extra_bytes: int) -> Dict:
        return {
            "error": "inbox_full",
            "max_mb": self.policy.max_mb,
            "current_mb": self.current_size_mb(),
            "need_mb": extra_bytes / (1024 * 1024),
            "on_full": self.policy.on_full,
        }

    @staticmethod
    def _build_original_copy_failed_error(original_copy_error: Optional[str]) -> Dict:
        return {
            "error": "original_copy_failed",
            "message": original_copy_error or "failed_to_copy_original",
        }

    @staticmethod
    def _build_variant_copy_failed_error(copy_failures: List[Dict], requested_count: int) -> Dict:
        return {
            "error": "variant_copy_failed",
            "message": "failed_to_copy_any_variant",
            "requested_count": int(requested_count or 0),
            "failures": list(copy_failures or []),
        }

    def _persist_added_record(
        self,
        existing_index: List[Dict],
        record: Dict,
        item_dir: str,
        duplicate_info: Dict,
        duplicate_mode: str,
        item_id: str,
    ) -> Tuple[bool, Dict]:
        with open(os.path.join(item_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        pending_delta = self._apply_duplicate_overwrite(existing_index, duplicate_info, duplicate_mode, item_id)
        existing_index.append(record)
        self.save_index(existing_index)
        record["pending_delta"] = pending_delta
        return True, record

    def _resolve_duplicate_add_result(self, duplicate_info: Dict, duplicate_mode: str) -> Tuple[bool, bool, Dict]:
        if duplicate_info["status"] == "same":
            existing = deepcopy(duplicate_info["record"])
            existing["already_exists"] = True
            existing["same_content"] = True
            existing["pending_delta"] = 0
            return True, True, existing
        if duplicate_info["status"] == "duplicate" and duplicate_mode == "reuse":
            existing = deepcopy(duplicate_info["record"])
            existing["already_exists"] = True
            existing["duplicate_conflict"] = True
            existing["pending_delta"] = 0
            return True, True, existing
        return False, False, {}

    def _copy_inbox_original(self, item_dir: str, original_path: str) -> Tuple[Optional[str], Optional[str]]:
        orig_dst = os.path.join(item_dir, "original" + os.path.splitext(original_path)[1].lower())
        try:
            shutil.copy2(original_path, orig_dst)
            return os.path.relpath(orig_dst, self.root), None
        except Exception as exc:
            shutil.rmtree(item_dir, ignore_errors=True)
            return None, str(exc)

    def _write_inbox_tags(self, item_dir: str, tag_text: str) -> Optional[str]:
        tag_dst = os.path.join(item_dir, "tags.txt")
        if not tag_text:
            return None
        with open(tag_dst, "w", encoding="utf-8") as f:
            f.write(tag_text)
        return os.path.relpath(tag_dst, self.root)

    def _copy_inbox_variants(self, item_dir: str, variant_paths: List[str], variants: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        copied_variants = []
        copy_failures = []
        for idx, src in enumerate(variant_paths):
            if not src or not os.path.exists(src):
                copy_failures.append({
                    "idx": idx,
                    "path": src or "",
                    "error": "missing_source_variant",
                })
                continue
            dst = os.path.join(item_dir, f"v{idx}.png")
            try:
                shutil.copy2(src, dst)
                copied_variants.append({
                    "idx": idx,
                    "path": os.path.relpath(dst, self.root),
                    "score": (variants[idx] or {}).get("score", 0),
                    "preset": (variants[idx] or {}).get("preset", "unknown"),
                    "thresholds": (variants[idx] or {}).get("thresholds"),
                    "warning": (variants[idx] or {}).get("warning"),
                    "metrics": (variants[idx] or {}).get("metrics"),
                    "detections": (variants[idx] or {}).get("detections"),
                    "visibility_ratio": (variants[idx] or {}).get("visibility_ratio"),
                    "is_vitpose": (variants[idx] or {}).get("is_vitpose", False),
                })
            except Exception as exc:
                copy_failures.append({
                    "idx": idx,
                    "path": src,
                    "error": str(exc),
                })
        return copied_variants, copy_failures

    def _build_inbox_record(
        self,
        *,
        item_id: str,
        item_dir: str,
        duplicate_info: Dict,
        progress_key: str,
        basename: str,
        control_type: str,
        original_path: str,
        tag_text: str,
        profile: str,
        best_score: float,
        prefilter: Optional[Dict],
        copied_original: Optional[str],
        copied_tags: Optional[str],
        copied_variants: List[Dict],
        variant_copy_failures: List[Dict],
    ) -> Dict:
        return {
            "id": item_id,
            "root_id": duplicate_info["base_id"],
            "revision": duplicate_info["next_revision"],
            "ts": _now_ts(),
            "session": self.session_id,
            "status": "pending",
            "content_signature": duplicate_info["content_signature"],
            "progress_key": progress_key,
            "basename": basename,
            "control_type": control_type,
            "original_src": original_path,
            "tags": tag_text,
            "profile": profile,
            "best_score": best_score,
            "prefilter": prefilter,
            "stored": {
                "root": os.path.relpath(item_dir, self.root),
                "original": copied_original,
                "tags": copied_tags,
                "variants": copied_variants,
                "variant_copy_failures": list(variant_copy_failures or []),
            },
        }

    def _apply_duplicate_overwrite(
        self,
        index: List[Dict],
        duplicate_info: Dict,
        duplicate_mode: str,
        item_id: str,
    ) -> int:
        pending_delta = 1
        if duplicate_info["status"] != "duplicate" or duplicate_mode != "overwrite":
            return pending_delta
        existing_record = index[duplicate_info["existing_index"]]
        if not isinstance(existing_record, dict):
            return pending_delta
        previous_status = existing_record.get("status", "pending")
        existing_record["archived_status"] = existing_record.get("status", "pending")
        existing_record["status"] = "superseded"
        existing_record["superseded_ts"] = _now_ts()
        existing_record["superseded_by"] = item_id
        existing_record["superseded_mode"] = "overwrite"
        if previous_status == "pending":
            return 0
        return pending_delta

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

    def mark_done_by_id(self, item_id: str) -> bool:
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

    def mark_done(self, progress_key: str) -> bool:
        records = self.load_index()
        matching = [
            r for r in records
            if isinstance(r, dict) and r.get("progress_key") == progress_key and r.get("status") == "pending"
        ]
        if matching:
            matching.sort(key=lambda r: (self._record_revision(r), r.get("ts", "")), reverse=True)
            return self.mark_done_by_id(matching[0].get("id", ""))

        changed = False
        for r in records:
            if not isinstance(r, dict):
                continue
            if r.get("progress_key") == progress_key and r.get("status") != "done":
                r["status"] = "done"
                r["done_ts"] = _now_ts()
                changed = True
        if changed:
            self.save_index(records)
        return changed
