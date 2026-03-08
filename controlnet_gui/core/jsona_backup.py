"""Utilities for JSONA backup, restore, import, inspection, and repair."""
import json
import os
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class JsonaBackupManager:
    """Manage JSONA backups and maintenance operations."""

    DEFAULT_NAMES = ('canny', 'pose', 'depth', 'metadata')
    KNOWN_TASKS = ('canny', 'pose', 'depth')
    TOP_LEVEL_KEYS = ('entries', 'items', 'data', 'records')

    def __init__(self, backup_interval_entries: int = 200, rolling_keep: int = 10, backup_interval_seconds: int = 600):
        self.backup_interval_entries = max(1, int(backup_interval_entries))
        self.rolling_keep = max(1, int(rolling_keep))
        self.backup_interval_seconds = max(60, int(backup_interval_seconds))
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._session_state: Dict[str, Dict[str, float]] = {}

    def _state_for(self, json_file: str) -> Dict[str, float]:
        return self._session_state.setdefault(json_file, {
            'entries_since_backup': 0,
            'startup_done': 0,
            'last_backup_at': 0.0,
        })

    def _backup_dir(self, json_file: str) -> str:
        json_path = Path(json_file)
        return str(json_path.parent / '.jsona_backups' / json_path.stem)

    def _lock_file(self, json_file: str) -> str:
        return json_file + '.lock'

    def _wait_for_unlock(self, json_file: str, timeout: float = 5.0) -> bool:
        lock_file = self._lock_file(json_file)
        deadline = time.time() + timeout
        while os.path.exists(lock_file) and time.time() < deadline:
            time.sleep(0.1)
        return not os.path.exists(lock_file)

    def _acquire_lock(self, json_file: str, max_retries: int = 10):
        lock_file = self._lock_file(json_file)
        for attempt in range(max_retries):
            try:
                lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(lock_fd)
                return lock_file
            except FileExistsError:
                time.sleep(0.05 * (attempt + 1))
        raise TimeoutError(f'Could not acquire JSONA lock for {json_file}')

    def _release_lock(self, lock_file: str):
        try:
            os.remove(lock_file)
        except OSError:
            pass

    def _cleanup_backups(self, json_file: str):
        backup_dir = self._backup_dir(json_file)
        if not os.path.isdir(backup_dir):
            return

        files = [
            os.path.join(backup_dir, name)
            for name in os.listdir(backup_dir)
            if name.lower().endswith('.jsona')
        ]
        startup_backups = sorted(
            [path for path in files if Path(path).name.startswith('startup_')],
            key=os.path.getmtime,
            reverse=True,
        )
        rolling_backups = sorted(
            [path for path in files if not Path(path).name.startswith('startup_')],
            key=os.path.getmtime,
            reverse=True,
        )

        for path in startup_backups[1:]:
            try:
                os.remove(path)
            except OSError:
                pass

        for path in rolling_backups[self.rolling_keep:]:
            try:
                os.remove(path)
            except OSError:
                pass

    def _copy_backup(self, source_file: str, backup_path: str) -> Optional[str]:
        if not os.path.exists(source_file):
            return None
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(source_file, backup_path)
        return backup_path

    def _read_json_data(self, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _write_json_data(self, json_file: str, data):
        temp_file = json_file + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, json_file)

    def _coerce_entries(self, raw_data) -> Tuple[Optional[List], Optional[str]]:
        if isinstance(raw_data, list):
            return raw_data, None
        if isinstance(raw_data, dict):
            for key in self.TOP_LEVEL_KEYS:
                value = raw_data.get(key)
                if isinstance(value, list):
                    return value, key
        return None, None

    def _normalize_path(self, value: str) -> str:
        if not isinstance(value, str):
            value = '' if value is None else str(value)
        return value.strip().replace('\\', '/')

    def _normalize_task_id(self, task_id: str) -> str:
        if not task_id:
            return ''
        task_id = str(task_id).strip().lower()
        if task_id == 'openpose':
            return 'pose'
        return task_id

    def _app_root(self) -> str:
        return str(Path(__file__).resolve().parents[2])

    def _path_candidates(self, value: str, source_path: str = '') -> List[str]:
        normalized = self._normalize_path(value)
        if not normalized:
            return []

        value_for_os = normalized.replace('/', os.sep)
        candidates: List[str] = []
        seen = set()

        def add_candidate(path: str):
            if not path:
                return
            candidate = os.path.normpath(path)
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

        add_candidate(value_for_os)
        if os.path.isabs(value_for_os):
            return candidates

        source_dir = os.path.dirname(source_path) if source_path else ''
        for base_dir in (self._app_root(), source_dir, os.getcwd()):
            if base_dir:
                add_candidate(os.path.join(base_dir, value_for_os))
        return candidates

    def _path_exists(self, value: str, source_path: str = '') -> bool:
        return any(os.path.exists(candidate) for candidate in self._path_candidates(value, source_path))

    def _infer_task_id_from_path(self, control_path: str) -> str:
        path = self._normalize_path(control_path).lower()
        if not path:
            return ''
        if '_canny' in path or '/canny/' in path:
            return 'canny'
        if '_openpose' in path or '/pose/' in path or '_pose' in path:
            return 'pose'
        if '_depth' in path or '/depth/' in path:
            return 'depth'
        return ''

    def _entry_key(self, entry: Dict) -> Tuple[str, str, str]:
        return (
            entry.get('hint_image_path', ''),
            entry.get('control_hints_path', ''),
            entry.get('task_id', ''),
        )

    def _target_name_to_expected(self, target_name: str) -> Optional[str]:
        if target_name in self.KNOWN_TASKS:
            return target_name
        if target_name == 'metadata':
            return 'metadata'
        return None

    def _normalize_entry(self, item, expected_name: Optional[str] = None) -> Tuple[Optional[Dict], List[str], int]:
        issues: List[str] = []
        fixes = 0

        if not isinstance(item, dict):
            return None, ['entry_not_object'], fixes

        entry = deepcopy(item)
        hint_path = self._normalize_path(entry.get('hint_image_path', ''))
        control_path = self._normalize_path(entry.get('control_hints_path', ''))
        hint_prompt = entry.get('hint_prompt', '')
        if hint_prompt is None:
            hint_prompt = ''
        elif not isinstance(hint_prompt, str):
            hint_prompt = str(hint_prompt)
            fixes += 1

        task_id = self._normalize_task_id(entry.get('task_id', ''))
        inferred_task = self._infer_task_id_from_path(control_path)

        if not task_id:
            if expected_name in self.KNOWN_TASKS:
                task_id = expected_name
                fixes += 1
            elif inferred_task:
                task_id = inferred_task
                fixes += 1

        if not hint_path:
            issues.append('missing_hint_image_path')
        if not control_path:
            issues.append('missing_control_hints_path')
        if not task_id:
            issues.append('missing_task_id')
        elif task_id not in self.KNOWN_TASKS:
            issues.append('invalid_task_id')
        elif expected_name in self.KNOWN_TASKS and task_id != expected_name:
            issues.append('task_id_mismatch')

        if issues:
            return None, issues, fixes

        normalized = {
            'hint_image_path': hint_path,
            'hint_prompt': hint_prompt,
            'control_hints_path': control_path,
            'task_id': task_id,
        }
        return normalized, [], fixes

    def detect_target_name(self, source_path: str, entries: Optional[List] = None) -> Optional[str]:
        stem = Path(source_path).stem.lower()
        if stem in self.DEFAULT_NAMES:
            return stem

        entries = entries or []
        task_ids = set()
        for item in entries:
            if isinstance(item, dict):
                task_id = self._normalize_task_id(item.get('task_id', ''))
                if task_id in self.KNOWN_TASKS:
                    task_ids.add(task_id)
        if len(task_ids) == 1:
            return next(iter(task_ids))
        if len(task_ids) > 1:
            return 'metadata'

        inferred = set()
        for item in entries:
            if isinstance(item, dict):
                task_id = self._infer_task_id_from_path(item.get('control_hints_path', ''))
                if task_id:
                    inferred.add(task_id)
        if len(inferred) == 1:
            return next(iter(inferred))
        if len(inferred) > 1:
            return 'metadata'
        return None

    def analyze_entries(self, entries: List, expected_name: Optional[str] = None, check_files: bool = False, source_path: str = '') -> Dict:
        expected_name = self._target_name_to_expected(expected_name) or expected_name
        seen = set()
        valid_entries: List[Dict] = []
        existing_entries: List[Dict] = []
        problems: List[Dict] = []
        duplicates: List[Dict] = []
        missing_files: List[Dict] = []
        type_mismatch_count = 0
        fixed_fields = 0

        for index, item in enumerate(entries):
            normalized, issues, fixes = self._normalize_entry(item, expected_name)
            fixed_fields += fixes
            if issues:
                if 'task_id_mismatch' in issues:
                    type_mismatch_count += 1
                problems.append({
                    'index': index,
                    'reason': ','.join(issues),
                    'entry': deepcopy(item),
                })
                continue

            entry_key = self._entry_key(normalized)
            if entry_key in seen:
                duplicates.append({
                    'index': index,
                    'reason': 'duplicate_entry',
                    'entry': normalized,
                })
                continue
            seen.add(entry_key)
            valid_entries.append(normalized)

            if check_files:
                missing_fields = []
                if not self._path_exists(normalized['hint_image_path'], source_path):
                    missing_fields.append('hint_image_path')
                if not self._path_exists(normalized['control_hints_path'], source_path):
                    missing_fields.append('control_hints_path')
                if missing_fields:
                    missing_files.append({
                        'index': index,
                        'reason': ','.join(missing_fields),
                        'entry': normalized,
                    })
                    continue

            existing_entries.append(normalized)

        actual_task_ids = sorted({entry['task_id'] for entry in valid_entries})
        return {
            'path': source_path,
            'expected_name': expected_name,
            'entry_count': len(entries),
            'valid_entry_count': len(valid_entries),
            'existing_entry_count': len(existing_entries),
            'invalid_entry_count': len(problems),
            'duplicate_count': len(duplicates),
            'missing_file_count': len(missing_files),
            'type_mismatch_count': type_mismatch_count,
            'fixed_field_count': fixed_fields,
            'actual_task_ids': actual_task_ids,
            'valid_entries': valid_entries,
            'existing_entries': existing_entries,
            'problem_entries': problems,
            'duplicate_entries': duplicates,
            'missing_entries': missing_files,
        }

    def inspect_jsona_file(self, json_file: str, expected_name: Optional[str] = None, check_files: bool = False) -> Dict:
        info = {
            'path': json_file,
            'name': Path(json_file).stem.lower(),
            'exists': os.path.exists(json_file),
            'entry_count': 0,
            'status': 'missing',
            'error': '',
            'modified_at': None,
            'backup_count': len(self.list_backups(json_file)),
            'top_level_fixed': False,
        }
        if not info['exists']:
            return info

        info['modified_at'] = datetime.fromtimestamp(os.path.getmtime(json_file))
        try:
            raw_data = self._read_json_data(json_file)
            entries, top_level_key = self._coerce_entries(raw_data)
            if entries is None:
                info['status'] = 'invalid'
                info['error'] = 'Top-level JSON is not a list.'
                return info

            analysis = self.analyze_entries(entries, expected_name=expected_name, check_files=check_files, source_path=json_file)
            info.update(analysis)
            info['top_level_fixed'] = top_level_key is not None
            issue_count = (
                info.get('invalid_entry_count', 0)
                + info.get('duplicate_count', 0)
                + info.get('missing_file_count', 0)
                + info.get('type_mismatch_count', 0)
            )
            if info.get('error'):
                info['status'] = 'invalid'
            elif issue_count > 0 or info.get('fixed_field_count', 0) > 0 or info.get('top_level_fixed', False):
                info['status'] = 'issues'
            else:
                info['status'] = 'ok'
        except Exception as exc:
            info['status'] = 'invalid'
            info['error'] = str(exc)
        return info

    def list_managed_files(self, output_dir: str) -> List[Dict]:
        results = []
        for name in self.DEFAULT_NAMES:
            json_file = os.path.join(output_dir, f'{name}.jsona')
            info = self.inspect_jsona_file(json_file, expected_name=name, check_files=False)
            info['name'] = name
            results.append(info)
        return results

    def ensure_session_backup(self, json_file: str) -> Optional[str]:
        state = self._state_for(json_file)
        if state['startup_done']:
            return None

        state['startup_done'] = 1
        if not os.path.exists(json_file) or os.path.getsize(json_file) == 0:
            return None

        self._wait_for_unlock(json_file)
        backup_path = os.path.join(self._backup_dir(json_file), f'startup_{self.session_id}.jsona')
        created = self._copy_backup(json_file, backup_path)
        if created:
            state['last_backup_at'] = time.time()
        self._cleanup_backups(json_file)
        return created

    def prepare_output_backups(self, output_dir: str):
        for info in self.list_managed_files(output_dir):
            if info['exists']:
                self.ensure_session_backup(info['path'])

    def create_backup(self, json_file: str, reason: str = 'manual') -> Optional[str]:
        if not os.path.exists(json_file):
            return None

        self._wait_for_unlock(json_file)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(self._backup_dir(json_file), f'{reason}_{timestamp}.jsona')
        created = self._copy_backup(json_file, backup_path)
        if created:
            self._state_for(json_file)['last_backup_at'] = time.time()
        self._cleanup_backups(json_file)
        return created

    def register_write(self, json_file: str, added_entries: int) -> Optional[str]:
        if added_entries <= 0:
            return None

        state = self._state_for(json_file)
        state['entries_since_backup'] += added_entries
        now = time.time()
        if state['entries_since_backup'] < self.backup_interval_entries and (now - state['last_backup_at']) < self.backup_interval_seconds:
            return None

        state['entries_since_backup'] = 0
        state['last_backup_at'] = now
        return self.create_backup(json_file, reason='rolling')

    def list_backups(self, json_file: str) -> List[Dict]:
        backup_dir = self._backup_dir(json_file)
        if not os.path.isdir(backup_dir):
            return []

        results = []
        for name in os.listdir(backup_dir):
            if not name.lower().endswith('.jsona'):
                continue
            path = os.path.join(backup_dir, name)
            results.append({
                'path': path,
                'name': name,
                'modified_at': datetime.fromtimestamp(os.path.getmtime(path)),
                'size': os.path.getsize(path),
            })
        results.sort(key=lambda item: item['modified_at'], reverse=True)
        return results

    def append_entries(self, json_file: str, entries: List[Dict]) -> int:
        if not entries:
            return 0

        self.ensure_session_backup(json_file)
        lock_file = self._acquire_lock(json_file)
        try:
            existing_data = []
            if os.path.exists(json_file):
                try:
                    raw_data = self._read_json_data(json_file)
                except Exception as exc:
                    raise ValueError(
                        f'Existing JSONA is invalid and was not appended: {json_file}. Repair or restore it before continuing. {exc}'
                    ) from exc

                existing_entries, _ = self._coerce_entries(raw_data)
                if not isinstance(existing_entries, list):
                    raise ValueError(
                        f'Existing JSONA top-level structure is invalid and was not appended: {json_file}. Repair or restore it before continuing.'
                    )
                existing_data = existing_entries

            normalized_existing = []
            seen = set()
            for item in existing_data:
                normalized, issues, _ = self._normalize_entry(item, None)
                if normalized and not issues:
                    key = self._entry_key(normalized)
                    if key not in seen:
                        seen.add(key)
                        normalized_existing.append(normalized)

            added = 0
            for entry in entries:
                normalized, issues, _ = self._normalize_entry(entry, None)
                if normalized and not issues:
                    key = self._entry_key(normalized)
                    if key not in seen:
                        seen.add(key)
                        normalized_existing.append(normalized)
                        added += 1

            if added == 0 and os.path.exists(json_file):
                return 0

            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            self._write_json_data(json_file, normalized_existing)
            self.register_write(json_file, added)
            return added
        finally:
            self._release_lock(lock_file)

    def replace_entries(self, json_file: str, entries: List[Dict], reason: str = 'repair') -> Optional[str]:
        backup_path = self.create_backup(json_file, reason='restorepoint') if os.path.exists(json_file) else None
        lock_file = self._acquire_lock(json_file)
        try:
            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            self._write_json_data(json_file, entries)
            self.create_backup(json_file, reason=reason)
            return backup_path
        finally:
            self._release_lock(lock_file)

    def restore_backup(self, json_file: str, backup_path: str) -> Optional[str]:
        if not os.path.exists(backup_path):
            raise FileNotFoundError(backup_path)

        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        self._wait_for_unlock(json_file)
        restore_point = None
        if os.path.exists(json_file):
            restore_point = self.create_backup(json_file, reason='restorepoint')

        temp_file = json_file + '.restore_tmp'
        shutil.copy2(backup_path, temp_file)
        os.replace(temp_file, json_file)
        self._cleanup_backups(json_file)
        return restore_point

    def repair_jsona_file(self, json_file: str, expected_name: Optional[str] = None) -> Dict:
        report = self.inspect_jsona_file(json_file, expected_name=expected_name, check_files=False)
        if report['status'] == 'invalid' and report.get('entry_count', 0) == 0 and not report.get('valid_entries'):
            raise ValueError(report.get('error') or 'JSONA file structure is invalid.')

        repaired_entries = report.get('valid_entries', [])
        backup_path = self.replace_entries(json_file, repaired_entries, reason='repaired')
        report['restore_point'] = backup_path
        report['written_entry_count'] = len(repaired_entries)
        return report

    def remove_missing_entries(self, json_file: str, expected_name: Optional[str] = None) -> Dict:
        report = self.inspect_jsona_file(json_file, expected_name=expected_name, check_files=True)
        cleaned_entries = report.get('existing_entries', [])
        backup_path = self.replace_entries(json_file, cleaned_entries, reason='missing_removed')
        report['restore_point'] = backup_path
        report['written_entry_count'] = len(cleaned_entries)
        return report

    def export_report(self, report: Dict, export_dir: Optional[str] = None) -> Dict[str, str]:
        source_path = report.get('path') or report.get('name', 'jsona')
        stem = Path(source_path).stem
        export_root = export_dir or os.path.dirname(source_path) or os.getcwd()
        os.makedirs(export_root, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        summary = {
            'path': report.get('path', ''),
            'expected_name': report.get('expected_name', ''),
            'entry_count': report.get('entry_count', 0),
            'valid_entry_count': report.get('valid_entry_count', 0),
            'existing_entry_count': report.get('existing_entry_count', 0),
            'invalid_entry_count': report.get('invalid_entry_count', 0),
            'duplicate_count': report.get('duplicate_count', 0),
            'missing_file_count': report.get('missing_file_count', 0),
            'type_mismatch_count': report.get('type_mismatch_count', 0),
            'fixed_field_count': report.get('fixed_field_count', 0),
            'actual_task_ids': report.get('actual_task_ids', []),
        }

        summary_path = os.path.join(export_root, f'{stem}_check_report_{timestamp}.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        issue_entries = [item['entry'] for item in report.get('missing_entries', [])]
        issue_entries.extend(item['entry'] for item in report.get('problem_entries', []))
        issue_entries.extend(item['entry'] for item in report.get('duplicate_entries', []))

        issue_path = ''
        if issue_entries:
            issue_path = os.path.join(export_root, f'{stem}_problem_entries_{timestamp}.jsona')
            with open(issue_path, 'w', encoding='utf-8') as f:
                json.dump(issue_entries, f, ensure_ascii=False, indent=2)

        return {
            'summary': summary_path,
            'issues': issue_path,
        }

    def import_jsona(self, source_path: str, output_dir: str, verify_files: bool = False) -> Dict:
        raw_data = self._read_json_data(source_path)
        entries, top_level_key = self._coerce_entries(raw_data)
        if entries is None:
            raise ValueError('JSONA top-level structure is not a list.')

        target_name = self.detect_target_name(source_path, entries)
        if not target_name:
            raise ValueError('Cannot determine JSONA target type. Please use canny/pose/depth/metadata naming or valid task_id values.')

        expected_name = self._target_name_to_expected(target_name)
        report = self.analyze_entries(entries, expected_name=expected_name, check_files=verify_files, source_path=source_path)
        report['top_level_fixed'] = top_level_key is not None
        report['target_name'] = target_name
        report['target_file'] = os.path.join(output_dir, f'{target_name}.jsona')
        report['imported_count'] = self.append_entries(report['target_file'], report['valid_entries'])
        return report
