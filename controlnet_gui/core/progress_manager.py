"""
Progress Manager - Crash recovery and progress tracking
"""
import json
import os
import threading
from typing import Dict, Set
from datetime import datetime


class ProgressManager:
    """Manage processing progress for crash recovery."""

    def __init__(self, progress_file: str = ".progress.json"):
        self.progress_file = progress_file
        self._lock = threading.RLock()
        self.data = self._create_default_data()
        self.save_counter = 0
        self.save_interval = 10
        self.load()

    def _create_default_data(self) -> Dict:
        return {
            'version': '1.0',
            'started_at': None,
            'last_updated': None,
            'processed': [],
            'accepted': [],
            'rejected': [],
            'user_confirmed': [],
            'user_discarded': [],
            'failed': [],
            'statistics': {
                'total_processed': 0,
                'auto_accepted': 0,
                'auto_rejected': 0,
                'user_confirmed': 0,
                'user_discarded': 0,
                'failed': 0,
                'retry_count': 0,
            },
        }

    def _merge_data(self, loaded_data: Dict):
        merged = self._create_default_data()
        merged.update({k: v for k, v in loaded_data.items() if k != 'statistics'})
        if isinstance(loaded_data.get('statistics'), dict):
            merged['statistics'].update(loaded_data['statistics'])
        self.data = merged

    def load(self):
        """Load progress from file."""
        with self._lock:
            if not os.path.exists(self.progress_file):
                return
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                self._merge_data(loaded_data)
                print(f"Progress loaded: {self.data['statistics']['total_processed']} items processed")
            except Exception as e:
                print(f"Failed to load progress: {e}")

    def save(self, force: bool = False):
        """Save progress to file atomically."""
        with self._lock:
            self.save_counter += 1
            if not force and self.save_counter < self.save_interval:
                return

            self.save_counter = 0
            self.data['last_updated'] = datetime.now().isoformat()
            temp_file = self.progress_file + '.tmp'

            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, indent=2, ensure_ascii=False)
                os.replace(temp_file, self.progress_file)
            except Exception as e:
                print(f"Failed to save progress: {e}")
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except OSError:
                    pass

    def start(self):
        """Mark processing as started."""
        with self._lock:
            if not self.data['started_at']:
                self.data['started_at'] = datetime.now().isoformat()
        self.save(force=True)

    def is_processed(self, item_key: str) -> bool:
        """Check if a review item has been processed."""
        with self._lock:
            return item_key in self.data['processed']

    def mark_processed(self, item_key: str):
        """Mark a review item as processed."""
        with self._lock:
            if item_key not in self.data['processed']:
                self.data['processed'].append(item_key)
                self.data['statistics']['total_processed'] += 1

    def mark_accepted(self, item_key: str, auto: bool = True):
        """Mark a review item as accepted."""
        with self._lock:
            self.mark_processed(item_key)
            if auto:
                if item_key not in self.data['accepted']:
                    self.data['accepted'].append(item_key)
                    self.data['statistics']['auto_accepted'] += 1
            else:
                if item_key not in self.data['user_confirmed']:
                    self.data['user_confirmed'].append(item_key)
                    self.data['statistics']['user_confirmed'] += 1
        self.save()

    def mark_rejected(self, item_key: str, auto: bool = True):
        """Mark a review item as rejected."""
        with self._lock:
            self.mark_processed(item_key)
            if auto:
                if item_key not in self.data['rejected']:
                    self.data['rejected'].append(item_key)
                    self.data['statistics']['auto_rejected'] += 1
            else:
                if item_key not in self.data['user_discarded']:
                    self.data['user_discarded'].append(item_key)
                    self.data['statistics']['user_discarded'] += 1
        self.save()

    def mark_failed(self, item_key: str, reason: str = ""):
        """Mark a review item as failed."""
        with self._lock:
            self.mark_processed(item_key)
            if item_key not in self.data['failed']:
                self.data['failed'].append(item_key)
                self.data['statistics']['failed'] += 1
        self.save()

    def increment_retry(self):
        """Increment retry counter."""
        with self._lock:
            self.data['statistics']['retry_count'] += 1

    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        with self._lock:
            return self.data['statistics'].copy()

    def get_processed_set(self) -> Set[str]:
        """Get set of processed review item keys."""
        with self._lock:
            return set(self.data['processed'])

    def reset(self):
        """Reset progress (start fresh)."""
        with self._lock:
            self.data = self._create_default_data()
            self.save_counter = 0
        self.save(force=True)

    def generate_report(self, output_file: str = "report.txt"):
        """Generate processing report."""
        with self._lock:
            stats = self.data['statistics'].copy()
            started_at = self.data['started_at']
            last_updated = self.data['last_updated']

        total = stats['total_processed']
        if total == 0:
            return

        report_lines = [
            "=" * 60,
            "ControlNet Data Processing Report",
            "=" * 60,
            "",
            f"Started: {started_at}",
            f"Last Updated: {last_updated}",
            "",
            "Processing Statistics:",
            "-" * 60,
            f"Total Processed: {total}",
            f"  - Auto Accepted: {stats['auto_accepted']} ({stats['auto_accepted']/total*100:.1f}%)",
            f"  - Auto Rejected: {stats['auto_rejected']} ({stats['auto_rejected']/total*100:.1f}%)",
            f"  - User Confirmed: {stats['user_confirmed']} ({stats['user_confirmed']/total*100:.1f}%)",
            f"  - User Discarded: {stats['user_discarded']} ({stats['user_discarded']/total*100:.1f}%)",
            f"  - Failed: {stats['failed']} ({stats['failed']/total*100:.1f}%)",
            "",
            f"Total Retries: {stats['retry_count']}",
            "",
            "Final Output:",
            "-" * 60,
            f"Accepted Images: {stats['auto_accepted'] + stats['user_confirmed']}",
            f"Rejected Images: {stats['auto_rejected'] + stats['user_discarded']}",
            "",
            "=" * 60,
        ]

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            print(f"Report generated: {output_file}")
        except Exception as e:
            print(f"Failed to generate report: {e}")
