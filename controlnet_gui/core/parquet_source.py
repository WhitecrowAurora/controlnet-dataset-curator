"""
Parquet Data Source - Extract data from parquet files or streaming datasets
"""
import os
import io
import json
import requests
from PIL import Image
from typing import Iterator, Dict, List, Optional, Callable
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue


class ParquetDataSource:
    """Extract data from local parquet files with multi-threading support"""

    def __init__(self, parquet_files: List[str], extract_dir: str, num_samples: int = 0, num_threads: int = 4):
        """
        Args:
            parquet_files: List of parquet file paths
            extract_dir: Directory to extract images and tags
            num_samples: Number of samples to extract (0 = all)
            num_threads: Number of threads for parallel extraction
        """
        self.parquet_files = parquet_files
        self.extract_dir = extract_dir
        self.num_samples = num_samples
        self.num_threads = num_threads
        self.image_dir = os.path.join(extract_dir, "images")
        self.tag_dir = os.path.join(extract_dir, "tags")

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.tag_dir, exist_ok=True)

        self.counter_lock = threading.Lock()
        self.counter = 0

    def extract(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> int:
        """
        Extract images and tags from parquet files using multi-threading

        Args:
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Number of images extracted
        """
        pq = self._load_local_pyarrow()
        if pq is None:
            return 0

        all_rows = self._collect_local_parquet_rows(pq)

        if not all_rows:
            print("No data to extract")
            return 0

        print(f"Extracting {len(all_rows)} images using {self.num_threads} threads...")
        self._run_local_extraction_pool(all_rows, progress_callback)

        print(f"Extraction complete: {self.counter} images extracted to {self.extract_dir}")
        return self.counter

    @staticmethod
    def _load_local_pyarrow():
        try:
            import pyarrow.parquet as pq
            return pq
        except ImportError:
            print("Error: pyarrow not installed. Run: pip install pyarrow")
            return None

    def _collect_local_parquet_rows(self, pq) -> List:
        all_rows = []
        total_rows = 0

        print("Reading parquet files...")
        for pq_file in self.parquet_files:
            if not os.path.exists(pq_file):
                print(f"Warning: File not found: {pq_file}")
                continue

            print(f"Reading: {pq_file}")
            parquet_file = pq.ParquetFile(pq_file)

            for batch in parquet_file.iter_batches(batch_size=1000):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    all_rows.append(row)
                    total_rows += 1
                    if self.num_samples > 0 and total_rows >= self.num_samples:
                        break

                if self.num_samples > 0 and total_rows >= self.num_samples:
                    break

            if self.num_samples > 0 and total_rows >= self.num_samples:
                break

        return all_rows

    def _run_local_extraction_pool(self, all_rows: List, progress_callback: Optional[Callable[[int, int], None]] = None):
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._extract_single_row, row) for row in all_rows]

            for i, future in enumerate(as_completed(futures)):
                try:
                    success = future.result()
                    if success:
                        with self.counter_lock:
                            self.counter += 1

                    if progress_callback:
                        progress_callback(i + 1, len(all_rows))
                except Exception as e:
                    print(f"Error in thread: {e}")

    def _extract_single_row(self, row) -> bool:
        """Extract a single row (thread-safe)"""
        try:
            # Extract image
            img_data = row.get('image')
            if img_data is None:
                return False

            # Handle different image formats
            if isinstance(img_data, dict) and 'bytes' in img_data:
                img_bytes = img_data['bytes']
            elif isinstance(img_data, bytes):
                img_bytes = img_data
            else:
                return False

            img = Image.open(io.BytesIO(img_bytes))

            # Extract tag
            tag = row.get('text') or row.get('caption') or row.get('tags', '')
            if isinstance(tag, list):
                tag = ", ".join(str(t) for t in tag)

            # Get unique index (thread-safe)
            with self.counter_lock:
                idx = self.counter
                self.counter += 1

            # Save with zero-padded index
            idx_str = f"{idx:08d}"
            img.save(os.path.join(self.image_dir, f"{idx_str}.png"))

            with open(os.path.join(self.tag_dir, f"{idx_str}.txt"), "w", encoding="utf-8") as f:
                f.write(str(tag).strip())

            return True

        except Exception as e:
            print(f"Error extracting row: {e}")
            return False

    def get_image_files(self) -> List[str]:
        """Get list of extracted image files"""
        if not os.path.exists(self.image_dir):
            return []

        files = [f for f in os.listdir(self.image_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        return sorted(files)


class StreamingDataSource:
    """Extract data from HuggingFace streaming dataset"""

    def __init__(self, dataset_id: str, split: str, extract_dir: str,
                 num_samples: int = 10000, hf_token: Optional[str] = None,
                 user_prefix: str = "", skip_count: int = 0, num_threads: int = 4):
        """
        Args:
            dataset_id: HuggingFace dataset ID or URL
                       Supports:
                       - https://huggingface.co/datasets/username/dataset-name
                       - username/dataset-name
            split: Dataset split (train/test/validation), leave empty for auto-detect
            extract_dir: Directory to extract images and tags
            num_samples: Number of samples to extract
            hf_token: HuggingFace token (optional, required for private repos)
            user_prefix: User prefix for filename (e.g., "user_a")
            skip_count: Number of samples to skip (for multi-user collaboration)
            num_threads: Number of threads for parallel downloading (1-8)
        """
        # Parse URL to extract dataset ID
        from .data_source import parse_huggingface_url
        self.dataset_id = parse_huggingface_url(dataset_id)
        self.split = split if split else "train"  # Default to train if empty
        self.extract_dir = extract_dir
        self.num_samples = num_samples
        self.hf_token = hf_token
        self.user_prefix = user_prefix
        self.skip_count = skip_count
        self.num_threads = max(1, min(8, num_threads))  # Clamp between 1-8
        self.image_dir = os.path.join(extract_dir, "images")
        self.tag_dir = os.path.join(extract_dir, "tags")
        self.cache_dir = os.path.join(extract_dir, ".cache")
        self.progress_file = os.path.join(extract_dir, ".progress.json")
        self.auto_detect_split = not bool(split)  # Auto-detect if split is empty

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.tag_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _save_progress(self, downloaded_files: List[str], processed_count: int):
        """Save extraction progress for resume"""
        progress_data = {
            'dataset_id': self.dataset_id,
            'split': self.split,
            'downloaded_files': downloaded_files,
            'processed_count': processed_count,
            'num_samples': self.num_samples,
            'user_prefix': self.user_prefix,
            'skip_count': self.skip_count
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f)

    def _load_progress(self) -> Optional[Dict]:
        """Load extraction progress if exists"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _make_logger(self, log_callback=None):
        """Create a logger callback with print fallback."""
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

        return log

    def _load_pyarrow_with_logging(self, log):
        """Import pyarrow parquet module with user-facing error logging."""
        try:
            import pyarrow.parquet as pq
            return pq
        except ImportError:
            log("[ERROR] pyarrow not installed. Run: pip install pyarrow")
            return None

    def _build_request_headers(self) -> Dict[str, str]:
        """Build optional authorization headers for HuggingFace requests."""
        if self.hf_token:
            return {'Authorization': f'Bearer {self.hf_token}'}
        return {}

    def _initialize_pyarrow_state(self, log):
        """Initialize extraction counters and resume state."""
        state = {
            'count': 0,
            'skipped': 0,
        }
        count_lock = threading.Lock()
        downloaded_files = []

        progress_data = self._load_progress()
        if progress_data and progress_data.get('dataset_id') == self.dataset_id:
            downloaded_files = progress_data.get('downloaded_files', [])
            state['count'] = progress_data.get('processed_count', 0)
            log(f"[INFO] Resuming from previous session: {state['count']} images already processed")
            log(f"[INFO] {len(downloaded_files)} parquet files already downloaded")

        return state, count_lock, downloaded_files

    def extract_via_pyarrow(self, log_callback=None, progress_callback=None) -> int:
        """
        Extract images directly from HuggingFace parquet files using pyarrow
        This method bypasses datasets library to avoid torch dependency issues

        Args:
            log_callback: Optional callback function(message: str) for logging
            progress_callback: Optional callback function(current: int, total: int) for progress

        Returns:
            Number of samples extracted
        """
        log = self._make_logger(log_callback)
        pq = self._load_pyarrow_with_logging(log)
        if pq is None:
            return 0

        try:
            log(f"[INFO] Loading dataset via pyarrow: {self.dataset_id}")
            log(f"[INFO] Split: {self.split}")
            log(f"[INFO] Target samples: {self.num_samples}")

            parquet_urls = self._fetch_parquet_urls(log)
            if parquet_urls is None:
                return 0

            state, count_lock, downloaded_files = self._initialize_pyarrow_state(log)
            self._log_pyarrow_startup(log)
            self._run_pyarrow_pipeline(
                parquet_urls,
                pq,
                state,
                count_lock,
                downloaded_files,
                log,
                progress_callback,
            )

            self._log_pyarrow_success(log, state['count'])
            self._clear_progress_file(log)
            return state['count']

        except Exception as e:
            log(f"[ERROR] Error: {e}")
            import traceback
            log(f"[ERROR] {traceback.format_exc()}")
            return 0

    def _fetch_parquet_urls(self, log) -> Optional[List[str]]:
        """Fetch parquet URLs from HuggingFace dataset API."""
        api_url = f"https://huggingface.co/api/datasets/{self.dataset_id}/parquet"
        log(f"[INFO] Fetching parquet file list...")
        try:
            response = requests.get(api_url, headers=self._build_request_headers(), timeout=10)
            response.raise_for_status()
            parquet_info = response.json()
            log(f"[DEBUG] Parquet API response keys: {list(parquet_info.keys())}")
            return self._resolve_parquet_urls_from_info(parquet_info, log)
        except Exception as e:
            log(f"[ERROR] Failed to get parquet file list: {e}")
            return None

    def _resolve_parquet_urls_from_info(self, parquet_info: Dict, log) -> Optional[List[str]]:
        """Resolve requested split URLs from API response structure."""
        parquet_urls = None

        for config_name, config_data in parquet_info.items():
            if isinstance(config_data, dict) and self.split in config_data:
                parquet_urls = config_data[self.split]
                log(f"[INFO] Found {len(parquet_urls)} parquet files in config '{config_name}', split '{self.split}'")
                break

        if parquet_urls is None:
            for config_name, config_data in parquet_info.items():
                if isinstance(config_data, dict):
                    available_splits = list(config_data.keys())
                    if available_splits:
                        actual_split = available_splits[0]
                        parquet_urls = config_data[actual_split]
                        log(f"[WARN] Split '{self.split}' not found. Using config '{config_name}', split '{actual_split}'")
                        log(f"[INFO] Found {len(parquet_urls)} parquet files")
                        break

        if parquet_urls is None:
            log(f"[ERROR] No parquet files found")
            return None
        return parquet_urls

    def _log_pyarrow_startup(self, log):
        """Emit startup messages for pyarrow extraction."""
        log(f"[INFO] Starting extraction with multi-threading...")
        log(f"[INFO] Thread pool size: {self.num_threads} workers")
        log(f"[INFO] Output directory: {self.extract_dir}")
        if self.user_prefix:
            log(f"[INFO] Filename prefix: {self.user_prefix}_")

    def _run_pyarrow_pipeline(
        self,
        parquet_urls: List[str],
        pq,
        state: Dict,
        count_lock: threading.Lock,
        downloaded_files: List[str],
        log,
        progress_callback=None,
    ):
        """Run download/process pipeline for parquet files."""
        log(f"[INFO] Processing {len(parquet_urls)} parquet files with pipeline parallelism")
        log(f"[INFO] Using {self.num_threads} threads for parallel image processing")

        download_queue = queue.Queue(maxsize=2)
        download_complete = threading.Event()

        download_thread = threading.Thread(
            target=self._download_worker,
            args=(parquet_urls, state, downloaded_files, download_queue, download_complete, log),
            daemon=True,
        )
        download_thread.start()

        while True:
            try:
                file_info = download_queue.get(timeout=1)
            except Exception:
                if download_complete.is_set() and download_queue.empty():
                    break
                continue

            if state['count'] >= self.num_samples:
                break

            file_idx = file_info[0]
            try:
                self._process_parquet_file(
                    file_info,
                    pq,
                    state,
                    count_lock,
                    downloaded_files,
                    log,
                    progress_callback,
                )
            except Exception as e:
                log(f"[ERROR] Error processing file {file_idx + 1}: {e}")

        download_thread.join(timeout=5)

    def _download_worker(
        self,
        parquet_urls: List[str],
        state: Dict,
        downloaded_files: List[str],
        download_queue: queue.Queue,
        download_complete: threading.Event,
        log,
    ):
        """Download parquet files and push bytes into processing queue."""
        try:
            for file_idx, parquet_file in enumerate(parquet_urls):
                if state['count'] >= self.num_samples:
                    break

                parquet_url = self._resolve_parquet_url(parquet_file)
                parquet_bytes = self._load_or_download_parquet_bytes(
                    file_idx,
                    parquet_url,
                    len(parquet_urls),
                    downloaded_files,
                    log,
                )
                download_queue.put((file_idx, parquet_url, parquet_bytes))
        finally:
            download_complete.set()

    def _resolve_parquet_url(self, parquet_file) -> str:
        """Normalize relative parquet path to full HuggingFace URL."""
        if isinstance(parquet_file, str) and not parquet_file.startswith('http'):
            return f"https://huggingface.co/datasets/{self.dataset_id}/resolve/main/{parquet_file}"
        return parquet_file

    def _load_or_download_parquet_bytes(
        self,
        file_idx: int,
        parquet_url: str,
        total_files: int,
        downloaded_files: List[str],
        log,
    ) -> bytes:
        """Load parquet file from cache or download from network."""
        import hashlib

        url_hash = hashlib.md5(parquet_url.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{url_hash}.parquet")

        if os.path.exists(cache_file) and parquet_url in downloaded_files:
            log(f"[INFO] Using cached file {file_idx + 1}/{total_files}")
            with open(cache_file, 'rb') as f:
                return f.read()

        log(f"[INFO] Downloading file {file_idx + 1}/{total_files}: {parquet_url}")
        response = requests.get(
            parquet_url,
            headers=self._build_request_headers(),
            timeout=60,
            stream=True,
        )
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded_bytes = 0
        chunks = []

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
                downloaded_bytes += len(chunk)
                if total_size > 0 and downloaded_bytes % (1024 * 1024) < 8192:
                    progress_mb = downloaded_bytes / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    log(f"[DOWNLOAD] File {file_idx + 1}: {progress_mb:.1f}MB / {total_mb:.1f}MB")

        parquet_bytes = b''.join(chunks)
        with open(cache_file, 'wb') as f:
            f.write(parquet_bytes)
        downloaded_files.append(parquet_url)
        return parquet_bytes

    def _process_parquet_file(
        self,
        file_info,
        pq,
        state: Dict,
        count_lock: threading.Lock,
        downloaded_files: List[str],
        log,
        progress_callback=None,
    ) -> int:
        """Process a parquet byte payload and persist extracted items."""
        file_idx, _, parquet_bytes = file_info
        try:
            table = pq.read_table(io.BytesIO(parquet_bytes))
            df = table.to_pandas()

            log(f"[INFO] Processing {len(df)} rows from file {file_idx + 1}")
            log(f"[INFO] Using {self.num_threads} worker threads for parallel processing")

            import time
            start_time = time.time()
            processed_threads = set()

            local_count = self._process_parquet_rows(
                df,
                state,
                count_lock,
                processed_threads,
                log,
                progress_callback,
            )

            self._save_progress(downloaded_files, state['count'])

            elapsed = time.time() - start_time
            if local_count > 0:
                speed = local_count / elapsed
                log(f"[INFO] File {file_idx + 1} processed: {local_count} images in {elapsed:.1f}s ({speed:.1f} it/s)")
                log(f"[INFO] Active threads used: {len(processed_threads)} (configured: {self.num_threads})")

            return local_count

        except Exception as e:
            log(f"[ERROR] Error fetching parquet file {file_idx + 1}: {e}")
            return 0

    def _process_parquet_rows(
        self,
        df,
        state: Dict,
        count_lock: threading.Lock,
        processed_threads: set,
        log,
        progress_callback=None,
    ) -> int:
        """Process dataframe rows in parallel and write results in batches."""
        local_count = 0
        with ThreadPoolExecutor(max_workers=self.num_threads) as row_executor:
            row_futures = []
            for idx, row in df.iterrows():
                if state['count'] >= self.num_samples:
                    break
                future = row_executor.submit(
                    self._process_parquet_row,
                    (idx, row),
                    state,
                    count_lock,
                    processed_threads,
                    log,
                )
                row_futures.append(future)

            write_batch = []
            for future in as_completed(row_futures):
                try:
                    result = future.result()
                    if result:
                        write_batch.append(result)
                        local_count += 1
                        if len(write_batch) >= 10:
                            self._write_pyarrow_batch(write_batch, progress_callback)
                            write_batch = []
                except Exception as e:
                    log(f"[WARN] Row processing error: {e}")

            self._write_pyarrow_batch(write_batch, progress_callback)

        return local_count

    def _process_parquet_row(
        self,
        row_info,
        state: Dict,
        count_lock: threading.Lock,
        processed_threads: set,
        log,
    ) -> Optional[Dict]:
        """Process one parquet row and return serialized item for batch write."""
        _, row = row_info

        thread_id = threading.current_thread().ident
        processed_threads.add(thread_id)

        with count_lock:
            if state['skipped'] < self.skip_count:
                state['skipped'] += 1
                return None
            if state['count'] >= self.num_samples:
                return None

            current_id = state['count']
            state['count'] += 1

        try:
            img_data = row.get('image', None)
            if img_data is None:
                return None

            if isinstance(img_data, dict) and 'bytes' in img_data:
                img = Image.open(io.BytesIO(img_data['bytes']))
            elif isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data))
            else:
                return None

            tag = ''
            for col in ['caption', 'text', 'tags', 'prompt']:
                if col in row:
                    tag = str(row[col])
                    break

            if self.user_prefix:
                filename = f"{self.user_prefix}_{current_id:08d}"
            else:
                filename = f"{current_id:08d}"

            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            return {
                'filename': filename,
                'img_bytes': img_bytes,
                'tag': tag.strip(),
                'current_id': current_id,
            }
        except Exception as e:
            log(f"[WARN] Error processing row: {e}")
            return None

    def _write_pyarrow_batch(self, write_batch: List[Dict], progress_callback=None):
        """Persist batch items and emit progress callback."""
        for item in write_batch:
            with open(os.path.join(self.image_dir, f"{item['filename']}.png"), "wb") as f:
                f.write(item['img_bytes'])
            with open(os.path.join(self.tag_dir, f"{item['filename']}.txt"), "w", encoding="utf-8") as f:
                f.write(item['tag'])
            if progress_callback:
                progress_callback(item['current_id'] + 1, self.num_samples)

    def _log_pyarrow_success(self, log, count: int):
        """Emit final extraction success logs."""
        log(f"[SUCCESS] Extraction complete: {count} images extracted")
        log(f"[SUCCESS] Images saved to: {self.image_dir}")
        log(f"[SUCCESS] Tags saved to: {self.tag_dir}")
        if self.user_prefix:
            log(f"[SUCCESS] Filename format: {self.user_prefix}_XXXXXXXX.png")

    def _clear_progress_file(self, log):
        """Delete resume progress file after successful extraction."""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
            log(f"[INFO] Progress file cleared")

    def extract(self, log_callback=None, progress_callback=None) -> int:
        """
        Extract images and tags from streaming dataset
        First tries pyarrow method (no torch dependency), falls back to datasets library

        Args:
            log_callback: Optional callback function(message: str) for logging
            progress_callback: Optional callback function(current: int, total: int) for progress

        Returns:
            Number of samples extracted
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

        # Try pyarrow method first (no torch dependency)
        log("[INFO] Attempting extraction via pyarrow (no torch required)...")
        try:
            return self.extract_via_pyarrow(log_callback, progress_callback)
        except Exception as e:
            log(f"[WARN] Pyarrow method failed: {e}")
            log("[INFO] Falling back to datasets library method...")

        # Fall back to datasets library method
        return self._extract_via_datasets(log_callback, progress_callback)

    def _extract_via_datasets(self, log_callback=None, progress_callback=None) -> int:
        """
        Extract images and tags from streaming dataset

        Args:
            log_callback: Optional callback function(message: str) for logging
            progress_callback: Optional callback function(current: int, total: int) for progress

        Returns:
            Number of samples extracted
        """
        log = self._make_logger(log_callback)
        self._prepare_datasets_runtime_environment()

        datasets_mod = self._import_datasets_module(log)
        if datasets_mod is None:
            return 0
        load_dataset, get_dataset_split_names = datasets_mod

        log(f"[INFO] Loading streaming dataset: {self.dataset_id}")
        detected_split = self._detect_streaming_split(get_dataset_split_names, log)

        try:
            ds = self._load_streaming_dataset(load_dataset, detected_split, log)
            ds, image_col, text_col = self._analyze_streaming_dataset_columns(ds, log)
            ds = self._apply_streaming_skip(ds, log)
            return self._extract_streaming_samples(ds, image_col, text_col, log, log_callback, progress_callback)
        except Exception as e:
            log(f"[ERROR] Error loading dataset: {e}")
            import traceback
            log(f"[ERROR] {traceback.format_exc()}")
            return 0

    def _prepare_datasets_runtime_environment(self):
        """Set environment flags and torch mock before importing datasets."""
        import sys

        os.environ['HF_DATASETS_DISABLE_TORCH'] = '1'
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '0'

        class _MockTorch:
            """Mock torch module to prevent imports."""

            def __getattr__(self, name):
                if name == '__version__':
                    return '0.0.0'
                raise ImportError("torch is disabled (HF_DATASETS_DISABLE_TORCH=1)")

        if 'torch' not in sys.modules:
            sys.modules['torch'] = _MockTorch()

    def _import_datasets_module(self, log):
        """Import datasets APIs with detailed diagnostics."""
        try:
            from datasets import load_dataset, get_dataset_split_names
            return load_dataset, get_dataset_split_names
        except ImportError as e:
            log(f"[ERROR] datasets not installed. Run: pip install datasets")
            log(f"[ERROR] Import error details: {e}")
            return None
        except Exception as e:
            error_msg = str(e)
            log(f"[ERROR] Failed to import datasets: {error_msg}")
            log(f"[ERROR] Error type: {type(e).__name__}")

            import traceback
            log(f"[ERROR] Traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    log(f"[ERROR] {line}")

            if "torch" in error_msg.lower() or "dll" in error_msg.lower() or "c10.dll" in error_msg.lower():
                self._log_torch_dll_failure_help(log)
            return None

    def _log_torch_dll_failure_help(self, log):
        """Emit torch DLL troubleshooting guidance."""
        log("[ERROR] ========================================")
        log("[ERROR] Torch DLL loading failed!")
        log("[ERROR] ========================================")
        log("[ERROR] ")
        log("[ERROR] This is a known issue with PyTorch on Windows.")
        log("[ERROR] ")
        log("[ERROR] Solutions:")
        log("[ERROR] 1. Install Visual C++ Redistributable:")
        log("[ERROR]    https://aka.ms/vs/17/release/vc_redist.x64.exe")
        log("[ERROR] ")
        log("[ERROR] 2. Reinstall PyTorch:")
        log("[ERROR]    Tools menu -> Install PyTorch")
        log("[ERROR] ")
        log("[ERROR] 3. Or manually run:")
        log("[ERROR]    pip uninstall torch torchvision")
        log("[ERROR]    pip install torch torchvision")
        log("[ERROR] ========================================")

    def _detect_streaming_split(self, get_dataset_split_names, log) -> str:
        """Resolve split with optional auto-detection fallback."""
        detected_split = self.split
        if not self.auto_detect_split:
            return detected_split

        try:
            log(f"[INFO] Auto-detecting available splits...")
            splits = get_dataset_split_names(self.dataset_id, token=self.hf_token)
            log(f"[INFO] Available splits: {splits}")

            if 'train' in splits:
                detected_split = 'train'
            elif 'validation' in splits:
                detected_split = 'validation'
            elif 'test' in splits:
                detected_split = 'test'
            elif splits:
                detected_split = splits[0]

            log(f"[INFO] Auto-detected split: {detected_split}")
        except Exception as e:
            log(f"[WARN] Failed to auto-detect split, using '{self.split}': {e}")
            detected_split = self.split
        return detected_split

    def _load_streaming_dataset(self, load_dataset, detected_split: str, log):
        """Load streaming dataset with temporary torch guard."""
        import sys

        log(f"[INFO] Connecting to HuggingFace Hub...")

        torch_was_imported = 'torch' in sys.modules
        original_torch = sys.modules.get('torch', None)

        if not torch_was_imported:
            class _TorchMock:
                __version__ = '2.0.0'
                Tensor = int

                def __getattr__(self, name):
                    return None

            sys.modules['torch'] = _TorchMock()

        try:
            return load_dataset(
                self.dataset_id,
                split=detected_split,
                streaming=True,
                token=self.hf_token,
            )
        finally:
            if not torch_was_imported:
                if original_torch is None:
                    sys.modules.pop('torch', None)
                else:
                    sys.modules['torch'] = original_torch

    def _analyze_streaming_dataset_columns(self, ds, log):
        """Detect image and text columns from first sample."""
        log(f"[INFO] Analyzing dataset structure...")
        sample = next(iter(ds))
        columns = list(sample.keys())
        log(f"[INFO] Dataset columns: {columns}")

        image_col = 'image' if 'image' in columns else columns[0]
        text_candidates = ['text', 'caption', 'tags', 'prompt']
        text_col = next((col for col in text_candidates if col in columns), None)

        if text_col:
            log(f"[INFO] Using image column: '{image_col}', text column: '{text_col}'")
        else:
            log(f"[INFO] Using image column: '{image_col}', no text column found")
        return ds, image_col, text_col

    def _apply_streaming_skip(self, ds, log):
        """Apply initial skip for collaboration offset."""
        if self.skip_count > 0:
            log(f"[INFO] Skipping first {self.skip_count} samples...")
            return ds.skip(self.skip_count)
        return ds

    def _extract_streaming_samples(self, ds, image_col: str, text_col: Optional[str], log, log_callback=None, progress_callback=None) -> int:
        """Extract samples from streaming dataset iterator."""
        log(f"[INFO] Starting extraction of {self.num_samples} samples...")
        log(f"[INFO] Output directory: {self.extract_dir}")
        if self.user_prefix:
            log(f"[INFO] Filename prefix: {self.user_prefix}_")

        count = 0
        for example in tqdm(
            ds.take(self.num_samples),
            total=self.num_samples,
            desc="Extracting",
            disable=bool(log_callback),
        ):
            try:
                img = example.get(image_col)
                if img is None:
                    continue

                tag = ''
                if text_col:
                    tag = example.get(text_col, '')
                    if isinstance(tag, list):
                        tag = ", ".join(str(t) for t in tag)

                if self.user_prefix:
                    idx = f"{self.user_prefix}_{count:08d}"
                else:
                    idx = f"{count:08d}"

                img.save(os.path.join(self.image_dir, f"{idx}.png"), format="PNG")
                with open(os.path.join(self.tag_dir, f"{idx}.txt"), "w", encoding="utf-8") as f:
                    f.write(str(tag).strip())

                count += 1
                if progress_callback:
                    progress_callback(count, self.num_samples)
                if count % 100 == 0:
                    log(f"[PROGRESS] Extracted {count}/{self.num_samples} samples...")
            except Exception as e:
                log(f"[WARN] Error extracting sample: {e}")
                continue

        log(f"[SUCCESS] Extraction complete: {count} images extracted")
        log(f"[SUCCESS] Images saved to: {self.image_dir}")
        log(f"[SUCCESS] Tags saved to: {self.tag_dir}")
        if self.user_prefix:
            log(f"[SUCCESS] Filename format: {self.user_prefix}_XXXXXXXX.png")
        return count

    def get_image_files(self) -> List[str]:
        """Get list of extracted image files"""
        if not os.path.exists(self.image_dir):
            return []

        files = [f for f in os.listdir(self.image_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        return sorted(files)
