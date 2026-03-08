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
        try:
            import pyarrow.parquet as pq
        except ImportError:
            print("Error: pyarrow not installed. Run: pip install pyarrow")
            return 0

        # Collect all rows from all parquet files
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
                for idx, row in df.iterrows():
                    all_rows.append(row)
                    total_rows += 1

                    if self.num_samples > 0 and total_rows >= self.num_samples:
                        break

                if self.num_samples > 0 and total_rows >= self.num_samples:
                    break

            if self.num_samples > 0 and total_rows >= self.num_samples:
                break

        if not all_rows:
            print("No data to extract")
            return 0

        print(f"Extracting {len(all_rows)} images using {self.num_threads} threads...")

        # Multi-threaded extraction
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for row in all_rows:
                future = executor.submit(self._extract_single_row, row)
                futures.append(future)

            # Progress tracking
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

        print(f"Extraction complete: {self.counter} images extracted to {self.extract_dir}")
        return self.counter

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
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

        try:
            import pyarrow.parquet as pq
        except ImportError:
            log("[ERROR] pyarrow not installed. Run: pip install pyarrow")
            return 0

        try:
            log(f"[INFO] Loading dataset via pyarrow: {self.dataset_id}")
            log(f"[INFO] Split: {self.split}")
            log(f"[INFO] Target samples: {self.num_samples}")

            # Get list of parquet files from HuggingFace API
            api_url = f"https://huggingface.co/api/datasets/{self.dataset_id}/parquet"
            log(f"[INFO] Fetching parquet file list...")

            try:
                response = requests.get(api_url, headers={'Authorization': f'Bearer {self.hf_token}'} if self.hf_token else {}, timeout=10)
                response.raise_for_status()
                parquet_info = response.json()

                log(f"[DEBUG] Parquet API response keys: {list(parquet_info.keys())}")

                # The structure is: {config: {split: [urls]}}
                # Find the right config and split
                parquet_urls = None

                # Try to find the split in any config
                for config_name, config_data in parquet_info.items():
                    if isinstance(config_data, dict) and self.split in config_data:
                        parquet_urls = config_data[self.split]
                        log(f"[INFO] Found {len(parquet_urls)} parquet files in config '{config_name}', split '{self.split}'")
                        break

                # If not found, try first available split in first config
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
                    return 0

            except Exception as e:
                log(f"[ERROR] Failed to get parquet file list: {e}")
                return 0

            count = 0
            skipped = 0
            count_lock = threading.Lock()

            # Check for existing progress
            progress_data = self._load_progress()
            downloaded_files = []

            if progress_data and progress_data.get('dataset_id') == self.dataset_id:
                downloaded_files = progress_data.get('downloaded_files', [])
                count = progress_data.get('processed_count', 0)
                log(f"[INFO] Resuming from previous session: {count} images already processed")
                log(f"[INFO] {len(downloaded_files)} parquet files already downloaded")

            log(f"[INFO] Starting extraction with multi-threading...")
            log(f"[INFO] Thread pool size: {self.num_threads} workers")
            log(f"[INFO] Output directory: {self.extract_dir}")
            if self.user_prefix:
                log(f"[INFO] Filename prefix: {self.user_prefix}_")

            def process_parquet_file(file_info):
                """Process a single parquet file"""
                file_idx, parquet_url, parquet_bytes = file_info
                nonlocal count, skipped, downloaded_files

                try:
                    # Read parquet from bytes
                    table = pq.read_table(io.BytesIO(parquet_bytes))
                    df = table.to_pandas()

                    log(f"[INFO] Processing {len(df)} rows from file {file_idx + 1}")
                    log(f"[INFO] Using {self.num_threads} worker threads for parallel processing")

                    import time
                    start_time = time.time()
                    processed_threads = set()

                    def process_row(row_info):
                        """Process a single row (image) - returns data for batch writing"""
                        idx, row = row_info
                        nonlocal count, skipped, processed_threads

                        # Track which threads are being used
                        import threading
                        thread_id = threading.current_thread().ident
                        processed_threads.add(thread_id)

                        # Check skip first (with lock)
                        with count_lock:
                            if skipped < self.skip_count:
                                skipped += 1
                                return None
                            if count >= self.num_samples:
                                return None

                            # Allocate ID immediately (reduce lock time)
                            current_id = count
                            count += 1

                        try:
                            # Extract and process image (WITHOUT lock - parallel processing)
                            img_data = row.get('image', None)
                            if img_data is None:
                                return None

                            # Handle different image formats
                            if isinstance(img_data, dict) and 'bytes' in img_data:
                                img = Image.open(io.BytesIO(img_data['bytes']))
                            elif isinstance(img_data, bytes):
                                img = Image.open(io.BytesIO(img_data))
                            else:
                                return None

                            # Extract text/caption
                            tag = ''
                            for col in ['caption', 'text', 'tags', 'prompt']:
                                if col in row:
                                    tag = str(row[col])
                                    break

                            # Generate filename using pre-allocated ID
                            if self.user_prefix:
                                filename = f"{self.user_prefix}_{current_id:08d}"
                            else:
                                filename = f"{current_id:08d}"

                            # Convert image to bytes in memory (parallel processing)
                            img_buffer = io.BytesIO()
                            img.save(img_buffer, format="PNG")
                            img_bytes = img_buffer.getvalue()

                            # Return data for batch writing
                            return {
                                'filename': filename,
                                'img_bytes': img_bytes,
                                'tag': tag.strip(),
                                'current_id': current_id
                            }

                        except Exception as e:
                            log(f"[WARN] Error processing row: {e}")
                            return None

                    # Use ThreadPoolExecutor to process rows in parallel
                    local_count = 0
                    with ThreadPoolExecutor(max_workers=self.num_threads) as row_executor:
                        row_futures = []
                        for idx, row in df.iterrows():
                            if count >= self.num_samples:
                                break
                            future = row_executor.submit(process_row, (idx, row))
                            row_futures.append(future)

                        # Collect results and write in batches
                        write_batch = []
                        for future in as_completed(row_futures):
                            try:
                                result = future.result()
                                if result:
                                    write_batch.append(result)
                                    local_count += 1

                                    # Write in batches of 10 to reduce I/O overhead
                                    if len(write_batch) >= 10:
                                        for item in write_batch:
                                            # Write image file
                                            with open(os.path.join(self.image_dir, f"{item['filename']}.png"), "wb") as f:
                                                f.write(item['img_bytes'])
                                            # Write tag file
                                            with open(os.path.join(self.tag_dir, f"{item['filename']}.txt"), "w", encoding="utf-8") as f:
                                                f.write(item['tag'])
                                            # Update progress
                                            if progress_callback:
                                                progress_callback(item['current_id'] + 1, self.num_samples)
                                        write_batch = []

                            except Exception as e:
                                log(f"[WARN] Row processing error: {e}")

                        # Write remaining items
                        for item in write_batch:
                            with open(os.path.join(self.image_dir, f"{item['filename']}.png"), "wb") as f:
                                f.write(item['img_bytes'])
                            with open(os.path.join(self.tag_dir, f"{item['filename']}.txt"), "w", encoding="utf-8") as f:
                                f.write(item['tag'])
                            if progress_callback:
                                progress_callback(item['current_id'] + 1, self.num_samples)

                    # Save progress after processing each file
                    self._save_progress(downloaded_files, count)

                    # Log performance stats
                    elapsed = time.time() - start_time
                    if local_count > 0:
                        speed = local_count / elapsed
                        log(f"[INFO] File {file_idx + 1} processed: {local_count} images in {elapsed:.1f}s ({speed:.1f} it/s)")
                        log(f"[INFO] Active threads used: {len(processed_threads)} (configured: {self.num_threads})")

                    return local_count

                except Exception as e:
                    log(f"[ERROR] Error fetching parquet file {file_idx + 1}: {e}")
                    return 0

            # Process files with pipeline parallelism:
            # Download next file while processing current file
            log(f"[INFO] Processing {len(parquet_urls)} parquet files with pipeline parallelism")
            log(f"[INFO] Using {self.num_threads} threads for parallel image processing")

            # Use a queue to pass downloaded files between download and process threads
            from queue import Queue
            download_queue = Queue(maxsize=2)  # Buffer 2 files
            download_complete = threading.Event()

            def download_worker():
                """Download files in background"""
                try:
                    for file_idx, parquet_file in enumerate(parquet_urls):
                        if count >= self.num_samples:
                            break

                        # Construct full URL
                        if isinstance(parquet_file, str) and not parquet_file.startswith('http'):
                            parquet_url = f"https://huggingface.co/datasets/{self.dataset_id}/resolve/main/{parquet_file}"
                        else:
                            parquet_url = parquet_file

                        # Generate cache filename
                        import hashlib
                        url_hash = hashlib.md5(parquet_url.encode()).hexdigest()
                        cache_file = os.path.join(self.cache_dir, f"{url_hash}.parquet")

                        # Check cache or download
                        if os.path.exists(cache_file) and parquet_url in downloaded_files:
                            log(f"[INFO] Using cached file {file_idx + 1}/{len(parquet_urls)}")
                            with open(cache_file, 'rb') as f:
                                parquet_bytes = f.read()
                        else:
                            log(f"[INFO] Downloading file {file_idx + 1}/{len(parquet_urls)}: {parquet_url}")
                            response = requests.get(parquet_url, headers={'Authorization': f'Bearer {self.hf_token}'} if self.hf_token else {}, timeout=60, stream=True)
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

                            # Save to cache
                            with open(cache_file, 'wb') as f:
                                f.write(parquet_bytes)

                            downloaded_files.append(parquet_url)

                        # Put in queue for processing
                        download_queue.put((file_idx, parquet_url, parquet_bytes))

                finally:
                    download_complete.set()

            # Start download thread
            download_thread = threading.Thread(target=download_worker, daemon=True)
            download_thread.start()

            # Process files as they are downloaded
            while True:
                try:
                    # Get next file from queue (with timeout)
                    file_idx, parquet_url, parquet_bytes = download_queue.get(timeout=1)
                except:
                    # Check if download is complete
                    if download_complete.is_set() and download_queue.empty():
                        break
                    continue

                if count >= self.num_samples:
                    break

                try:
                    # Process the file
                    process_parquet_file((file_idx, parquet_url, parquet_bytes))
                except Exception as e:
                    log(f"[ERROR] Error processing file {file_idx + 1}: {e}")

            # Wait for download thread to finish
            download_thread.join(timeout=5)

            log(f"[SUCCESS] Extraction complete: {count} images extracted")
            log(f"[SUCCESS] Images saved to: {self.image_dir}")
            log(f"[SUCCESS] Tags saved to: {self.tag_dir}")
            if self.user_prefix:
                log(f"[SUCCESS] Filename format: {self.user_prefix}_XXXXXXXX.png")

            # Clear progress file on successful completion
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
                log(f"[INFO] Progress file cleared")

            return count

        except Exception as e:
            log(f"[ERROR] Error: {e}")
            import traceback
            log(f"[ERROR] {traceback.format_exc()}")
            return 0

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

    def _extract_via_datasets(self, log_callback=None) -> int:
        """
        Extract images and tags from streaming dataset

        Args:
            log_callback: Optional callback function(message: str) for logging

        Returns:
            Number of samples extracted
        """
        def log(msg):
            """Helper to log message"""
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

        # Disable torch formatter to avoid torch import errors
        import os
        import sys
        os.environ['HF_DATASETS_DISABLE_TORCH'] = '1'
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '0'  # Keep online mode

        # Mock torch to prevent any imports
        class _MockTorch:
            """Mock torch module to prevent imports"""
            def __getattr__(self, name):
                if name == '__version__':
                    return '0.0.0'
                raise ImportError(f"torch is disabled (HF_DATASETS_DISABLE_TORCH=1)")

        # Only mock if torch is not already imported
        if 'torch' not in sys.modules:
            sys.modules['torch'] = _MockTorch()

        try:
            from datasets import load_dataset, get_dataset_split_names
        except ImportError as e:
            log(f"[ERROR] datasets not installed. Run: pip install datasets")
            log(f"[ERROR] Import error details: {e}")
            return 0
        except Exception as e:
            error_msg = str(e)
            log(f"[ERROR] Failed to import datasets: {error_msg}")
            log(f"[ERROR] Error type: {type(e).__name__}")

            # Show detailed traceback
            import traceback
            log(f"[ERROR] Traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    log(f"[ERROR] {line}")

            if "torch" in error_msg.lower() or "dll" in error_msg.lower() or "c10.dll" in error_msg.lower():
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
            return 0

        log(f"[INFO] Loading streaming dataset: {self.dataset_id}")

        # Auto-detect split if enabled
        detected_split = self.split
        if self.auto_detect_split:
            try:
                log(f"[INFO] Auto-detecting available splits...")
                splits = get_dataset_split_names(self.dataset_id, token=self.hf_token)
                log(f"[INFO] Available splits: {splits}")

                # Priority: train > validation > test > first available
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

        try:
            # Load streaming dataset
            log(f"[INFO] Connecting to HuggingFace Hub...")

            # Temporarily mock torch to prevent DLL loading
            import sys
            torch_was_imported = 'torch' in sys.modules
            original_torch = sys.modules.get('torch', None)

            if not torch_was_imported:
                # Create a minimal mock that satisfies datasets' checks
                class _TorchMock:
                    __version__ = '2.0.0'
                    Tensor = int  # Fake Tensor type

                    def __getattr__(self, name):
                        # Return mock for any attribute access
                        return None

                sys.modules['torch'] = _TorchMock()

            try:
                ds = load_dataset(
                    self.dataset_id,
                    split=detected_split,
                    streaming=True,
                    token=self.hf_token
                )
            finally:
                # Restore original torch state
                if not torch_was_imported:
                    if original_torch is None:
                        sys.modules.pop('torch', None)
                    else:
                        sys.modules['torch'] = original_torch

            # Preview column names
            log(f"[INFO] Analyzing dataset structure...")
            sample = next(iter(ds))
            columns = list(sample.keys())
            log(f"[INFO] Dataset columns: {columns}")

            # Detect image and text columns
            image_col = 'image' if 'image' in columns else columns[0]
            text_cols = ['text', 'caption', 'tags', 'prompt']
            text_col = next((col for col in text_cols if col in columns), None)

            if text_col:
                log(f"[INFO] Using image column: '{image_col}', text column: '{text_col}'")
            else:
                log(f"[INFO] Using image column: '{image_col}', no text column found")

            # Skip samples if needed
            if self.skip_count > 0:
                log(f"[INFO] Skipping first {self.skip_count} samples...")
                ds = ds.skip(self.skip_count)

            log(f"[INFO] Starting extraction of {self.num_samples} samples...")
            log(f"[INFO] Output directory: {self.extract_dir}")
            if self.user_prefix:
                log(f"[INFO] Filename prefix: {self.user_prefix}_")

            count = 0
            for example in tqdm(ds.take(self.num_samples), total=self.num_samples, desc="Extracting", disable=bool(log_callback)):
                try:
                    # Extract image
                    img = example.get(image_col)
                    if img is None:
                        continue

                    # Extract tag
                    tag = ''
                    if text_col:
                        tag = example.get(text_col, '')
                        if isinstance(tag, list):
                            tag = ", ".join(str(t) for t in tag)

                    # Generate filename with user prefix
                    if self.user_prefix:
                        idx = f"{self.user_prefix}_{count:08d}"
                    else:
                        idx = f"{count:08d}"

                    img.save(os.path.join(self.image_dir, f"{idx}.png"), format="PNG")

                    with open(os.path.join(self.tag_dir, f"{idx}.txt"), "w", encoding="utf-8") as f:
                        f.write(str(tag).strip())

                    count += 1

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

        except Exception as e:
            log(f"[ERROR] Error loading dataset: {e}")
            import traceback
            log(f"[ERROR] {traceback.format_exc()}")
            return 0

    def get_image_files(self) -> List[str]:
        """Get list of extracted image files"""
        if not os.path.exists(self.image_dir):
            return []

        files = [f for f in os.listdir(self.image_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        return sorted(files)
