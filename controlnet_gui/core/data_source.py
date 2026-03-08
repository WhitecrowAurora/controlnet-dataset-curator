"""
ControlNet GUI - 数据源管理模块
支持本地文件和HuggingFace流式数据源
"""

import os
import glob
import re
from abc import ABC, abstractmethod
from typing import Optional, Iterator, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Lazy import for datasets to avoid loading torch at startup
HF_AVAILABLE = False


def parse_huggingface_url(url_or_id: str) -> str:
    """
    Parse HuggingFace URL or dataset ID and return the dataset ID.

    Supports:
    - Full URL: https://huggingface.co/datasets/username/dataset-name
    - Short URL: huggingface.co/datasets/username/dataset-name
    - Dataset ID: username/dataset-name

    Args:
        url_or_id: HuggingFace URL or dataset ID

    Returns:
        Dataset ID in format "username/dataset-name"

    Examples:
        >>> parse_huggingface_url("https://huggingface.co/datasets/user/data")
        'user/data'
        >>> parse_huggingface_url("user/data")
        'user/data'
    """
    url_or_id = url_or_id.strip()

    # Pattern 1: Full URL with https://
    # https://huggingface.co/datasets/username/dataset-name
    match = re.match(r'https?://huggingface\.co/datasets/([^/]+/[^/?#]+)', url_or_id)
    if match:
        return match.group(1)

    # Pattern 2: URL without protocol
    # huggingface.co/datasets/username/dataset-name
    match = re.match(r'huggingface\.co/datasets/([^/]+/[^/?#]+)', url_or_id)
    if match:
        return match.group(1)

    # Pattern 3: Already a dataset ID
    # username/dataset-name
    if '/' in url_or_id and not url_or_id.startswith('http'):
        # Remove any query parameters or fragments
        dataset_id = url_or_id.split('?')[0].split('#')[0]
        return dataset_id

    # If none of the patterns match, return as-is
    return url_or_id


@dataclass
class ImageData:
    """图像数据容器"""
    image: Image.Image
    image_path: str
    basename: str
    tag: Optional[str] = None
    depth: Optional[Image.Image] = None
    depth_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DataSource(ABC):
    """数据源抽象基类"""
    
    @abstractmethod
    def __iter__(self) -> Iterator[ImageData]:
        """迭代返回图像数据"""
        pass
    
    @abstractmethod
    def has_more(self) -> bool:
        """检查是否还有更多数据"""
        pass
    
    @abstractmethod
    def reset(self):
        """重置数据源"""
        pass
    
    @abstractmethod
    def get_total_count(self) -> Optional[int]:
        """获取总数据量（如果可知）"""
        pass


class LocalDataSource(DataSource):
    """
    本地文件数据源
    支持从本地文件夹读取图片、标签和深度图
    """
    
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    def __init__(
        self,
        image_dir: str,
        tag_dir: Optional[str] = None,
        depth_dir: Optional[str] = None,
        recursive: bool = False
    ):
        """
        初始化本地数据源
        
        Args:
            image_dir: 图片目录
            tag_dir: 标签目录（可选）
            depth_dir: 深度图目录（可选）
            recursive: 是否递归搜索子目录
        """
        self.image_dir = Path(image_dir)
        self.tag_dir = Path(tag_dir) if tag_dir else None
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.recursive = recursive
        
        self._file_list: List[Path] = []
        self._index = 0
        self._scanned = False
        
        self._scan_files()
    
    def _scan_files(self):
        """扫描目录中的图片文件"""
        if self._scanned:
            return
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {self.image_dir}")
        
        # 收集所有图片文件
        pattern = '**/*' if self.recursive else '*'
        for ext in self.SUPPORTED_EXTENSIONS:
            # glob需要 *.png 格式，扩展名已包含点号
            self._file_list.extend(self.image_dir.glob(f"{pattern}{ext}"))
        
        # 排序保证顺序一致
        self._file_list.sort(key=lambda x: x.name)
        self._scanned = True
    
    def __iter__(self) -> Iterator[ImageData]:
        """迭代返回图像数据"""
        self._index = 0
        
        while self._index < len(self._file_list):
            file_path = self._file_list[self._index]
            self._index += 1
            
            try:
                image_data = self._load_image_data(file_path)
                if image_data:
                    yield image_data
            except Exception as e:
                print(f"跳过文件 {file_path}: {e}")
                continue
    
    def _load_image_data(self, image_path: Path) -> Optional[ImageData]:
        """加载单个图像数据"""
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        basename = image_path.stem
        
        # 加载标签
        tag = None
        if self.tag_dir:
            tag_path = self.tag_dir / f"{basename}.txt"
            if tag_path.exists():
                with open(tag_path, 'r', encoding='utf-8') as f:
                    tag = f.read().strip()
        
        # 加载深度图
        depth = None
        depth_path = None
        if self.depth_dir:
            for ext in self.SUPPORTED_EXTENSIONS:
                candidate = self.depth_dir / f"{basename}{ext}"
                if candidate.exists():
                    depth_path = str(candidate)
                    depth = Image.open(candidate).convert('RGB')
                    break
        
        return ImageData(
            image=image,
            image_path=str(image_path),
            basename=basename,
            tag=tag,
            depth=depth,
            depth_path=depth_path
        )
    
    def has_more(self) -> bool:
        """检查是否还有更多数据"""
        return self._index < len(self._file_list)
    
    def reset(self):
        """重置数据源"""
        self._index = 0
    
    def get_total_count(self) -> int:
        """获取总数据量"""
        return len(self._file_list)
    
    def get_file_list(self) -> List[str]:
        """获取文件列表"""
        return [str(f) for f in self._file_list]


class StreamingDataSource(DataSource):
    """
    HuggingFace流式数据源
    支持从HuggingFace数据集流式加载图片
    """
    
    def __init__(
        self,
        dataset_id: str,
        split: str = "train",
        hf_token: Optional[str] = None,
        num_samples: Optional[int] = None,
        image_column: str = "image",
        text_column: Optional[str] = None,
        conditioning_column: Optional[str] = None,
        depth_column: Optional[str] = None,
        auto_detect_split: bool = True
    ):
        """
        初始化流式数据源

        Args:
            dataset_id: HuggingFace数据集ID或完整URL
                       支持格式：
                       - https://huggingface.co/datasets/username/dataset-name
                       - huggingface.co/datasets/username/dataset-name
                       - username/dataset-name
            split: 数据集分割（如果 auto_detect_split=True，会自动探测）
            hf_token: HuggingFace token（私有数据集需要）
            num_samples: 要加载的样本数量（None表示全部）
            image_column: 图片列名
            text_column: 文本/标签列名（可选，自动检测）
            conditioning_column: 控制图列名（可选）
            depth_column: 深度图列名（可选）
            auto_detect_split: 是否自动探测可用的 split
        """
        if not HF_AVAILABLE:
            raise ImportError("需要安装 datasets 库: pip install datasets")

        # Parse URL to extract dataset ID
        self.dataset_id = parse_huggingface_url(dataset_id)
        self.split = split
        self.hf_token = hf_token
        self.num_samples = num_samples
        self.image_column = image_column
        self.text_column = text_column
        self.conditioning_column = conditioning_column
        self.depth_column = depth_column
        self.auto_detect_split = auto_detect_split

        self._dataset = None
        self._iterator = None
        self._count = 0
        self._columns = None
        self._detected_split = None
    
    def _load_dataset(self):
        """加载数据集"""
        if self._dataset is not None:
            return

        # Lazy import datasets only when actually needed
        try:
            from datasets import load_dataset as hf_load_dataset
            from datasets import get_dataset_config_names, get_dataset_split_names
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library is required for streaming data sources. "
                "Install it with: pip install datasets"
            )

        # Auto-detect split if enabled
        if self.auto_detect_split and not self._detected_split:
            try:
                # Try to get available splits
                # First, try default config
                splits = get_dataset_split_names(self.dataset_id, token=self.hf_token)

                # Priority order: train > validation > test > first available
                if 'train' in splits:
                    self._detected_split = 'train'
                elif 'validation' in splits:
                    self._detected_split = 'validation'
                elif 'test' in splits:
                    self._detected_split = 'test'
                elif splits:
                    self._detected_split = splits[0]
                else:
                    self._detected_split = self.split  # Fallback to user-specified

                print(f"Auto-detected split: {self._detected_split} (available: {splits})")
            except Exception as e:
                print(f"Failed to auto-detect split, using '{self.split}': {e}")
                self._detected_split = self.split
        else:
            self._detected_split = self.split

        self._dataset = hf_load_dataset(
            self.dataset_id,
            split=self._detected_split,
            streaming=True,
            token=self.hf_token
        )

        # 获取列信息（需要peek一个样本）
        sample = next(iter(self._dataset))
        self._columns = list(sample.keys())

        # 自动检测文本列
        if self.text_column is None:
            for col in ['text', 'caption', 'tags', 'prompt']:
                if col in self._columns:
                    self.text_column = col
                    break

        # 自动检测深度图列
        if self.depth_column is None:
            for col in ['depth_map', 'depth', 'depth_image']:
                if col in self._columns:
                    self.depth_column = col
                    break

        # 重新加载数据集以避免丢失第一个样本
        self._dataset = hf_load_dataset(
            self.dataset_id,
            split=self._detected_split,
            streaming=True,
            token=self.hf_token
        )
    
    def __iter__(self) -> Iterator[ImageData]:
        """迭代返回图像数据"""
        self._load_dataset()
        self._count = 0
        self._iterator = iter(self._dataset)
        
        while True:
            if self.num_samples and self._count >= self.num_samples:
                break
            
            try:
                sample = next(self._iterator)
                image_data = self._convert_sample(sample)
                if image_data:
                    self._count += 1
                    yield image_data
            except StopIteration:
                break
            except Exception as e:
                print(f"跳过样本: {e}")
                continue
    
    def _convert_sample(self, sample: Dict) -> Optional[ImageData]:
        """将HuggingFace样本转换为ImageData"""
        # 获取图片
        image = sample.get(self.image_column)
        if image is None:
            return None
        
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        else:
            image = image.convert('RGB')
        
        # 获取标签
        tag = None
        if self.text_column and self.text_column in sample:
            tag = sample[self.text_column]
            if isinstance(tag, list):
                tag = ", ".join(tag)
        
        # 获取深度图
        depth = None
        if self.depth_column and self.depth_column in sample:
            depth_img = sample[self.depth_column]
            if isinstance(depth_img, Image.Image):
                depth = depth_img.convert('RGB')
        
        # 获取控制图
        conditioning = None
        if self.conditioning_column and self.conditioning_column in sample:
            cond = sample[self.conditioning_column]
            if isinstance(cond, Image.Image):
                conditioning = cond
        
        return ImageData(
            image=image,
            image_path=f"{self.dataset_id}:{self._count}",
            basename=f"{self._count:08d}",
            tag=str(tag) if tag else None,
            depth=depth,
            metadata={
                'conditioning': conditioning,
                'source': 'huggingface',
                'dataset_id': self.dataset_id
            }
        )
    
    def has_more(self) -> bool:
        """检查是否还有更多数据"""
        if self.num_samples and self._count >= self.num_samples:
            return False
        return self._iterator is not None
    
    def reset(self):
        """重置数据源"""
        self._dataset = None
        self._iterator = None
        self._count = 0
    
    def get_total_count(self) -> Optional[int]:
        """获取总数据量（流式数据源通常未知）"""
        return self.num_samples


class DataSourceManager:
    """
    数据源管理器
    统一管理不同类型的数据源
    """
    
    def __init__(self):
        self._source: Optional[DataSource] = None
        self._source_type: Optional[str] = None
    
    def setup_local(
        self,
        image_dir: str,
        tag_dir: Optional[str] = None,
        depth_dir: Optional[str] = None,
        recursive: bool = False
    ):
        """配置本地数据源"""
        self._source = LocalDataSource(
            image_dir=image_dir,
            tag_dir=tag_dir,
            depth_dir=depth_dir,
            recursive=recursive
        )
        self._source_type = "local"
    
    def setup_streaming(
        self,
        dataset_id: str,
        split: str = "train",
        hf_token: Optional[str] = None,
        num_samples: Optional[int] = None,
        **kwargs
    ):
        """配置流式数据源"""
        self._source = StreamingDataSource(
            dataset_id=dataset_id,
            split=split,
            hf_token=hf_token,
            num_samples=num_samples,
            **kwargs
        )
        self._source_type = "streaming"
    
    def get_source(self) -> Optional[DataSource]:
        """获取当前数据源"""
        return self._source
    
    def get_source_type(self) -> Optional[str]:
        """获取数据源类型"""
        return self._source_type
    
    def __iter__(self) -> Iterator[ImageData]:
        """迭代数据"""
        if self._source:
            yield from self._source
    
    def has_more(self) -> bool:
        """检查是否还有更多数据"""
        return self._source.has_more() if self._source else False
    
    def reset(self):
        """重置数据源"""
        if self._source:
            self._source.reset()
    
    def get_total_count(self) -> Optional[int]:
        """获取总数据量"""
        return self._source.get_total_count() if self._source else None