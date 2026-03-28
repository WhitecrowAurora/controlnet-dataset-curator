"""
相似图片检测模块
使用感知哈希算法检测重复或相似的图片
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import hashlib


class ImageSimilarityDetector:
    """图片相似度检测器"""

    def __init__(self, hash_size: int = 8):
        """
        初始化

        Args:
            hash_size: 哈希大小，默认8x8
        """
        self.hash_size = hash_size

    def compute_phash(self, image: Image.Image) -> str:
        """
        计算图片的感知哈希值

        Args:
            image: PIL图片对象

        Returns:
            哈希值字符串
        """
        # 转换为灰度图
        img = image.convert('L')

        # 缩放到hash_size x hash_size
        img = img.resize((self.hash_size, self.hash_size), Image.Resampling.LANCZOS)

        # 转换为numpy数组
        pixels = np.array(img).flatten()

        # 计算平均值
        avg = pixels.mean()

        # 生成哈希：大于平均值为1，否则为0
        hash_bits = (pixels > avg).astype(int)

        # 转换为十六进制字符串
        hash_str = ''.join(str(bit) for bit in hash_bits)
        return hash_str

    def compute_dhash(self, image: Image.Image) -> str:
        """
        计算图片的差异哈希值（更适合检测相似图片）

        Args:
            image: PIL图片对象

        Returns:
            哈希值字符串
        """
        # 转换为灰度图
        img = image.convert('L')

        # 缩放到(hash_size+1) x hash_size
        img = img.resize((self.hash_size + 1, self.hash_size), Image.Resampling.LANCZOS)

        # 转换为numpy数组
        pixels = np.array(img)

        # 计算水平梯度
        diff = pixels[:, 1:] > pixels[:, :-1]

        # 转换为字符串
        hash_str = ''.join('1' if bit else '0' for bit in diff.flatten())
        return hash_str

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        计算两个哈希值的汉明距离

        Args:
            hash1: 第一个哈希值
            hash2: 第二个哈希值

        Returns:
            汉明距离（不同位的数量）
        """
        if len(hash1) != len(hash2):
            raise ValueError("Hash lengths must be equal")

        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def calculate_similarity(self, hash1: str, hash2: str) -> float:
        """
        计算两个哈希值的相似度

        Args:
            hash1: 第一个哈希值
            hash2: 第二个哈希值

        Returns:
            相似度（0-1之间，1表示完全相同）
        """
        distance = self.hamming_distance(hash1, hash2)
        max_distance = len(hash1)
        return 1.0 - (distance / max_distance)

    def find_similar_images(
        self,
        images: List[Tuple[str, Image.Image]],
        threshold: float = 0.9
    ) -> List[List[Tuple[str, str]]]:
        """
        查找相似的图片组

        Args:
            images: 图片列表，每个元素为(标识符, PIL图片对象)
            threshold: 相似度阈值，默认0.9

        Returns:
            相似图片组列表，每组包含相似图片的标识符和哈希值
        """
        # 计算所有图片的哈希值
        hashes = []
        for identifier, img in images:
            try:
                hash_val = self.compute_dhash(img)
                hashes.append((identifier, hash_val))
            except Exception as e:
                print(f"Error computing hash for {identifier}: {e}")
                continue

        # 查找相似图片
        similar_groups = []
        processed = set()

        for i, (id1, hash1) in enumerate(hashes):
            if id1 in processed:
                continue

            group = [(id1, hash1)]

            for j, (id2, hash2) in enumerate(hashes[i+1:], start=i+1):
                if id2 in processed:
                    continue

                similarity = self.calculate_similarity(hash1, hash2)
                if similarity >= threshold:
                    group.append((id2, hash2))
                    processed.add(id2)

            if len(group) > 1:
                similar_groups.append(group)
                processed.add(id1)

        return similar_groups

    def find_duplicates(
        self,
        images: List[Tuple[str, Image.Image]]
    ) -> List[List[str]]:
        """
        查找完全重复的图片

        Args:
            images: 图片列表，每个元素为(标识符, PIL图片对象)

        Returns:
            重复图片组列表
        """
        return self.find_similar_images(images, threshold=1.0)

    def compute_md5(self, image: Image.Image) -> str:
        """
        计算图片的MD5哈希值（用于精确匹配）

        Args:
            image: PIL图片对象

        Returns:
            MD5哈希值
        """
        img_bytes = image.tobytes()
        return hashlib.md5(img_bytes).hexdigest()
