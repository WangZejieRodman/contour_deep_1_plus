"""
RetrievalDataset: 方向1的数据集实现
功能:
1. 加载BEV缓存
2. 构建三元组(anchor, positive, negatives)
3. 数据归一化和增强
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_utils import (
    load_pickle,
    normalize_bev,
    stack_bev_with_vcd,
    apply_augmentation
)


class RetrievalDataset(Dataset):
    """方向1: 检索特征提取数据集"""

    def __init__(self,
                 queries_pickle: str,
                 cache_root: str,
                 split: str = 'train',
                 num_negatives: int = 10,
                 augmentation_config: Optional[Dict] = None,
                 resolution: float = 0.2,
                 use_cache: bool = True,
                 max_cache_size: int = 1000):
        """
        初始化数据集

        Args:
            queries_pickle: 查询pickle文件路径
            cache_root: BEV缓存根目录
            split: 'train' 或 'test'
            num_negatives: 每个anchor采样的负样本数量
            augmentation_config: 数据增强配置
            resolution: BEV分辨率(用于增强)
            use_cache: 是否使用内存缓存
            max_cache_size: 最大缓存数量(LRU)
        """
        self.split = split
        self.cache_dir = os.path.join(cache_root, split)
        self.num_negatives = num_negatives
        self.aug_config = augmentation_config
        self.resolution = resolution
        self.use_cache = use_cache

        # 加载查询数据
        self.queries = load_pickle(queries_pickle)
        self.query_keys = sorted(self.queries.keys())

        print(f"[{split}] Loaded {len(self.query_keys)} queries")

        # 内存缓存(LRU)
        if self.use_cache:
            from collections import OrderedDict
            self.memory_cache = OrderedDict()
            self.max_cache_size = max_cache_size

        # 统计信息
        self._compute_statistics()

    def _compute_statistics(self):
        """计算数据集统计信息"""
        total_positives = 0
        total_negatives = 0

        for query_data in self.queries.values():
            total_positives += len(query_data['positives'])
            total_negatives += len(query_data['negatives'])

        self.stats = {
            'total_queries': len(self.queries),
            'avg_positives': total_positives / len(self.queries),
            'avg_negatives': total_negatives / len(self.queries),
        }

        print(f"[{self.split}] Statistics:")
        print(f"  Avg positives: {self.stats['avg_positives']:.1f}")
        print(f"  Avg negatives: {self.stats['avg_negatives']:.1f}")

    def __len__(self) -> int:
        return len(self.query_keys)

    def _load_bev_from_cache(self, query_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        从缓存加载BEV

        Args:
            query_idx: 查询索引

        Returns:
            (bev_layers, vcd) 或 None
        """
        # 检查内存缓存
        if self.use_cache and query_idx in self.memory_cache:
            # LRU: 移到末尾
            self.memory_cache.move_to_end(query_idx)
            return self.memory_cache[query_idx]

        # 从磁盘加载
        cache_filename = f"{query_idx:06d}.npz"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        if not os.path.exists(cache_path):
            print(f"Warning: Cache not found for query {query_idx}")
            return None

        try:
            data = np.load(cache_path, mmap_mode='r' if not self.use_cache else None)
            bev_layers = data['bev_layers']
            vcd = data['vcd']

            # 更新内存缓存
            if self.use_cache:
                # 如果缓存满了，删除最旧的
                if len(self.memory_cache) >= self.max_cache_size:
                    self.memory_cache.popitem(last=False)

                # 复制到内存(避免mmap引用)
                self.memory_cache[query_idx] = (
                    bev_layers.copy(),
                    vcd.copy()
                )

            return bev_layers, vcd

        except Exception as e:
            print(f"Error loading cache {query_idx}: {e}")
            return None

    def _sample_triplet(self, query_key: int) -> Tuple[int, int, List[int]]:
        """
        采样三元组

        Args:
            query_key: anchor查询键

        Returns:
            (anchor_idx, positive_idx, negative_indices)
        """
        query_data = self.queries[query_key]

        # Anchor
        anchor_idx = query_key

        # Positive: 随机选1个
        positives = query_data['positives']
        if len(positives) == 0:
            # 没有正样本，返回自身
            positive_idx = anchor_idx
        else:
            positive_idx = random.choice(positives)

        # Negatives: 随机选num_negatives个
        negatives = query_data['negatives']
        if len(negatives) >= self.num_negatives:
            negative_indices = random.sample(negatives, self.num_negatives)
        else:
            # 不足则重复采样
            negative_indices = random.choices(negatives, k=self.num_negatives)

        return anchor_idx, positive_idx, negative_indices

    def _preprocess_bev(self, bev_layers: np.ndarray, vcd: np.ndarray,
                        apply_aug: bool = False) -> torch.Tensor:
        """
        预处理BEV: 归一化、堆叠、增强、转tensor

        Args:
            bev_layers: [8, H, W]
            vcd: [H, W]
            apply_aug: 是否应用数据增强

        Returns:
            tensor: [9, H, W]
        """
        # 1. 归一化
        bev_norm, vcd_norm = normalize_bev(bev_layers, vcd)

        # 2. 堆叠
        stacked = stack_bev_with_vcd(bev_norm, vcd_norm)  # [9, H, W]

        # 3. 数据增强(仅训练集)
        if apply_aug and self.aug_config is not None:
            stacked = apply_augmentation(stacked, self.aug_config, self.resolution)

        # 4. 转tensor
        tensor = torch.from_numpy(stacked).float()

        return tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本

        Returns:
            {
                'anchor': [9, H, W],
                'positive': [9, H, W],
                'negatives': [num_neg, 9, H, W],
                'anchor_idx': int,
                'positive_idx': int,
                'negative_indices': [num_neg]
            }
        """
        query_key = self.query_keys[idx]

        # 1. 采样三元组
        anchor_idx, positive_idx, negative_indices = self._sample_triplet(query_key)

        # 2. 加载BEV
        anchor_bev = self._load_bev_from_cache(anchor_idx)
        positive_bev = self._load_bev_from_cache(positive_idx)

        if anchor_bev is None or positive_bev is None:
            # 加载失败，返回零张量
            dummy = torch.zeros(9, 200, 200)
            return {
                'anchor': dummy,
                'positive': dummy,
                'negatives': torch.zeros(self.num_negatives, 9, 200, 200),
                'anchor_idx': anchor_idx,
                'positive_idx': positive_idx,
                'negative_indices': negative_indices
            }

        # 3. 预处理
        apply_aug = (self.split == 'train')  # 仅训练集增强

        anchor_tensor = self._preprocess_bev(*anchor_bev, apply_aug=apply_aug)
        positive_tensor = self._preprocess_bev(*positive_bev, apply_aug=apply_aug)

        # 4. 加载负样本
        negative_tensors = []
        valid_negative_indices = []

        for neg_idx in negative_indices:
            neg_bev = self._load_bev_from_cache(neg_idx)
            if neg_bev is not None:
                neg_tensor = self._preprocess_bev(*neg_bev, apply_aug=False)  # 负样本不增强
                negative_tensors.append(neg_tensor)
                valid_negative_indices.append(neg_idx)

        # 如果负样本不足，用零填充
        while len(negative_tensors) < self.num_negatives:
            negative_tensors.append(torch.zeros_like(anchor_tensor))
            valid_negative_indices.append(-1)

        negatives_tensor = torch.stack(negative_tensors)  # [num_neg, 9, H, W]

        return {
            'anchor': anchor_tensor,
            'positive': positive_tensor,
            'negatives': negatives_tensor,
            'anchor_idx': anchor_idx,
            'positive_idx': positive_idx,
            'negative_indices': valid_negative_indices
        }


def create_dataloader(dataset: RetrievalDataset,
                      batch_size: int = 8,
                      num_workers: int = 4,
                      shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    创建DataLoader

    Args:
        dataset: RetrievalDataset实例
        batch_size: 批大小
        num_workers: 工作进程数
        shuffle: 是否打乱

    Returns:
        DataLoader
    """

    def collate_fn(batch):
        """自定义batch整理函数"""
        anchors = torch.stack([item['anchor'] for item in batch])
        positives = torch.stack([item['positive'] for item in batch])
        negatives = torch.stack([item['negatives'] for item in batch])

        return {
            'anchor': anchors,  # [B, 9, H, W]
            'positive': positives,  # [B, 9, H, W]
            'negatives': negatives,  # [B, num_neg, 9, H, W]
            'anchor_idx': [item['anchor_idx'] for item in batch],
            'positive_idx': [item['positive_idx'] for item in batch],
            'negative_indices': [item['negative_indices'] for item in batch]
        }

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # 加速GPU传输
        persistent_workers=True if num_workers > 0 else False
    )

    return dataloader


# ========== 测试代码 ==========
if __name__ == "__main__":
    """
    测试RetrievalDataset
    用法: python data/dataset_retrieval.py
    """
    import yaml
    import time

    print("Testing RetrievalDataset...")

    # 1. 加载配置
    config_path = "/home/wzj/pan1/contour_deep_1/configs/config_base.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 创建训练数据集
    print("\n=== 创建训练数据集 ===")
    train_dataset = RetrievalDataset(
        queries_pickle="/home/wzj/pan1/contour_deep_1/data/test_queries_chilean_period.pickle",
        cache_root="/home/wzj/pan1/contour_deep_1/data/Chilean_BEV_Cache/",
        split='train',
        num_negatives=10,
        augmentation_config=config['augmentation'],
        resolution=config['bev']['resolution'],
        use_cache=True,
        max_cache_size=500
    )

    print(f"数据集大小: {len(train_dataset)}")

    # 3. 测试单个样本
    print("\n=== 测试单个样本 ===")
    sample = train_dataset[0]
    print(f"Anchor shape: {sample['anchor'].shape}")
    print(f"Positive shape: {sample['positive'].shape}")
    print(f"Negatives shape: {sample['negatives'].shape}")
    print(f"Anchor idx: {sample['anchor_idx']}")
    print(f"Positive idx: {sample['positive_idx']}")
    print(f"Negative indices: {sample['negative_indices']}")

    # 4. 创建DataLoader并测试
    print("\n=== 测试DataLoader ===")
    train_loader = create_dataloader(
        train_dataset,
        batch_size=8,
        num_workers=4,
        shuffle=True
    )

    print(f"Batches per epoch: {len(train_loader)}")

    # 测试加载速度
    print("\n=== 测试加载速度 ===")
    start_time = time.time()
    num_batches_to_test = 10

    for i, batch in enumerate(train_loader):
        if i >= num_batches_to_test:
            break

        print(f"Batch {i + 1}:")
        print(f"  Anchor: {batch['anchor'].shape}")
        print(f"  Positive: {batch['positive'].shape}")
        print(f"  Negatives: {batch['negatives'].shape}")

    elapsed = time.time() - start_time
    samples_per_sec = (num_batches_to_test * 8) / elapsed

    print(f"\n加载速度: {samples_per_sec:.1f} samples/sec")
    print(f"目标: >50 samples/sec")

    if samples_per_sec > 50:
        print("✓ 加载速度达标!")
    else:
        print("✗ 加载速度不足，需要优化")

    # 5. 可视化测试
    print("\n=== 可视化测试 ===")
    import matplotlib.pyplot as plt

    # 取第一个batch
    batch = next(iter(train_loader))

    # 可视化第一个样本的anchor/positive/negative[0]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Anchor (显示前3层BEV)
    for i in range(3):
        axes[0, i].imshow(batch['anchor'][0, i].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'Anchor - Layer {i}')
        axes[0, i].axis('off')

    # Positive
    for i in range(3):
        axes[1, i].imshow(batch['positive'][0, i].cpu().numpy(), cmap='gray')
        axes[1, i].set_title(f'Positive - Layer {i}')
        axes[1, i].axis('off')

    # Negative[0]
    for i in range(3):
        axes[2, i].imshow(batch['negatives'][0, 0, i].cpu().numpy(), cmap='gray')
        axes[2, i].set_title(f'Negative - Layer {i}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('test_retrieval_dataset_visualization.png', dpi=150)
    print("可视化已保存: test_retrieval_dataset_visualization.png")

    # 6. 创建测试数据集
    print("\n=== 创建测试数据集 ===")
    test_dataset = RetrievalDataset(
        queries_pickle="/home/wzj/pan1/contour_deep_1/data/test_queries_chilean_period.pickle",
        cache_root="/home/wzj/pan1/contour_deep_1/data/Chilean_BEV_Cache/",
        split='test',
        num_negatives=10,
        augmentation_config=None,  # 测试集不增强
        resolution=config['bev']['resolution'],
        use_cache=True
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=8,
        num_workers=4,
        shuffle=False  # 测试集不打乱
    )

    print(f"测试集大小: {len(test_dataset)}")
    print(f"测试集batches: {len(test_loader)}")

    print("\n✓ RetrievalDataset测试完成!")