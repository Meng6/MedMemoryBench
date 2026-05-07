"""
Benchmark 数据集模块

提供各类评测数据集的处理实现
"""

from pathlib import Path
from typing import Dict, Type, Optional

from .base import BaseDataset, Session, Query, EvaluationUnit
from .medmemorybench import MedMemoryBenchDataset
from .locomo import LoCoMoDataset

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {
    "medmemorybench": MedMemoryBenchDataset,
    "locomo": LoCoMoDataset,
}


def create_dataset(
    dataset_name: str,
    data_dir: Path,
    config: Dict,
) -> BaseDataset:
    """
    创建数据集实例

    Args:
        dataset_name: 数据集名称
        data_dir: 数据目录
        config: 数据集配置

    Returns:
        数据集实例
    """
    dataset_class = DATASET_REGISTRY.get(dataset_name.lower())
    if dataset_class is None:
        raise ValueError(f"未知的数据集: {dataset_name}，可用: {list(DATASET_REGISTRY.keys())}")

    return dataset_class(data_dir, config)


__all__ = [
    "BaseDataset",
    "Session",
    "Query",
    "EvaluationUnit",
    "MedMemoryBenchDataset",
    "LoCoMoDataset",
    "create_dataset",
    "DATASET_REGISTRY",
]
