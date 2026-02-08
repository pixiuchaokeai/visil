# -*- coding: utf-8 -*-
"""
config.py
统一的配置管理文件
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    """实验配置类"""

    # 目录配置
    output_dir: str = "reproduce_results"
    frames_dir: str = "output/frames"
    features_dir: str = "output/features"
    video_dir: str = "datasets/FIVR-200K"
    video_pattern: str = "{id}.mp4"

    # FIVR-5K配置 - 使用过滤后的29个查询
    fivr5k_queries: int = 29  # 你的过滤列表中有29个查询
    fivr5k_database_size: int = 3234  # 过滤后的数据库大小

    # 论文结果配置
    paper_results: Dict = None

    def __post_init__(self):
        """初始化后的处理"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

        # 初始化论文结果
        if self.paper_results is None:
            self.paper_results = self.load_paper_results()

    def load_paper_results(self) -> Dict:
        """加载论文结果"""
        return {
            # 表2: Feature Extraction Comparison
            "table2": {
                "MAC": {"DSVR": 0.747, "CSVR": 0.730, "ISVR": 0.684},
                "SPoC": {"DSVR": 0.735, "CSVR": 0.722, "ISVR": 0.669},
                "R-MAC": {"DSVR": 0.777, "CSVR": 0.764, "ISVR": 0.707},
                "GeM": {"DSVR": 0.776, "CSVR": 0.768, "ISVR": 0.711},
                "iMAC": {"DSVR": 0.755, "CSVR": 0.749, "ISVR": 0.689},
                "L2-iMAC4x": {"DSVR": 0.814, "CSVR": 0.810, "ISVR": 0.738},
                "L3-iMAC9x": {"DSVR": 0.838, "CSVR": 0.832, "ISVR": 0.739},
            },
            # 表3: Ablation Study
            "table3": {
                "ViSiLf": {"DSVR": 0.838, "CSVR": 0.832, "ISVR": 0.739},
                "ViSiLf+W": {"DSVR": 0.844, "CSVR": 0.837, "ISVR": 0.750},
                "ViSiLf+W+A": {"DSVR": 0.856, "CSVR": 0.848, "ISVR": 0.768},
                "ViSiLsym": {"DSVR": 0.830, "CSVR": 0.823, "ISVR": 0.731},
                "ViSiLv": {"DSVR": 0.880, "CSVR": 0.869, "ISVR": 0.777},
            },
            # 表4: Regularization Impact
            "table4": {
                "without_reg": {"DSVR": 0.859, "CSVR": 0.842, "ISVR": 0.756},
                "with_reg": {"DSVR": 0.880, "CSVR": 0.869, "ISVR": 0.777},
            },
            # 表6: FIVR-200K Results
            "table6": {
                "ViSiLf": {"DSVR": 0.843, "CSVR": 0.797, "ISVR": 0.660},
                "ViSiLsym": {"DSVR": 0.833, "CSVR": 0.792, "ISVR": 0.654},
                "ViSiLv": {"DSVR": 0.892, "CSVR": 0.841, "ISVR": 0.702},
            }
        }


@dataclass
class ModelConfig:
    """模型配置类"""

    # 模型选择
    model_type: str = "ViSiLv"  # ViSiLf, ViSiLsym, ViSiLv

    # 根据论文表3，修正配置
    # ViSiLf: L3-iMAC9x 特征
    # ViSiLsym: L3-iMAC9x + 对称
    # ViSiLv: L3-iMAC9x + 白化 + 注意力 + 视频比较器

    # 修正：根据模型类型设置正确的配置
    @property
    def symmetric(self) -> bool:
        """对称性：仅ViSiLsym使用"""
        return self.model_type.lower() == "visilsym"

    @property
    def whiteninig(self) -> bool:
        """白化：ViSiLf+W, ViSiLf+W+A, ViSiLsym, ViSiLv使用"""
        return self.model_type.lower() in ["visilf+w", "visilf+w+a", "visilsym", "visilv"]

    @property
    def attention(self) -> bool:
        """注意力：ViSiLf+W+A, ViSiLsym, ViSiLv使用"""
        return self.model_type.lower() in ["visilf+w+a", "visilsym", "visilv"]

    @property
    def video_comperator(self) -> bool:
        """视频比较器：仅ViSiLv使用"""
        return self.model_type.lower() == "visilv"

    @property
    def feature_dim(self) -> int:
        """特征维度：3840维（L3-iMAC9x），512维是PCA降维后的"""
        if self.pretrained:
            return 512  # 预训练模型使用PCA降维到512维
        else:
            return 3840  # 原始L3-iMAC9x维度

    # 其他固定配置
    pretrained: bool = True
    network: str = "resnet50"

    # 设备配置
    gpu_id: int = 0
    batch_size: int = 32
    query_batch_size: int = 5


@dataclass
class DatasetConfig:
    """数据集配置类"""

    name: str = "FIVR-5K"
    queries_file: str = "datasets/fivr-5k-queries-filtered.txt"  # 使用过滤后的查询文件
    database_file: str = "datasets/fivr-5k-database-filtered.txt"  # 使用过滤后的数据库文件
    annotations_file: str = "datasets/fivr-filtered.pickle"  # 使用生成的pickle标注文件

    # 论文设置：使用完整的数据库视频进行评估
    evaluation_size: int = 3234  # 过滤后的数据库大小

    # 任务定义
    tasks: Dict = None

    def __post_init__(self):
        """初始化任务定义"""
        if self.tasks is None:
            self.tasks = {
                "DSVR": ["ND", "DS"],  # Duplicate Scene Video Retrieval
                "CSVR": ["ND", "DS", "CS"],  # Complementary Scene Video Retrieval
                "ISVR": ["ND", "DS", "CS", "IS"],  # Incident Scene Video Retrieval
            }


# 全局配置实例
experiment_config = ExperimentConfig()
model_config = ModelConfig()
dataset_config = DatasetConfig()