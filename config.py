# config.py - 配置管理（新增默认实验列表，支持灵活选择）
"""
ViSiL实验配置管理
修复点1: 新增 DEFAULT_ABLATION_EXPERIMENTS，仅包含需要的消融模型
修复点2: 保留完整配置供用户自定义
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import torch
import hashlib


@dataclass
class FeatureExtractionConfig:
    """特征提取配置 - 专注L2/L3-iMAC"""
    name: str = "L3-iMAC9x"
    network: str = "resnet50"
    feature_dim: int = 3840
    regions: int = 9
    level: int = 3
    use_pca: bool = False
    use_attention: bool = False
    use_video_comparator: bool = False
    symmetric: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    def get_hash(self) -> str:
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


@dataclass
class EvaluationConfig:
    """评估配置 - 统一参数"""
    dataset: str = "FIVR-5K"
    video_dir: str = "datasets/FIVR-200K"
    video_pattern: str = "{id}.mp4"
    feature_config: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)

    max_frames: int = 50
    batch_size: int = 4
    feature_batch_size: int = 16
    query_batch_size: int = 10
    num_workers: int = 1

    video_size: int = 224
    video_fps: int = 1

    gpu_id: int = 0
    cpu_only: bool = False

    base_output_dir: str = "output"
    experiment_id: str = "default_experiment"
    clear_cache: bool = False

    def __post_init__(self):
        if self.experiment_id == "default_experiment":
            import time
            timestamp = str(int(time.time()))[-6:]
            self.experiment_id = f"{self.feature_config.name}_{timestamp}"

        # 目录结构
        self.frames_dir = os.path.join(self.base_output_dir, "frames")
        feature_name = self.feature_config.name
        if feature_name == "L2-iMAC4x":
            feature_dir_name = "l2_imac"
        elif feature_name == "L3-iMAC9x":
            feature_dir_name = "ViSiL_f"
        else:
            feature_dir_name = feature_name.replace("_", "").lower()
        self.features_dir = os.path.join(self.base_output_dir, "features1", feature_dir_name)
        self.output_dir = os.path.join(self.base_output_dir, "results", self.experiment_id)

        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        if self.cpu_only or not torch.cuda.is_available():
            self.device = torch.device('cpu')
            self.feature_batch_size = min(self.feature_batch_size, 8)
            self.num_workers = min(self.num_workers, 2)
        else:
            self.device = torch.device(f'cuda:{self.gpu_id}')
            self.feature_batch_size = min(self.feature_batch_size, 32)

        print(f"> 实验配置初始化:")
        print(f"  实验ID: {self.experiment_id}")
        print(f"  特征配置: {self.feature_config.name}")
        print(f"  特征目录: {self.features_dir}")
        print(f"  特征批次大小: {self.feature_batch_size}")
        print(f"  并行数: {self.num_workers}")

    def to_dict(self) -> Dict:
        config_dict = asdict(self)
        config_dict['feature_config'] = self.feature_config.to_dict()
        config_dict['device'] = str(self.device)
        config_dict['frames_dir'] = self.frames_dir
        config_dict['features_dir'] = self.features_dir
        config_dict['output_dir'] = self.output_dir
        return config_dict

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ==================== 修复点1：特征比较实验配置 ====================
FEATURE_CONFIGS = {
    "L3-iMAC9x": FeatureExtractionConfig(
        name="L3-iMAC9x",
        network="resnet50",
        feature_dim=3840,
        regions=9,
        level=3,
        use_pca=False,
        use_attention=False,
        use_video_comparator=False,
        symmetric=False
    )
}

# ==================== 修复点2：消融实验配置（保留完整，但默认仅运行部分）====================
ABLATION_CONFIGS = {
    "visil_v": FeatureExtractionConfig(
        name="visil_v",
        network="resnet50",
        feature_dim=3840,
        regions=9,
        level=3,
        use_pca=True,
        use_attention=True,
        use_video_comparator=True,
        symmetric=False
    ),
    "visil_f_w_a": FeatureExtractionConfig(
        name="visil_f_w_a",
        network="resnet50",
        feature_dim=3840,
        regions=9,
        level=3,
        use_pca=True,
        use_attention=True,
        use_video_comparator=False,
        symmetric=False
    ),
    "visil_f_w": FeatureExtractionConfig(
        name="visil_f_w",
        network="resnet50",
        feature_dim=3840,
        regions=9,
        level=3,
        use_pca=True,
        use_attention=False,
        use_video_comparator=False,
        symmetric=False
    ),
    "visil_f": FeatureExtractionConfig(
        name="visil_f",
        network="resnet50",
        feature_dim=3840,
        regions=9,
        level=3,
        use_pca=False,
        use_attention=False,
        use_video_comparator=False,
        symmetric=False
    ),
    "visil_sym": FeatureExtractionConfig(
        name="visil_sym",
        network="resnet50",
        feature_dim=3840,
        regions=9,
        level=3,
        use_pca=True,
        use_attention=True,
        use_video_comparator=True,
        symmetric=True
    )
}

# ==================== 修复点3：默认运行的消融实验列表 ====================
EXPERIMENT_PHASES = ["visil_v", "visil_f_w_a", "visil_f_w"]