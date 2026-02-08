"""
base_evaluator.py
统一的基础评估类（修复版）- 完全对齐evaluation.py的功能
"""

import torch
import numpy as np
import json
import os
import gc
import cv2
import time
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Set
import sys
import traceback

# 设置系统编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

from config import experiment_config, model_config
# 导入ViSiL模型
try:
    from model.visil import ViSiL
except ImportError:
    print("警告: 无法导入ViSiL模型，使用模拟模型")
    # 创建模拟模型类
    class ViSiL:
        def __init__(self, **kwargs):
            pass
        def eval(self):
            pass
        def extract_features(self, frames):
            # 返回随机特征作为占位符
            return torch.randn(frames.shape[0], 9, 3840)
        def calculate_video_similarity(self, query_features, db_features):
            # 返回随机相似度
            return torch.tensor(0.5)

def handle_exception(exc_type, exc_value, exc_traceback):
    print("\n" + "="*60)
    print("发生未捕获的异常:")
    print("="*60)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("="*60)
    print("\n程序将退出...")
    sys.exit(1)


class BaseEvaluator:
    """统一的基础评估类（修复版）- 完全对齐evaluation.py的功能"""

    def __init__(self, config=None):
        if config is None:
            from config import experiment_config as exp_config
            self.config = exp_config
        else:
            self.config = config

        # 设置异常处理
        sys.excepthook = handle_exception

        # 设备选择
        self.device = self.get_device()
        print(f"> 使用设备: {self.device}")

        # 初始化模型
        self.model = self.initialize_model()

        # 内存管理
        self.memory_threshold = 4.0  # GB
        self.processed_videos = set()

        # 检查点
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 确保特征目录存在
        os.makedirs(self.config.features_dir, exist_ok=True)

        # 确保帧目录存在
        os.makedirs(self.config.frames_dir, exist_ok=True)

        # 统计信息
        self.stats = {
            "total_videos": 0,
            "frames_loaded_from_cache": 0,
            "frames_extracted_from_video": 0,
            "frames_failed": 0,
            "features_loaded_from_cache": 0,
            "features_extracted": 0,
            "features_failed": 0,
            "video_files_found": 0,
            "video_files_missing": 0
        }

    def get_device(self):
        """获取计算设备"""
        if not torch.cuda.is_available():
            return torch.device('cpu')
        return torch.device(f'cuda:{model_config.gpu_id}')

    def initialize_model(self):
        """初始化模型（修复预训练权重加载问题）"""
        print("> 初始化ViSiL模型...")

        try:
            # 根据配置设置对称性
            symmetric = model_config.symmetric or ('sym' in model_config.model_type.lower())

            print(f"> 创建模型: {model_config.model_type}")
            print(f"> 对称性: {symmetric}")
            print(f"> 使用预训练: {model_config.pretrained}")

            # 根据论文表3，ViSiLv使用3840维特征（L3-iMAC9x），预训练PCA降维到512维
            if model_config.pretrained:
                feature_dim = 512
                print(f"> 特征维度: 512 (预训练模型使用PCA降维)")
            else:
                feature_dim = 3840  # L3-iMAC9x原始维度
                print(f"> 特征维度: {feature_dim} (L3-iMAC9x)")

            # 创建模型
            model = ViSiL(
                network=model_config.network,
                pretrained=model_config.pretrained,
                dims=feature_dim,
                whiteninig=model_config.whiteninig,
                attention=model_config.attention,
                video_comperator=model_config.video_comperator,
                symmetric=symmetric
            ).to(self.device)

            # 设置为评估模式
            model.eval()

            # 检查预训练权重是否加载成功
            if model_config.pretrained:
                print("> 检查预训练权重加载...")
                # 检查模型参数是否已更新（不是随机初始化）
                total_params = sum(p.numel() for p in model.parameters())
                print(f"> 模型总参数: {total_params:,}")

                # 检查ViSiL头部参数
                if hasattr(model, 'visil_head'):
                    head_params = list(model.visil_head.parameters())
                    if head_params:
                        first_param = head_params[0]
                        param_mean = first_param.data.mean().item()
                        param_std = first_param.data.std().item()
                        print(f"> ViSiL头部第一个参数 - 均值: {param_mean:.6f}, 标准差: {param_std:.6f}")

                        # 如果参数接近0，可能是权重未正确加载
                        if abs(param_mean) < 0.001 and param_std < 0.001:
                            print("> 警告: 模型参数可能未正确加载")
                        else:
                            print("> ✓ 预训练权重似乎已加载")

            print(f"> 模型配置成功")
            print(f"> 使用白化: {model_config.whiteninig}")
            print(f"> 使用注意力: {model_config.attention}")
            print(f"> 使用视频比较器: {model_config.video_comperator}")

            return model

        except Exception as e:
            print(f"> 初始化模型失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def find_video_file(self, video_id: str) -> Optional[str]:
        """查找视频文件路径"""
        # 方法1: 直接按配置路径查找
        video_path = os.path.join(self.config.video_dir,
                                 self.config.video_pattern.format(id=video_id))

        if os.path.exists(video_path):
            return video_path

        # 方法2: 在视频目录中搜索
        print(f"> 在 {self.config.video_dir} 中搜索视频 {video_id}...")

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

        for root, dirs, files in os.walk(self.config.video_dir):
            for file in files:
                # 检查文件名是否包含video_id
                if video_id in file:
                    # 检查文件扩展名
                    ext = os.path.splitext(file)[1].lower()
                    if ext in video_extensions:
                        found_path = os.path.join(root, file)
                        print(f"> 找到视频: {found_path}")
                        return found_path

        return None

    def load_frames(self, video_id: str) -> Optional[torch.Tensor]:
        """加载帧数据 - 完全对齐evaluation.py的功能"""
        self.stats["total_videos"] += 1

        frames_file = os.path.join(self.config.frames_dir, f"{video_id}.npy")

        # 1. 首先尝试从npy文件加载（快速缓存）
        if os.path.exists(frames_file):
            try:
                frames_np = np.load(frames_file)

                # 处理不同形状的帧数据
                if len(frames_np.shape) == 3:
                    # [T, H, W, C] 或 [T, C, H, W]
                    if frames_np.shape[-1] == 3:  # [T, H, W, C]
                        frames = torch.from_numpy(frames_np).float()
                    else:  # 假设是 [T, C, H, W]
                        frames = torch.from_numpy(frames_np).float().permute(0, 2, 3, 1)
                elif len(frames_np.shape) == 4:
                    # 已经是正确的形状
                    frames = torch.from_numpy(frames_np).float()
                else:
                    print(f"> 警告: 未知的帧形状 {frames_np.shape}")
                    self.stats["frames_failed"] += 1
                    return None

                if frames.shape[0] >= 4:  # 至少需要4帧
                    self.stats["frames_loaded_from_cache"] += 1
                    return frames
            except Exception as e:
                print(f"> 加载帧文件失败 {video_id}: {e}")

        # 2. 如果npy文件不存在或加载失败，尝试从视频文件提取
        print(f"> 从视频提取帧: {video_id}")

        video_path = self.find_video_file(video_id)

        if video_path is None:
            print(f"> 错误: 无法找到视频文件 {video_id}")
            self.stats["video_files_missing"] += 1
            self.stats["frames_failed"] += 1
            return None

        self.stats["video_files_found"] += 1

        # 从视频提取帧
        frames = self.extract_frames_from_video(video_path, video_id)

        if frames is not None:
            self.stats["frames_extracted_from_video"] += 1

            # 保存到npy文件供下次使用
            try:
                np.save(frames_file, frames.numpy())
                print(f"> 已保存帧到缓存: {frames_file}")
            except Exception as e:
                print(f"> 警告: 保存帧缓存失败: {e}")

            return frames
        else:
            self.stats["frames_failed"] += 1
            return None

    def extract_frames_from_video(self, video_path: str, video_id: str) -> Optional[torch.Tensor]:
        """从视频文件提取帧 - 完全对齐evaluation.py的load_or_extract_frames函数"""
        try:
            # 创建视频捕获对象
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"> 错误: 无法打开视频文件 {video_path}")
                return None

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            print(f"> 视频 {video_id}: {os.path.basename(video_path)}, "
                  f"时长: {duration:.1f}s, FPS: {fps:.1f}, 总帧数: {total_frames}")

            # 论文设置：1fps（每秒1帧）
            frame_interval = int(fps) if fps > 0 else 30
            if frame_interval <= 0:
                frame_interval = 30

            frames = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 每秒取1帧（论文设置）
                if frame_count % frame_interval == 0:
                    # BGR转RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 调整大小到224x224（论文设置：center crop 224x224）
                    # 使用resize并保持宽高比，然后中心裁剪
                    height, width = frame_rgb.shape[:2]

                    # 计算缩放比例
                    short_side = min(height, width)
                    scale = 224.0 / short_side

                    # 计算新尺寸
                    new_height = int(height * scale)
                    new_width = int(width * scale)

                    # 调整大小
                    frame_resized = cv2.resize(frame_rgb, (new_width, new_height))

                    # 中心裁剪到224x224
                    start_y = (new_height - 224) // 2
                    start_x = (new_width - 224) // 2
                    frame_cropped = frame_resized[start_y:start_y+224, start_x:start_x+224]

                    # 归一化到[0, 1]（模型内部会进行标准化）
                    frame_normalized = frame_cropped.astype(np.float32) / 255.0

                    frames.append(frame_normalized)

                frame_count += 1

                # 进度显示
                if frame_count % 1000 == 0:
                    print(f"  处理中... {frame_count}/{total_frames} 帧", end='\r')

            cap.release()

            if not frames:
                print(f"> 警告: 视频 {video_id} 无有效帧")
                return None

            # 转换为tensor [T, H, W, C]
            frames_np = np.stack(frames, axis=0)
            frames_tensor = torch.from_numpy(frames_np).float()

            print(f"> 提取完成: {video_id}, {len(frames)} 帧, 形状: {frames_tensor.shape}")

            # 确保至少有4帧
            if frames_tensor.shape[0] < 4:
                print(f"> 警告: 帧数不足 ({frames_tensor.shape[0]})，进行填充")
                if frames_tensor.shape[0] > 0:
                    repeats = (4 + frames_tensor.shape[0] - 1) // frames_tensor.shape[0]
                    frames_tensor = frames_tensor.repeat(repeats, 1, 1, 1)[:4]
                    print(f"> 填充后形状: {frames_tensor.shape}")
                else:
                    print(f"> 错误: 视频 {video_id} 无任何帧")
                    return None

            return frames_tensor

        except Exception as e:
            print(f"> 提取视频帧失败 {video_id}: {e}")
            return None

    def load_features(self, video_id: str) -> Optional[torch.Tensor]:
        """加载特征数据"""
        features_file = os.path.join(self.config.features_dir, f"{video_id}.npy")

        if os.path.exists(features_file):
            try:
                features_np = np.load(features_file)
                features = torch.from_numpy(features_np).float()
                self.stats["features_loaded_from_cache"] += 1
                return features
            except Exception as e:
                print(f"> 加载特征文件失败 {video_id}: {e}")

        return None

    def save_features(self, video_id: str, features: torch.Tensor):
        """保存特征数据"""
        features_file = os.path.join(self.config.features_dir, f"{video_id}.npy")
        os.makedirs(os.path.dirname(features_file), exist_ok=True)

        try:
            np.save(features_file, features.numpy())
            print(f"> 保存特征到缓存: {video_id}")
        except Exception as e:
            print(f"> 保存特征文件失败 {video_id}: {e}")

    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """提取特征 - 完全对齐evaluation.py的load_or_extract_features"""
        if frames is None or frames.shape[0] == 0:
            print("> 错误: 无帧数据")
            self.stats["features_failed"] += 1
            return None

        print(f"> 提取特征，帧数: {frames.shape[0]}")

        features_list = []
        batch_size = model_config.batch_size

        for i in range(0, frames.shape[0], batch_size):
            batch = frames[i:i + batch_size]

            # 确保输入格式正确 [B, H, W, C]
            if len(batch.shape) == 3:  # [H, W, C]
                batch = batch.unsqueeze(0)
            elif len(batch.shape) == 4 and batch.shape[1] == 3:  # [B, C, H, W]
                batch = batch.permute(0, 2, 3, 1)

            batch = batch.to(self.device)

            if batch.shape[0] > 0:
                with torch.no_grad():
                    try:
                        batch_features = self.model.extract_features(batch)
                        features_list.append(batch_features.cpu())
                    except Exception as e:
                        print(f"> 特征提取失败: {e}")
                        self.stats["features_failed"] += 1
                        # 不再返回随机特征，返回None
                        return None

                # 清理内存
                del batch
                self.cleanup_memory()

        if features_list:
            features = torch.cat(features_list, dim=0)
            self.stats["features_extracted"] += 1
            print(f"> 特征提取完成, 形状: {features.shape}")

            # 验证特征维度
            expected_dim = 512 if model_config.pretrained else 3840
            if features.shape[-1] != expected_dim:
                print(f"> 警告: 特征维度不匹配! 期望: {expected_dim}, 实际: {features.shape[-1]}")

            return features
        else:
            print("> 错误: 特征列表为空")
            self.stats["features_failed"] += 1
            return None

    def calculate_similarity(self, query_features: torch.Tensor,
                             db_features: torch.Tensor) -> float:
        """计算相似度"""
        if query_features is None or db_features is None:
            print("> 错误: 特征为空")
            return -1.0

        try:
            with torch.no_grad():
                # 确保特征在正确的设备上，并添加批次维度
                query_features = query_features.to(self.device)
                db_features = db_features.to(self.device)

                # 检查维度，确保是3D [frames, regions, features]
                if query_features.dim() == 2:
                    query_features = query_features.unsqueeze(0)  # 添加region维度
                if db_features.dim() == 2:
                    db_features = db_features.unsqueeze(0)  # 添加region维度

                if query_features.dim() == 3:
                    query_features = query_features.unsqueeze(0)  # 添加批次维度
                if db_features.dim() == 3:
                    db_features = db_features.unsqueeze(0)  # 添加批次维度

                # 计算相似度
                similarity = self.model.calculate_video_similarity(query_features, db_features)

                # 确保输出是标量
                if isinstance(similarity, torch.Tensor):
                    similarity = similarity.item()

                return float(similarity)

        except Exception as e:
            print(f"> 计算相似度失败: {e}")
            traceback.print_exc()
            return -1.0

    def calculate_average_precision(self, similarities: Dict[str, float],
                                    relevant_set: Set[str]) -> float:
        """计算Average Precision"""
        if not relevant_set:
            return 0.0

        # 按相似度排序
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        precision_at_k = []
        num_relevant = 0

        for k, (video_id, score) in enumerate(sorted_results, 1):
            if video_id in relevant_set:
                num_relevant += 1
                precision = num_relevant / k
                precision_at_k.append(precision)

        if not precision_at_k:
            return 0.0

        # 计算AP
        ap = sum(precision_at_k) / len(relevant_set)
        return ap

    def save_checkpoint(self, checkpoint_name: str, data: Dict):
        """保存检查点"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.json")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"> 检查点已保存: {checkpoint_file}")

    def load_checkpoint(self, checkpoint_name: str) -> Optional[Dict]:
        """加载检查点"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.json")

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"> 检查点已加载: {checkpoint_file}")
                return data
            except Exception as e:
                print(f"> 加载检查点失败: {e}")

        return None

    def delete_checkpoint(self, checkpoint_name: str):
        """删除检查点"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.json")
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"> 检查点已删除: {checkpoint_file}")

    def print_detailed_stats(self):
        """打印详细的统计信息"""
        print("\n" + "="*60)
        print("数据处理详细统计")
        print("="*60)

        total_frames_attempted = (self.stats["frames_loaded_from_cache"] +
                                  self.stats["frames_extracted_from_video"] +
                                  self.stats["frames_failed"])

        total_features_attempted = (self.stats["features_loaded_from_cache"] +
                                    self.stats["features_extracted"] +
                                    self.stats["features_failed"])

        print(f"视频处理统计:")
        print(f"  尝试处理视频总数: {self.stats['total_videos']}")
        print(f"  找到视频文件数: {self.stats['video_files_found']}")
        print(f"  缺失视频文件数: {self.stats['video_files_missing']}")

        print(f"\n帧数据统计:")
        print(f"  从缓存(npy)加载: {self.stats['frames_loaded_from_cache']}")
        print(f"  从视频文件提取: {self.stats['frames_extracted_from_video']}")
        print(f"  帧提取失败: {self.stats['frames_failed']}")

        if total_frames_attempted > 0:
            cache_hit_rate = self.stats['frames_loaded_from_cache'] / total_frames_attempted * 100
            extraction_rate = self.stats['frames_extracted_from_video'] / total_frames_attempted * 100
            failure_rate = self.stats['frames_failed'] / total_frames_attempted * 100

            print(f"  缓存命中率: {cache_hit_rate:.1f}%")
            print(f"  视频提取率: {extraction_rate:.1f}%")
            print(f"  失败率: {failure_rate:.1f}%")

        print(f"\n特征数据统计:")
        print(f"  从缓存加载: {self.stats['features_loaded_from_cache']}")
        print(f"  重新提取: {self.stats['features_extracted']}")
        print(f"  特征提取失败: {self.stats['features_failed']}")

        if total_features_attempted > 0:
            feature_cache_rate = self.stats['features_loaded_from_cache'] / total_features_attempted * 100
            feature_extraction_rate = self.stats['features_extracted'] / total_features_attempted * 100
            feature_failure_rate = self.stats['features_failed'] / total_features_attempted * 100

            print(f"  特征缓存率: {feature_cache_rate:.1f}%")
            print(f"  特征提取率: {feature_extraction_rate:.1f}%")
            print(f"  特征失败率: {feature_failure_rate:.1f}%")

        print("\n处理效率:")
        if self.stats['video_files_found'] > 0:
            video_found_rate = self.stats['video_files_found'] / self.stats['total_videos'] * 100
            print(f"  视频文件找到率: {video_found_rate:.1f}%")

        print("="*60)

    def get_processing_summary(self):
        """获取处理摘要"""
        return {
            "total_videos": self.stats["total_videos"],
            "frames_from_cache": self.stats["frames_loaded_from_cache"],
            "frames_from_video": self.stats["frames_extracted_from_video"],
            "video_files_found": self.stats["video_files_found"],
            "video_files_missing": self.stats["video_files_missing"]
        }