# base_evaluator.py - 基础评估器（修复帧提取，仿照evaluation.py）
"""
基础评估器类 - 仿照evaluation.py修复帧提取
修复点：完全仿照evaluation.py的帧提取逻辑
修复点：使用VideoGenerator正确提取帧
修复点：修复特征提取缓存逻辑
"""

import torch
import os
import json
import gc
import numpy as np
import time
from tqdm import tqdm
import traceback

from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator

from config import EvaluationConfig


class BaseEvaluator:
    """基础评估器 - 仿照evaluation.py修复版"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = config.device

        # ==================== 修复点1：验证配置正确性 ====================
        print(f"\n> 验证实验配置:")
        print(f"  实验ID: {config.experiment_id}")
        print(f"  特征配置: {config.feature_config.name}")
        print(f"  设备: {self.device}")
        print(f"  帧目录: {config.frames_dir}")
        print(f"  特征目录: {config.features_dir}")

        # 打印特征配置详情
        feature_cfg = config.feature_config
        print(f"  特征参数:")
        print(f"    网络: {feature_cfg.network}")
        print(f"    特征维度: {feature_cfg.feature_dim}")
        print(f"    区域: {feature_cfg.regions}")
        print(f"    级别: L{feature_cfg.level}")
        print(f"    PCA: {feature_cfg.use_pca}")
        print(f"    注意力: {feature_cfg.use_attention}")
        print(f"    视频比较器: {feature_cfg.use_video_comparator}")
        print(f"    对称: {feature_cfg.symmetric}")

        # 清理缓存（如果配置要求）
        if config.clear_cache:
            self.clear_cached_files()

        # 初始化模型
        self.model = self.init_model()

        # 加载数据集
        self.dataset = self.load_dataset()

        print(f"> 初始化完成")

    def clear_cached_files(self):
        """清理缓存文件 - 只清理特征缓存，不清理帧缓存"""
        print("> 清理特征缓存文件...")
        import shutil

        # 只清理当前配置的特征缓存目录
        if os.path.exists(self.config.features_dir):
            try:
                shutil.rmtree(self.config.features_dir)
                print(f"> 已清理: {self.config.features_dir}")
            except Exception as e:
                print(f"> 清理失败 {self.config.features_dir}: {e}")

        # 重新创建特征目录
        os.makedirs(self.config.features_dir, exist_ok=True)

    def init_model(self) -> ViSiL:
        """初始化ViSiL模型 - 根据配置正确设置参数"""
        print(f"> 初始化模型: {self.config.feature_config.name}")

        # 根据特征配置设置模型参数
        feature_config = self.config.feature_config

        try:
            # ==================== 修复点2：正确传递模型参数 ====================
            model = ViSiL(
                pretrained=True,
                symmetric=feature_config.symmetric,
                whiteninig=feature_config.use_pca,
                attention=feature_config.use_attention,
                video_comperator=feature_config.use_video_comparator,
                dims=feature_config.feature_dim
            ).to(self.device)

            model.eval()

            # 验证模型配置
            print(f"> 模型加载成功，验证配置:")
            print(f"  实际对称设置: {model.visil_head.symmetric}")

            if feature_config.use_attention:
                if model.visil_head.attention is None:
                    print("> 警告: 配置要求注意力机制，但模型未启用")
                else:
                    print("> 注意力机制: 已启用")

            if feature_config.use_video_comparator:
                if model.visil_head.video_comperator is None:
                    print("> 警告: 配置要求视频比较器，但模型未启用")
                else:
                    print("> 视频比较器: 已启用")

            if feature_config.use_pca:
                if model.cnn.pca is None:
                    print("> 警告: 配置要求PCA，但模型未启用")
                else:
                    print("> PCA: 已启用")

            return model

        except Exception as e:
            print(f"> 模型初始化失败: {e}")
            traceback.print_exc()
            raise

    def load_dataset(self):
        """加载数据集 - 修复版本"""
        dataset_name = self.config.dataset

        try:
            if 'FIVR' in dataset_name:
                from datasets import FIVR
                version = dataset_name.split('-')[1].lower() if '-' in dataset_name else '5k'
                dataset = FIVR(version=version)
                print(f"> 数据集加载成功: {dataset.name}")
                print(f"> 查询视频数: {len(dataset.get_queries())}")
                print(f"> 数据库视频数: {len(dataset.get_database())}")
                return dataset
            else:
                raise ValueError(f"未知的数据集: {dataset_name}")

        except ImportError as e:
            print(f"> 错误: 无法导入数据集模块 - {e}")
            raise

    def extract_frames_from_video(self, video_path: str, video_id: str) -> torch.Tensor:
        """
        从视频提取帧 - 完全仿照evaluation.py的实现
        修复点：使用与evaluation.py完全相同的逻辑
        """
        frames_file = os.path.join(self.config.frames_dir, f"{video_id}.npy")

        # ==================== 修复点3：优先检查共享帧缓存 ====================
        if os.path.exists(frames_file):
            try:
                file_size = os.path.getsize(frames_file)
                if file_size > 1024:
                    frames = np.load(frames_file)
                    if frames is not None and frames.shape[0] >= 4:
                        print(f"> 加载缓存帧: {video_id} ({frames.shape[0]}帧)")
                        return torch.from_numpy(frames).float()
            except Exception as e:
                print(f"> 加载帧文件失败 {video_id}: {e}")
                try:
                    os.remove(frames_file)
                except:
                    pass

        print(f"> 提取帧: {video_id}")
        try:
            # ==================== 修复点4：完全仿照evaluation.py的帧提取逻辑 ====================
            # 创建临时文件列表（仿照evaluation.py）
            temp_list_file = f"temp_{video_id}.txt"
            with open(temp_list_file, 'w') as f:
                f.write(f"{video_id} {video_path}")

            # 使用VideoGenerator提取帧（与evaluation.py参数一致）
            generator = VideoGenerator(temp_list_file)
            loader = DataLoader(generator, num_workers=0, batch_size=1)

            frames = None
            for frames_batch, video_id_batch in loader:
                frames = frames_batch[0]  # [T, H, W, C]
                break

            if frames is None or frames.shape[0] < 4:
                print(f"> 视频无有效帧: {video_id}")
                if os.path.exists(temp_list_file):
                    os.remove(temp_list_file)
                return None

            # 限制最大帧数（仿照evaluation.py）
            if frames.shape[0] > self.config.max_frames:
                step = max(1, frames.shape[0] // self.config.max_frames)
                indices = list(range(0, frames.shape[0], step))[:self.config.max_frames]
                frames = frames[indices]

            # 保存为npy文件
            frames_np = frames.cpu().numpy()
            np.save(frames_file, frames_np)

            # 清理临时文件
            if os.path.exists(temp_list_file):
                os.remove(temp_list_file)

            # 验证保存的文件
            if os.path.exists(frames_file):
                saved_frames = np.load(frames_file)
                print(f"> 保存帧完成: {video_id} ({saved_frames.shape[0]}帧)")
            else:
                print(f"> 警告: 帧文件未保存成功: {video_id}")

            return frames

        except Exception as e:
            print(f"> 提取视频 {video_id} 帧失败: {e}")
            # 清理临时文件
            if os.path.exists(f'temp_{video_id}.txt'):
                try:
                    os.remove(f'temp_{video_id}.txt')
                except:
                    pass
            return None

    def extract_features_from_frames(self, frames: torch.Tensor, video_id: str) -> torch.Tensor:
        """
        从帧提取特征 - 修复缓存逻辑
        修复点：正确保存和加载特征缓存
        """
        features_file = os.path.join(self.config.features_dir, f"{video_id}.npy")

        # ==================== 修复点5：优先检查特征缓存 ====================
        if os.path.exists(features_file):
            try:
                file_size = os.path.getsize(features_file)
                if file_size > 1024:
                    features_np = np.load(features_file)
                    if features_np is not None and features_np.shape[0] > 0:
                        features = torch.from_numpy(features_np).float()
                        print(f"> 加载缓存特征: {video_id} (形状: {features_np.shape})")
                        return features
            except Exception as e:
                print(f"> 加载特征文件失败 {video_id}: {e}")
                try:
                    os.remove(features_file)
                except:
                    pass

        try:
            # ==================== 修复点6：正确提取特征 ====================
            if frames.dim() == 3:
                frames = frames.unsqueeze(0)  # [1, T, C, H, W]

            # 分批提取特征
            features_list = []
            batch_size = min(self.config.batch_size, frames.shape[1])

            for i in range(0, frames.shape[1], batch_size):
                end_idx = min(i + batch_size, frames.shape[1])

                if i >= end_idx:
                    break

                # 获取当前批次的帧
                batch_frames = frames[:, i:end_idx]  # [1, batch_size, C, H, W]

                if batch_frames.shape[1] == 0:
                    continue

                # 重塑为ViSiL期望的格式 [batch_size, C, H, W]
                batch_frames = batch_frames.squeeze(0)

                # 确保维度正确
                if batch_frames.dim() == 3:
                    batch_frames = batch_frames.unsqueeze(0)

                batch_frames = batch_frames.to(self.device).float()

                with torch.no_grad():
                    batch_features = self.model.extract_features(batch_frames)

                features_list.append(batch_features.cpu())

            if not features_list:
                print(f"> 未提取到特征: {video_id}")
                return None

            # 合并所有批次的特征
            features = torch.cat(features_list, dim=0)

            # 保存特征
            features_np = features.cpu().numpy()
            np.save(features_file, features_np)

            # 验证保存的文件
            if os.path.exists(features_file):
                saved_features = np.load(features_file)
                print(f"> 保存特征完成: {video_id} (形状: {saved_features.shape})")
            else:
                print(f"> 警告: 特征文件未保存成功: {video_id}")

            return features

        except Exception as e:
            print(f"> 提取特征失败 {video_id}: {e}")
            traceback.print_exc()
            return None

    def process_videos(self, video_ids: list, video_dir: str, pattern: str) -> dict:
        """处理视频列表 - 提取特征"""
        features = {}
        failed_videos = []

        print(f"> 处理视频 ({len(video_ids)}个)")

        for idx, video_id in enumerate(tqdm(video_ids, desc="处理视频")):
            try:
                # 构建视频路径
                video_path = os.path.join(video_dir, pattern.format(id=video_id))

                # 检查文件是否存在
                if not os.path.exists(video_path):
                    # 尝试其他可能的后缀
                    for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
                        alt_path = video_path.rsplit('.', 1)[0] + ext
                        if os.path.exists(alt_path):
                            video_path = alt_path
                            break

                if not os.path.exists(video_path):
                    print(f"> 视频文件不存在: {video_id}")
                    failed_videos.append(video_id)
                    continue

                # 提取帧
                frames = self.extract_frames_from_video(video_path, video_id)
                if frames is None or frames.shape[0] < 4:
                    print(f"> 视频 {video_id} 帧数不足")
                    failed_videos.append(video_id)
                    continue

                # 提取特征
                video_features = self.extract_features_from_frames(frames, video_id)
                if video_features is None:
                    print(f"> 视频 {video_id} 特征提取失败")
                    failed_videos.append(video_id)
                    continue

                features[video_id] = video_features

                # 内存清理
                if (idx + 1) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                print(f"> 处理视频 {video_id} 失败: {str(e)[:200]}")
                failed_videos.append(video_id)

        print(f"> 视频处理完成: 成功 {len(features)}, 失败 {len(failed_videos)}")
        return features, failed_videos

    def calculate_similarities(self, query_features: dict, database_ids: list) -> dict:
        """计算相似度 - 简化版本"""
        print(f"\n" + "=" * 60)
        print("计算相似度")
        print("=" * 60)
        print(f"查询视频数: {len(query_features)}")
        print(f"数据库视频数: {len(database_ids)}")

        similarities = {}
        batch_size = self.config.query_batch_size

        # 分批处理查询视频
        query_ids = list(query_features.keys())

        for batch_start in range(0, len(query_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(query_ids))
            batch_query_ids = query_ids[batch_start:batch_end]

            print(f"> 处理查询批次 {batch_start // batch_size + 1}/{(len(query_ids) + batch_size - 1) // batch_size}")

            for query_id in tqdm(batch_query_ids, desc="计算相似度"):
                if query_id not in query_features:
                    continue

                query_feat = query_features[query_id].to(self.device)
                similarities[query_id] = {}

                # 分批处理数据库视频
                db_batch_size = 100
                for db_start in range(0, len(database_ids), db_batch_size):
                    db_end = min(db_start + db_batch_size, len(database_ids))
                    db_batch_ids = database_ids[db_start:db_end]

                    # 批量加载数据库特征
                    db_batch_features = {}
                    for db_id in db_batch_ids:
                        features_file = os.path.join(self.config.features_dir, f"{db_id}.npy")
                        if os.path.exists(features_file):
                            try:
                                features_np = np.load(features_file)
                                if features_np is not None and features_np.shape[0] > 0:
                                    db_batch_features[db_id] = torch.from_numpy(features_np).float()
                            except:
                                pass

                    # 计算相似度
                    for db_id, db_feat in db_batch_features.items():
                        try:
                            # 确保有批次维度
                            q_feat_batch = query_feat.unsqueeze(0) if query_feat.dim() == 2 else query_feat
                            db_feat_batch = db_feat.unsqueeze(0).to(self.device) if db_feat.dim() == 2 else db_feat.to(
                                self.device)

                            with torch.no_grad():
                                sim = self.model.calculate_video_similarity(q_feat_batch, db_feat_batch)

                            similarities[query_id][db_id] = float(sim.item())
                        except Exception as e:
                            print(f"> 计算相似度失败 {query_id}-{db_id}: {str(e)[:100]}")
                            similarities[query_id][db_id] = 0.0

        return similarities

    def evaluate(self) -> dict:
        """执行完整的评估流程"""
        start_time = time.time()

        print("=" * 60)
        print(f"开始评估: {self.config.dataset}")
        print(f"特征配置: {self.config.feature_config.name}")
        print(f"实验ID: {self.config.experiment_id}")
        print("=" * 60)

        try:
            # 获取视频ID
            query_ids = self.dataset.get_queries()
            database_ids = self.dataset.get_database()

            print(f"> 查询视频数: {len(query_ids)}")
            print(f"> 数据库视频数: {len(database_ids)}")

            # ========== 第一阶段：处理查询视频 ==========
            print("\n" + "=" * 60)
            print("第一阶段：处理查询视频")
            print("=" * 60)

            query_features, query_failed = self.process_videos(
                query_ids,
                self.config.video_dir,
                self.config.video_pattern
            )

            if not query_features:
                raise ValueError("没有成功提取查询特征")

            # ========== 第二阶段：处理数据库视频 ==========
            print("\n" + "=" * 60)
            print("第二阶段：处理数据库视频")
            print("=" * 60)

            # 只处理数据库视频，但不加载到内存
            database_features, database_failed = self.process_videos(
                database_ids,
                self.config.video_dir,
                self.config.video_pattern
            )

            # ========== 第三阶段：计算相似度 ==========
            print("\n" + "=" * 60)
            print("第三阶段：计算相似度")
            print("=" * 60)

            similarities = self.calculate_similarities(query_features, database_ids)

            # ========== 第四阶段：评估结果 ==========
            print("\n" + "=" * 60)
            print("第四阶段：评估结果")
            print("=" * 60)

            evaluation_results = self.dataset.evaluate(similarities)

            # ========== 第五阶段：保存结果 ==========
            print("\n" + "=" * 60)
            print("第五阶段：保存结果")
            print("=" * 60)

            results = self.save_results(similarities, evaluation_results, start_time)

            total_time = time.time() - start_time
            print(f"\n> 评估完成!")
            print(f"> 总耗时: {total_time:.2f} 秒 ({total_time / 60:.1f} 分钟)")

            return results

        except Exception as e:
            print(f"> 评估失败: {e}")
            traceback.print_exc()
            raise

    def save_results(self, similarities: dict, evaluation_results: dict, start_time: float) -> dict:
        """保存结果文件"""
        # 保存完整相似度结果
        result_file = os.path.join(self.config.output_dir, "similarities.json")
        try:
            with open(result_file, 'w') as f:
                json.dump(similarities, f, separators=(',', ':'))
            print(f"> 相似度结果已保存: {result_file}")
        except Exception as e:
            print(f"> 保存相似度结果失败: {e}")

        # 保存排序版本
        sorted_file = os.path.join(self.config.output_dir, "sorted_results.json")
        try:
            sorted_results = {}
            for query_id, query_sims in similarities.items():
                sorted_items = sorted(query_sims.items(), key=lambda x: x[1], reverse=True)
                sorted_results[query_id] = dict(sorted_items[:100])

            with open(sorted_file, 'w') as f:
                json.dump(sorted_results, f, indent=2)
            print(f"> 排序结果已保存: {sorted_file}")
        except Exception as e:
            print(f"> 保存排序结果失败: {e}")

        # 保存评估结果
        eval_file = os.path.join(self.config.output_dir, "evaluation_results.json")
        try:
            with open(eval_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"> 评估结果已保存: {eval_file}")
        except Exception as e:
            print(f"> 保存评估结果失败: {e}")

        total_time = time.time() - start_time

        # 返回汇总结果
        return {
            "config": self.config.feature_config.name,
            "experiment_id": self.config.experiment_id,
            "dataset": self.config.dataset,
            "evaluation": evaluation_results,
            "total_time": total_time,
            "output_dir": self.config.output_dir
        }