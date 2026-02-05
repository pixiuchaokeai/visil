"""
calculate_similarity_immediate.py
立即保存为npy文件的视频相似度计算器
"""

import json
import torch
import numpy as np
import argparse
import gc
import os
import sys
import time
import psutil
from pathlib import Path
from tqdm import tqdm
import shutil

# 导入自定义模块
from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator


class ImmediateSaver:
    """立即保存为npy文件的视频处理类"""

    def __init__(self, args):
        self.args = args

        # 设备选择
        if args.cpu_only or not torch.cuda.is_available():
            self.device = torch.device('cpu')
            args.batch_sz = min(args.batch_sz, 4)
            print("> 使用CPU进行计算")
        else:
            self.device = torch.device(f'cuda:{args.gpu_id}')
            print(f"> 使用GPU设备: {self.device}")

        args.device = self.device

        # 创建输出目录
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # 创建帧存储目录
        self.frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        # 创建特征存储目录
        self.features_dir = os.path.join(self.output_dir, "features")
        os.makedirs(self.features_dir, exist_ok=True)

        print(f"> 输出目录: {self.output_dir}")
        print(f"> 帧存储目录: {self.frames_dir}")
        print(f"> 特征存储目录: {self.features_dir}")

        # 立即显示目录内容
        print(f"> 当前帧目录文件数: {len(os.listdir(self.frames_dir)) if os.path.exists(self.frames_dir) else 0}")
        print(f"> 当前特征目录文件数: {len(os.listdir(self.features_dir)) if os.path.exists(self.features_dir) else 0}")

        # 内存监控
        self.memory_warning_threshold = args.max_memory_gb

    def check_memory(self):
        """检查内存使用情况"""
        try:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 ** 3)  # GB

            if memory_usage > self.memory_warning_threshold:
                print(f"\n⚠️ 内存使用过高: {memory_usage:.2f}GB > {self.memory_warning_threshold}GB")
                return True
        except:
            pass
        return False

    def cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def save_frames_immediate(self, frames, video_id, save_dir):
        """立即将帧保存为npy文件"""
        try:
            # 转换为numpy数组
            if isinstance(frames, torch.Tensor):
                frames_np = frames.cpu().numpy()
            else:
                frames_np = frames

            # 限制最大帧数
            if frames_np.shape[0] > self.args.max_frames:
                # 均匀采样
                step = max(1, frames_np.shape[0] // self.args.max_frames)
                indices = list(range(0, frames_np.shape[0], step))[:self.args.max_frames]
                frames_np = frames_np[indices]

            # 保存为npy文件
            npy_path = os.path.join(save_dir, f"{video_id}.npy")
            np.save(npy_path, frames_np)

            # 立即验证文件
            time.sleep(0.01)  # 等待文件写入
            if not os.path.exists(npy_path):
                print(f"> 错误: 文件未保存成功: {npy_path}")
                return False

            # 检查文件大小
            file_size = os.path.getsize(npy_path) / 1024  # KB
            if file_size < 1:
                print(f"> 警告: 文件过小 ({file_size:.1f}KB): {video_id}")
                # 尝试重新保存
                np.save(npy_path, frames_np)
                file_size = os.path.getsize(npy_path) / 1024
                if file_size < 1:
                    print(f"> 文件仍然过小，可能有问题: {video_id}")

            # 立即打印保存信息
            print(f"> 已保存: {video_id}.npy ({file_size:.1f}KB, {frames_np.shape[0]}帧)")
            return True

        except Exception as e:
            print(f"> 保存npy文件失败 {video_id}: {e}")
            return False

    def extract_and_save_all_frames(self, video_file, is_query=False):
        """提取所有视频帧并立即保存为npy文件"""
        phase_name = "查询" if is_query else "数据库"
        print(f"\n" + "=" * 60)
        print(f"阶段1: 提取{phase_name}视频帧并立即保存")
        print("=" * 60)

        # 读取视频列表
        video_infos = []
        try:
            with open(video_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            video_id = parts[0]
                            video_path = ' '.join(parts[1:])
                            video_infos.append((video_id, video_path))
            print(f"> 从 {video_file} 读取到 {len(video_infos)} 个视频")
        except Exception as e:
            print(f"> 读取视频列表失败: {e}")
            return [], []

        # 如果帧目录不为空，显示已有文件
        if os.path.exists(self.frames_dir):
            existing_files = [f for f in os.listdir(self.frames_dir) if f.endswith('.npy')]
            print(f"> 帧目录已有 {len(existing_files)} 个npy文件")
            if existing_files:
                print(f"> 示例文件: {existing_files[:3]}")

        video_ids = []
        failed_videos = []

        # 直接处理每个视频，不使用DataLoader（避免内存积累）
        for idx, (video_id, video_path) in enumerate(video_infos):
            try:
                # 检查文件是否存在
                if not os.path.exists(video_path):
                    print(f"\n> 视频文件不存在: {video_path}")
                    failed_videos.append((video_id, "文件不存在"))
                    continue

                # 使用VideoGenerator读取视频
                temp_list_file = f"temp_{video_id}.txt"
                with open(temp_list_file, 'w') as f:
                    f.write(f"{video_id} {video_path}")

                try:
                    generator = VideoGenerator(temp_list_file)
                    loader = DataLoader(generator, num_workers=0, batch_size=1)

                    # 获取视频帧
                    for frames_batch, video_id_batch in loader:
                        frames = frames_batch[0]

                        # 检查帧数据
                        if frames is None or frames.nelement() == 0:
                            print(f"\n> 视频无有效帧: {video_id}")
                            failed_videos.append((video_id, "无有效帧"))
                            break

                        if frames.shape[0] < 4:
                            print(f"\n> 帧数不足: {video_id} ({frames.shape[0]}帧)")
                            failed_videos.append((video_id, f"帧数不足 ({frames.shape[0]})"))
                            break

                        # 立即保存为npy文件
                        success = self.save_frames_immediate(frames, video_id, self.frames_dir)
                        if success:
                            video_ids.append(video_id)

                            # 每保存5个文件显示一次进度
                            if len(video_ids) % 5 == 0:
                                print(f"> 进度: 已保存 {len(video_ids)}/{len(video_infos)} 个视频")
                        else:
                            failed_videos.append((video_id, "保存失败"))

                        # 清理内存
                        del frames
                        break  # 只处理第一个批次
                finally:
                    if os.path.exists(temp_list_file):
                        os.remove(temp_list_file)

                # 每处理10个视频检查一次内存
                if idx % 10 == 0:
                    if self.check_memory():
                        print("> 清理内存...")
                        self.cleanup_memory()

            except Exception as e:
                error_msg = str(e)[:100]
                print(f"\n> 处理视频 {video_id} 出错: {error_msg}")
                failed_videos.append((video_id, error_msg))

        # 显示最终结果
        print(f"\n> {phase_name}视频抽帧完成!")
        print(f"成功: {len(video_ids)} 个视频")
        print(f"失败: {len(failed_videos)} 个视频")

        # 立即显示保存的文件数量
        if os.path.exists(self.frames_dir):
            saved_files = [f for f in os.listdir(self.frames_dir) if f.endswith('.npy')]
            print(f"> 帧目录中现有 {len(saved_files)} 个npy文件")
            if saved_files:
                print(f"> 最后保存的5个文件: {saved_files[-5:] if len(saved_files) > 5 else saved_files}")

        return video_ids, failed_videos

    def load_frames_from_npy(self, video_id):
        """从npy文件加载帧"""
        try:
            npy_path = os.path.join(self.frames_dir, f"{video_id}.npy")

            if not os.path.exists(npy_path):
                return None

            # 检查文件大小
            file_size = os.path.getsize(npy_path) / 1024  # KB
            if file_size < 1:
                return None

            # 加载npy文件
            frames_np = np.load(npy_path)

            if frames_np is None or frames_np.size == 0:
                return None

            # 转换为torch tensor
            frames_tensor = torch.from_numpy(frames_np).float()

            # 检查帧数
            if frames_tensor.shape[0] < 4:
                return None

            return frames_tensor

        except Exception as e:
            print(f"> 加载npy文件失败 {video_id}: {e}")
            return None

    def extract_and_save_all_features(self, model, video_ids, video_type="query"):
        """提取所有特征并立即保存为npy文件"""
        phase_name = "查询" if video_type == "query" else "数据库"
        print(f"\n" + "=" * 60)
        print(f"阶段2: 提取{phase_name}视频特征并立即保存")
        print("=" * 60)

        # 如果特征目录不为空，显示已有文件
        if os.path.exists(self.features_dir):
            existing_files = [f for f in os.listdir(self.features_dir) if f.endswith('.npy')]
            print(f"> 特征目录已有 {len(existing_files)} 个npy文件")
            if existing_files:
                print(f"> 示例文件: {existing_files[:3]}")

        features_dict = {}
        failed_videos = []

        print(f"> 开始处理 {len(video_ids)} 个{phase_name}视频")

        for idx, video_id in enumerate(video_ids):
            try:
                # 从npy文件加载帧
                frames = self.load_frames_from_npy(video_id)
                if frames is None:
                    failed_videos.append((video_id, "加载帧失败"))
                    continue

                # 提取特征
                features = self.extract_features(model, frames)
                if features is None:
                    failed_videos.append((video_id, "特征提取失败"))
                    continue

                # 保存特征为npy文件
                success = self.save_features_immediate(features, video_id, self.features_dir)
                if success:
                    features_dict[video_id] = features
                else:
                    failed_videos.append((video_id, "保存特征失败"))

                # 每处理5个视频显示一次进度
                if (idx + 1) % 5 == 0:
                    print(f"> 进度: 已处理 {idx + 1}/{len(video_ids)} 个视频")

                # 每处理10个视频清理一次内存
                if idx % 10 == 0:
                    if self.check_memory():
                        self.cleanup_memory()

                # 释放帧数据
                del frames

            except Exception as e:
                error_msg = str(e)[:100]
                print(f"\n> 提取视频 {video_id} 特征出错: {error_msg}")
                failed_videos.append((video_id, error_msg))

        # 显示最终结果
        print(f"\n> {phase_name}特征提取完成!")
        print(f"成功: {len(features_dict)} 个视频")
        print(f"失败: {len(failed_videos)} 个视频")

        # 立即显示保存的文件数量
        if os.path.exists(self.features_dir):
            saved_files = [f for f in os.listdir(self.features_dir) if f.endswith('.npy')]
            print(f"> 特征目录中现有 {len(saved_files)} 个npy文件")
            if saved_files:
                print(f"> 最后保存的5个文件: {saved_files[-5:] if len(saved_files) > 5 else saved_files}")

        return features_dict, failed_videos

    def extract_features(self, model, frames):
        """提取特征"""
        try:
            batch_sz = self.args.batch_sz

            features_list = []
            for i in range(0, frames.shape[0], batch_sz):
                batch = frames[i:i + batch_sz]
                if batch.shape[0] > 0:
                    batch = batch.to(self.device).float()

                    # 确保正确的维度
                    if len(batch.shape) == 3:  # [B, H, W]
                        batch = batch.unsqueeze(1)  # 添加通道维度

                    with torch.no_grad():
                        batch_features = model.extract_features(batch)

                    features_list.append(batch_features.cpu())

                    # 清理
                    del batch

            if features_list:
                features = torch.cat(features_list, dim=0)
                return features
            else:
                return None

        except Exception as e:
            print(f"> 特征提取失败: {e}")
            return None

    def save_features_immediate(self, features, video_id, save_dir):
        """立即将特征保存为npy文件"""
        try:
            # 转换为numpy数组
            if isinstance(features, torch.Tensor):
                features_np = features.cpu().numpy()
            else:
                features_np = features

            # 保存为npy文件
            npy_path = os.path.join(save_dir, f"{video_id}.npy")
            np.save(npy_path, features_np)

            # 立即验证文件
            time.sleep(0.01)  # 等待文件写入
            if not os.path.exists(npy_path):
                print(f"> 错误: 特征文件未保存成功: {npy_path}")
                return False

            # 检查文件大小
            file_size = os.path.getsize(npy_path) / 1024  # KB
            if file_size < 1:
                print(f"> 警告: 特征文件过小 ({file_size:.1f}KB): {video_id}")
                # 尝试重新保存
                np.save(npy_path, features_np)

            # 立即打印保存信息
            print(f"> 已保存特征: {video_id}.npy ({file_size:.1f}KB, 形状: {features_np.shape})")
            return True

        except Exception as e:
            print(f"> 保存特征npy文件失败 {video_id}: {e}")
            return False

    def calculate_similarities(self, model, query_features, database_features):
        """计算相似度"""
        print(f"\n" + "=" * 60)
        print(f"阶段3: 计算相似度")
        print("=" * 60)

        if not query_features or not database_features:
            print("> 错误: 没有有效的特征数据")
            return {}

        query_ids = list(query_features.keys())
        database_ids = list(database_features.keys())

        print(f"> 查询视频数: {len(query_ids)}")
        print(f"> 数据库视频数: {len(database_ids)}")

        similarities = {}

        # 分批处理查询视频
        query_batch_size = min(self.args.query_batch_size, len(query_ids))

        for batch_start in range(0, len(query_ids), query_batch_size):
            batch_end = min(batch_start + query_batch_size, len(query_ids))
            batch_query_ids = query_ids[batch_start:batch_end]

            batch_num = batch_start // query_batch_size + 1
            total_batches = (len(query_ids) + query_batch_size - 1) // query_batch_size

            print(f"\n> 处理查询批次 {batch_num}/{total_batches}")

            # 准备当前批次的查询特征
            batch_queries = []
            valid_query_ids = []

            for qid in batch_query_ids:
                if qid in query_features:
                    batch_queries.append(query_features[qid].to(self.device))
                    valid_query_ids.append(qid)
                else:
                    # 尝试从磁盘加载
                    features = self.load_features_from_npy(qid)
                    if features is not None:
                        batch_queries.append(features.to(self.device))
                        valid_query_ids.append(qid)
                    else:
                        print(f"> 跳过查询 {qid}: 特征加载失败")

            if not batch_queries:
                print("> 当前批次无有效查询特征，跳过")
                continue

            # 处理数据库视频
            for db_id in database_ids:
                try:
                    # 获取数据库视频特征
                    if db_id in database_features:
                        db_feat = database_features[db_id]
                    else:
                        # 尝试从磁盘加载
                        features = self.load_features_from_npy(db_id)
                        if features is None:
                            continue
                        db_feat = features

                    # 移动到设备
                    db_feat = db_feat.to(self.device)

                    # 计算相似度
                    for i, query_feat in enumerate(batch_queries):
                        query_id = valid_query_ids[i]

                        if query_id not in similarities:
                            similarities[query_id] = {}

                        with torch.no_grad():
                            query_feat_expanded = query_feat.unsqueeze(0)
                            db_feat_expanded = db_feat.unsqueeze(0)
                            sim_score = model.calculate_video_similarity(
                                query_feat_expanded,
                                db_feat_expanded
                            )

                        similarities[query_id][db_id] = float(sim_score.item())

                    # 每处理50个数据库视频显示一次进度
                    if len(similarities.get(valid_query_ids[0], {})) % 50 == 0:
                        print(f"> 进度: 已处理 {len(similarities[valid_query_ids[0]])}/{len(database_ids)} 个数据库视频")

                    # 释放数据库特征
                    del db_feat

                except Exception as e:
                    print(f"\n> 计算视频 {db_id} 相似度出错: {str(e)[:100]}")

            # 清理当前批次的查询特征
            for qfeat in batch_queries:
                del qfeat
            self.cleanup_memory()

            # 保存检查点
            if batch_num % 2 == 0 or batch_num == total_batches:
                self.save_checkpoint(similarities, batch_num)

        return similarities

    def load_features_from_npy(self, video_id):
        """从npy文件加载特征"""
        try:
            npy_path = os.path.join(self.features_dir, f"{video_id}.npy")

            if not os.path.exists(npy_path):
                return None

            # 检查文件大小
            file_size = os.path.getsize(npy_path) / 1024  # KB
            if file_size < 1:
                return None

            # 加载npy文件
            features_np = np.load(npy_path)

            if features_np is None or features_np.size == 0:
                return None

            # 转换为torch tensor
            features_tensor = torch.from_numpy(features_np).float()
            return features_tensor

        except Exception as e:
            print(f"> 加载特征npy文件失败 {video_id}: {e}")
            return None

    def save_checkpoint(self, similarities, batch_num):
        """保存检查点"""
        try:
            checkpoint_file = os.path.join(self.output_dir, f"checkpoint_{batch_num:03d}.json")

            # 保存部分结果
            checkpoint_data = {}
            query_keys = list(similarities.keys())

            # 保存最近处理的查询
            for qid in query_keys[-10:]:
                checkpoint_data[qid] = {}
                db_keys = list(similarities[qid].keys())
                for dbid in db_keys[:50]:
                    checkpoint_data[qid][dbid] = similarities[qid][dbid]

            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            print(f"> 检查点已保存: {checkpoint_file}")
        except Exception as e:
            print(f"> 保存检查点失败: {e}")

    def run(self):
        """运行完整的计算流程"""
        print("=" * 80)
        print("ViSiL 视频相似度计算 (立即保存npy版)")
        print("=" * 80)

        start_time = time.time()

        try:
            # 初始化模型
            print("\n> 初始化ViSiL模型...")
            model = ViSiL(
                pretrained=True,
                symmetric=('symmetric' in self.args.similarity_function)
            ).to(self.device)
            model.eval()

            # ==================== 第一阶段：抽帧 ====================
            print("\n" + "=" * 60)
            print("第一阶段：抽帧（立即保存为npy）")
            print("=" * 60)

            # 1.1 抽取查询视频帧
            query_ids, query_failed = self.extract_and_save_all_frames(
                self.args.query_file, is_query=True
            )

            if not query_ids:
                print("> 错误: 没有成功提取查询视频帧")
                return

            # 1.2 抽取数据库视频帧
            database_ids, database_failed = self.extract_and_save_all_frames(
                self.args.database_file, is_query=False
            )

            print(f"\n> 抽帧阶段完成!")
            print(f"查询视频: {len(query_ids)} 个成功，{len(query_failed)} 个失败")
            print(f"数据库视频: {len(database_ids)} 个成功，{len(database_failed)} 个失败")

            # 立即显示目录内容
            if os.path.exists(self.frames_dir):
                frame_files = [f for f in os.listdir(self.frames_dir) if f.endswith('.npy')]
                print(f"> 帧目录现有 {len(frame_files)} 个npy文件")

            # ==================== 第二阶段：提取特征 ====================
            print("\n" + "=" * 60)
            print("第二阶段：提取特征（立即保存为npy）")
            print("=" * 60)

            # 2.1 提取查询视频特征
            query_features, query_failed_features = self.extract_and_save_all_features(
                model, query_ids, video_type="query"
            )

            if not query_features:
                print("> 错误: 没有成功提取查询视频特征")
                return

            # 2.2 提取数据库视频特征
            database_features, database_failed_features = self.extract_and_save_all_features(
                model, database_ids, video_type="database"
            )

            print(f"\n> 特征提取阶段完成!")
            print(f"查询视频特征: {len(query_features)} 个成功")
            print(f"数据库视频特征: {len(database_features)} 个成功")

            # 立即显示目录内容
            if os.path.exists(self.features_dir):
                feature_files = [f for f in os.listdir(self.features_dir) if f.endswith('.npy')]
                print(f"> 特征目录现有 {len(feature_files)} 个npy文件")

            # ==================== 第三阶段：计算相似度 ====================
            print("\n" + "=" * 60)
            print("第三阶段：计算相似度")
            print("=" * 60)

            similarities = self.calculate_similarities(
                model, query_features, database_features
            )

            # ==================== 第四阶段：保存结果 ====================
            print("\n" + "=" * 60)
            print("第四阶段：保存结果")
            print("=" * 60)

            self.save_final_results(
                similarities,
                query_ids, query_failed,
                database_ids, database_failed,
                start_time
            )

            total_time = time.time() - start_time
            print(f"\n> 所有阶段完成!")
            print(f"> 总耗时: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")

            # 显示最终的文件统计
            if os.path.exists(self.frames_dir):
                frame_files = [f for f in os.listdir(self.frames_dir) if f.endswith('.npy')]
                print(f"> 帧目录最终文件数: {len(frame_files)}")

            if os.path.exists(self.features_dir):
                feature_files = [f for f in os.listdir(self.features_dir) if f.endswith('.npy')]
                print(f"> 特征目录最终文件数: {len(feature_files)}")

        except KeyboardInterrupt:
            print("\n> 用户中断")
        except Exception as e:
            print(f"\n> 发生错误: {e}")
            import traceback
            traceback.print_exc()

    def save_final_results(self, similarities, query_ids, query_failed,
                          database_ids, database_failed, start_time):
        """保存最终结果"""
        # 保存完整结果
        result_file = os.path.join(self.output_dir, "similarities.json")
        with open(result_file, 'w') as f:
            json.dump(similarities, f, separators=(',', ':'))

        # 保存排序结果
        sorted_file = os.path.join(self.output_dir, "sorted_results.json")
        sorted_results = {}
        for query_id in similarities:
            sorted_items = sorted(similarities[query_id].items(),
                                key=lambda x: x[1], reverse=True)
            sorted_results[query_id] = dict(sorted_items[:100])

        with open(sorted_file, 'w') as f:
            json.dump(sorted_results, f, indent=2)

        # 保存统计信息
        stats_file = os.path.join(self.output_dir, "stats.txt")
        total_time = time.time() - start_time

        with open(stats_file, 'w') as f:
            f.write("ViSiL 视频相似度计算结果统计\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"计算时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总耗时: {total_time:.2f} 秒\n\n")
            f.write(f"查询视频总数: {len(query_ids) + len(query_failed)}\n")
            f.write(f"成功提取查询视频: {len(query_ids)}\n")
            f.write(f"失败的查询视频: {len(query_failed)}\n\n")
            f.write(f"数据库视频总数: {len(database_ids) + len(database_failed)}\n")
            f.write(f"成功提取数据库视频: {len(database_ids)}\n")
            f.write(f"失败的据库视频: {len(database_failed)}\n\n")
            f.write(f"帧数据目录: {self.frames_dir} (npy格式)\n")
            f.write(f"特征数据目录: {self.features_dir} (npy格式)\n")
            f.write(f"相似度结果: {result_file}\n")
            f.write(f"排序结果: {sorted_file}\n")
            f.write(f"文件格式: 所有中间数据都保存为NumPy (.npy) 格式\n")

        print(f"> 相似度结果已保存: {result_file}")
        print(f"> 排序结果已保存: {sorted_file}")
        print(f"> 统计信息已保存: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='立即保存为npy文件的视频相似度计算器')

    # 基本参数
    parser.add_argument('--query_file', type=str,
                        default='datasets/fivr-5k-queries-filtered.txt',
                        help='查询视频列表文件')
    parser.add_argument('--database_file', type=str,
                        default='datasets/fivr-5k-database-filtered.txt',
                        help='数据库视频列表文件')

    # 帧处理参数
    parser.add_argument('--max_frames', type=int, default=50,
                        help='每个视频最大帧数，默认：50')
    parser.add_argument('--batch_sz', type=int, default=4,
                        help='特征提取批次大小，默认：4')

    # 批处理参数
    parser.add_argument('--query_batch_size', type=int, default=10,
                        help='查询视频批处理大小，默认：10')

    # 设备参数
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU设备ID，默认：0')
    parser.add_argument('--cpu_only', action='store_true',
                        help='强制使用CPU')

    # 内存参数
    parser.add_argument('--max_memory_gb', type=float, default=8.0,
                        help='内存警告阈值(GB)，默认：8.0GB')

    # 模型参数
    parser.add_argument('--similarity_function', type=str, default='symmetric_chamfer',
                        choices=['chamfer', 'symmetric_chamfer'],
                        help='相似度函数，默认：symmetric_chamfer')

    args = parser.parse_args()

    # 运行计算器
    calculator = ImmediateSaver(args)
    calculator.run()


if __name__ == '__main__':
    main()