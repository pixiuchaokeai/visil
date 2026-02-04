"""
optimized_calculate_similarity.py
内存优化的视频相似度计算
"""

import json
import torch
import argparse
import gc
import os
import sys

from tqdm import tqdm
from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator
from evaluation import extract_features, calculate_similarities_to_queries


class MemoryOptimizedSimilarityCalculator:
    """内存优化的相似度计算器"""

    def __init__(self, args):
        self.args = args

        # 设备选择
        if args.cpu_only or not torch.cuda.is_available():
            self.device = torch.device('cpu')
            args.batch_sz = min(args.batch_sz, 16)  # CPU使用更小批次
            args.batch_sz_sim = min(args.batch_sz_sim, 256)
            print("> 使用CPU进行计算（内存优化模式）")
        else:
            self.device = torch.device(f'cuda:{args.gpu_id}')
            print(f"> 使用GPU设备: {self.device}")

        args.device = self.device

        # 内存监控
        self.max_memory_usage = 0
        self.check_memory_interval = 10

    def check_memory_usage(self):
        """检查内存使用情况"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 ** 3)  # GB

        if memory_usage > self.max_memory_usage:
            self.max_memory_usage = memory_usage

        if memory_usage > self.args.max_cpu_memory_gb:
            print(f"\n> 警告: 内存使用超过阈值 ({memory_usage:.2f}GB > {self.args.max_cpu_memory_gb}GB)")
            print("> 正在清理内存...")
            self.cleanup_memory()

        return memory_usage

    def cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("> 内存清理完成")

    def process_queries(self, model):
        """处理查询视频（内存优化版）"""
        print("\n" + "=" * 60)
        print("阶段1: 处理查询视频（内存优化）")
        print("=" * 60)

        generator = VideoGenerator(self.args.query_file)
        loader = DataLoader(generator, num_workers=0, batch_size=1)

        queries, queries_ids = [], []
        failed_queries = []

        pbar = tqdm(loader, desc="处理查询视频")
        for idx, video in enumerate(pbar):
            frames = video[0][0]
            video_id = video[1][0]

            # 限制最大帧数，减少内存使用
            max_frames = 100  # 限制每视频最多100帧
            if frames.shape[0] > max_frames:
                print(f"\n  > 视频 {video_id} 帧数过多 ({frames.shape[0]} > {max_frames})，随机采样")
                indices = torch.randperm(frames.shape[0])[:max_frames]
                frames = frames[indices]

            if frames.shape[0] < 4:
                print(f"\n  > 跳过查询视频 {video_id}: 帧数不足 ({frames.shape[0]})")
                failed_queries.append(video_id)
                continue

            try:
                # 分批处理，避免内存峰值
                features = self.extract_features_with_memory_limit(model, frames)
                features = features.cpu()  # 立即转移到CPU
                queries.append(features)
                queries_ids.append(video_id)

                pbar.set_postfix(query=video_id, 内存=f"{self.check_memory_usage():.1f}GB")

            except Exception as e:
                print(f"\n  > 处理查询视频 {video_id} 出错: {str(e)[:100]}")
                failed_queries.append(video_id)

            # 定期清理
            if idx % self.check_memory_interval == 0:
                self.cleanup_memory()

        print(f"\n> 成功处理 {len(queries)} 个查询视频")
        if failed_queries:
            print(f"> 失败 {len(failed_queries)} 个: {failed_queries[:5]}...")

        if not queries:
            print("> 错误: 没有成功处理的查询视频")
            sys.exit(1)

        return queries, queries_ids, failed_queries

    def extract_features_with_memory_limit(self, model, frames):
        """内存限制的特征提取"""
        batch_sz = self.args.batch_sz

        # 动态调整批次大小
        if frames.shape[0] > 100:
            batch_sz = max(8, batch_sz // 2)

        features = []
        for i in range(0, frames.shape[0], batch_sz):
            batch = frames[i:i + batch_sz]
            if batch.shape[0] > 0:
                batch = batch.to(self.device).float()
                batch_features = model.extract_features(batch)
                features.append(batch_features.cpu())  # 立即转移到CPU

                # 每批后检查内存
                if i % (batch_sz * 5) == 0:
                    self.check_memory_usage()

        if features:
            return torch.cat(features, 0)
        else:
            return torch.randn(4, 9, 512)

    def process_database(self, model, queries, queries_ids):
        """处理数据库视频（内存优化版）"""
        print("\n" + "=" * 60)
        print("阶段2: 处理数据库视频（内存优化）")
        print("=" * 60)

        generator = VideoGenerator(self.args.database_file)
        loader = DataLoader(generator, num_workers=0, batch_size=1)

        similarities = {query_id: {} for query_id in queries_ids}
        failed_database = []
        processed_count = 0

        # 分批处理查询特征，避免一次性加载所有查询
        query_batch_size = 10  # 每次处理10个查询
        total_queries = len(queries)

        pbar = tqdm(loader, desc="处理数据库视频")
        for idx, video in enumerate(pbar):
            frames = video[0][0]
            video_id = video[1][0]

            # 限制最大帧数
            max_frames = 100
            if frames.shape[0] > max_frames:
                indices = torch.randperm(frames.shape[0])[:max_frames]
                frames = frames[indices]

            if frames.shape[0] < 4:
                failed_database.append(video_id)
                continue

            try:
                # 分批提取特征
                target_features = self.extract_features_with_memory_limit(model, frames)
                target_features = target_features.cpu()

                # 分批计算相似度
                for q_start in range(0, total_queries, query_batch_size):
                    q_end = min(q_start + query_batch_size, total_queries)

                    # 获取当前批次的查询特征
                    current_queries = []
                    for q_idx in range(q_start, q_end):
                        # 确保查询特征在CPU上
                        if queries[q_idx].device != torch.device('cpu'):
                            current_queries.append(queries[q_idx].cpu())
                        else:
                            current_queries.append(queries[q_idx])

                    # 计算相似度
                    sims = self.calculate_similarities_memory_optimized(
                        model, current_queries, target_features
                    )

                    # 存储结果
                    for i, s in enumerate(sims):
                        similarities[queries_ids[q_start + i]][video_id] = float(s)

                processed_count += 1
                pbar.set_postfix(
                    video=video_id[:10],
                    已处理=processed_count,
                    内存=f"{self.check_memory_usage():.1f}GB"
                )

            except Exception as e:
                print(f"\n  > 处理数据库视频 {video_id} 出错: {str(e)[:100]}")
                failed_database.append(video_id)

            # 定期清理和保存检查点
            if idx % self.check_memory_interval == 0:
                self.cleanup_memory()

            if processed_count % 100 == 0:
                self.save_checkpoint(similarities, processed_count)
                # 重新打开文件以释放内存
                del target_features
                self.cleanup_memory()

        return similarities, failed_database, processed_count

    def calculate_similarities_memory_optimized(self, model, queries, target):
        """内存优化的相似度计算"""
        similarities = []

        # 更小的批次大小用于相似度计算
        sim_batch_size = min(128, self.args.batch_sz_sim // 4)

        for query in queries:
            query = query.to(self.device)

            sim_values = []
            for b in range(0, target.shape[0], sim_batch_size):
                batch = target[b:b + sim_batch_size]
                if batch.shape[0] >= 4:
                    batch = batch.to(self.device)
                    batch_sim = model.calculate_video_similarity(query.unsqueeze(0), batch.unsqueeze(0))
                    sim_values.append(batch_sim.cpu())
                    del batch
                    self.cleanup_memory()

            if sim_values:
                sim_tensor = torch.stack(sim_values, 0)
                sim_value = torch.mean(sim_tensor)
                similarities.append(sim_value.detach().cpu().numpy())
            else:
                similarities.append(0.0)

            del query
            self.cleanup_memory()

        return similarities

    def save_checkpoint(self, similarities, count):
        """保存检查点"""
        checkpoint_file = f"{self.args.output_file}.checkpoint_{count}.json"
        with open(checkpoint_file, 'w') as f:
            # 只保存部分结果以减少文件大小
            checkpoint_data = {}
            for query_id in list(similarities.keys())[:10]:  # 只保存前10个查询的结果
                checkpoint_data[query_id] = similarities[query_id]
            json.dump(checkpoint_data, f, indent=1)

        print(f"\n> 检查点已保存: {checkpoint_file}")

    def run(self):
        """运行完整的相似度计算流程"""
        print("=" * 80)
        print("ViSiL 视频相似度计算（内存优化版）")
        print("=" * 80)

        # 初始化模型
        print("\n> 初始化ViSiL模型...")
        model = ViSiL(
            pretrained=True,
            symmetric=('symmetric' in self.args.similarity_function)
        ).to(self.device)
        model.eval()

        # 阶段1: 处理查询视频
        queries, queries_ids, failed_queries = self.process_queries(model)

        # 阶段2: 处理数据库视频
        similarities, failed_database, processed_count = self.process_database(
            model, queries, queries_ids
        )

        # 阶段3: 保存最终结果
        self.save_final_results(similarities, failed_queries, failed_database, processed_count)

        print(f"\n> 最大内存使用: {self.max_memory_usage:.2f}GB")
        print("> 完成！")

    def save_final_results(self, similarities, failed_queries, failed_database, processed_count):
        """保存最终结果"""
        print("\n" + "=" * 60)
        print("阶段3: 保存最终结果")
        print("=" * 60)

        # 保存主结果（压缩格式以节省磁盘空间）
        with open(self.args.output_file, 'w') as f:
            json.dump(similarities, f, separators=(',', ':'))  # 紧凑格式

        # 保存统计信息
        stats_file = f"{self.args.output_file}.stats.txt"
        with open(stats_file, 'w') as f:
            f.write(f"ViSiL 相似度计算结果统计\n")
            f.write(f"查询视频总数: {len(queries_ids)}\n")
            f.write(f"失败的查询视频: {len(failed_queries)}\n")
            f.write(f"处理的数据库视频: {processed_count}\n")
            f.write(f"失败的数据库视频: {len(failed_database)}\n")
            f.write(f"最大内存使用: {self.max_memory_usage:.2f}GB\n")

        print(f"> 结果已保存到: {self.args.output_file}")
        print(f"> 统计信息已保存到: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='内存优化的视频相似度计算')

    # 基本参数
    parser.add_argument('--query_file', type=str, required=True,
                        help='查询视频列表文件路径')
    parser.add_argument('--database_file', type=str, required=True,
                        help='数据库视频列表文件路径')
    parser.add_argument('--output_file', type=str, default='results.json',
                        help='输出文件路径')

    # 批处理参数
    parser.add_argument('--batch_sz', type=int, default=16,
                        help='特征提取批次大小（建议: CPU=16, GPU=32）')
    parser.add_argument('--batch_sz_sim', type=int, default=256,
                        help='相似度计算批次大小')

    # 设备参数
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU设备ID')
    parser.add_argument('--cpu_only', action='store_true',
                        help='强制使用CPU')

    # 模型参数
    parser.add_argument('--similarity_function', type=str, default='symmetric_chamfer',
                        choices=['chamfer', 'symmetric_chamfer'],
                        help='相似度函数类型')

    # 内存管理参数
    parser.add_argument('--max_cpu_memory_gb', type=float, default=12.0,
                        help='最大CPU内存使用（GB）')

    args = parser.parse_args()

    # 运行内存优化的计算器
    calculator = MemoryOptimizedSimilarityCalculator(args)
    calculator.run()


if __name__ == '__main__':
    main()