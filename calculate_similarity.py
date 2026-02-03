import json
import torch
import argparse
import gc
import psutil  # 直接导入，假设已安装
import os

from tqdm import tqdm
from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator
from evaluation import extract_features, calculate_similarities_to_queries


def check_memory_usage():
    """检查内存使用情况"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = total_memory - allocated_memory
        return free_memory, total_memory
    return None, None


def adaptive_batch_size(base_batch_sz, model_size_estimate=0):
    """
    根据可用内存自适应调整批次大小

    参数:
        base_batch_sz: 基础批次大小
        model_size_estimate: 模型大小估计（MB）
    """
    if torch.cuda.is_available():
        free_memory, total_memory = check_memory_usage()
        if free_memory:
            # 保留10%的内存作为安全余量
            available_memory = free_memory * 0.9

            # 估计每帧特征所需内存（经验值）
            memory_per_frame = 1.0 * 1024 * 1024  # 1MB per frame

            # 计算最大批次大小
            max_batch = int(available_memory / memory_per_frame)

            # 如果模型很大，进一步减少批次大小
            if model_size_estimate > 0:
                max_batch = max(1, int(max_batch * (1 - model_size_estimate / 1000)))

            return min(base_batch_sz, max(1, max_batch))

    # CPU环境下使用较小的批次大小
    return min(base_batch_sz, 32)


if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='基于ViSiL网络的视频相似度计算代码，支持CPU/GPU混合环境。',
                                     formatter_class=formatter)

    # ===== 输入输出文件参数 =====
    parser.add_argument('--query_file', type=str, required=True,
                        help='包含查询视频列表的文件路径（文本格式，每行：视频ID 视频路径）')
    parser.add_argument('--database_file', type=str, required=True,
                        help='包含数据库视频列表的文件路径（文本格式，每行：视频ID 视频路径）')
    parser.add_argument('--output_file', type=str, default='results.json',
                        help='输出文件名，保存查询视频与数据库视频的相似度结果（JSON格式）')

    # ===== 批处理大小参数 =====
    parser.add_argument('--batch_sz', type=int, default=128,
                        help='特征提取时每批包含的帧数。默认：128（影响GPU显存占用）')
    parser.add_argument('--batch_sz_sim', type=int, default=2048,
                        help='相似度计算时每批包含的特征张量数。默认：2048（影响计算效率）')

    # ===== 内存优化参数 =====
    parser.add_argument('--adaptive_batch', action='store_true',
                        help='启用自适应批次大小调整，根据可用内存自动调整批次大小')
    parser.add_argument('--cpu_only', action='store_true',
                        help='强制使用CPU，即使GPU可用')
    parser.add_argument('--max_cpu_memory_gb', type=float, default=16.0,
                        help='CPU模式下允许使用的最大内存（GB），超过时会清空缓存')

    # ===== 计算设备参数 =====
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='使用的GPU设备ID。默认：0（使用第一块GPU）')

    # ===== 内存优化参数 =====
    parser.add_argument('--load_queries', action='store_true',
                        help='指示是否将查询视频特征加载到GPU内存中的标志。'
                             '如果查询视频数量多，特征大会占用大量显存')

    # ===== 相似度计算参数 =====
    parser.add_argument('--similarity_function', type=str, default='chamfer',
                        choices=["chamfer", "symmetric_chamfer"],
                        help='用于计算查询-目标帧和视频之间相似度的函数。'
                             'chamfer: 单向Chamfer距离；symmetric_chamfer: 对称Chamfer距离')

    # ===== 数据加载参数 =====
    parser.add_argument('--workers', type=int, default=8,
                        help='视频加载时使用的工作进程数。默认：8（多进程加速数据加载）')

    # ===== 性能优化参数 =====
    parser.add_argument('--enable_garbage_collection', action='store_true',
                        help='启用更频繁的垃圾回收，减少内存占用')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='每处理N个视频后保存检查点，防止程序中断')

    # ===== 视频处理参数 =====
    parser.add_argument('--skip_failed_videos', action='store_true',
                        help='跳过无法读取的视频文件')
    parser.add_argument('--min_frames', type=int, default=4,
                        help='视频最少需要多少帧才进行处理，默认4帧')

    args = parser.parse_args()

    # 设备选择
    if args.cpu_only or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("> 使用CPU进行计算")
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"> 使用GPU设备: {device}")

    # 自适应批次大小调整
    if args.adaptive_batch:
        args.batch_sz = adaptive_batch_size(args.batch_sz)
        args.batch_sz_sim = adaptive_batch_size(args.batch_sz_sim // 16) * 16  # 保持16的倍数
        print(f"> 自适应批次大小: batch_sz={args.batch_sz}, batch_sz_sim={args.batch_sz_sim}")

    # 将device添加到args，便于后续函数使用
    args.device = device

    # 初始化ViSiL模型
    model = ViSiL(pretrained=True, symmetric='symmetric' in args.similarity_function).to(device)
    model.eval()

    # 估计模型大小
    model_params = sum(p.numel() for p in model.parameters())
    model_size_mb = model_params * 4 / (1024 * 1024)  # 假设float32
    print(f"> 模型参数数量: {model_params:,}，估计大小: {model_size_mb:.2f} MB")

    # ===== 第一阶段：提取查询视频特征 =====
    print('> 开始提取查询视频的特征')
    generator = VideoGenerator(args.query_file)
    loader = DataLoader(generator, num_workers=args.workers)

    queries, queries_ids = [], []
    failed_queries = []
    pbar = tqdm(loader, desc='正在处理的查询视频')

    for idx, video in enumerate(pbar):
        frames = video[0][0]
        video_id = video[1][0]

        # 检查视频是否有足够的帧
        if frames.shape[0] < args.min_frames:
            print(f"\n警告: 查询视频 {video_id} 帧数不足 ({frames.shape[0]} < {args.min_frames})，跳过")
            failed_queries.append(video_id)
            if not args.skip_failed_videos:
                # 如果设置了不跳过失败视频，创建占位符特征
                placeholder_features = torch.zeros((4, 512), device=device)
                if not args.load_queries or device.type == 'cpu':
                    placeholder_features = placeholder_features.cpu()
                queries.append(placeholder_features)
                queries_ids.append(video_id)
            continue

        # CPU环境下的内存管理
        if device.type == 'cpu':
            process = psutil.Process()
            memory_usage_gb = process.memory_info().rss / (1024 ** 3)

            if memory_usage_gb > args.max_cpu_memory_gb:
                print(f"\n> 内存使用超过阈值 ({memory_usage_gb:.2f} GB > {args.max_cpu_memory_gb} GB)")
                print("> 清理缓存并回收内存...")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()

        try:
            features = extract_features(model, frames, args)

            # 根据参数决定特征存储位置
            if not args.load_queries or device.type == 'cpu':
                features = features.cpu()

            queries.append(features)
            queries_ids.append(video_id)

            # 更新进度条
            mem_usage_gb = process.memory_info().rss / (1024 ** 3) if device.type == 'cpu' else 0
            pbar.set_postfix(query_id=video_id, mem=f"{mem_usage_gb:.1f}GB" if device.type == 'cpu' else "")

        except Exception as e:
            print(f"\n错误: 处理查询视频 {video_id} 时出错: {e}")
            failed_queries.append(video_id)
            if not args.skip_failed_videos:
                # 创建占位符特征
                placeholder_features = torch.zeros((4, 512), device=device)
                if not args.load_queries or device.type == 'cpu':
                    placeholder_features = placeholder_features.cpu()
                queries.append(placeholder_features)
                queries_ids.append(video_id)
            continue

        # 定期垃圾回收
        if args.enable_garbage_collection and idx % 10 == 0:
            gc.collect()

    # 报告失败的查询视频
    if failed_queries:
        print(f"\n> 警告: {len(failed_queries)} 个查询视频处理失败:")
        for vid in failed_queries[:10]:  # 只显示前10个
            print(f"  - {vid}")
        if len(failed_queries) > 10:
            print(f"  ... 还有 {len(failed_queries) - 10} 个")

    print(f"> 成功提取 {len(queries)} 个查询视频的特征")

    # ===== 第二阶段：计算与数据库视频的相似度 =====
    print('\n> 开始计算查询-目标视频相似度')
    generator = VideoGenerator(args.database_file)
    loader = DataLoader(generator, num_workers=args.workers)

    similarities = dict({query: dict() for query in queries_ids})

    pbar = tqdm(loader, desc='正在处理的数据库视频')
    processed_count = 0
    failed_database = []

    for idx, video in enumerate(pbar):
        frames = video[0][0]
        video_id = video[1][0]

        if frames.shape[0] < args.min_frames:
            print(f"\n警告: 数据库视频 {video_id} 帧数不足 ({frames.shape[0]} < {args.min_frames})，跳过")
            failed_database.append(video_id)
            continue

        try:
            features = extract_features(model, frames, args)
            sims = calculate_similarities_to_queries(model, queries, features, args)

            for i, s in enumerate(sims):
                similarities[queries_ids[i]][video_id] = float(s)

            processed_count += 1

            # 更新进度条
            if device.type == 'cpu':
                mem_usage_gb = psutil.Process().memory_info().rss / (1024 ** 3)
                pbar.set_postfix(video_id=video_id, mem=f"{mem_usage_gb:.1f}GB")
            else:
                pbar.set_postfix(video_id=video_id)

        except Exception as e:
            print(f"\n错误: 处理数据库视频 {video_id} 时出错: {e}")
            failed_database.append(video_id)
            continue

        # 检查点保存
        if args.checkpoint_interval > 0 and (idx + 1) % args.checkpoint_interval == 0:
            checkpoint_file = f"{args.output_file}.checkpoint_{idx + 1}"
            with open(checkpoint_file, 'w') as f:
                json.dump(similarities, f, indent=1)
            print(f"> 检查点已保存: {checkpoint_file}")

        # 定期垃圾回收
        if args.enable_garbage_collection and idx % 10 == 0:
            gc.collect()

    # 报告失败的数据库视频
    if failed_database:
        print(f"\n> 警告: {len(failed_database)} 个数据库视频处理失败:")
        for vid in failed_database[:10]:  # 只显示前10个
            print(f"  - {vid}")
        if len(failed_database) > 10:
            print(f"  ... 还有 {len(failed_database) - 10} 个")

    print(f"> 成功处理 {processed_count} 个数据库视频")

    # ===== 第三阶段：保存结果 =====
    print('> 保存相似度结果到JSON文件')
    with open(args.output_file, 'w') as f:
        json.dump(similarities, f, indent=1)

    print(f'> 完成！结果已保存到 {args.output_file}')

    # 保存处理日志
    log_file = f"{args.output_file}.log"
    with open(log_file, 'w') as f:
        f.write(f"查询视频总数: {len(queries_ids)}\n")
        f.write(f"失败的查询视频: {len(failed_queries)}\n")
        if failed_queries:
            f.write("失败的查询视频ID:\n")
            for vid in failed_queries:
                f.write(f"  {vid}\n")

        f.write(f"\n数据库视频总数: {processed_count + len(failed_database)}\n")
        f.write(f"成功处理的数据库视频: {processed_count}\n")
        f.write(f"失败的数据库视频: {len(failed_database)}\n")
        if failed_database:
            f.write("失败的数据库视频ID:\n")
            for vid in failed_database:
                f.write(f"  {vid}\n")

    print(f'> 处理日志已保存到 {log_file}')

    # 最终内存清理
    if args.enable_garbage_collection:
        del queries, similarities
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None