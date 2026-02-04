import json
import torch
import argparse
import warnings
import gc
import psutil

from tqdm import tqdm
from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator
from evaluation import extract_features, calculate_similarities_to_queries

if __name__ == '__main__':

    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(
        description='This is the code for video similarity calculation based on ViSiL network.',
        formatter_class=formatter)
    parser.add_argument('--query_file', type=str, required=True,
                        help='Path to file that contains the query videos')
    parser.add_argument('--database_file', type=str, required=True,
                        help='Path to file that contains the database videos')
    parser.add_argument('--output_file', type=str, default='results.json',
                        help='Name of the output file.')
    parser.add_argument('--batch_sz', type=int, default=128,
                        help='Number of frames contained in each batch during feature extraction. Default: 128')
    parser.add_argument('--batch_sz_sim', type=int, default=2048,
                        help='Number of feature tensors in each batch during similarity calculation.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Id of the GPU used.')
    parser.add_argument('--load_queries', action='store_true',
                        help='Flag that indicates that the queries will be loaded to the GPU memory.')
    parser.add_argument('--similarity_function', type=str, default='chamfer', choices=["chamfer", "symmetric_chamfer"],
                        help='Function that will be used to calculate similarity '
                             'between query-target frames and videos.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers used for video loading.')
    # 新增内存管理参数
    parser.add_argument('--cpu_only', action='store_true',
                        help='Force to use CPU even if GPU is available')
    parser.add_argument('--max_cpu_memory_gb', type=float, default=16.0,
                        help='Maximum CPU memory usage in GB')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='Save checkpoint every N videos')

    args = parser.parse_args()

    # 设备选择
    if args.cpu_only or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("> 使用CPU进行计算")
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"> 使用GPU设备: {device}")

    # 将设备信息添加到args
    args.device = device

    # 调整批次大小以适应CPU内存
    if device.type == 'cpu':
        args.batch_sz = min(args.batch_sz, 32)
        args.batch_sz_sim = min(args.batch_sz_sim, 512)
        print(f"> CPU模式下使用较小的批次大小: batch_sz={args.batch_sz}, batch_sz_sim={args.batch_sz_sim}")

    # Create a video generator for the queries
    generator = VideoGenerator(args.query_file)
    loader = DataLoader(generator, num_workers=min(args.workers, 4) if device.type == 'cpu' else args.workers)

    # Initialize ViSiL model
    model = ViSiL(pretrained=True, symmetric='symmetric' in args.similarity_function).to(device)
    model.eval()

    # Extract features of the queries
    queries, queries_ids = [], []
    pbar = tqdm(loader)
    print('> Extract features of the query videos')

    failed_videos = []
    for idx, video in enumerate(pbar):
        frames = video[0][0]
        video_id = video[1][0]

        # 检查帧数是否足够
        if frames.shape[0] < 4:
            print(f"\n> Warning: Video {video_id} has too few frames ({frames.shape[0]} < 4), skipping")
            failed_videos.append(video_id)
            continue

        try:
            # CPU内存管理
            if device.type == 'cpu':
                memory_usage = psutil.Process().memory_info().rss / (1024 ** 3)
                if memory_usage > args.max_cpu_memory_gb:
                    print(f"\n> Memory usage ({memory_usage:.2f} GB) exceeds threshold ({args.max_cpu_memory_gb} GB)")
                    print("> Clearing cache and garbage collecting...")
                    gc.collect()

            features = extract_features(model, frames, args)
            if not args.load_queries:
                features = features.cpu()
            queries.append(features)
            queries_ids.append(video_id)
            pbar.set_postfix(query_id=video_id)

        except Exception as e:
            print(f"\n> Error processing query video {video_id}: {e}")
            failed_videos.append(video_id)
            continue

        # 定期垃圾回收
        if idx % 10 == 0:
            gc.collect()

    if failed_videos:
        print(f"\n> Warning: {len(failed_videos)} query videos failed to process")
        if len(failed_videos) <= 10:
            for vid in failed_videos:
                print(f"  - {vid}")

    if not queries:
        print("> Error: No query features extracted. Exiting.")
        exit(1)

    # Create a video generator for the database video
    generator = VideoGenerator(args.database_file)
    loader = DataLoader(generator, num_workers=min(args.workers, 4) if device.type == 'cpu' else args.workers)

    # Calculate similarities between the queries and the database videos
    similarities = dict({query: dict() for query in queries_ids})
    pbar = tqdm(loader)
    print('\n> Calculate query-target similarities')

    db_failed_videos = []
    for idx, video in enumerate(pbar):
        frames = video[0][0]
        video_id = video[1][0]

        # 检查帧数是否足够
        if frames.shape[0] < 4:
            print(f"\n> Warning: Database video {video_id} has too few frames ({frames.shape[0]} < 4), skipping")
            db_failed_videos.append(video_id)
            continue

        try:
            features = extract_features(model, frames, args)
            sims = calculate_similarities_to_queries(model, queries, features, args)
            for i, s in enumerate(sims):
                similarities[queries_ids[i]][video_id] = float(s)
            pbar.set_postfix(video_id=video_id)

        except Exception as e:
            print(f"\n> Error processing database video {video_id}: {e}")
            db_failed_videos.append(video_id)
            continue

        # 检查点保存
        if args.checkpoint_interval > 0 and (idx + 1) % args.checkpoint_interval == 0:
            checkpoint_file = f"{args.output_file}.checkpoint_{idx + 1}"
            with open(checkpoint_file, 'w') as f:
                json.dump(similarities, f, indent=1)
            print(f"\n> Checkpoint saved: {checkpoint_file}")

        # 定期垃圾回收
        if idx % 10 == 0:
            gc.collect()

    if db_failed_videos:
        print(f"\n> Warning: {len(db_failed_videos)} database videos failed to process")

    # Save similarities to a json file
    with open(args.output_file, 'w') as f:
        json.dump(similarities, f, indent=1)

    print(f'\n> Results saved to {args.output_file}')

    # 保存失败视频列表
    if failed_videos or db_failed_videos:
        failed_file = f"{args.output_file}.failed.json"
        with open(failed_file, 'w') as f:
            json.dump({
                'failed_queries': failed_videos,
                'failed_database': db_failed_videos
            }, f, indent=1)
        print(f'> Failed videos list saved to {failed_file}')