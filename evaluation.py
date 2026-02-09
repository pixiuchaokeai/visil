import torch
import argparse
import os
import json
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import cv2

from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator


def extract_frames_from_video(video_path: str, video_id: str, frames_dir: str, max_frames: int = 50):
    """从视频文件提取帧并保存为npy"""
    frames_file = os.path.join(frames_dir, f"{video_id}.npy")

    # 如果已存在，直接加载
    if os.path.exists(frames_file):
        try:
            frames = np.load(frames_file)
            return torch.from_numpy(frames).float()
        except Exception as e:
            print(f"> 加载帧文件失败 {video_id}: {e}")
            os.remove(frames_file)  # 删除损坏的文件

    # 使用VideoGenerator提取帧
    try:
        # 创建临时文件列表
        temp_list_file = f"temp_{video_id}.txt"
        with open(temp_list_file, 'w') as f:
            f.write(f"{video_id} {video_path}")

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

        # 限制最大帧数
        if frames.shape[0] > max_frames:
            step = max(1, frames.shape[0] // max_frames)
            indices = list(range(0, frames.shape[0], step))[:max_frames]
            frames = frames[indices]

        # 保存为npy
        frames_np = frames.cpu().numpy()
        np.save(frames_file, frames_np)

        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)

        print(f"> 提取帧完成: {video_id} ({frames.shape[0]}帧)")
        return frames

    except Exception as e:
        print(f"> 提取视频 {video_id} 帧失败: {e}")
        if os.path.exists(temp_list_file):
            try:
                os.remove(temp_list_file)
            except:
                pass
        return None


def extract_features_from_frames(model, frames: torch.Tensor, video_id: str, features_dir: str,
                                 device: torch.device, batch_size: int = 4):
    """从帧数据提取特征并保存为npy"""
    features_file = os.path.join(features_dir, f"{video_id}.npy")

    # 如果已存在，直接加载
    if os.path.exists(features_file):
        try:
            features_np = np.load(features_file)
            features = torch.from_numpy(features_np).float()
            return features
        except Exception as e:
            print(f"> 加载特征文件失败 {video_id}: {e}")
            os.remove(features_file)

    try:
        # 确保帧数据格式正确 [T, H, W, C]
        if frames.dim() == 3:
            # [H, W, C] -> [1, H, W, C]
            frames = frames.unsqueeze(0)

        # 分批提取特征
        features_list = []
        for i in range(0, frames.shape[0], batch_size):
            batch = frames[i:i + batch_size]
            if batch.shape[0] > 0:
                batch = batch.to(device).float()

                # ViSiL模型期望 [B, H, W, C] 格式
                if len(batch.shape) == 3:
                    batch = batch.unsqueeze(0)

                with torch.no_grad():
                    batch_features = model.extract_features(batch)

                features_list.append(batch_features.cpu())

        if not features_list:
            print(f"> 未提取到特征: {video_id}")
            return None

        features = torch.cat(features_list, dim=0)

        # 保存特征
        features_np = features.cpu().numpy()
        np.save(features_file, features_np)

        print(f"> 提取特征完成: {video_id} (形状: {features.shape})")
        return features

    except Exception as e:
        print(f"> 提取特征失败 {video_id}: {e}")
        return None


def process_query_videos(query_ids, video_dir, pattern, model, device,
                         frames_dir, features_dir, batch_size=4, max_frames=50):
    """处理查询视频"""
    query_features = {}
    failed_queries = []

    print(f"> 处理查询视频 ({len(query_ids)}个)")

    for idx, query_id in enumerate(tqdm(query_ids, desc="查询视频")):
        try:
            # 构建视频路径
            video_path = os.path.join(video_dir, pattern.format(id=query_id))

            if not os.path.exists(video_path):
                # 尝试其他可能的后缀
                for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
                    alt_path = video_path.rsplit('.', 1)[0] + ext
                    if os.path.exists(alt_path):
                        video_path = alt_path
                        break

            if not os.path.exists(video_path):
                print(f"> 查询视频文件不存在: {query_id}")
                failed_queries.append(query_id)
                continue

            # 提取帧
            frames = extract_frames_from_video(video_path, query_id, frames_dir, max_frames)
            if frames is None or frames.shape[0] < 4:
                print(f"> 查询视频 {query_id} 帧数不足")
                failed_queries.append(query_id)
                continue

            # 提取特征
            features = extract_features_from_frames(model, frames, query_id, features_dir,
                                                    device, batch_size)
            if features is None:
                print(f"> 查询视频 {query_id} 特征提取失败")
                failed_queries.append(query_id)
                continue

            query_features[query_id] = features

            # 每处理5个视频清理一次内存
            if (idx + 1) % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"> 处理查询视频 {query_id} 失败: {str(e)[:200]}")
            failed_queries.append(query_id)

    print(f"> 查询视频处理完成: 成功 {len(query_features)}, 失败 {len(failed_queries)}")
    return query_features, failed_queries


def main():
    parser = argparse.ArgumentParser(description='增强版视频相似度评估')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default="FIVR-5K",
                        choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE"],
                        help='评估数据集名称 (默认: FIVR-5K)')

    # 视频路径参数
    parser.add_argument('--video_dir', type=str, default='datasets/FIVR-200K',
                        help='包含数据库视频的文件路径 (默认: datasets/FIVR-200K)')
    parser.add_argument('--pattern', type=str, default='{id}.mp4',
                        help='视频文件模式 (默认: {id}.mp4)')

    # 帧处理参数
    parser.add_argument('--max_frames', type=int, default=50,
                        help='每个视频最大帧数 (默认: 50)')
    parser.add_argument('--batch_sz', type=int, default=4,
                        help='特征提取批次大小 (默认: 4)')

    # 设备参数
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='使用的GPU设备ID (默认: 0)')
    parser.add_argument('--cpu_only', action='store_true',
                        help='强制使用CPU')

    # 相似度计算参数
    parser.add_argument('--similarity_function', type=str, default='symmetric_chamfer',
                        choices=["chamfer", "symmetric_chamfer"],
                        help='相似度计算函数 (默认: symmetric_chamfer)')

    # 输出目录参数
    parser.add_argument('--frames_dir', type=str, default='output/frames',
                        help='帧文件目录 (默认: output/frames)')
    parser.add_argument('--features_dir', type=str, default='output/features',
                        help='特征文件目录 (默认: output/features)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出目录 (默认: output)')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.frames_dir, exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)

    # 设备选择
    if args.cpu_only or not torch.cuda.is_available():
        device = torch.device('cpu')
        args.batch_sz = min(args.batch_sz, 4)
        print("> 使用CPU进行计算")
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"> 使用GPU设备: {device}")

    # 初始化模型 - 使用与calculate_similarity.py相同的参数
    try:
        model = ViSiL(
            pretrained=True,
            symmetric=('symmetric' in args.similarity_function)
        ).to(device)
        model.eval()
        print("> 模型初始化成功")
    except Exception as e:
        print(f"错误: 初始化模型失败 - {e}")
        import traceback
        traceback.print_exc()
        return

    # 加载数据集
    try:
        if 'CC_WEB' in args.dataset:
            from datasets import CC_WEB_VIDEO
            dataset = CC_WEB_VIDEO()
        elif 'FIVR' in args.dataset:
            from datasets import FIVR
            version = args.dataset.split('-')[1].lower() if '-' in args.dataset else '200k'
            dataset = FIVR(version=version)
        elif 'EVVE' in args.dataset:
            from datasets import EVVE
            dataset = EVVE()
        elif 'SVD' in args.dataset:
            from datasets import SVD
            dataset = SVD()
        else:
            raise ValueError(f"未知的数据集: {args.dataset}")

        print(f"> 已加载数据集: {dataset.name}")
        print(f"> 视频目录: {args.video_dir}")
        print(f"> 帧目录: {args.frames_dir}")
        print(f"> 特征目录: {args.features_dir}")

    except ImportError as e:
        print(f"错误: 无法导入数据集模块 - {e}")
        print("请确保已安装所需的数据集模块")
        return

    # 获取查询和数据库视频ID
    query_ids = dataset.get_queries()
    database_ids = dataset.get_database()

    print(f"> 查询视频数: {len(query_ids)}")
    print(f"> 数据库视频数: {len(database_ids)}")

    # ========== 第一阶段：处理查询视频 ==========
    print("\n" + "=" * 60)
    print("第一阶段：处理查询视频")
    print("=" * 60)

    query_features, failed_queries = process_query_videos(
        query_ids,
        args.video_dir,
        args.pattern,
        model,
        device,
        args.frames_dir,
        args.features_dir,
        args.batch_sz,
        args.max_frames
    )

    if not query_features:
        print("> 错误: 没有成功提取查询特征")
        return

    # ========== 第二阶段：处理数据库视频 ==========
    print("\n" + "=" * 60)
    print("第二阶段：处理数据库视频")
    print("=" * 60)

    # 加载检查点（如果存在）
    checkpoint_file = os.path.join(args.output_dir, "checkpoint.json")
    similarities = {}
    processed_db = set()

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            similarities = checkpoint.get('similarities', {})
            processed_db = set(checkpoint.get('processed_db', []))
            print(f"> 加载检查点，已处理 {len(processed_db)} 个数据库视频")
        except Exception as e:
            print(f"> 加载检查点失败: {e}")

    # 过滤已处理的数据库视频
    db_to_process = [db_id for db_id in database_ids if db_id not in processed_db]

    if not db_to_process:
        print("> 所有数据库视频已处理")
    else:
        print(f"> 待处理数据库视频: {len(db_to_process)}")

        # 分批处理数据库视频
        db_batch_size = 1  # 逐个处理以减少内存使用
        for batch_start in range(0, len(db_to_process), db_batch_size):
            batch_end = min(batch_start + db_batch_size, len(db_to_process))
            batch_db = db_to_process[batch_start:batch_end]

            batch_num = batch_start // db_batch_size + 1
            total_batches = (len(db_to_process) + db_batch_size - 1) // db_batch_size

            print(f"\n> 处理数据库批次 {batch_num}/{total_batches}")

            for db_id in tqdm(batch_db, desc=f"批次 {batch_num}"):
                try:
                    # 构建视频路径
                    video_path = os.path.join(args.video_dir, args.pattern.format(id=db_id))

                    # 检查文件是否存在
                    if not os.path.exists(video_path):
                        # 尝试其他可能的后缀
                        for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
                            alt_path = video_path.rsplit('.', 1)[0] + ext
                            if os.path.exists(alt_path):
                                video_path = alt_path
                                break

                    if not os.path.exists(video_path):
                        print(f"> 数据库视频文件不存在: {db_id}")
                        processed_db.add(db_id)
                        continue

                    # 提取帧
                    frames = extract_frames_from_video(video_path, db_id, args.frames_dir, args.max_frames)
                    if frames is None or frames.shape[0] < 4:
                        print(f"> 数据库视频 {db_id} 帧数不足")
                        processed_db.add(db_id)
                        continue

                    # 提取特征
                    db_features = extract_features_from_frames(model, frames, db_id, args.features_dir,
                                                               device, args.batch_sz)
                    if db_features is None:
                        print(f"> 数据库视频 {db_id} 特征提取失败")
                        processed_db.add(db_id)
                        continue

                    # 计算与所有查询视频的相似度
                    print(f"> 计算相似度: {db_id}...")

                    db_features = db_features.to(device)

                    for query_id, q_feat in query_features.items():
                        q_feat_device = q_feat.to(device)

                        with torch.no_grad():
                            # 确保有批次维度
                            if q_feat_device.dim() == 2:
                                q_feat_device = q_feat_device.unsqueeze(0)
                            if db_features.dim() == 2:
                                db_features = db_features.unsqueeze(0)

                            # 计算相似度
                            sim = model.calculate_video_similarity(q_feat_device, db_features)

                        if query_id not in similarities:
                            similarities[query_id] = {}
                        similarities[query_id][db_id] = float(sim.item())

                    # 标记为已处理
                    processed_db.add(db_id)

                    # 每处理5个视频保存一次检查点
                    if len(processed_db) % 5 == 0:
                        checkpoint = {
                            'similarities': similarities,
                            'processed_db': list(processed_db)
                        }
                        with open(checkpoint_file, 'w') as f:
                            json.dump(checkpoint, f, indent=2)
                        print(f"> 保存检查点，已处理 {len(processed_db)}/{len(database_ids)} 个数据库视频")

                    # 清理内存
                    del frames, db_features
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                except Exception as e:
                    print(f"> 处理数据库视频 {db_id} 失败: {str(e)[:200]}")
                    processed_db.add(db_id)

            # 每批处理完后保存检查点
            checkpoint = {
                'similarities': similarities,
                'processed_db': list(processed_db)
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

    # ========== 第三阶段：保存结果 ==========
    print("\n" + "=" * 60)
    print("第三阶段：保存结果")
    print("=" * 60)

    # 保存完整结果
    result_file = os.path.join(args.output_dir, "similarities.json")
    with open(result_file, 'w') as f:
        json.dump(similarities, f, separators=(',', ':'))
    print(f"> 结果已保存: {result_file}")

    # 保存排序版本
    sorted_file = os.path.join(args.output_dir, "sorted_results.json")
    sorted_results = {}
    for query_id, query_sims in similarities.items():
        sorted_items = sorted(query_sims.items(), key=lambda x: x[1], reverse=True)
        sorted_results[query_id] = dict(sorted_items[:100])

    with open(sorted_file, 'w') as f:
        json.dump(sorted_results, f, indent=2)
    print(f"> 排序结果已保存: {sorted_file}")

    # 删除检查点
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"> 删除检查点文件: {checkpoint_file}")

    # ========== 第四阶段：评估 ==========
    print("\n" + "=" * 60)
    print("第四阶段：评估")
    print("=" * 60)

    try:
        # 评估
        evaluation_results = dataset.evaluate(similarities)

        # 保存评估结果
        eval_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"> 评估结果已保存: {eval_file}")

        # 打印关键指标
        if 'DSVR' in evaluation_results:
            print(f"\n> 评估指标:")
            print(f"  DSVR mAP: {evaluation_results['DSVR']:.4f}")
            print(f"  CSVR mAP: {evaluation_results['CSVR']:.4f}")
            print(f"  ISVR mAP: {evaluation_results['ISVR']:.4f}")
            # for key, value in evaluation_results['DSVR'].items():
            #     if isinstance(value, dict):
            #         for subkey, subvalue in value.items():
            #             print(f"  {key}_{subkey}: {subvalue:.4f}")
            #     else:
            #         print(f"  {key}: {value:.4f}")

        print("\n> 评估完成!")

    except Exception as e:
        print(f"> 评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()