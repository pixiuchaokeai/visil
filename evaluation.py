import torch
import argparse
import os
import json
import gc
import numpy as np
from tqdm import tqdm
import time

from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator


def extract_frames_from_video(video_path: str, video_id: str, frames_dir: str, max_frames: int = 50):
    """从视频文件提取帧并保存为npy，优先使用缓存"""
    frames_file = os.path.join(frames_dir, f"{video_id}.npy")

    if os.path.exists(frames_file):
        try:
            frames = np.load(frames_file)
            if frames.shape[0] >= 4:
                return torch.from_numpy(frames).float()
        except Exception:
            try:
                os.remove(frames_file)
            except:
                pass

    try:
        temp_list_file = f"temp_{video_id}.txt"
        with open(temp_list_file, 'w') as f:
            f.write(f"{video_id} {video_path}")

        generator = VideoGenerator(temp_list_file)
        loader = DataLoader(generator, num_workers=0, batch_size=1)

        frames = None
        for frames_batch, _ in loader:
            frames = frames_batch[0]
            break

        if frames is None or frames.shape[0] < 4:
            if os.path.exists(temp_list_file):
                os.remove(temp_list_file)
            return None

        if frames.shape[0] > max_frames:
            step = max(1, frames.shape[0] // max_frames)
            indices = list(range(0, frames.shape[0], step))[:max_frames]
            frames = frames[indices]

        np.save(frames_file, frames.cpu().numpy())
        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)
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
    """从帧数据提取特征并保存为npy，优先使用缓存"""
    features_file = os.path.join(features_dir, f"{video_id}.npy")

    if os.path.exists(features_file):
        try:
            features_np = np.load(features_file)
            return torch.from_numpy(features_np).float()
        except Exception:
            try:
                os.remove(features_file)
            except:
                pass

    try:
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)

        features_list = []
        total_frames = frames.shape[0]
        for i in range(0, total_frames, batch_size):
            end = min(i + batch_size, total_frames)
            batch = frames[i:end].to(device).float()

            if len(batch.shape) == 3:
                batch = batch.unsqueeze(0)

            with torch.no_grad():
                batch_features = model.extract_features(batch)
            features_list.append(batch_features.cpu())
            del batch, batch_features
            if device.type == 'cpu':
                gc.collect()

        if not features_list:
            return None

        features = torch.cat(features_list, dim=0)
        np.save(features_file, features.cpu().numpy())
        return features

    except Exception as e:
        print(f"> 提取特征失败 {video_id}: {e}")
        return None


def process_query_videos(query_ids, video_dir, pattern, model, device,
                         frames_dir, features_dir, batch_size=4, max_frames=50):
    """处理查询视频，返回特征字典和失败列表"""
    query_features = {}
    failed_queries = []

    print(f"> 处理查询视频 ({len(query_ids)}个)")
    for idx, query_id in enumerate(tqdm(query_ids, desc="查询视频")):
        try:
            video_path = os.path.join(video_dir, pattern.format(id=query_id))
            if not os.path.exists(video_path):
                for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
                    alt_path = video_path.rsplit('.', 1)[0] + ext
                    if os.path.exists(alt_path):
                        video_path = alt_path
                        break
            if not os.path.exists(video_path):
                failed_queries.append(query_id)
                continue

            frames = extract_frames_from_video(video_path, query_id, frames_dir, max_frames)
            if frames is None or frames.shape[0] < 4:
                failed_queries.append(query_id)
                continue

            features = extract_features_from_frames(model, frames, query_id, features_dir,
                                                    device, batch_size)
            if features is None:
                failed_queries.append(query_id)
                continue

            query_features[query_id] = features

            if (idx + 1) % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            failed_queries.append(query_id)

    print(f"> 查询视频完成: 成功 {len(query_features)}, 失败 {len(failed_queries)}")
    return query_features, failed_queries


def main():
    parser = argparse.ArgumentParser(description='增强版视频相似度评估')
    parser.add_argument('--dataset', type=str, default="FIVR-5K",
                        choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE"],
                        help='评估数据集名称')
    parser.add_argument('--video_dir', type=str, default='datasets/FIVR-200K',
                        help='视频文件根目录')
    parser.add_argument('--pattern', type=str, default='{id}.mp4',
                        help='视频文件名模式')
    parser.add_argument('--max_frames', type=int, default=50,
                        help='每个视频最大帧数')
    parser.add_argument('--batch_sz', type=int, default=4,
                        help='特征提取批次大小')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--cpu_only', action='store_true',
                        help='强制使用CPU')
    parser.add_argument('--similarity_function', type=str, default='chamfer',
                        choices=["chamfer", "symmetric_chamfer"],
                        help='相似度函数')
    parser.add_argument('--frames_dir', type=str, default='output/frames',
                        help='帧文件目录')
    parser.add_argument('--features_dir', type=str, default='output/features1/ViSiL_v',
                        help='特征文件目录')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出目录')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.frames_dir, exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)

    if args.cpu_only or not torch.cuda.is_available():
        device = torch.device('cpu')
        args.batch_sz = min(args.batch_sz, 4)
        print("> 使用CPU")
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"> 使用GPU: {device}")

    try:
        model = ViSiL(
            pretrained=True,
            symmetric=('symmetric' in args.similarity_function)
        ).to(device)
        model.eval()
        print("> 模型加载成功")
    except Exception as e:
        print(f"错误: 模型加载失败 - {e}")
        return

    try:
        if 'FIVR' in args.dataset:
            from datasets import FIVR
            version = args.dataset.split('-')[1].lower() if '-' in args.dataset else '5k'
            dataset = FIVR(version=version)
        else:
            raise ValueError(f"未知数据集: {args.dataset}")
        print(f"> 数据集: {dataset.name}")
        query_ids = dataset.get_queries()
        database_ids = dataset.get_database()
        print(f"> 查询: {len(query_ids)}, 数据库: {len(database_ids)}")
    except Exception as e:
        print(f"> 数据集加载失败: {e}")
        return

    # ---------- 第一阶段：处理查询 ----------
    print("\n" + "=" * 60)
    print("第一阶段：处理查询视频")
    print("=" * 60)
    query_features, q_failed = process_query_videos(
        query_ids, args.video_dir, args.pattern,
        model, device, args.frames_dir, args.features_dir,
        args.batch_sz, args.max_frames
    )
    if not query_features:
        print("> 错误: 无查询特征")
        return

    # ---------- 第二阶段：处理数据库 ----------
    print("\n" + "=" * 60)
    print("第二阶段：处理数据库视频")
    print("=" * 60)
    checkpoint_file = os.path.join(args.output_dir, "checkpoint.json")
    similarities = {}
    processed_db = set()

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            similarities = checkpoint.get('similarities', {})
            processed_db = set(checkpoint.get('processed_db', []))
            print(f"> 加载检查点，已处理 {len(processed_db)}/{len(database_ids)}")
        except Exception as e:
            print(f"> 检查点加载失败: {e}")

    db_to_process = [db_id for db_id in database_ids if db_id not in processed_db]
    if not db_to_process:
        print("> 所有数据库视频已处理")
    else:
        print(f"> 待处理 {len(db_to_process)} 个数据库视频")
        # [修改点] 使用单个全局进度条，避免多个进度条刷屏
        pbar = tqdm(total=len(db_to_process), desc="数据库视频", unit="vid")
        for idx, db_id in enumerate(db_to_process):
            try:
                video_path = os.path.join(args.video_dir, args.pattern.format(id=db_id))
                if not os.path.exists(video_path):
                    for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
                        alt_path = video_path.rsplit('.', 1)[0] + ext
                        if os.path.exists(alt_path):
                            video_path = alt_path
                            break
                if not os.path.exists(video_path):
                    processed_db.add(db_id)
                    pbar.update(1)
                    continue

                frames = extract_frames_from_video(video_path, db_id, args.frames_dir, args.max_frames)
                if frames is None or frames.shape[0] < 4:
                    processed_db.add(db_id)
                    pbar.update(1)
                    continue

                db_features = extract_features_from_frames(model, frames, db_id, args.features_dir,
                                                           device, args.batch_sz)
                if db_features is None:
                    processed_db.add(db_id)
                    pbar.update(1)
                    continue

                db_features = db_features.to(device)

                for query_id, q_feat in query_features.items():
                    q_feat_device = q_feat.to(device)
                    with torch.no_grad():
                        if q_feat_device.dim() == 2:
                            q_feat_device = q_feat_device.unsqueeze(0)
                        if db_features.dim() == 2:
                            db_features = db_features.unsqueeze(0)
                        sim = model.calculate_video_similarity(q_feat_device, db_features)
                    if query_id not in similarities:
                        similarities[query_id] = {}
                    similarities[query_id][db_id] = float(sim.item())

                processed_db.add(db_id)

                # [修改点] 每处理 50 个视频保存一次检查点，避免频繁 I/O
                if len(processed_db) % 50 == 0:
                    checkpoint = {
                        'similarities': similarities,
                        'processed_db': list(processed_db)
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint, f)
                    # [修改点] 在进度条后添加简短信息，不单独占行
                    pbar.set_postfix(saved=f"{len(processed_db)}/{len(database_ids)}")

                del frames, db_features
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n> 处理 {db_id} 失败: {str(e)[:100]}")
                processed_db.add(db_id)
            finally:
                pbar.update(1)

        pbar.close()

        # 最终保存一次
        checkpoint = {
            'similarities': similarities,
            'processed_db': list(processed_db)
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)

    # ---------- 第三阶段：保存结果 ----------
    print("\n" + "=" * 60)
    print("第三阶段：保存结果")
    print("=" * 60)
    result_file = os.path.join(args.output_dir, "similarities.json")
    with open(result_file, 'w') as f:
        json.dump(similarities, f, separators=(',', ':'))
    print(f"> 相似度保存: {result_file}")

    sorted_file = os.path.join(args.output_dir, "sorted_results.json")
    sorted_results = {}
    for qid, qsims in similarities.items():
        sorted_items = sorted(qsims.items(), key=lambda x: x[1], reverse=True)[:100]
        sorted_results[qid] = dict(sorted_items)
    with open(sorted_file, 'w') as f:
        json.dump(sorted_results, f, indent=2)
    print(f"> 排序结果保存: {sorted_file}")

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"> 检查点已删除")

    # ---------- 第四阶段：评估 ----------
    print("\n" + "=" * 60)
    print("第四阶段：评估")
    print("=" * 60)
    try:
        eval_results = dataset.evaluate(similarities)
        eval_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"> 评估结果保存: {eval_file}")
        if 'DSVR' in eval_results:
            print(f"\n> DSVR: {eval_results['DSVR']:.4f}")
            print(f"> CSVR: {eval_results['CSVR']:.4f}")
            print(f"> ISVR: {eval_results['ISVR']:.4f}")
        print("\n> 评估完成!")
    except Exception as e:
        print(f"> 评估失败: {e}")

    print("\n" + "=" * 60)
    print("所有阶段完成")
    print("=" * 60)


if __name__ == '__main__':
    main()