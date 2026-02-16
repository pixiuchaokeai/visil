import torch
import argparse
import os
import json
import gc
import numpy as np
from tqdm import tqdm
import time
import traceback

from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator


def extract_frames_from_video(video_path: str, video_id: str, frames_dir: str, max_frames: int = 50):
    """从视频文件提取帧并保存为npy，优先使用缓存，并统一保存为 [T, C, H, W] 格式"""
    frames_file = os.path.join(frames_dir, f"{video_id}.npy")

    if os.path.exists(frames_file):
        try:
            frames_np = np.load(frames_file)
            frames = torch.from_numpy(frames_np).float()
            # 验证帧维度并统一为 [T, C, H, W]
            if frames.dim() == 4:
                if frames.shape[1] == 3:  # 已经是 [T, C, H, W]
                    if frames.shape[2] == frames.shape[3] == 224:
                        return frames
                elif frames.shape[3] == 3:  # 是 [T, H, W, C]
                    frames = frames.permute(0, 3, 1, 2).contiguous()
                    if frames.shape[2] == frames.shape[3] == 224:
                        np.save(frames_file, frames.cpu().numpy())
                        return frames
            # 若维度不符合，删除并重新提取
            print(f"> 帧文件 {video_id}.npy 维度异常 {frames.shape}，重新提取")
            os.remove(frames_file)
        except Exception:
            try:
                os.remove(frames_file)
            except:
                pass

    try:
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            print(f"> 视频文件不存在: {video_path}")
            return None

        temp_list_file = f"temp_{video_id}.txt"
        with open(temp_list_file, 'w') as f:
            f.write(f"{video_id} {video_path}")

        generator = VideoGenerator(temp_list_file)
        loader = DataLoader(generator, num_workers=0, batch_size=1)

        frames = None
        for frames_batch, _ in loader:
            frames = frames_batch[0]  # [T, C, H, W]
            break

        if frames is None or frames.shape[0] < 4:
            print(f"> 视频 {video_id} 帧数不足 ({frames.shape[0] if frames is not None else 0})")
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
        traceback.print_exc()
        if os.path.exists(temp_list_file):
            try:
                os.remove(temp_list_file)
            except:
                pass
        return None


def extract_features_from_frames(model, frames: torch.Tensor, video_id: str, features_dir: str,
                                 device: torch.device, batch_size: int = 4):
    """从帧数据提取特征并保存为npy，优先使用缓存，并验证特征维度（第二维必须为9）"""
    features_file = os.path.join(features_dir, f"{video_id}.npy")

    # 如果存在缓存文件，尝试加载并验证维度
    if os.path.exists(features_file):
        try:
            features_np = np.load(features_file)
            features = torch.from_numpy(features_np).float()
            # 验证特征维度：应为 [帧数, 9, dims]（第二维必须为9）
            if features.dim() == 3 and features.shape[1] == 9:
                return features
            else:
                print(f"> 特征文件 {video_id}.npy 维度异常 {features.shape}（期望第二维为9），重新提取")
                os.remove(features_file)
        except Exception:
            try:
                os.remove(features_file)
            except:
                pass

    try:
        # 确保帧为 [T, C, H, W] 格式（通道在前）
        if frames.dim() == 4 and frames.shape[1] != 3 and frames.shape[3] == 3:
            frames = frames.permute(0, 3, 1, 2).contiguous()
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)

        features_list = []
        total_frames = frames.shape[0]
        for i in range(0, total_frames, batch_size):
            end = min(i + batch_size, total_frames)
            batch = frames[i:end].to(device).float()

            # [修改点] 转换为模型期望的 [B, H, W, C] 格式
            if batch.dim() == 4:
                # 如果当前是 [B, C, H, W]（通道在第二维）
                if batch.shape[1] == 3:
                    batch = batch.permute(0, 2, 3, 1).contiguous()  # -> [B, H, W, C]
                # 如果已经是 [B, H, W, C]（通道在最后一维），则保持
                # 否则尝试转换
                elif batch.shape[3] != 3:
                    # 尝试假设是 [B, C, H, W] 但通道数可能不是3？简单转换
                    batch = batch.permute(0, 2, 3, 1).contiguous()

            with torch.no_grad():
                batch_features = model.extract_features(batch)
            features_list.append(batch_features.cpu())
            del batch, batch_features
            if device.type == 'cpu':
                gc.collect()

        if not features_list:
            print(f"> 视频 {video_id} 特征提取为空")
            return None

        features = torch.cat(features_list, dim=0)  # [总帧数, 9, dims]

        # 验证特征维度：第二维必须为9（区域数）
        if features.dim() != 3 or features.shape[1] != 9:
            print(f"> 提取的特征 {video_id} 维度异常: {features.shape}，期望第二维为9")
            return None

        np.save(features_file, features.cpu().numpy())
        return features

    except Exception as e:
        print(f"> 提取特征失败 {video_id}: {e}")
        traceback.print_exc()
        return None


def process_query_videos(query_ids, id_to_path, model, device,
                         frames_dir, features_dir, batch_size=4, max_frames=50):
    """处理查询视频，返回特征字典和失败列表"""
    query_features = {}
    failed_queries = []

    print(f"> 处理查询视频 ({len(query_ids)}个)")
    for idx, query_id in enumerate(tqdm(query_ids, desc="查询视频")):
        try:
            # 获取视频路径：优先从 id_to_path 字典获取
            if query_id in id_to_path:
                video_path = id_to_path[query_id]
            else:
                # 若字典中无此 ID，记录失败
                failed_queries.append(query_id)
                continue

            if not os.path.exists(video_path):
                print(f"> 视频文件不存在: {video_path}")
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
            print(f"> 处理查询 {query_id} 异常: {e}")
            failed_queries.append(query_id)

    print(f"> 查询视频完成: 成功 {len(query_features)}, 失败 {len(failed_queries)}")
    return query_features, failed_queries


def main():
    parser = argparse.ArgumentParser(description='增强版视频相似度评估')
    parser.add_argument('--dataset', type=str, default="FIVR-5K",
                        choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE"],
                        help='评估数据集名称')
    parser.add_argument('--video_dir', type=str, default='datasets/FIVR-200K',
                        help='视频文件根目录（当不使用列表文件时使用）')
    parser.add_argument('--pattern', type=str, default='{id}.mp4',
                        help='视频文件名模式（当不使用列表文件时使用）')
    # 新增视频列表文件参数，与 calculate_similarity.py 保持一致
    parser.add_argument('--query_file', type=str, default=None,
                        help='查询视频列表文件（每行：id 路径），优先于 video_dir+pattern')
    parser.add_argument('--database_file', type=str, default=None,
                        help='数据库视频列表文件（每行：id 路径），优先于 video_dir+pattern')
    parser.add_argument('--max_frames', type=int, default=50,
                        help='每个视频最大帧数')
    parser.add_argument('--batch_sz', type=int, default=4,
                        help='特征提取批次大小')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--cpu_only', action='store_true',
                        help='强制使用CPU')
    parser.add_argument('--similarity_function', type=str, default='symmetric_chamfer',
                        choices=["chamfer", "symmetric_chamfer"],
                        help='相似度函数')
    parser.add_argument('--frames_dir', type=str, default='output/frames',
                        help='帧文件目录')
    parser.add_argument('--features_dir', type=str, default='output/features1',
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

    # 构建 id 到路径的映射
    id_to_path = {}
    query_ids = []
    database_ids = []

    # 如果提供了查询列表文件，则从文件读取 ID 和路径
    if args.query_file:
        try:
            with open(args.query_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            vid = parts[0]
                            path = ' '.join(parts[1:])
                            id_to_path[vid] = path
                            query_ids.append(vid)
            print(f"> 从 {args.query_file} 读取到 {len(query_ids)} 个查询视频")
        except Exception as e:
            print(f"> 读取查询列表文件失败: {e}")
            return
    else:
        # 否则从数据集模块获取 ID，并准备用 pattern 拼接路径
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
            print(f"> 查询: {len(query_ids)}, 数据库: {len(database_ids)} (将使用 video_dir+pattern 拼接路径)")
        except Exception as e:
            print(f"> 数据集加载失败: {e}")
            return

    # 如果提供了数据库列表文件，同样读取并覆盖 database_ids
    if args.database_file:
        try:
            db_ids_from_file = []
            with open(args.database_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            vid = parts[0]
                            path = ' '.join(parts[1:])
                            id_to_path[vid] = path
                            db_ids_from_file.append(vid)
            database_ids = db_ids_from_file
            print(f"> 从 {args.database_file} 读取到 {len(database_ids)} 个数据库视频")
        except Exception as e:
            print(f"> 读取数据库列表文件失败: {e}")
            return

    # 若使用 pattern 拼接方式，需要为每个 ID 构建路径（当 id_to_path 中不存在时）
    for vid in query_ids + database_ids:
        if vid not in id_to_path:
            # 使用 video_dir + pattern 拼接
            path_candidate = os.path.join(args.video_dir, args.pattern.format(id=vid))
            # 尝试常见扩展名
            if not os.path.exists(path_candidate):
                base = os.path.splitext(path_candidate)[0]
                found = False
                for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
                    alt_path = base + ext
                    if os.path.exists(alt_path):
                        path_candidate = alt_path
                        found = True
                        break
                if not found:
                    # 如果都不存在，路径仍为原拼接路径，后续会检查失败
                    pass
            id_to_path[vid] = path_candidate

    # ---------- 第一阶段：处理查询 ----------
    print("\n" + "=" * 60)
    print("第一阶段：处理查询视频")
    print("=" * 60)
    query_features, q_failed = process_query_videos(
        query_ids, id_to_path, model, device,
        args.frames_dir, args.features_dir,
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
    failed_db = set()

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
        pbar = tqdm(total=len(db_to_process), desc="数据库视频", unit="vid")
        for idx, db_id in enumerate(db_to_process):
            try:
                video_path = id_to_path.get(db_id)
                if not video_path or not os.path.exists(video_path):
                    # 尝试用 pattern 拼接（如果之前未构建）
                    if video_path is None:
                        video_path = os.path.join(args.video_dir, args.pattern.format(id=db_id))
                    if not os.path.exists(video_path):
                        failed_db.add(db_id)
                        processed_db.add(db_id)
                        pbar.update(1)
                        continue

                frames = extract_frames_from_video(video_path, db_id, args.frames_dir, args.max_frames)
                if frames is None or frames.shape[0] < 4:
                    failed_db.add(db_id)
                    processed_db.add(db_id)
                    pbar.update(1)
                    continue

                db_features = extract_features_from_frames(model, frames, db_id, args.features_dir,
                                                           device, args.batch_sz)
                if db_features is None:
                    failed_db.add(db_id)
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

                if len(processed_db) % 50 == 0:
                    checkpoint = {
                        'similarities': similarities,
                        'processed_db': list(processed_db)
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint, f)
                    pbar.set_postfix(saved=f"{len(processed_db)}/{len(database_ids)}")

                del frames, db_features
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n> 处理 {db_id} 失败: {str(e)[:100]}")
                failed_db.add(db_id)
                processed_db.add(db_id)
            finally:
                pbar.update(1)

        pbar.close()

        checkpoint = {
            'similarities': similarities,
            'processed_db': list(processed_db)
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)

        print(f"> 数据库视频完成: 成功 {len(processed_db) - len(failed_db)}/{len(database_ids)}, 失败 {len(failed_db)}")

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
        # 注意：评估需要 dataset 对象，若未加载则需重新加载
        if 'dataset' not in locals():
            if 'FIVR' in args.dataset:
                from datasets import FIVR
                version = args.dataset.split('-')[1].lower() if '-' in args.dataset else '5k'
                dataset = FIVR(version=version)
            else:
                raise ValueError(f"未知数据集: {args.dataset}")
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