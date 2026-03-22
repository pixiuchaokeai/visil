import torch
import torch.nn.functional as F
import argparse
import os
import json
import gc
import numpy as np
from tqdm import tqdm
import time
import traceback
import warnings
from PIL import Image  # [修改点] 用于保存JPG
warnings.filterwarnings("ignore")

# 设置 OpenCV ffmpeg 日志级别为 quiet
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = 'quiet'
try:
    import cv2
    cv2.setLogLevel(0)
except:
    pass

from model.visil import ViSiL
from utils import (
    extract_keyframes,
    extract_features_from_frames,
    extract_dense_frames_range_and_save,
    VideoFrameReader,
    save_interval_metadata,
    load_interval_metadata,
    merge_interval_pairs,
    get_sorted_frames_from_dir
)
from model.similarities import (
    compute_keyframe_similarity_matrix,
    find_candidate_pairs,
    save_similarity_matrix,
    load_similarity_matrix,
    load_features,
    compute_dense_cosine_similarity
)
from visualization import compute_and_save_heatmap
from datasets.generators import VideoGenerator


def main():
    parser = argparse.ArgumentParser(description='四阶段视频相似度评估（关键帧提取→矩阵计算→密集帧提取→相似度计算+亮线）')
    parser.add_argument('--dataset', type=str, default="FIVR-5K",
                        choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE"],
                        help='评估数据集名称')
    parser.add_argument('--video_dir', type=str, default='datasets/FIVR-200K',
                        help='视频文件根目录')
    parser.add_argument('--pattern', type=str, default='{id}.mp4',
                        help='视频文件名模式')
    parser.add_argument('--query_file', type=str, default="datasets/fivr-5k-queries-filtered.txt",
                        help='查询视频列表文件')
    parser.add_argument('--database_file', type=str, default="datasets/fivr-5k-database-filtered.txt",
                        help='数据库视频列表文件')
    # 第一阶段参数
    parser.add_argument('--first_stage_method', type=str, default='iframe',
                        choices=['default', '2s', 'local_maxima', 'iframe', 'i_p_mixed'],
                        help='第一阶段关键帧提取方法')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='相似度阈值，用于第二阶段筛选和第三阶段区间扩展')
    parser.add_argument('--max_keyframes', type=int, default=0,
                        help='第一阶段最大关键帧数（0表示不限制）')
    parser.add_argument('--lm_threshold', type=float, default=0.6,
                        help='local_maxima方法的差异阈值')
    # 密集采样参数
    parser.add_argument('--dense_fps', type=float, default=1.0,
                        help='第三阶段密集采样帧率（每秒帧数）')
    # 模型和设备
    parser.add_argument('--batch_sz', type=int, default=4,
                        help='特征提取批次大小')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--cpu_only', action='store_true', default=False,
                        help='强制使用CPU')
    parser.add_argument('--similarity_function', type=str, default='chamfer',
                        choices=["chamfer", "symmetric_chamfer"],
                        help='相似度函数：chamfer(visil_v) 或 symmetric_chamfer(visil_sym)')
    # 目录参数
    parser.add_argument('--frames_dir', type=str, default='output/frames1',
                        help='基础帧文件目录')
    parser.add_argument('--features_dir', type=str, default='output/features1',
                        help='基础特征文件目录')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出目录')
    parser.add_argument('--ffprobe_path', type=str, default='ffprobe',
                        help='ffprobe 可执行文件路径')
    # 阶段控制参数
    parser.add_argument('--stage', type=str, default='4',
                        choices=['1', '2', '3', '4', 'all'],
                        help='要运行的阶段：1=关键帧提取与特征计算，2=相似度矩阵计算，3=密集帧提取，4=相似度计算+亮线，all=全部顺序执行')
    # 目录参数
    parser.add_argument('--sim_matrices_dir', type=str, default=None,
                        help='第二阶段相似度矩阵保存目录（默认自动生成）')
    parser.add_argument('--dense_frames_dir', type=str, default=None,
                        help='第三阶段密集帧JPG保存目录（默认自动生成）')
    parser.add_argument('--intervals_meta', type=str, default=None,
                        help='第三阶段区间元数据文件路径（默认自动生成）')
    parser.add_argument('--heatmaps_dir', type=str, default=None,
                        help='第四阶段亮线热图保存目录（默认自动生成）')
    parser.add_argument('--dense_features_dir', type=str, default=None,
                        help='第四阶段密集帧特征缓存目录（默认自动生成）')
    # [修改点] 查询全局密集特征缓存目录，现在合并到 dense_features_dir 中
    parser.add_argument('--dense_query_features_dir', type=str, default=None,
                        help='查询视频全局密集帧特征缓存目录（默认自动生成）')
    parser.add_argument('--sim_matrices_dense_dir', type=str, default=None,
                        help='第四阶段密集帧相似度矩阵保存目录（默认自动生成）')
    # 兼容旧参数 --only_first_stage，将其映射为 stage=1
    parser.add_argument('--only_first_stage', action='store_true', default=False,
                        help='[已废弃] 请使用 --stage 1')

    args = parser.parse_args()

    # 兼容旧参数
    if args.only_first_stage:
        args.stage = '1'
        print("> 检测到 --only_first_stage，已自动转换为 --stage 1")

    # 根据相似度函数确定模型类型和维度
    is_symmetric = 'symmetric' in args.similarity_function
    expected_dims = 512 if is_symmetric else 3840
    model_suffix = '_sym' if is_symmetric else '_v'

    # 第一阶段方法后缀
    method_suffix = args.first_stage_method
    if args.first_stage_method == 'local_maxima':
        method_suffix = f'local_maxima_{args.lm_threshold}'

    # 设置目录
    actual_frames_dir = os.path.join(args.frames_dir, method_suffix)
    actual_features_dir = os.path.join(args.features_dir, method_suffix + model_suffix)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(actual_frames_dir, exist_ok=True)
    os.makedirs(actual_features_dir, exist_ok=True)

    if args.sim_matrices_dir is None:
        args.sim_matrices_dir = os.path.join(args.output_dir, 'sim_matrices_cos_rough', method_suffix + model_suffix)
    if args.dense_frames_dir is None:
        args.dense_frames_dir = os.path.join(args.output_dir, 'dense_frames', f"{method_suffix}{model_suffix}_dense_{args.dense_fps}fps")
    if args.intervals_meta is None:
        args.intervals_meta = os.path.join(args.dense_frames_dir, 'intervals_meta.json')
    if args.heatmaps_dir is None:
        args.heatmaps_dir = os.path.join(args.output_dir, 'heatmaps', method_suffix + model_suffix)
    if args.dense_features_dir is None:
        args.dense_features_dir = os.path.join(args.output_dir, 'dense_features', f"{method_suffix}{model_suffix}_dense_{args.dense_fps}fps")
    # [修改点] 将查询特征缓存目录合并到 dense_features_dir
    if args.dense_query_features_dir is None:
        args.dense_query_features_dir = args.dense_features_dir  # 不再单独创建目录
    if args.sim_matrices_dense_dir is None:
        args.sim_matrices_dense_dir = os.path.join(args.output_dir, 'sim_matrices_chamfer_dense', method_suffix + model_suffix)

    # 创建必要的目录（注意 dense_query_features_dir 已合并，不再单独创建）
    for d in [args.sim_matrices_dir, args.dense_frames_dir, args.heatmaps_dir, args.dense_features_dir, args.sim_matrices_dense_dir]:
        os.makedirs(d, exist_ok=True)

    # 设备
    if args.cpu_only or not torch.cuda.is_available():
        device = torch.device('cpu')
        args.batch_sz = min(args.batch_sz, 4)
        print("> 使用CPU")
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"> 使用GPU: {device}")

    total_start = time.time()

    # 加载模型（阶段1、3、4需要）
    model = None
    if args.stage in ['1', '3', '4', 'all']:
        try:
            model = ViSiL(pretrained=True, symmetric=is_symmetric).to(device)
            model.eval()
            print("> 模型加载成功")
        except Exception as e:
            print(f"错误: 模型加载失败 - {e}")
            return

    # 构建 id_to_path 和列表
    id_to_path = {}
    query_ids = []
    database_ids = []

    # 从文件读取
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
            print(f"> 读取查询列表: {len(query_ids)} 个")
        except Exception as e:
            print(f"> 读取查询列表失败: {e}")
            return

    if args.database_file:
        try:
            with open(args.database_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            vid = parts[0]
                            path = ' '.join(parts[1:])
                            id_to_path[vid] = path
                            database_ids.append(vid)
            print(f"> 读取数据库列表: {len(database_ids)} 个")
        except Exception as e:
            print(f"> 读取数据库列表失败: {e}")
            return

    # 补全路径
    for vid in query_ids + database_ids:
        if vid not in id_to_path:
            path_candidate = os.path.join(args.video_dir, args.pattern.format(id=vid))
            if not os.path.exists(path_candidate):
                base = os.path.splitext(path_candidate)[0]
                for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
                    alt = base + ext
                    if os.path.exists(alt):
                        path_candidate = alt
                        break
            id_to_path[vid] = path_candidate

    # ========== 阶段1：关键帧提取与特征计算 ==========
    if args.stage in ['1', 'all']:
        print("\n" + "=" * 60)
        print("第一阶段：关键帧提取与特征计算")
        print("=" * 60)
        stage1_start = time.time()

        db_meta = {}  # video_id -> (indices, fps, total_frames)

        # 处理数据库视频
        print(f"处理数据库视频 ({len(database_ids)} 个)...")
        for db_id in tqdm(database_ids, desc="数据库关键帧", unit="vid"):
            video_path = id_to_path.get(db_id)
            if not video_path or not os.path.exists(video_path):
                continue
            indices, frames, fps, total_frames = extract_keyframes(
                video_path, db_id, actual_frames_dir, args.first_stage_method,
                args.max_keyframes, args.lm_threshold, args.ffprobe_path
            )
            if indices is None or frames is None:
                continue
            db_meta[db_id] = (indices, fps, total_frames)

            _ = extract_features_from_frames(
                model, frames, db_id, actual_features_dir, device,
                args.batch_sz, expected_dims, use_cache=True
            )
            del frames
            gc.collect()

        # 处理查询视频
        print(f"\n处理查询视频 ({len(query_ids)} 个)...")
        query_keyfeat = {}  # query_id -> (indices, features, fps, total_frames)
        for q_id in tqdm(query_ids, desc="查询关键帧", unit="vid"):
            video_path = id_to_path.get(q_id)
            if not video_path or not os.path.exists(video_path):
                continue
            indices, frames, fps, total_frames = extract_keyframes(
                video_path, q_id, actual_frames_dir, args.first_stage_method,
                args.max_keyframes, args.lm_threshold, args.ffprobe_path
            )
            if indices is None or frames is None:
                continue
            feats = extract_features_from_frames(
                model, frames, q_id, actual_features_dir, device,
                args.batch_sz, expected_dims, use_cache=True
            )
            if feats is not None:
                query_keyfeat[q_id] = (indices, feats, fps, total_frames)
            del frames

        stage1_time = time.time() - stage1_start
        print(f"> 第一阶段完成，耗时 {stage1_time:.2f}s")

    # ========== 阶段2：关键帧相似度矩阵计算与存储 ==========
    if args.stage in ['2', 'all']:
        print("\n" + "=" * 60)
        print("第二阶段：关键帧相似度矩阵计算与存储")
        print("=" * 60)
        stage2_start = time.time()

        # 检查点文件
        stage2_checkpoint = os.path.join(args.sim_matrices_dir, 'stage2_checkpoint.json')
        processed_pairs = set()
        if os.path.exists(stage2_checkpoint):
            try:
                with open(stage2_checkpoint, 'r') as f:
                    processed_pairs = set(json.load(f))
                print(f"> 加载检查点，已处理 {len(processed_pairs)} 对")
            except Exception as e:
                print(f"> 检查点加载失败: {e}")

        total_pairs = len(query_ids) * len(database_ids)
        pbar = tqdm(total=total_pairs, desc="计算相似度矩阵", unit="pair")
        for q_id in query_ids:
            q_feat = load_features(q_id, actual_features_dir, expected_dims)
            if q_feat is None:
                pbar.update(len(database_ids))
                continue
            for db_id in database_ids:
                pair_key = f"{q_id}_{db_id}"
                if pair_key in processed_pairs:
                    pbar.update(1)
                    continue
                db_feat = load_features(db_id, actual_features_dir, expected_dims)
                if db_feat is None:
                    pbar.update(1)
                    continue
                sim_matrix = compute_keyframe_similarity_matrix(q_feat, db_feat)
                save_similarity_matrix(sim_matrix, q_id, db_id, args.sim_matrices_dir)
                processed_pairs.add(pair_key)
                if len(processed_pairs) % 100 == 0:
                    with open(stage2_checkpoint, 'w') as f:
                        json.dump(list(processed_pairs), f)
                pbar.update(1)
                del db_feat
                gc.collect()
            del q_feat
            gc.collect()
        pbar.close()

        if os.path.exists(stage2_checkpoint):
            os.remove(stage2_checkpoint)
        stage2_time = time.time() - stage2_start
        print(f"> 第二阶段完成，耗时 {stage2_time:.2f}s，矩阵保存至 {args.sim_matrices_dir}")

    # ========== 阶段3：候选区间定位与密集帧提取 ==========
    if args.stage in ['3', 'all']:
        print("\n" + "=" * 60)
        print("第三阶段：候选区间定位与密集帧提取")
        print("=" * 60)
        stage3_start = time.time()

        # [修改点] 预处理所有查询视频的全局密集帧特征，并保存JPG
        print("预处理查询视频全局密集帧特征并保存JPG...")
        query_dense_feat = {}  # q_id -> (features, fps, total_frames, indices, step)
        for q_id in tqdm(query_ids, desc="查询全局密集帧"):
            video_path = id_to_path.get(q_id)
            if not video_path or not os.path.exists(video_path):
                continue
            # 读取视频信息获取总帧数和fps
            try:
                reader = VideoFrameReader(video_path)
                total_frames = len(reader)
                fps = reader.fps
                reader.close()
            except:
                continue
            # 计算采样步长
            step = max(1, int(round(fps / args.dense_fps)))
            indices = list(range(0, total_frames, step))
            if not indices:
                continue
            # 检查是否已有缓存特征（现在使用 args.dense_features_dir/query/）
            q_feat_cache = os.path.join(args.dense_features_dir, 'query', q_id, f"{q_id}.npy")
            if os.path.exists(q_feat_cache):
                try:
                    feat_np = np.load(q_feat_cache)
                    feat = torch.from_numpy(feat_np).float()
                    if feat.dim() == 3 and feat.shape[1] == 9 and feat.shape[2] == expected_dims:
                        query_dense_feat[q_id] = (feat, fps, total_frames, indices, step)
                        continue
                except:
                    pass
            # 提取帧
            frames_np = None
            try:
                reader = VideoFrameReader(video_path)
                frames_np = reader.get_frames(indices)
                reader.close()
            except:
                continue
            if frames_np is None or frames_np.size == 0:
                continue
            # [修改点] 保存JPG到 dense_frames/query/<video_id>/
            query_jpg_dir = os.path.join(args.dense_frames_dir, 'query', q_id)
            os.makedirs(query_jpg_dir, exist_ok=True)
            for i, idx in enumerate(indices):
                jpg_path = os.path.join(query_jpg_dir, f"frame_{idx:06d}.jpg")
                Image.fromarray(frames_np[i]).save(jpg_path, quality=90)
            # 缩放到224x224
            frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
            if frames_tensor.shape[2] != 224 or frames_tensor.shape[3] != 224:
                frames_tensor = F.interpolate(frames_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            frames_tensor = frames_tensor.permute(0, 2, 3, 1)  # [T, H, W, C]
            # 特征保存到 args.dense_features_dir/query/<video_id>/
            feat = extract_features_from_frames(model, frames_tensor, q_id,
                                                os.path.join(args.dense_features_dir, 'query'),
                                                device, args.batch_sz, expected_dims, use_cache=True)
            if feat is not None:
                query_dense_feat[q_id] = (feat, fps, total_frames, indices, step)

        # 加载区间元数据
        intervals_meta = {}
        if os.path.exists(args.intervals_meta):
            try:
                with open(args.intervals_meta, 'r') as f:
                    intervals_meta = json.load(f)
                print(f"> 加载区间元数据，已有 {len(intervals_meta)} 个区间")
            except Exception as e:
                print(f"> 加载区间元数据失败: {e}")

        total_pairs = len(query_ids) * len(database_ids)
        pbar = tqdm(total=total_pairs, desc="处理矩阵", unit="pair")
        for q_id in query_ids:
            if q_id not in query_dense_feat:
                pbar.update(len(database_ids))
                continue
            q_feat, q_fps, q_total, q_indices, q_step = query_dense_feat[q_id]

            # 加载查询元数据（关键帧信息）
            indices_file = os.path.join(actual_frames_dir, q_id, 'indices.json')
            info_file = os.path.join(actual_frames_dir, q_id, 'info.json')
            if not (os.path.exists(indices_file) and os.path.exists(info_file)):
                pbar.update(len(database_ids))
                continue
            with open(indices_file, 'r') as f:
                q_key_indices = json.load(f)
            with open(info_file, 'r') as f:
                q_info = json.load(f)
            q_fps = q_info.get('fps', 30.0)
            q_total = q_info.get('total_frames', 0)

            for db_id in database_ids:
                # 加载数据库元数据
                db_indices_file = os.path.join(actual_frames_dir, db_id, 'indices.json')
                db_info_file = os.path.join(actual_frames_dir, db_id, 'info.json')
                if not (os.path.exists(db_indices_file) and os.path.exists(db_info_file)):
                    pbar.update(1)
                    continue
                with open(db_indices_file, 'r') as f:
                    db_key_indices = json.load(f)
                with open(db_info_file, 'r') as f:
                    db_info = json.load(f)
                db_fps = db_info.get('fps', 30.0)
                db_total = db_info.get('total_frames', 0)

                # 加载相似度矩阵
                sim_matrix = load_similarity_matrix(q_id, db_id, args.sim_matrices_dir)
                if sim_matrix is None:
                    pbar.update(1)
                    continue

                # 查找候选对
                candidate_pairs = find_candidate_pairs(sim_matrix, args.threshold)
                if not candidate_pairs:
                    pbar.update(1)
                    continue

                # 收集该视频对的所有候选区间对
                interval_pairs = []
                for i, j in candidate_pairs:
                    q_isolated = True
                    if i > 0 and sim_matrix[i-1, j] >= args.threshold:
                        q_isolated = False
                    if i < len(q_key_indices)-1 and sim_matrix[i+1, j] >= args.threshold:
                        q_isolated = False

                    db_isolated = True
                    if j > 0 and sim_matrix[i, j-1] >= args.threshold:
                        db_isolated = False
                    if j < len(db_key_indices)-1 and sim_matrix[i, j+1] >= args.threshold:
                        db_isolated = False

                    isolated = q_isolated and db_isolated

                    if isolated:
                        q_center = q_key_indices[i]
                        q_start_frame = max(0, q_center - int(q_fps))
                        q_end_frame = min(q_total - 1, q_center + int(q_fps))

                        db_center = db_key_indices[j]
                        db_start_frame = max(0, db_center - int(db_fps))
                        db_end_frame = min(db_total - 1, db_center + int(db_fps))

                        q_step_dense = max(1, int(round(q_fps / args.dense_fps)))
                        db_step_dense = max(1, int(round(db_fps / args.dense_fps)))
                    else:
                        db_k_start = j
                        while db_k_start > 0 and sim_matrix[i, db_k_start - 1] >= args.threshold:
                            db_k_start -= 1

                        db_k_end = j
                        while db_k_end < len(db_key_indices) - 1 and sim_matrix[i, db_k_end + 1] >= args.threshold:
                            db_k_end += 1

                        q_k_start = i
                        while q_k_start > 0 and sim_matrix[q_k_start - 1, j] >= args.threshold:
                            q_k_start -= 1

                        q_k_end = i
                        while q_k_end < len(q_key_indices) - 1 and sim_matrix[q_k_end + 1, j] >= args.threshold:
                            q_k_end += 1

                        db_start_frame = db_key_indices[db_k_start]
                        db_end_frame   = db_key_indices[db_k_end]
                        q_start_frame  = q_key_indices[q_k_start]
                        q_end_frame    = q_key_indices[q_k_end]

                        q_step_dense = max(1, int(round(q_fps / args.dense_fps)))
                        db_step_dense = max(1, int(round(db_fps / args.dense_fps)))

                    interval_pairs.append({
                        'q_start': q_start_frame,
                        'q_end': q_end_frame,
                        'db_start': db_start_frame,
                        'db_end': db_end_frame,
                        'q_step': q_step_dense,
                        'db_step': db_step_dense
                    })

                merged_pairs = merge_interval_pairs(interval_pairs)

                for idx, pair in enumerate(merged_pairs):
                    q_start = pair['q_start']
                    q_end = pair['q_end']
                    db_start = pair['db_start']
                    db_end = pair['db_end']
                    q_step_dense = pair['q_step']
                    db_step_dense = pair['db_step']

                    interval_id = f"{q_id}_{db_id}_m{idx}_q{q_start}_{q_end}_db{db_start}_{db_end}"

                    if interval_id in intervals_meta:
                        continue

                    # 提取数据库区间密集帧（查询区间已全局提取，此处不再重复）
                    db_save_dir = os.path.join(args.dense_frames_dir, 'database', db_id, interval_id)
                    db_frames_saved = extract_dense_frames_range_and_save(
                        id_to_path[db_id], db_start, db_end, db_step_dense,
                        db_save_dir, max_size=None
                    )
                    if not db_frames_saved:
                        continue

                    # 记录区间元数据（包含查询区间信息，但不保存查询帧）
                    intervals_meta[interval_id] = {
                        'query_id': q_id,
                        'db_id': db_id,
                        'q_start_frame': q_start,
                        'q_end_frame': q_end,
                        'q_step': q_step_dense,
                        'db_start_frame': db_start,
                        'db_end_frame': db_end,
                        'db_step': db_step_dense,
                        'db_save_dir': db_save_dir,
                    }

                pbar.update(1)
                if len(intervals_meta) % 50 == 0:
                    save_interval_metadata(intervals_meta, args.intervals_meta)
        pbar.close()

        save_interval_metadata(intervals_meta, args.intervals_meta)
        stage3_time = time.time() - stage3_start
        print(f"> 第三阶段完成，耗时 {stage3_time:.2f}s，共生成 {len(intervals_meta)} 个区间，数据库密集帧保存至 {args.dense_frames_dir}/database，查询密集帧保存至 {args.dense_frames_dir}/query")

    # ========== 阶段4：密集帧相似度计算与亮线检测（按区间独立计算，输出定位信息）==========
    if args.stage in ['4', 'all']:
        print("\n" + "=" * 60)
        print("第四阶段：密集帧相似度计算与亮线检测（按区间独立计算，输出定位信息）")
        print("=" * 60)
        stage4_start = time.time()

        if not os.path.exists(args.intervals_meta):
            print(f"> 错误：区间元数据文件不存在 {args.intervals_meta}")
            return
        intervals_meta = load_interval_metadata(args.intervals_meta)

        # [修改点] 加载查询全局密集特征（从 args.dense_features_dir/query/）
        query_dense_feat = {}  # q_id -> (features, fps, total_frames, indices, step)
        for q_id in query_ids:
            feat_path = os.path.join(args.dense_features_dir, 'query', q_id, f"{q_id}.npy")
            if os.path.exists(feat_path):
                try:
                    feat_np = np.load(feat_path)
                    feat = torch.from_numpy(feat_np).float()
                    if feat.dim() == 3 and feat.shape[1] == 9 and feat.shape[2] == expected_dims:
                        # 重新获取视频信息（也可从阶段3缓存的信息中读取，这里简化处理）
                        video_path = id_to_path.get(q_id)
                        if video_path and os.path.exists(video_path):
                            try:
                                reader = VideoFrameReader(video_path)
                                total_frames = len(reader)
                                fps = reader.fps
                                reader.close()
                                step = max(1, int(round(fps / args.dense_fps)))
                                indices = list(range(0, total_frames, step))
                                query_dense_feat[q_id] = (feat, fps, total_frames, indices, step)
                            except:
                                pass
                except:
                    pass
        print(f"> 加载查询全局密集特征: {len(query_dense_feat)} 个")

        # 数据库区间特征缓存（按区间ID）
        db_interval_feat_cache = {}

        # 存储每个区间的详细结果
        interval_results = []  # 每个元素为字典

        # 用于聚合视频对相似度（取区间最高分）
        similarities = {}

        processed_intervals = set()
        # 检查点文件（按区间）
        stage4_checkpoint = os.path.join(args.heatmaps_dir, 'stage4_checkpoint_intervals.json')
        if os.path.exists(stage4_checkpoint):
            try:
                with open(stage4_checkpoint, 'r') as f:
                    cp = json.load(f)
                interval_results = cp.get('interval_results', [])
                similarities = cp.get('similarities', {})
                processed_intervals = set(cp.get('processed_intervals', []))
                print(f"> 加载检查点，已处理 {len(processed_intervals)} 个区间")
            except Exception as e:
                print(f"> 检查点加载失败: {e}")

        total_intervals = len(intervals_meta)
        pbar = tqdm(total=total_intervals, desc="处理区间", unit="interval")
        for interval_id, meta in intervals_meta.items():
            if interval_id in processed_intervals:
                pbar.update(1)
                continue

            q_id = meta['query_id']
            db_id = meta['db_id']
            q_start = meta['q_start_frame']
            q_end = meta['q_end_frame']
            q_step = meta['q_step']
            db_start = meta['db_start_frame']
            db_end = meta['db_end_frame']
            db_step = meta['db_step']
            db_save_dir = meta['db_save_dir']

            if q_id not in query_dense_feat:
                pbar.update(1)
                continue
            q_feat_full, q_fps, q_total, q_indices_full, q_step_full = query_dense_feat[q_id]

            # 从查询全局特征中切片得到查询区间特征
            # 首先找到q_start和q_end在q_indices_full中的索引范围
            start_idx_in_full = None
            end_idx_in_full = None
            for i, idx in enumerate(q_indices_full):
                if idx >= q_start:
                    start_idx_in_full = i
                    break
            for i, idx in enumerate(q_indices_full):
                if idx <= q_end:
                    end_idx_in_full = i
                else:
                    break
            if start_idx_in_full is None or end_idx_in_full is None:
                pbar.update(1)
                continue
            q_feat_interval = q_feat_full[start_idx_in_full:end_idx_in_full+1]  # [T_q_interval, 9, D]

            # 获取数据库区间特征
            if db_interval_feat_cache.get(interval_id) is not None:
                db_feat_interval = db_interval_feat_cache[interval_id]
            else:
                # 从数据库区间JPG目录加载帧并提取特征
                db_frames_info = get_sorted_frames_from_dir(db_save_dir)
                if db_frames_info is None:
                    pbar.update(1)
                    continue
                db_frames = [img for _, img in db_frames_info]
                db_frames_np = np.stack(db_frames, axis=0)  # [T, H, W, 3]
                db_frames_tensor = torch.from_numpy(db_frames_np).permute(0, 3, 1, 2).float()
                if db_frames_tensor.shape[2] != 224 or db_frames_tensor.shape[3] != 224:
                    db_frames_tensor = F.interpolate(db_frames_tensor, size=(224, 224), mode='bilinear', align_corners=False)
                db_frames_tensor = db_frames_tensor.permute(0, 2, 3, 1)
                db_feat_interval = extract_features_from_frames(model, db_frames_tensor, f"{interval_id}_db",
                                                                None, device, args.batch_sz, expected_dims, use_cache=False)
                if db_feat_interval is None:
                    pbar.update(1)
                    continue
                # 可选缓存
                db_interval_feat_cache[interval_id] = db_feat_interval

            if q_feat_interval is None or db_feat_interval is None:
                pbar.update(1)
                continue

            # 使用余弦相似度计算视频级相似度
            sim_val = compute_dense_cosine_similarity(q_feat_interval, db_feat_interval)

            # [修改点] 生成该区间的热图（do_plot=True）
            heatmap_path = os.path.join(args.heatmaps_dir, q_id, f"{db_id}_{interval_id}.png")
            matrix_save_path = os.path.join(args.sim_matrices_dense_dir, q_id, f"{db_id}_{interval_id}.npy")
            compute_and_save_heatmap(model, q_feat_interval, db_feat_interval, heatmap_path, matrix_save_path, do_plot=True)

            # 记录区间结果
            interval_result = {
                'interval_id': interval_id,
                'query_id': q_id,
                'db_id': db_id,
                'q_start_frame': q_start,
                'q_end_frame': q_end,
                'q_start_sec': q_start / q_fps if q_fps > 0 else 0,
                'q_end_sec': q_end / q_fps if q_fps > 0 else 0,
                'db_start_frame': db_start,
                'db_end_frame': db_end,
                'db_start_sec': db_start / q_fps if q_fps > 0 else 0,  # 注意数据库帧率可能与查询不同，但这里用数据库自己的帧率更准确，暂时用q_fps简化
                'db_end_sec': db_end / q_fps if q_fps > 0 else 0,
                'similarity': sim_val
            }
            interval_results.append(interval_result)

            # 聚合视频对相似度（取最大值）
            if q_id not in similarities:
                similarities[q_id] = {}
            if db_id not in similarities[q_id] or sim_val > similarities[q_id][db_id]:
                similarities[q_id][db_id] = sim_val

            processed_intervals.add(interval_id)

            if len(processed_intervals) % 50 == 0:
                cp_data = {
                    'interval_results': interval_results,
                    'similarities': similarities,
                    'processed_intervals': list(processed_intervals)
                }
                with open(stage4_checkpoint, 'w') as f:
                    json.dump(cp_data, f)
            pbar.update(1)

        pbar.close()

        # 保存区间详细结果
        interval_results_file = os.path.join(args.output_dir, f"interval_results_{method_suffix}{model_suffix}_th{args.threshold}.json")
        with open(interval_results_file, 'w') as f:
            json.dump(interval_results, f, indent=2)
        print(f"> 区间详细结果保存至: {interval_results_file}")

        # 保存聚合相似度
        final_sim_file = os.path.join(args.output_dir, f"similarities_{method_suffix}{model_suffix}_th{args.threshold}.json")
        with open(final_sim_file, 'w') as f:
            json.dump(similarities, f, separators=(',', ':'))
        print(f"> 聚合相似度保存至: {final_sim_file}")

        if os.path.exists(stage4_checkpoint):
            os.remove(stage4_checkpoint)

        stage4_time = time.time() - stage4_start
        total_time = time.time() - total_start

        # ========== 评估 ==========
        print("\n" + "=" * 60)
        print("评估")
        print("=" * 60)
        try:
            if 'FIVR' in args.dataset:
                from datasets import FIVR
                version = args.dataset.split('-')[1].lower() if '-' in args.dataset else '5k'
                dataset = FIVR(version=version)
            else:
                raise ValueError(f"未知数据集: {args.dataset}")
            eval_results = dataset.evaluate(similarities)
            eval_file = os.path.join(args.output_dir, f"evaluation_{method_suffix}{model_suffix}_th{args.threshold}.json")
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"> DSVR: {eval_results.get('DSVR', 0):.4f}")
            print(f"> CSVR: {eval_results.get('CSVR', 0):.4f}")
            print(f"> ISVR: {eval_results.get('ISVR', 0):.4f}")
        except Exception as e:
            print(f"> 评估失败: {e}")

        summary = {
            'dataset': args.dataset,
            'first_stage_method': args.first_stage_method,
            'threshold': args.threshold,
            'dense_fps': args.dense_fps,
            'stage1_time': stage1_time if 'stage1_time' in locals() else 0,
            'stage2_time': stage2_time if 'stage2_time' in locals() else 0,
            'stage3_time': stage3_time if 'stage3_time' in locals() else 0,
            'stage4_time': stage4_time if 'stage4_time' in locals() else 0,
            'total_time': total_time,
            'device': 'GPU' if not args.cpu_only and torch.cuda.is_available() else 'CPU'
        }
        summary_file = os.path.join(args.output_dir, f"summary_{method_suffix}{model_suffix}_th{args.threshold}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"> 总结保存: {summary_file}")
        print("\n" + "=" * 60)
        print("完成")
        print("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n> 用户中断，程序退出")
    except Exception as e:
        print(f"\n> 未捕获异常: {e}")
        traceback.print_exc()