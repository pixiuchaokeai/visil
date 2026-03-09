import torch
import argparse
import os
import json
import gc
import numpy as np
from tqdm import tqdm
import time
import traceback
import warnings
warnings.filterwarnings("ignore")

# 设置 OpenCV ffmpeg 日志级别为 quiet
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = 'quiet'
try:
    import cv2
    cv2.setLogLevel(0)
except:
    pass

from model.visil import ViSiL
from stage_utils import (
    extract_keyframes,
    extract_features_from_frames,
    compute_keyframe_similarity_matrix,
    find_candidate_pairs,
    extract_dense_frames_range_and_save,
    compute_dense_similarity,
    VideoFrameReader,
    save_similarity_matrix,
    load_similarity_matrix,
    load_features,
    save_interval_metadata,
    load_interval_metadata,
    compute_and_save_heatmap,
    merge_interval_pairs,
    load_dense_frames_from_dir,
    get_sorted_frames_from_dir  # [修改点] 新增函数用于获取排序去重的帧列表
)
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
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='相似度阈值，用于第二阶段筛选和第三阶段区间扩展')
    parser.add_argument('--max_keyframes', type=int, default=0,
                        help='第一阶段最大关键帧数（0表示不限制）')
    parser.add_argument('--lm_threshold', type=float, default=0.6,
                        help='local_maxima方法的差异阈值')
    # 密集采样参数
    parser.add_argument('--dense_fps', type=float, default=2.0,
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
    parser.add_argument('--stage', type=str, default='3',
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
    # [修改点] 第四阶段矩阵保存目录，改为 sim_matrices_chamfer_dense
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

    # [修改点] 自动生成各阶段目录，基础目录改为 sim_matrices_cos_rough 和 sim_matrices_chamfer_dense
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
    # [修改点] 生成第四阶段矩阵目录，使用 sim_matrices_chamfer_dense，子目录保留方法后缀
    if args.sim_matrices_dense_dir is None:
        args.sim_matrices_dense_dir = os.path.join(args.output_dir, 'sim_matrices_chamfer_dense', method_suffix + model_suffix)

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

    # 加载模型（阶段1和阶段4需要）
    model = None
    if args.stage in ['1', '4', 'all']:
        try:
            model = ViSiL(pretrained=True, symmetric=is_symmetric).to(device)
            model.eval()
            print("> 模型加载成功")
            # 解释第一阶段特征含义
            print("> 第一阶段提取的特征为：每个关键帧的 9 个区域特征（来自 ResNet50 中间层），"
                  f"形状 [关键帧数, 9, {expected_dims}]，随后可计算帧间余弦相似度。")
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

        # 清理检查点
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

        # 加载区间元数据（若已存在则跳过已处理区间）
        intervals_meta = {}
        if os.path.exists(args.intervals_meta):
            try:
                with open(args.intervals_meta, 'r') as f:
                    intervals_meta = json.load(f)
                print(f"> 加载区间元数据，已有 {len(intervals_meta)} 个区间")
            except Exception as e:
                print(f"> 加载区间元数据失败: {e}")

        # 遍历所有查询-数据库对
        total_pairs = len(query_ids) * len(database_ids)
        pbar = tqdm(total=total_pairs, desc="处理矩阵", unit="pair")
        for q_id in query_ids:
            # 加载查询元数据
            indices_file = os.path.join(actual_frames_dir, q_id, 'indices.json')
            info_file = os.path.join(actual_frames_dir, q_id, 'info.json')
            if os.path.exists(indices_file) and os.path.exists(info_file):
                with open(indices_file, 'r') as f:
                    q_indices = json.load(f)
                with open(info_file, 'r') as f:
                    q_info = json.load(f)
                q_fps = q_info.get('fps', 30.0)
                q_total = q_info.get('total_frames', 0)  # [修改点] 获取视频总帧数
            else:
                pbar.update(len(database_ids))
                continue

            for db_id in database_ids:
                # 加载数据库元数据
                db_indices_file = os.path.join(actual_frames_dir, db_id, 'indices.json')
                db_info_file = os.path.join(actual_frames_dir, db_id, 'info.json')
                if not (os.path.exists(db_indices_file) and os.path.exists(db_info_file)):
                    pbar.update(1)
                    continue
                with open(db_indices_file, 'r') as f:
                    db_indices = json.load(f)
                with open(db_info_file, 'r') as f:
                    db_info = json.load(f)
                db_fps = db_info.get('fps', 30.0)
                db_total = db_info.get('total_frames', 0)  # [修改点] 获取视频总帧数

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
                    # [修改点] 判断当前关键帧对是否孤立
                    q_isolated = True
                    if i > 0 and sim_matrix[i-1, j] >= args.threshold:
                        q_isolated = False
                    if i < len(q_indices)-1 and sim_matrix[i+1, j] >= args.threshold:
                        q_isolated = False

                    db_isolated = True
                    if j > 0 and sim_matrix[i, j-1] >= args.threshold:
                        db_isolated = False
                    if j < len(db_indices)-1 and sim_matrix[i, j+1] >= args.threshold:
                        db_isolated = False

                    isolated = q_isolated and db_isolated

                    if isolated:
                        # [修改点] 孤立关键帧：以当前关键帧为中心前后各扩展1秒
                        q_center = q_indices[i]
                        q_start_frame = max(0, q_center - int(q_fps))
                        q_end_frame = min(q_total - 1, q_center + int(q_fps))

                        db_center = db_indices[j]
                        db_start_frame = max(0, db_center - int(db_fps))
                        db_end_frame = min(db_total - 1, db_center + int(db_fps))

                        # 步长仍按原方式计算
                        q_step = max(1, int(round(q_fps / args.dense_fps)))
                        db_step = max(1, int(round(db_fps / args.dense_fps)))
                    else:
                        # [修改点] 非孤立：原扩展逻辑，但移除强制扩展的 if 语句
                        db_k_start = j
                        while db_k_start > 0 and sim_matrix[i, db_k_start - 1] >= args.threshold:
                            db_k_start -= 1
                        # [修改点] 移除强制扩展

                        db_k_end = j
                        while db_k_end < len(db_indices) - 1 and sim_matrix[i, db_k_end + 1] >= args.threshold:
                            db_k_end += 1
                        # [修改点] 移除强制扩展

                        q_k_start = i
                        while q_k_start > 0 and sim_matrix[q_k_start - 1, j] >= args.threshold:
                            q_k_start -= 1
                        # [修改点] 移除强制扩展

                        q_k_end = i
                        while q_k_end < len(q_indices) - 1 and sim_matrix[q_k_end + 1, j] >= args.threshold:
                            q_k_end += 1
                        # [修改点] 移除强制扩展

                        db_start_frame = db_indices[db_k_start]
                        db_end_frame   = db_indices[db_k_end]
                        q_start_frame  = q_indices[q_k_start]
                        q_end_frame    = q_indices[q_k_end]

                        q_step = max(1, int(round(q_fps / args.dense_fps)))
                        db_step = max(1, int(round(db_fps / args.dense_fps)))

                    interval_pairs.append({
                        'q_start': q_start_frame,
                        'q_end': q_end_frame,
                        'db_start': db_start_frame,
                        'db_end': db_end_frame,
                        'q_step': q_step,
                        'db_step': db_step
                    })

                # 合并重叠的区间对
                merged_pairs = merge_interval_pairs(interval_pairs)

                # 为每个合并后的区间对提取密集帧
                for idx, pair in enumerate(merged_pairs):
                    q_start = pair['q_start']
                    q_end = pair['q_end']
                    db_start = pair['db_start']
                    db_end = pair['db_end']
                    q_step = pair['q_step']
                    db_step = pair['db_step']

                    interval_id = f"{q_id}_{db_id}_m{idx}_q{q_start}_{q_end}_db{db_start}_{db_end}"

                    # 检查是否已处理
                    if interval_id in intervals_meta:
                        continue

                    # 去掉 intervals 层级，直接将区间文件夹放在视频 ID 下
                    q_save_dir = os.path.join(args.dense_frames_dir, 'queries', q_id, interval_id)
                    db_save_dir = os.path.join(args.dense_frames_dir, 'database', db_id, interval_id)

                    # 提取并保存查询区间密集帧
                    q_frames_saved = extract_dense_frames_range_and_save(
                        id_to_path[q_id], q_start, q_end, q_step,
                        q_save_dir, max_size=None
                    )
                    if not q_frames_saved:
                        continue

                    # 提取并保存数据库区间密集帧
                    db_frames_saved = extract_dense_frames_range_and_save(
                        id_to_path[db_id], db_start, db_end, db_step,
                        db_save_dir, max_size=None
                    )
                    if not db_frames_saved:
                        continue

                    # 记录区间元数据
                    intervals_meta[interval_id] = {
                        'query_id': q_id,
                        'db_id': db_id,
                        'q_start_frame': q_start,
                        'q_end_frame': q_end,
                        'q_step': q_step,
                        'db_start_frame': db_start,
                        'db_end_frame': db_end,
                        'db_step': db_step,
                        'q_save_dir': q_save_dir,
                        'db_save_dir': db_save_dir
                    }

                pbar.update(1)
                # 定期保存元数据
                if len(intervals_meta) % 50 == 0:
                    save_interval_metadata(intervals_meta, args.intervals_meta)
        pbar.close()

        save_interval_metadata(intervals_meta, args.intervals_meta)
        stage3_time = time.time() - stage3_start
        print(f"> 第三阶段完成，耗时 {stage3_time:.2f}s，共生成 {len(intervals_meta)} 个区间，密集帧保存至 {args.dense_frames_dir}")

    # ========== 阶段4：密集帧相似度计算与亮线检测 ==========
    if args.stage in ['4', 'all']:
        print("\n" + "=" * 60)
        print("第四阶段：密集帧相似度计算与亮线检测（按查询-数据库对合并区间）")
        print("=" * 60)
        stage4_start = time.time()

        # 加载区间元数据
        if not os.path.exists(args.intervals_meta):
            print(f"> 错误：区间元数据文件不存在 {args.intervals_meta}")
            return
        intervals_meta = load_interval_metadata(args.intervals_meta)

        # [修改点] 按 (query_id, db_id) 分组收集区间信息
        pair_to_intervals = {}  # (q_id, db_id) -> list of interval_meta
        for interval_id, meta in intervals_meta.items():
            key = (meta['query_id'], meta['db_id'])
            if key not in pair_to_intervals:
                pair_to_intervals[key] = []
            pair_to_intervals[key].append(meta)

        # 特征缓存字典（可选）
        feature_cache = {}

        similarities = {}  # query_id -> {db_id: score}
        processed_pairs = set()  # [修改点] 记录已处理的 (q_id, db_id) 对

        # 检查点文件（按对记录）
        stage4_checkpoint = os.path.join(args.heatmaps_dir, 'stage4_checkpoint_pairs.json')
        if os.path.exists(stage4_checkpoint):
            try:
                with open(stage4_checkpoint, 'r') as f:
                    cp = json.load(f)
                similarities = cp.get('similarities', {})
                processed_pairs = set(tuple(p) for p in cp.get('processed_pairs', []))
                print(f"> 加载检查点，已处理 {len(processed_pairs)} 个查询-数据库对")
            except Exception as e:
                print(f"> 检查点加载失败: {e}")

        total_pairs = len(pair_to_intervals)
        pbar = tqdm(total=total_pairs, desc="处理查询-数据库对", unit="pair")
        for (q_id, db_id), intervals_list in pair_to_intervals.items():
            if (q_id, db_id) in processed_pairs:
                pbar.update(1)
                continue

            # [修改点] 收集查询视频的所有帧（去重排序）
            q_frame_dirs = [meta['q_save_dir'] for meta in intervals_list]
            all_q_frames = []  # 存储所有查询帧的 (frame_idx, img_array)
            for dir_path in q_frame_dirs:
                frames_info = get_sorted_frames_from_dir(dir_path)  # 返回 [(idx, img), ...]
                if frames_info is None:
                    continue
                all_q_frames.extend(frames_info)
            if not all_q_frames:
                pbar.update(1)
                continue
            # 按帧索引排序并去重（保留第一次出现的帧）
            all_q_frames.sort(key=lambda x: x[0])
            unique_q_frames = []
            seen = set()
            for idx, img in all_q_frames:
                if idx not in seen:
                    seen.add(idx)
                    unique_q_frames.append((idx, img))
            # 提取查询帧张量
            q_imgs = [img for _, img in unique_q_frames]
            q_frames_np = np.stack(q_imgs, axis=0)  # [T, H, W, 3]
            q_frames_tensor = torch.from_numpy(q_frames_np).permute(0, 3, 1, 2).float()
            if q_frames_tensor.shape[2] != 224 or q_frames_tensor.shape[3] != 224:
                q_frames_tensor = torch.nn.functional.interpolate(
                    q_frames_tensor, size=(224, 224), mode='bilinear', align_corners=False
                )
            q_frames_tensor = q_frames_tensor.permute(0, 2, 3, 1)  # 转回 [T, H, W, C] 以便特征提取

            # [修改点] 收集数据库视频的所有帧（去重排序）
            db_frame_dirs = [meta['db_save_dir'] for meta in intervals_list]
            all_db_frames = []
            for dir_path in db_frame_dirs:
                frames_info = get_sorted_frames_from_dir(dir_path)
                if frames_info is None:
                    continue
                all_db_frames.extend(frames_info)
            if not all_db_frames:
                pbar.update(1)
                continue
            all_db_frames.sort(key=lambda x: x[0])
            unique_db_frames = []
            seen = set()
            for idx, img in all_db_frames:
                if idx not in seen:
                    seen.add(idx)
                    unique_db_frames.append((idx, img))
            db_imgs = [img for _, img in unique_db_frames]
            db_frames_np = np.stack(db_imgs, axis=0)
            db_frames_tensor = torch.from_numpy(db_frames_np).permute(0, 3, 1, 2).float()
            if db_frames_tensor.shape[2] != 224 or db_frames_tensor.shape[3] != 224:
                db_frames_tensor = torch.nn.functional.interpolate(
                    db_frames_tensor, size=(224, 224), mode='bilinear', align_corners=False
                )
            db_frames_tensor = db_frames_tensor.permute(0, 2, 3, 1)

            # 提取特征
            q_feat = extract_features_from_frames(
                model, q_frames_tensor, f"{q_id}_{db_id}_q",  # 临时名称，不保存缓存
                None, device, args.batch_sz, expected_dims, use_cache=False
            )
            db_feat = extract_features_from_frames(
                model, db_frames_tensor, f"{q_id}_{db_id}_db",
                None, device, args.batch_sz, expected_dims, use_cache=False
            )

            if q_feat is None or db_feat is None:
                pbar.update(1)
                continue

            # 计算相似度
            sim_val = compute_dense_similarity(model, q_feat, db_feat)

            # [修改点] 生成一张整体的热图，保存拼接后的矩阵
            heatmap_path = os.path.join(args.heatmaps_dir, q_id, f"{db_id}_merged.png")
            matrix_save_path = os.path.join(args.sim_matrices_dense_dir, q_id, f"{db_id}_merged.npy")
            compute_and_save_heatmap(model, q_feat, db_feat, heatmap_path, matrix_save_path)

            # 聚合分数
            if q_id not in similarities:
                similarities[q_id] = {}
            if db_id not in similarities[q_id] or sim_val > similarities[q_id][db_id]:
                similarities[q_id][db_id] = sim_val

            processed_pairs.add((q_id, db_id))

            # 定期保存检查点
            if len(processed_pairs) % 50 == 0:
                cp_data = {
                    'similarities': similarities,
                    'processed_pairs': list(processed_pairs)
                }
                with open(stage4_checkpoint, 'w') as f:
                    json.dump(cp_data, f)
            pbar.update(1)

        pbar.close()

        # 保存最终相似度文件
        final_sim_file = os.path.join(args.output_dir, f"similarities_{method_suffix}{model_suffix}_th{args.threshold}.json")
        with open(final_sim_file, 'w') as f:
            json.dump(similarities, f, separators=(',', ':'))
        print(f"> 最终相似度保存至: {final_sim_file}")

        # 清理检查点
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