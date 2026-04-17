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
import glob
import contextlib
import sys
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
    merge_db_intervals,
    load_dense_frames_from_dir,
    get_sorted_frames_from_dir,
    extract_full_dense_frames
)
from datasets.generators import VideoGenerator


# [修改点] 数据集预设配置 - 修正 VCDB 的 database_file 路径，使用清洗后的文件
DATASET_PRESETS = {
    "FIVR-5K": {
        "video_dir": "datasets/FIVR-200K",
        "query_file": "datasets/FIVR-200K-Downloader/fivr-5k-queries-filtered.txt",
        "database_file": "datasets/FIVR-200K-Downloader/fivr-5k-database-filtered.txt",
        "pattern": "{id}.mp4"
    },
    "FIVR-200K": {
        "video_dir": "datasets/FIVR-200K",
        "query_file": "datasets/FIVR-200K-Downloader/fivr-200k-queries-filtered.txt",
        "database_file": "datasets/FIVR-200K-Downloader/fivr-200k-database-filtered.txt",
        "pattern": "{id}.mp4"
    },
    "VCDB": {
        "video_dir": "datasets/VCDB/core_dataset/core_dataset",
        "query_file": "datasets/VCDB/core_dataset/vcdb_cleaned_queries.txt",
        "database_file": "datasets/VCDB/core_dataset/vcdb_cleaned_database.txt",
        "pattern": "{id}.flv"
    },
    "CC_WEB_VIDEO": {
        "video_dir": "datasets/CC_WEB_VIDEO",
        "query_file": "datasets/cc_web_video_queries.txt",
        "database_file": "datasets/cc_web_video_database.txt",
        "pattern": "{id}.mp4"
    },
    "SVD": {
        "video_dir": "datasets/SVD",
        "query_file": "datasets/svd_queries.txt",
        "database_file": "datasets/svd_database.txt",
        "pattern": "{id}.mp4"
    },
    "EVVE": {
        "video_dir": "datasets/EVVE",
        "query_file": "datasets/evve_queries.txt",
        "database_file": "datasets/evve_database.txt",
        "pattern": "{id}.mp4"
    }
}


def main():
    parser = argparse.ArgumentParser(description='四阶段视频相似度评估')
    parser.add_argument('--dataset', type=str, default="VCDB",
                        choices=list(DATASET_PRESETS.keys()),
                        help='评估数据集名称')
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--pattern', type=str, default=None)
    parser.add_argument('--query_file', type=str, default=None)
    parser.add_argument('--database_file', type=str, default=None)
    parser.add_argument('--first_stage_method', type=str, default='iframe',
                        choices=['default', '2s', 'local_maxima', 'iframe', 'i_p_mixed'])
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--max_keyframes', type=int, default=0)
    parser.add_argument('--lm_threshold', type=float, default=0.6)
    parser.add_argument('--dense_fps', type=float, default=1.0)
    parser.add_argument('--batch_sz', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--cpu_only', action='store_true', default=False)
    parser.add_argument('--similarity_function', type=str, default='chamfer',
                        choices=["chamfer", "symmetric_chamfer"])
    parser.add_argument('--frames_dir', type=str, default='frames1')
    parser.add_argument('--features_dir', type=str, default='features1')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--ffprobe_path', type=str, default='ffprobe')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['1', '2', '3', '4', 'all'])
    parser.add_argument('--sim_matrices_dir', type=str, default=None)
    parser.add_argument('--dense_frames_dir', type=str, default=None)
    parser.add_argument('--intervals_meta', type=str, default=None)
    parser.add_argument('--heatmaps_dir', type=str, default=None)
    parser.add_argument('--dense_features_dir', type=str, default=None)
    parser.add_argument('--sim_matrices_dense_dir', type=str, default=None)
    parser.add_argument('--only_first_stage', action='store_true', default=False)

    args = parser.parse_args()

    if args.only_first_stage:
        args.stage = '1'
        print("> 检测到 --only_first_stage，已自动转换为 --stage 1")

    # 应用数据集预设
    preset = DATASET_PRESETS.get(args.dataset, {})
    if args.video_dir is None:
        args.video_dir = preset.get("video_dir", "datasets")
    if args.pattern is None:
        args.pattern = preset.get("pattern", "{id}.mp4")
    if args.query_file is None:
        args.query_file = preset.get("query_file")
    if args.database_file is None:
        args.database_file = preset.get("database_file")

    if args.output_dir == 'output':
        args.output_dir = f"output_{args.dataset}"
        print(f"> 输出目录自动设置为: {args.output_dir}")

    if not os.path.isabs(args.frames_dir):
        args.frames_dir = os.path.join(args.output_dir, args.frames_dir)
    if not os.path.isabs(args.features_dir):
        args.features_dir = os.path.join(args.output_dir, args.features_dir)

    is_symmetric = 'symmetric' in args.similarity_function
    expected_dims = 512 if is_symmetric else 3840
    model_suffix = '_sym' if is_symmetric else '_v'

    method_suffix = args.first_stage_method
    if args.first_stage_method == 'local_maxima':
        method_suffix = f'local_maxima_{args.lm_threshold}'

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
    if args.sim_matrices_dense_dir is None:
        args.sim_matrices_dense_dir = os.path.join(args.output_dir, 'sim_matrices_chamfer_dense', method_suffix + model_suffix)

    for d in [args.sim_matrices_dir, args.dense_frames_dir, args.heatmaps_dir, args.dense_features_dir, args.sim_matrices_dense_dir]:
        os.makedirs(d, exist_ok=True)

    if args.cpu_only or not torch.cuda.is_available():
        device = torch.device('cpu')
        args.batch_sz = min(args.batch_sz, 4)
        print("> 使用CPU")
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"> 使用GPU: {device}")

    total_start = time.time()

    id_to_path = {}
    query_ids = []
    database_ids = []

    # [修改点] 构建列表 (VCDB 直接使用清洗后的文本文件，不再依赖 VCDB 类的 get_queries/get_database)
    if args.dataset == 'VCDB':
        # 评估阶段仍需要 VCDB 类提供 ground truth，但使用清洗后的 pickle
        from datasets import VCDB
        vcdb_root = "datasets/VCDB/core_dataset"
        vcdb = VCDB(root_dir=vcdb_root, pickle_path=os.path.join(vcdb_root, 'vcdb_cleaned.pickle'))
        print(f"> 构建 VCDB 查询/数据库列表 (使用清洗后文件)...")

        if args.query_file and os.path.exists(args.query_file):
            with open(args.query_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        query_id = parts[0]          # 事件名
                        video_path = ' '.join(parts[1:])
                        id_to_path[query_id] = video_path
                        query_ids.append(query_id)
            print(f"> 查询数: {len(query_ids)} (事件名)")
        else:
            print(f"> 错误: 查询文件不存在 {args.query_file}")
            return

        if args.database_file and os.path.exists(args.database_file):
            with open(args.database_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        vid = parts[0]
                        path = ' '.join(parts[1:])
                        id_to_path[vid] = path
                        database_ids.append(vid)
            print(f"> 数据库视频数: {len(database_ids)}")
        else:
            print(f"> 错误: 数据库文件不存在 {args.database_file}")
            return

    else:
        if args.query_file and os.path.exists(args.query_file):
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
        else:
            print(f"> 警告: 未指定有效的查询列表文件")

        if args.database_file and os.path.exists(args.database_file):
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
        else:
            print(f"> 警告: 未指定有效的数据库列表文件")

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

    # [修改点] 分批特征提取函数，加入 224x224 缩放，并检查缓存
    def _extract_features_from_list(model, frames_list, video_id, features_dir, device, batch_size, expected_dims):
        """
        frames_list: list of numpy array (H, W, 3) uint8, RGB
        返回: Tensor [T, 9, D] 或 None
        """
        feat_path = os.path.join(features_dir, video_id, f"{video_id}.npy")
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)

        # 检查已有有效缓存
        if os.path.exists(feat_path):
            try:
                feat_np = np.load(feat_path)
                feat = torch.from_numpy(feat_np).float()
                if feat.dim() == 3 and feat.shape[1] == 9 and feat.shape[2] == expected_dims:
                    return feat
            except:
                pass

        if not frames_list:
            return None

        # [修改点] 将所有帧缩放至 224x224
        resized_frames = []
        for img in frames_list:
            if img.shape[0] != 224 or img.shape[1] != 224:
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(img)
        frames_list = resized_frames

        total_frames = len(frames_list)
        features_list = []
        model.eval()

        with torch.no_grad():
            for i in range(0, total_frames, batch_size):
                end = min(i + batch_size, total_frames)
                batch_frames_np = np.stack(frames_list[i:end], axis=0)  # [B, H, W, 3] uint8
                batch_tensor = torch.from_numpy(batch_frames_np).float().to(device)  # [B, H, W, 3] float
                batch_feat = model.extract_features(batch_tensor)  # [B, 9, D]
                features_list.append(batch_feat.cpu())
                del batch_tensor, batch_feat
                if device.type == 'cpu':
                    gc.collect()

        if not features_list:
            return None

        features = torch.cat(features_list, dim=0)  # [T, 9, D]
        if features.shape[1] != 9 or features.shape[2] != expected_dims:
            return None

        np.save(feat_path, features.cpu().numpy())
        return features

    # [修改点] 阶段1：流式处理，严格检查缓存，明确区分帧提取和特征提取子阶段
    def run_stage1(model, device):
        print("\n" + "=" * 60)
        print("第一阶段：关键帧提取与特征计算")
        print("=" * 60)
        stage1_start = time.time()

        log_file = os.path.join(args.output_dir, 'stage1_failures.log')
        fail_log = open(log_file, 'w', encoding='utf-8')
        fail_log.write(f"Stage1 failures for dataset {args.dataset}\n")
        fail_log.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fail_log.write("=" * 80 + "\n")

        def check_frames_exist(video_id):
            """检查帧缓存是否完整（indices.json 及所有对应 jpg 均存在）"""
            sub_dir = os.path.join(actual_frames_dir, video_id)
            indices_path = os.path.join(sub_dir, 'indices.json')
            if not os.path.isfile(indices_path):
                return False
            try:
                with open(indices_path, 'r') as f:
                    indices = json.load(f)
                for idx in indices:
                    jpg_path = os.path.join(sub_dir, f"{idx:06d}.jpg")
                    if not os.path.isfile(jpg_path):
                        return False
                return True
            except:
                return False

        def load_frames_from_disk(video_id):
            """从已保存的 JPG 加载帧，返回 (indices, frames_np_list)"""
            sub_dir = os.path.join(actual_frames_dir, video_id)
            with open(os.path.join(sub_dir, 'indices.json'), 'r') as f:
                indices = json.load(f)
            frames = []
            for idx in indices:
                jpg_path = os.path.join(sub_dir, f"{idx:06d}.jpg")
                img = cv2.imread(jpg_path)
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    return None, None
            return indices, frames

        def feature_exists(video_id):
            """检查特征文件是否存在且维度正确"""
            feat_path = os.path.join(actual_features_dir, video_id, f"{video_id}.npy")
            if not os.path.exists(feat_path):
                return False
            try:
                feat_np = np.load(feat_path)
                feat = torch.from_numpy(feat_np).float()
                if feat.dim() == 3 and feat.shape[1] == 9 and feat.shape[2] == expected_dims:
                    return True
            except:
                pass
            return False

        # [修改点] 处理单个视频的辅助函数：帧提取（如需要）+ 特征提取（如需要）
        def process_video(video_id, video_type):
            video_path = id_to_path.get(video_id)
            if not video_path or not os.path.exists(video_path):
                fail_log.write(f"[{video_type}] {video_id}: 视频文件不存在或路径无效 ({video_path})\n")
                return False, False  # (frame_success, feat_success)

            frames_ready = False
            feat_ready = False

            # 先检查特征缓存，若已存在则直接成功
            if feature_exists(video_id):
                return True, True

            # 否则需要帧数据
            frames_np_list = None
            try:
                if check_frames_exist(video_id):
                    indices, frames_np_list = load_frames_from_disk(video_id)
                    if indices is not None and frames_np_list is not None:
                        frames_ready = True
                else:
                    # 抽取关键帧
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                            indices, frames_np_list, _, _ = extract_keyframes(
                                video_path, video_id, actual_frames_dir, args.first_stage_method,
                                args.max_keyframes, args.lm_threshold, args.ffprobe_path
                            )
                    if indices is not None and frames_np_list is not None and len(frames_np_list) > 0:
                        frames_ready = True
                    else:
                        fail_log.write(f"[{video_type}] {video_id}: 关键帧抽取失败\n")
                        return False, False

                if frames_ready:
                    # 提取特征
                    feat = _extract_features_from_list(
                        model, frames_np_list, video_id, actual_features_dir, device,
                        args.batch_sz, expected_dims
                    )
                    if feat is not None:
                        feat_ready = True
                    else:
                        fail_log.write(f"[{video_type}] {video_id}: 特征提取返回 None\n")
                return frames_ready, feat_ready
            except Exception as e:
                fail_log.write(f"[{video_type}] {video_id}: 处理异常 - {type(e).__name__}: {e}\n")
                return False, False
            finally:
                del frames_np_list
                gc.collect()

        # [修改点] 子阶段1：帧提取/加载（若缓存不存在）
        print("\n[子阶段1] 帧提取/加载...")
        # 第一轮：确保所有视频的帧缓存存在
        need_frame_db = []
        need_frame_query = []
        for db_id in database_ids:
            if not check_frames_exist(db_id):
                need_frame_db.append(db_id)
        for q_id in query_ids:
            if not check_frames_exist(q_id):
                need_frame_query.append(q_id)

        if need_frame_db or need_frame_query:
            print(f"  需要抽取帧: 数据库 {len(need_frame_db)} 个, 查询 {len(need_frame_query)} 个")
            # 抽取数据库缺失帧
            for db_id in tqdm(need_frame_db, desc="数据库帧抽取", unit="vid"):
                video_path = id_to_path.get(db_id)
                if not video_path or not os.path.exists(video_path):
                    fail_log.write(f"[DATABASE] {db_id}: 视频文件不存在 ({video_path})\n")
                    continue
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        extract_keyframes(
                            video_path, db_id, actual_frames_dir, args.first_stage_method,
                            args.max_keyframes, args.lm_threshold, args.ffprobe_path
                        )
            # 抽取查询缺失帧
            for q_id in tqdm(need_frame_query, desc="查询帧抽取", unit="vid"):
                video_path = id_to_path.get(q_id)
                if not video_path or not os.path.exists(video_path):
                    fail_log.write(f"[QUERY] {q_id}: 视频文件不存在 ({video_path})\n")
                    continue
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        extract_keyframes(
                            video_path, q_id, actual_frames_dir, args.first_stage_method,
                            args.max_keyframes, args.lm_threshold, args.ffprobe_path
                        )
        else:
            print("  所有帧缓存已存在，跳过帧抽取。")

        # [修改点] 子阶段2：特征提取（利用缓存），增加缓存命中统计
        print("\n[子阶段2] 特征提取...")
        db_success = 0
        db_failed = 0
        db_cached = 0
        for db_id in tqdm(database_ids, desc="数据库特征提取", unit="vid"):
            if feature_exists(db_id):
                db_success += 1
                db_cached += 1
                continue
            frames_ok, feat_ok = process_video(db_id, "DATABASE")
            if feat_ok:
                db_success += 1
            else:
                db_failed += 1

        query_success = 0
        query_failed = 0
        query_cached = 0
        for q_id in tqdm(query_ids, desc="查询特征提取", unit="vid"):
            if feature_exists(q_id):
                query_success += 1
                query_cached += 1
                continue
            frames_ok, feat_ok = process_video(q_id, "QUERY")
            if feat_ok:
                query_success += 1
            else:
                query_failed += 1

        fail_log.close()
        stage1_time = time.time() - stage1_start
        print(f"\n> 第一阶段完成，耗时 {stage1_time:.2f}s")
        print(f"   数据库: 成功 {db_success}/{len(database_ids)} (缓存命中 {db_cached})，失败 {db_failed}")
        print(f"   查询: 成功 {query_success}/{len(query_ids)} (缓存命中 {query_cached})，失败 {query_failed}")
        if db_failed > 0 or query_failed > 0:
            print(f"   失败详情已写入: {log_file}")
        return stage1_time

    # [修改点] 第二阶段：增加缓存命中统计
    def run_stage2():
        print("\n" + "=" * 60)
        print("第二阶段：关键帧相似度矩阵计算与存储")
        print("=" * 60)
        stage2_start = time.time()

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
        cached_pairs = 0
        computed_pairs = 0
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
                    cached_pairs += 1
                    continue
                db_feat = load_features(db_id, actual_features_dir, expected_dims)
                if db_feat is None:
                    pbar.update(1)
                    continue
                # [修改点] 检查矩阵是否已存在
                existing = load_similarity_matrix(q_id, db_id, args.sim_matrices_dir)
                if existing is not None:
                    processed_pairs.add(pair_key)
                    pbar.update(1)
                    cached_pairs += 1
                    continue
                sim_matrix = compute_keyframe_similarity_matrix(q_feat, db_feat)
                save_similarity_matrix(sim_matrix, q_id, db_id, args.sim_matrices_dir)
                processed_pairs.add(pair_key)
                computed_pairs += 1
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
        print(f"> 第二阶段完成，耗时 {stage2_time:.2f}s")
        print(f"   总对数: {total_pairs}, 缓存命中: {cached_pairs}, 新计算: {computed_pairs}")
        print(f"   矩阵保存至 {args.sim_matrices_dir}")
        return stage2_time

    # 初始化时间变量
    stage1_time = 0.0
    stage2_time = 0.0
    stage3_time = 0.0
    stage4_time = 0.0

    # ========== 阶段1 ==========
    if args.stage in ['1', 'all']:
        model = ViSiL(pretrained=True, symmetric=is_symmetric).to(device)
        model.eval()
        print("> 模型加载成功")
        stage1_time = run_stage1(model, device)

    # ========== 阶段2 ==========
    if args.stage in ['2', 'all']:
        if args.stage != '2' and args.stage != 'all':
            if 'model' not in locals() or model is None:
                model = ViSiL(pretrained=True, symmetric=is_symmetric).to(device)
                model.eval()
                print("> 模型加载成功")
        stage2_time = run_stage2()

    # ========== 阶段3+4 ==========
    if args.stage in ['3', 'all']:
        if 'model' not in locals() or model is None:
            try:
                model = ViSiL(pretrained=True, symmetric=is_symmetric).to(device)
                model.eval()
                print("> 模型加载成功")
            except Exception as e:
                print(f"错误: 模型加载失败 - {e}")
                return

        # 检查阶段1特征是否存在
        need_stage1 = False
        if query_ids and database_ids:
            test_q = query_ids[0]
            test_db = database_ids[0]
            q_feat_path = os.path.join(actual_features_dir, test_q, f"{test_q}.npy")
            db_feat_path = os.path.join(actual_features_dir, test_db, f"{test_db}.npy")
            if not (os.path.exists(q_feat_path) and os.path.exists(db_feat_path)):
                need_stage1 = True
        else:
            need_stage1 = True

        if need_stage1:
            print("> 检测到阶段1特征缺失，自动执行阶段1...")
            stage1_time = run_stage1(model, device)

        # 检查阶段2矩阵是否存在
        sim_mat_files = []
        if os.path.exists(args.sim_matrices_dir):
            for root, dirs, files in os.walk(args.sim_matrices_dir):
                for f in files:
                    if f.endswith('.npy'):
                        sim_mat_files.append(f)
                        break
                if sim_mat_files:
                    break
        need_stage2 = len(sim_mat_files) == 0

        if need_stage2:
            print("> 检测到阶段2矩阵缺失，自动执行阶段2...")
            stage2_time = run_stage2()

        # 阶段3：密集帧区间提取
        print("\n" + "=" * 60)
        print("第三阶段：候选区间定位与密集帧提取")
        print("=" * 60)
        stage3_start = time.time()

        intervals_meta = {}
        if os.path.exists(args.intervals_meta):
            try:
                with open(args.intervals_meta, 'r') as f:
                    intervals_meta = json.load(f)
                print(f"> 加载区间元数据，已有 {len(intervals_meta)} 个区间")
            except Exception as e:
                print(f"> 加载区间元数据失败: {e}")

        query_full_frames_extracted = set()

        total_pairs = len(query_ids) * len(database_ids)
        pbar = tqdm(total=total_pairs, desc="处理矩阵", unit="pair")
        for q_id in query_ids:
            indices_file = os.path.join(actual_frames_dir, q_id, 'indices.json')
            info_file = os.path.join(actual_frames_dir, q_id, 'info.json')
            if os.path.exists(indices_file) and os.path.exists(info_file):
                with open(indices_file, 'r') as f:
                    q_indices = json.load(f)
                with open(info_file, 'r') as f:
                    q_info = json.load(f)
                q_fps = q_info.get('fps', 30.0)
                q_total = q_info.get('total_frames', 0)
            else:
                pbar.update(len(database_ids))
                continue

            if q_id not in query_full_frames_extracted:
                num_frames = extract_full_dense_frames(id_to_path[q_id], q_id, args.dense_frames_dir, args.dense_fps)
                if num_frames > 0:
                    query_full_frames_extracted.add(q_id)

            for db_id in database_ids:
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
                db_total = db_info.get('total_frames', 0)

                sim_matrix = load_similarity_matrix(q_id, db_id, args.sim_matrices_dir)
                if sim_matrix is None:
                    pbar.update(1)
                    continue

                candidate_pairs = find_candidate_pairs(sim_matrix, args.threshold)
                if not candidate_pairs:
                    pbar.update(1)
                    continue

                db_intervals = []
                for i, j in candidate_pairs:
                    db_isolated = True
                    if j > 0 and sim_matrix[i, j-1] >= args.threshold:
                        db_isolated = False
                    if j < len(db_indices)-1 and sim_matrix[i, j+1] >= args.threshold:
                        db_isolated = False

                    if db_isolated:
                        db_center = db_indices[j]
                        db_start = max(0, db_center - int(db_fps))
                        db_end = min(db_total - 1, db_center + int(db_fps))
                    else:
                        db_k_start = j
                        while db_k_start > 0 and sim_matrix[i, db_k_start - 1] >= args.threshold:
                            db_k_start -= 1
                        db_k_end = j
                        while db_k_end < len(db_indices) - 1 and sim_matrix[i, db_k_end + 1] >= args.threshold:
                            db_k_end += 1
                        db_start = db_indices[db_k_start]
                        db_end = db_indices[db_k_end]

                    step = max(1, int(round(db_fps / args.dense_fps)))
                    db_intervals.append({
                        'db_start': db_start,
                        'db_end': db_end,
                        'db_step': step
                    })

                merged_db = merge_db_intervals(db_intervals)

                for idx, db_int in enumerate(merged_db):
                    db_start = db_int['db_start']
                    db_end = db_int['db_end']
                    db_step = db_int['db_step']

                    interval_id = f"{q_id}_{db_id}_db{db_start}_{db_end}_step{db_step}"
                    db_save_dir = os.path.join(args.dense_frames_dir, 'database', db_id, interval_id)

                    if interval_id in intervals_meta:
                        continue

                    db_frames_saved = extract_dense_frames_range_and_save(
                        id_to_path[db_id], db_start, db_end, db_step,
                        db_save_dir, max_size=None
                    )
                    if db_frames_saved == 0:
                        continue

                    intervals_meta[interval_id] = {
                        'query_id': q_id,
                        'db_id': db_id,
                        'db_start_frame': db_start,
                        'db_end_frame': db_end,
                        'db_step': db_step,
                        'db_save_dir': db_save_dir,
                        'query_full_dir': os.path.join(args.dense_frames_dir, 'queries', q_id, 'full')
                    }

                pbar.update(1)
                if len(intervals_meta) % 50 == 0:
                    save_interval_metadata(intervals_meta, args.intervals_meta)
        pbar.close()

        save_interval_metadata(intervals_meta, args.intervals_meta)
        stage3_time = time.time() - stage3_start
        print(f"> 第三阶段完成，耗时 {stage3_time:.2f}s，共生成 {len(intervals_meta)} 个数据库区间")

        if len(intervals_meta) == 0:
            print("> 警告：未找到任何候选区间，跳过第四阶段和评估。")
            return

        # 阶段4：密集帧相似度计算
        print("\n" + "=" * 60)
        print("第四阶段：密集帧相似度计算与亮线检测")
        print("=" * 60)
        stage4_start = time.time()

        pair_to_intervals = {}
        for interval_id, meta in intervals_meta.items():
            key = (meta['query_id'], meta['db_id'])
            if key not in pair_to_intervals:
                pair_to_intervals[key] = []
            pair_to_intervals[key].append(meta)

        feature_cache = {}
        similarities = {}
        processed_pairs = set()

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

            # 加载查询全局特征
            query_feat_key = f"query_full_{q_id}"
            if query_feat_key in feature_cache:
                q_feat = feature_cache[query_feat_key]
            else:
                query_full_dir = intervals_list[0]['query_full_dir']
                query_feat_path = os.path.join(args.dense_features_dir, 'queries', q_id, f"{q_id}_full.npy")
                if os.path.exists(query_feat_path):
                    try:
                        q_feat_np = np.load(query_feat_path)
                        q_feat = torch.from_numpy(q_feat_np).float()
                        if q_feat.dim() == 3 and q_feat.shape[1] == 9 and q_feat.shape[2] == expected_dims:
                            feature_cache[query_feat_key] = q_feat
                        else:
                            q_feat = None
                    except:
                        q_feat = None
                else:
                    q_feat = None

                if q_feat is None:
                    q_frames_info = get_sorted_frames_from_dir(query_full_dir)
                    if q_frames_info is None:
                        pbar.update(1)
                        continue
                    q_imgs = [img for _, img in q_frames_info]
                    # 分批提取特征（内部已包含 224 缩放）
                    q_feat = _extract_features_from_list(
                        model, q_imgs, f"{q_id}_full", args.dense_features_dir,
                        device, args.batch_sz, expected_dims
                    )
                    if q_feat is None:
                        pbar.update(1)
                        continue
                    feature_cache[query_feat_key] = q_feat

            # 加载数据库区间合并特征
            db_feat_key = f"db_{db_id}_" + "_".join(sorted([meta['db_save_dir'] for meta in intervals_list]))
            if db_feat_key in feature_cache:
                db_feat = feature_cache[db_feat_key]
            else:
                db_merged_feat_path = os.path.join(args.dense_features_dir, 'database', db_id, f"{db_id}_merged.npy")
                if os.path.exists(db_merged_feat_path):
                    try:
                        db_feat_np = np.load(db_merged_feat_path)
                        db_feat = torch.from_numpy(db_feat_np).float()
                        if db_feat.dim() == 3 and db_feat.shape[1] == 9 and db_feat.shape[2] == expected_dims:
                            feature_cache[db_feat_key] = db_feat
                        else:
                            db_feat = None
                    except:
                        db_feat = None
                else:
                    db_feat = None

                if db_feat is None:
                    all_db_frames = []
                    for meta in intervals_list:
                        db_dir = meta['db_save_dir']
                        frames_info = get_sorted_frames_from_dir(db_dir)
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
                    db_feat = _extract_features_from_list(
                        model, db_imgs, f"{db_id}_merged", args.dense_features_dir,
                        device, args.batch_sz, expected_dims
                    )
                    if db_feat is None:
                        pbar.update(1)
                        continue
                    feature_cache[db_feat_key] = db_feat

            sim_val = compute_dense_similarity(model, q_feat, db_feat)

            heatmap_path = os.path.join(args.heatmaps_dir, q_id, f"{db_id}_merged.png")
            matrix_save_path = os.path.join(args.sim_matrices_dense_dir, q_id, f"{db_id}_merged.npy")
            compute_and_save_heatmap(model, q_feat, db_feat, heatmap_path, matrix_save_path)

            if q_id not in similarities:
                similarities[q_id] = {}
            if db_id not in similarities[q_id] or sim_val > similarities[q_id][db_id]:
                similarities[q_id][db_id] = sim_val

            processed_pairs.add((q_id, db_id))

            if len(processed_pairs) % 50 == 0:
                cp_data = {
                    'similarities': similarities,
                    'processed_pairs': list(processed_pairs)
                }
                with open(stage4_checkpoint, 'w') as f:
                    json.dump(cp_data, f)
            pbar.update(1)

        pbar.close()

        final_sim_file = os.path.join(args.output_dir, f"similarities_{method_suffix}{model_suffix}_th{args.threshold}.json")
        with open(final_sim_file, 'w') as f:
            json.dump(similarities, f, separators=(',', ':'))
        print(f"> 最终相似度保存至: {final_sim_file}")

        if os.path.exists(stage4_checkpoint):
            os.remove(stage4_checkpoint)

        stage4_time = time.time() - stage4_start
        total_time = time.time() - total_start

        # 评估
        print("\n" + "=" * 60)
        print("评估")
        print("=" * 60)
        try:
            if 'FIVR' in args.dataset:
                from datasets import FIVR
                version = args.dataset.split('-')[1].lower() if '-' in args.dataset else '5k'
                dataset = FIVR(version=version)
            elif args.dataset == 'VCDB':
                from datasets import VCDB
                vcdb_root = "datasets/VCDB/core_dataset"
                dataset = VCDB(root_dir=vcdb_root, pickle_path=os.path.join(vcdb_root, 'vcdb_cleaned.pickle'))
            else:
                raise ValueError(f"未知数据集: {args.dataset}")
            eval_results = dataset.evaluate(similarities)
            eval_file = os.path.join(args.output_dir, f"evaluation_{method_suffix}{model_suffix}_th{args.threshold}.json")
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            if args.dataset == 'VCDB':
                print(f"> mAP: {eval_results.get('mAP', 0):.4f}")
            else:
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
            'stage1_time': stage1_time,
            'stage2_time': stage2_time,
            'stage3_time': stage3_time,
            'stage4_time': stage4_time,
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