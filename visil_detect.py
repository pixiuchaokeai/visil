"""
visil_detect.py

分阶段视频拷贝检测评估脚本（基于 ViSiL 关键帧特征）：

阶段1：关键帧提取 + 特征提取 + 帧间 Chamfer 相似度矩阵计算 + 矩阵统计量记录
阶段2：基于矩阵计算视频级相似度（经 VideoComperator CNN）
阶段3：检测评估（直接阈值，输出 mAP 及二分类指标，含视频级和多种矩阵统计量方式）
阶段4：热图输出（帧间相似度矩阵热图，自动去重）

可通过 --stage 参数控制执行阶段，便于调节检测阈值多次评估。
"""

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
import contextlib
import sys
import glob
warnings.filterwarnings("ignore")

os.environ['OPENCV_FFMPEG_LOGLEVEL'] = 'quiet'
try:
    import cv2
    cv2.setLogLevel(0)
except:
    pass

from model.visil import ViSiL
from stage_utils import (
    extract_keyframes,
    load_features,
    compute_and_save_heatmap,
)


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
        "query_file": None,
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
    parser = argparse.ArgumentParser(description='分阶段视频拷贝检测评估 (ViSiL)')
    parser.add_argument('--dataset', type=str, default="VCDB",
                        choices=list(DATASET_PRESETS.keys()),
                        help='评估数据集名称')
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--pattern', type=str, default=None)
    parser.add_argument('--query_file', type=str, default=None)
    parser.add_argument('--database_file', type=str, default=None)
    parser.add_argument('--first_stage_method', type=str, default='iframe',
                        choices=['default', '2s', 'local_maxima', 'iframe', 'i_p_mixed', 'shot'])
    parser.add_argument('--max_keyframes', type=int, default=0)
    parser.add_argument('--lm_threshold', type=float, default=0.6)
    parser.add_argument('--batch_sz', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--cpu_only', action='store_true', default=False)
    parser.add_argument('--similarity_function', type=str, default='chamfer',
                        choices=["chamfer", "symmetric_chamfer"])
    parser.add_argument('--frames_dir', type=str, default='frames1')
    parser.add_argument('--features_dir', type=str, default='features1')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--ffprobe_path', type=str, default='ffprobe')
    # [修改点] 重命名视频级相似度阈值参数
    parser.add_argument('--video_threshold', type=float, default=-0.5,
                        help='二分类检测阈值（视频级相似度）')
    # [修改点] 重命名帧间矩阵最大值阈值参数
    parser.add_argument('--max_threshold', type=float, default=0.06,
                        help='帧间相似度矩阵检测阈值，若矩阵中存在任意值 >= 该阈值则判定为同源')
    # [修改点] 双向平均阈值参数
    parser.add_argument('--algebraic_threshold', type=float, default=0.0325,
                        help='行/列最大值代数均值阈值，若均值 >= 该阈值则判定为同源')
    parser.add_argument('--harmonic_threshold', type=float, default=0.03125,
                        help='行/列最大值调和均值阈值，若均值 >= 该阈值则判定为同源')
    # [修改点] 删除归一化阈值参数（不再使用）
    parser.add_argument('--sim_matrices_dir', type=str, default=None)
    parser.add_argument('--heatmaps_dir', type=str, default=None)
    parser.add_argument('--stage', type=str, default='3',
                        choices=['1', '2', '3', '4', 'all'],
                        help='执行阶段：1-矩阵计算，2-视频相似度，3-检测评估，4-热图输出，all-全部')

    args = parser.parse_args()

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

    if args.output_dir is None:
        args.output_dir = f"output_{args.dataset}_visil"
    print(f"> 输出目录: {args.output_dir}")

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
        args.sim_matrices_dir = os.path.join(args.output_dir, 'sim_matrices_chamfer', method_suffix + model_suffix)
    if args.heatmaps_dir is None:
        args.heatmaps_dir = os.path.join(args.output_dir, 'heatmaps', method_suffix + model_suffix)

    os.makedirs(args.sim_matrices_dir, exist_ok=True)
    os.makedirs(args.heatmaps_dir, exist_ok=True)

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

    # [修改点] VCDB 数据集特殊处理：查询集从 pickle 的 query_to_database 键获取
    if args.dataset == 'VCDB':
        from datasets import VCDB
        vcdb_root = "datasets/VCDB/core_dataset"
        dataset = VCDB(root_dir=vcdb_root, pickle_path=os.path.join(vcdb_root, 'vcdb_cleaned.pickle'))
        print(f"> VCDB 数据集加载成功")

        query_ids = dataset.get_queries()
        print(f"> 查询视频数: {len(query_ids)}")

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

        for qid in query_ids:
            if qid not in id_to_path:
                path_candidate = os.path.join(args.video_dir, args.pattern.format(id=qid))
                if not os.path.exists(path_candidate):
                    base = os.path.splitext(path_candidate)[0]
                    for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv']:
                        alt = base + ext
                        if os.path.exists(alt):
                            path_candidate = alt
                            break
                id_to_path[qid] = path_candidate

        valid_queries = [q for q in query_ids if q in id_to_path and os.path.exists(id_to_path[q])]
        if len(valid_queries) < len(query_ids):
            print(f"> 警告: {len(query_ids) - len(valid_queries)} 个查询视频路径无效，将被忽略")
        query_ids = valid_queries
        print(f"> 有效查询视频数: {len(query_ids)}")

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

    # 加载模型（阶段1、2、4都需要）
    if args.stage in ['1', '2', '4', 'all']:
        model = ViSiL(pretrained=True, symmetric=is_symmetric).to(device)
        model.eval()
        print("> 模型加载成功")

    # [修改点] 定义矩阵统计量缓存文件路径（保留均值、标准差等统计量）
    matrix_stats_cache_file = os.path.join(args.output_dir, f"matrix_stats_{method_suffix}{model_suffix}.json")

    # ========== 阶段1 ==========
    if args.stage in ['1', 'all']:
        print("\n" + "=" * 60)
        print("阶段1：关键帧提取、特征计算与帧间相似度矩阵及统计量")
        print("=" * 60)
        stage1_start = time.time()

        log_file = os.path.join(args.output_dir, 'stage1_failures.log')
        fail_log = open(log_file, 'w', encoding='utf-8')

        def check_frames_exist(video_id):
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

        def feature_exists(video_id):
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

        def process_video(video_id, video_type):
            video_path = id_to_path.get(video_id)
            if not video_path or not os.path.exists(video_path):
                fail_log.write(f"[{video_type}] {video_id}: 视频文件不存在\n")
                return False

            if feature_exists(video_id):
                return True

            if not check_frames_exist(video_id):
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        indices, frames_np_list, _, _ = extract_keyframes(
                            video_path, video_id, actual_frames_dir, args.first_stage_method,
                            args.max_keyframes, args.lm_threshold, args.ffprobe_path
                        )
                if indices is None or frames_np_list is None or len(frames_np_list) == 0:
                    fail_log.write(f"[{video_type}] {video_id}: 关键帧抽取失败\n")
                    return False
            else:
                sub_dir = os.path.join(actual_frames_dir, video_id)
                with open(os.path.join(sub_dir, 'indices.json'), 'r') as f:
                    indices = json.load(f)
                frames_np_list = []
                for idx in indices:
                    jpg_path = os.path.join(sub_dir, f"{idx:06d}.jpg")
                    img = cv2.imread(jpg_path)
                    if img is not None:
                        frames_np_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    else:
                        fail_log.write(f"[{video_type}] {video_id}: JPG 读取失败\n")
                        return False

            feat_path = os.path.join(actual_features_dir, video_id, f"{video_id}.npy")
            os.makedirs(os.path.dirname(feat_path), exist_ok=True)
            resized_frames = []
            for img in frames_np_list:
                if img.shape[0] != 224 or img.shape[1] != 224:
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
                resized_frames.append(img)
            features_list = []
            with torch.no_grad():
                for i in range(0, len(resized_frames), args.batch_sz):
                    batch_np = np.stack(resized_frames[i:i+args.batch_sz], axis=0)
                    batch_tensor = torch.from_numpy(batch_np).float().to(device)
                    batch_feat = model.extract_features(batch_tensor)
                    features_list.append(batch_feat.cpu())
            if not features_list:
                fail_log.write(f"[{video_type}] {video_id}: 特征提取失败\n")
                return False
            features = torch.cat(features_list, dim=0)
            np.save(feat_path, features.cpu().numpy())
            return True

        print("\n[子阶段1] 帧提取/加载...")
        need_frame_db = [db_id for db_id in database_ids if not check_frames_exist(db_id)]
        need_frame_query = [q_id for q_id in query_ids if not check_frames_exist(q_id)]
        if need_frame_db or need_frame_query:
            print(f"  需要抽取帧: 数据库 {len(need_frame_db)} 个, 查询 {len(need_frame_query)} 个")
            for db_id in tqdm(need_frame_db, desc="数据库帧抽取", unit="vid"):
                video_path = id_to_path.get(db_id)
                if not video_path or not os.path.exists(video_path):
                    continue
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        extract_keyframes(video_path, db_id, actual_frames_dir, args.first_stage_method,
                                          args.max_keyframes, args.lm_threshold, args.ffprobe_path)
            for q_id in tqdm(need_frame_query, desc="查询帧抽取", unit="vid"):
                video_path = id_to_path.get(q_id)
                if not video_path or not os.path.exists(video_path):
                    continue
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        extract_keyframes(video_path, q_id, actual_frames_dir, args.first_stage_method,
                                          args.max_keyframes, args.lm_threshold, args.ffprobe_path)
        else:
            print("  所有帧缓存已存在，跳过帧抽取。")

        print("\n[子阶段2] 特征提取...")
        db_success, query_success = 0, 0
        for db_id in tqdm(database_ids, desc="数据库特征", unit="vid"):
            if process_video(db_id, "DATABASE"):
                db_success += 1
        for q_id in tqdm(query_ids, desc="查询特征", unit="vid"):
            if process_video(q_id, "QUERY"):
                query_success += 1
        fail_log.close()

        print("\n[子阶段3] 帧间相似度矩阵计算与统计量记录...")
        stage1_matrix_start = time.time()
        matrices_computed = 0
        matrices_cached = 0

        matrix_stats = {}
        if os.path.exists(matrix_stats_cache_file):
            try:
                with open(matrix_stats_cache_file, 'r') as f:
                    matrix_stats = json.load(f)
                print(f"> 加载已有矩阵统计量缓存，{len(matrix_stats)} 条记录")
            except:
                print("> 警告: 统计量缓存损坏，将重新计算")

        pbar = tqdm(total=len(query_ids) * len(database_ids), desc="矩阵计算", unit="pair")
        for q_id in query_ids:
            q_feat = load_features(q_id, actual_features_dir, expected_dims)
            if q_feat is None:
                pbar.update(len(database_ids))
                continue
            q_feat_device = q_feat.to(device)
            for db_id in database_ids:
                pair_str = f"{q_id}_{db_id}"
                matrix_path = os.path.join(args.sim_matrices_dir, f"{pair_str}.npy")

                if os.path.exists(matrix_path) and pair_str in matrix_stats:
                    matrices_cached += 1
                    pbar.update(1)
                    continue

                if not os.path.exists(matrix_path):
                    db_feat = load_features(db_id, actual_features_dir, expected_dims)
                    if db_feat is None:
                        pbar.update(1)
                        continue
                    db_feat_device = db_feat.to(device)
                    with torch.no_grad():
                        sim_tensor = model.calculate_f2f_matrix(q_feat_device.unsqueeze(0), db_feat_device.unsqueeze(0))
                        sim_matrix = sim_tensor.squeeze(0).cpu().numpy()
                    np.save(matrix_path, sim_matrix)
                    matrices_computed += 1
                    del db_feat_device
                else:
                    sim_matrix = np.load(matrix_path)

                # [修改点] 计算并记录矩阵统计量（含均值、标准差、最值等）
                try:
                    max_val = float(np.max(sim_matrix))
                    row_mean = float(np.mean(np.max(sim_matrix, axis=1)))
                    col_mean = float(np.mean(np.max(sim_matrix, axis=0)))
                    alg_mean = (row_mean + col_mean) / 2.0
                    if row_mean + col_mean > 0:
                        harm_mean = 2.0 * row_mean * col_mean / (row_mean + col_mean + 1e-8)
                    else:
                        harm_mean = 0.0

                    matrix_mean = float(np.mean(sim_matrix))
                    matrix_std = float(np.std(sim_matrix))
                    matrix_min = float(np.min(sim_matrix))
                    matrix_max = float(np.max(sim_matrix))

                    stats = {
                        'max': max_val,
                        'row_mean': row_mean,
                        'col_mean': col_mean,
                        'alg_mean': alg_mean,
                        'harm_mean': harm_mean,
                        'matrix_mean': matrix_mean,
                        'matrix_std': matrix_std,
                        'matrix_min': matrix_min,
                        'matrix_max': matrix_max
                    }
                except:
                    stats = {'max': -1.0, 'row_mean': -1.0, 'col_mean': -1.0,
                             'alg_mean': -1.0, 'harm_mean': -1.0,
                             'matrix_mean': 0.0, 'matrix_std': 0.0,
                             'matrix_min': 0.0, 'matrix_max': 0.0}

                matrix_stats[pair_str] = stats
                pbar.update(1)

                if len(matrix_stats) % 1000 == 0:
                    with open(matrix_stats_cache_file, 'w') as f:
                        json.dump(matrix_stats, f)
            del q_feat_device

        pbar.close()
        with open(matrix_stats_cache_file, 'w') as f:
            json.dump(matrix_stats, f)
        print(f"> 矩阵统计量缓存已保存: {matrix_stats_cache_file}")

        stage1_matrix_time = time.time() - stage1_matrix_start
        stage1_time = time.time() - stage1_start
        print(f"> 阶段1完成，总耗时 {stage1_time:.2f}s")
        print(f"   数据库成功: {db_success}/{len(database_ids)}，查询成功: {query_success}/{len(query_ids)}")
        print(f"   矩阵计算: {matrices_computed} 新, {matrices_cached} 缓存, 耗时 {stage1_matrix_time:.2f}s")

    # ========== 阶段2 ==========
    if args.stage in ['2', 'all']:
        print("\n" + "=" * 60)
        print("阶段2：视频级相似度计算（含 CNN 后处理）")
        print("=" * 60)
        stage2_start = time.time()

        if 'model' not in locals():
            model = ViSiL(pretrained=True, symmetric=is_symmetric).to(device)
            model.eval()
            print("> 模型加载成功")

        def pad_features_to_min_frames(feat, min_frames=4):
            if feat.shape[0] >= min_frames:
                return feat
            pad_size = min_frames - feat.shape[0]
            last_frame = feat[-1:].clone()
            padding = last_frame.repeat(pad_size, 1, 1)
            return torch.cat([feat, padding], dim=0)

        similarities = {}
        checkpoint_file = os.path.join(args.output_dir, 'stage2_checkpoint.json')
        processed_pairs = set()
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                cp = json.load(f)
                similarities = cp.get('similarities', {})
                processed_pairs = set(tuple(p) for p in cp.get('processed_pairs', []))
            print(f"> 加载检查点，已处理 {len(processed_pairs)} 对")

        total_pairs = len(query_ids) * len(database_ids)
        pbar = tqdm(total=total_pairs, desc="视频相似度", unit="pair", initial=len(processed_pairs))
        for q_id in query_ids:
            q_feat = load_features(q_id, actual_features_dir, expected_dims)
            if q_feat is None:
                pbar.update(len(database_ids) - len([p for p in processed_pairs if p[0] == q_id]))
                continue
            q_feat = pad_features_to_min_frames(q_feat, min_frames=4)
            q_feat_device = q_feat.to(device)
            if q_id not in similarities:
                similarities[q_id] = {}
            for db_id in database_ids:
                pair_key = (q_id, db_id)
                if pair_key in processed_pairs:
                    continue
                db_feat = load_features(db_id, actual_features_dir, expected_dims)
                if db_feat is None:
                    processed_pairs.add(pair_key)
                    pbar.update(1)
                    continue
                db_feat = pad_features_to_min_frames(db_feat, min_frames=4)
                db_feat_device = db_feat.to(device)

                with torch.no_grad():
                    sim_val = model.calculate_video_similarity(q_feat_device.unsqueeze(0), db_feat_device.unsqueeze(0))
                similarities[q_id][db_id] = float(sim_val.item())
                processed_pairs.add(pair_key)
                pbar.update(1)

                if len(processed_pairs) % 1000 == 0:
                    with open(checkpoint_file, 'w') as f:
                        json.dump({'similarities': similarities, 'processed_pairs': list(processed_pairs)}, f)
                del db_feat_device
            del q_feat_device
            gc.collect()
        pbar.close()

        stage2_time = time.time() - stage2_start
        print(f"> 阶段2完成，耗时 {stage2_time:.2f}s")

        sim_file = os.path.join(args.output_dir, f"similarities_{method_suffix}{model_suffix}.json")
        with open(sim_file, 'w') as f:
            json.dump(similarities, f, separators=(',', ':'))
        print(f"> 相似度保存至: {sim_file}")

        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

    # ========== 阶段3：检测评估 ==========
    if args.stage in ['3', 'all']:
        print("\n" + "=" * 60)
        print("阶段3：检测评估")
        print("=" * 60)
        stage3_start = time.time()

        sim_file = os.path.join(args.output_dir, f"similarities_{method_suffix}{model_suffix}.json")
        if not os.path.exists(sim_file):
            print(f"> 错误: 相似度文件不存在 {sim_file}，请先运行阶段2")
            return
        with open(sim_file, 'r') as f:
            similarities = json.load(f)

        # [修改点] 加载矩阵统计量缓存（若无则现场计算）
        matrix_stats = {}
        if os.path.exists(matrix_stats_cache_file):
            with open(matrix_stats_cache_file, 'r') as f:
                matrix_stats = json.load(f)
            print(f"> 加载矩阵统计量缓存，共 {len(matrix_stats)} 条")
        else:
            print("> 未找到矩阵统计量缓存，将现场计算并保存...")
            pbar = tqdm(total=len(query_ids) * len(database_ids), desc="计算矩阵统计量", unit="pair")
            for q_id in query_ids:
                for db_id in database_ids:
                    pair_str = f"{q_id}_{db_id}"
                    matrix_path = os.path.join(args.sim_matrices_dir, f"{pair_str}.npy")
                    if os.path.exists(matrix_path):
                        try:
                            matrix = np.load(matrix_path)
                            max_val = float(np.max(matrix))
                            row_mean = float(np.mean(np.max(matrix, axis=1)))
                            col_mean = float(np.mean(np.max(matrix, axis=0)))
                            alg_mean = (row_mean + col_mean) / 2.0
                            if row_mean + col_mean > 0:
                                harm_mean = 2.0 * row_mean * col_mean / (row_mean + col_mean + 1e-8)
                            else:
                                harm_mean = 0.0

                            matrix_mean = float(np.mean(matrix))
                            matrix_std = float(np.std(matrix))
                            matrix_min = float(np.min(matrix))
                            matrix_max = float(np.max(matrix))

                            stats = {
                                'max': max_val,
                                'row_mean': row_mean,
                                'col_mean': col_mean,
                                'alg_mean': alg_mean,
                                'harm_mean': harm_mean,
                                'matrix_mean': matrix_mean,
                                'matrix_std': matrix_std,
                                'matrix_min': matrix_min,
                                'matrix_max': matrix_max
                            }
                        except:
                            stats = {'max': -1.0, 'row_mean': -1.0, 'col_mean': -1.0,
                                     'alg_mean': -1.0, 'harm_mean': -1.0,
                                     'matrix_mean': 0.0, 'matrix_std': 0.0,
                                     'matrix_min': 0.0, 'matrix_max': 0.0}
                    else:
                        stats = {'max': -1.0, 'row_mean': -1.0, 'col_mean': -1.0,
                                 'alg_mean': -1.0, 'harm_mean': -1.0,
                                 'matrix_mean': 0.0, 'matrix_std': 0.0,
                                 'matrix_min': 0.0, 'matrix_max': 0.0}
                    matrix_stats[pair_str] = stats
                    pbar.update(1)
            pbar.close()
            with open(matrix_stats_cache_file, 'w') as f:
                json.dump(matrix_stats, f)
            print(f"> 矩阵统计量缓存已生成并保存: {matrix_stats_cache_file}")

        # 加载数据集对象
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

        # 评估 mAP
        print("\n> 检索排序评估 (mAP)...")
        try:
            eval_results = dataset.evaluate(similarities)
            eval_file = os.path.join(args.output_dir, f"evaluation_{method_suffix}{model_suffix}.json")
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            if args.dataset == 'VCDB':
                print(f"  mAP: {eval_results.get('mAP', 0):.4f}")
            else:
                print(f"  DSVR: {eval_results.get('DSVR', 0):.4f}")
                print(f"  CSVR: {eval_results.get('CSVR', 0):.4f}")
                print(f"  ISVR: {eval_results.get('ISVR', 0):.4f}")
        except Exception as e:
            print(f"> 评估失败: {e}")

        # ---------- 二分类检测评估（基于视频级相似度） ----------
        print(f"\n> 二分类检测评估 [视频级相似度] (阈值 = {args.video_threshold})...")
        y_true_video, raw_scores_video = [], []
        if args.dataset == 'VCDB':
            positive_set = {}
            for qid in query_ids:
                pos_list = dataset.query_to_database.get(qid, [])
                valid_pos = set(pos_list).intersection(set(database_ids))
                positive_set[qid] = valid_pos
            total_pos_pairs = sum(len(v) for v in positive_set.values())
            print(f"> 正样本视频对总数: {total_pos_pairs}")
            for qid in query_ids:
                pos_videos = positive_set.get(qid, set())
                for db_id in database_ids:
                    true_label = 1 if db_id in pos_videos else 0
                    score = similarities.get(qid, {}).get(db_id, -1.0)
                    y_true_video.append(true_label)
                    raw_scores_video.append(score)
        elif 'FIVR' in args.dataset:
            positive_set = {}
            for q_id in query_ids:
                relevant_dict = dataset.annotation.get(q_id, {})
                pos_list = sum([relevant_dict.get(label, []) for label in ['ND', 'DS']], [])
                positive_set[q_id] = set(pos_list)
            for q_id in query_ids:
                for db_id in database_ids:
                    true_label = 1 if db_id in positive_set.get(q_id, set()) else 0
                    score = similarities.get(q_id, {}).get(db_id, -1.0)
                    y_true_video.append(true_label)
                    raw_scores_video.append(score)

        if y_true_video:
            y_true_video = np.array(y_true_video)
            raw_scores_video = np.array(raw_scores_video)
            pred_labels_video = (raw_scores_video >= args.video_threshold).astype(int)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            acc_v = accuracy_score(y_true_video, pred_labels_video)
            prec_v = precision_score(y_true_video, pred_labels_video, zero_division=0)
            rec_v = recall_score(y_true_video, pred_labels_video, zero_division=0)
            f1_v = f1_score(y_true_video, pred_labels_video, zero_division=0)
            cm_v = confusion_matrix(y_true_video, pred_labels_video)
            tn_v, fp_v, fn_v, tp_v = cm_v.ravel() if cm_v.size == 4 else (0, 0, 0, 0)
            print(f"  准确率: {acc_v:.4f}")
            print(f"  精确率: {prec_v:.4f}")
            print(f"  召回率: {rec_v:.4f}")
            print(f"  F1分数: {f1_v:.4f}")
            print("  混淆矩阵:")
            print(f"    TP: {tp_v:6d}  FP: {fp_v:6d}")
            print(f"    FN: {fn_v:6d}  TN: {tn_v:6d}")

        # ---------- 二分类检测评估（基于帧间矩阵最大值） ----------
        print(f"\n> 二分类检测评估 [帧间矩阵最大值] (阈值 = {args.max_threshold})...")
        y_true_matrix, pred_labels_matrix = [], []
        missing_matrix = 0
        for q_id in query_ids:
            if args.dataset == 'VCDB':
                pos_set = set(dataset.query_to_database.get(q_id, [])).intersection(set(database_ids))
            elif 'FIVR' in args.dataset:
                relevant_dict = dataset.annotation.get(q_id, {})
                pos_list = sum([relevant_dict.get(label, []) for label in ['ND', 'DS']], [])
                pos_set = set(pos_list)
            else:
                pos_set = set()

            for db_id in database_ids:
                true_label = 1 if db_id in pos_set else 0
                y_true_matrix.append(true_label)

                pair_str = f"{q_id}_{db_id}"
                stats = matrix_stats.get(pair_str, {})
                max_val = stats.get('max', -1.0)
                if max_val == -1.0 and not os.path.exists(os.path.join(args.sim_matrices_dir, f"{pair_str}.npy")):
                    missing_matrix += 1
                pred = 1 if max_val >= args.max_threshold else 0
                pred_labels_matrix.append(pred)

        if missing_matrix > 0:
            print(f"  警告: {missing_matrix} 个矩阵缺失，预测置为 0")

        if y_true_matrix:
            y_true_matrix = np.array(y_true_matrix)
            pred_labels_matrix = np.array(pred_labels_matrix)
            acc_m = accuracy_score(y_true_matrix, pred_labels_matrix)
            prec_m = precision_score(y_true_matrix, pred_labels_matrix, zero_division=0)
            rec_m = recall_score(y_true_matrix, pred_labels_matrix, zero_division=0)
            f1_m = f1_score(y_true_matrix, pred_labels_matrix, zero_division=0)
            cm_m = confusion_matrix(y_true_matrix, pred_labels_matrix)
            tn_m, fp_m, fn_m, tp_m = cm_m.ravel() if cm_m.size == 4 else (0, 0, 0, 0)
            print(f"  准确率: {acc_m:.4f}")
            print(f"  精确率: {prec_m:.4f}")
            print(f"  召回率: {rec_m:.4f}")
            print(f"  F1分数: {f1_m:.4f}")
            print("  混淆矩阵:")
            print(f"    TP: {tp_m:6d}  FP: {fp_m:6d}")
            print(f"    FN: {fn_m:6d}  TN: {tn_m:6d}")

        # ---------- 二分类检测评估（基于代数均值） ----------
        print(f"\n> 二分类检测评估 [代数均值] (阈值 = {args.algebraic_threshold})...")
        y_true_alg, pred_labels_alg = [], []
        for q_id in query_ids:
            if args.dataset == 'VCDB':
                pos_set = set(dataset.query_to_database.get(q_id, [])).intersection(set(database_ids))
            elif 'FIVR' in args.dataset:
                relevant_dict = dataset.annotation.get(q_id, {})
                pos_list = sum([relevant_dict.get(label, []) for label in ['ND', 'DS']], [])
                pos_set = set(pos_list)
            else:
                pos_set = set()

            for db_id in database_ids:
                true_label = 1 if db_id in pos_set else 0
                y_true_alg.append(true_label)

                pair_str = f"{q_id}_{db_id}"
                stats = matrix_stats.get(pair_str, {})
                alg_mean = stats.get('alg_mean', -1.0)
                pred = 1 if alg_mean >= args.algebraic_threshold else 0
                pred_labels_alg.append(pred)

        if y_true_alg:
            y_true_alg = np.array(y_true_alg)
            pred_labels_alg = np.array(pred_labels_alg)
            acc_alg = accuracy_score(y_true_alg, pred_labels_alg)
            prec_alg = precision_score(y_true_alg, pred_labels_alg, zero_division=0)
            rec_alg = recall_score(y_true_alg, pred_labels_alg, zero_division=0)
            f1_alg = f1_score(y_true_alg, pred_labels_alg, zero_division=0)
            cm_alg = confusion_matrix(y_true_alg, pred_labels_alg)
            tn_alg, fp_alg, fn_alg, tp_alg = cm_alg.ravel() if cm_alg.size == 4 else (0, 0, 0, 0)
            print(f"  准确率: {acc_alg:.4f}")
            print(f"  精确率: {prec_alg:.4f}")
            print(f"  召回率: {rec_alg:.4f}")
            print(f"  F1分数: {f1_alg:.4f}")
            print("  混淆矩阵:")
            print(f"    TP: {tp_alg:6d}  FP: {fp_alg:6d}")
            print(f"    FN: {fn_alg:6d}  TN: {tn_alg:6d}")

        # ---------- 二分类检测评估（基于调和均值） ----------
        print(f"\n> 二分类检测评估 [调和均值] (阈值 = {args.harmonic_threshold})...")
        y_true_harm, pred_labels_harm = [], []
        for q_id in query_ids:
            if args.dataset == 'VCDB':
                pos_set = set(dataset.query_to_database.get(q_id, [])).intersection(set(database_ids))
            elif 'FIVR' in args.dataset:
                relevant_dict = dataset.annotation.get(q_id, {})
                pos_list = sum([relevant_dict.get(label, []) for label in ['ND', 'DS']], [])
                pos_set = set(pos_list)
            else:
                pos_set = set()

            for db_id in database_ids:
                true_label = 1 if db_id in pos_set else 0
                y_true_harm.append(true_label)

                pair_str = f"{q_id}_{db_id}"
                stats = matrix_stats.get(pair_str, {})
                harm_mean = stats.get('harm_mean', -1.0)
                pred = 1 if harm_mean >= args.harmonic_threshold else 0
                pred_labels_harm.append(pred)

        if y_true_harm:
            y_true_harm = np.array(y_true_harm)
            pred_labels_harm = np.array(pred_labels_harm)
            acc_harm = accuracy_score(y_true_harm, pred_labels_harm)
            prec_harm = precision_score(y_true_harm, pred_labels_harm, zero_division=0)
            rec_harm = recall_score(y_true_harm, pred_labels_harm, zero_division=0)
            f1_harm = f1_score(y_true_harm, pred_labels_harm, zero_division=0)
            cm_harm = confusion_matrix(y_true_harm, pred_labels_harm)
            tn_harm, fp_harm, fn_harm, tp_harm = cm_harm.ravel() if cm_harm.size == 4 else (0, 0, 0, 0)
            print(f"  准确率: {acc_harm:.4f}")
            print(f"  精确率: {prec_harm:.4f}")
            print(f"  召回率: {rec_harm:.4f}")
            print(f"  F1分数: {f1_harm:.4f}")
            print("  混淆矩阵:")
            print(f"    TP: {tp_harm:6d}  FP: {fp_harm:6d}")
            print(f"    FN: {fn_harm:6d}  TN: {tn_harm:6d}")

        stage3_time = time.time() - stage3_start
        print(f"> 阶段3完成，耗时 {stage3_time:.2f}s")

    # ========== 阶段4：热图输出 ==========
    if args.stage in ['4', 'all']:
        print("\n" + "=" * 60)
        print("阶段4：热图输出")
        print("=" * 60)
        stage4_start = time.time()

        if 'model' not in locals():
            model = ViSiL(pretrained=True, symmetric=is_symmetric).to(device)
            model.eval()
            print("> 模型加载成功")

        all_video_ids = set(query_ids) | set(database_ids)
        all_video_ids = list(all_video_ids)
        print(f"> 共 {len(all_video_ids)} 个独立视频")

        heatmap_pairs = set()
        for q_id in query_ids:
            for db_id in database_ids:
                pair = tuple(sorted([q_id, db_id]))
                heatmap_pairs.add(pair)

        print(f"> 待生成热图的视频对数量: {len(heatmap_pairs)}")

        feature_cache = {}
        def get_features(vid):
            if vid in feature_cache:
                return feature_cache[vid]
            feat = load_features(vid, actual_features_dir, expected_dims)
            if feat is not None:
                feature_cache[vid] = feat
            return feat

        heatmap_count = 0
        pbar = tqdm(heatmap_pairs, desc="热图生成", unit="pair")
        for vid1, vid2 in pbar:
            heatmap_path = os.path.join(args.heatmaps_dir, f"{vid1}_{vid2}.png")
            if os.path.exists(heatmap_path):
                continue

            feat1 = get_features(vid1)
            feat2 = get_features(vid2)
            if feat1 is None or feat2 is None:
                continue

            feat1_device = feat1.to(device)
            feat2_device = feat2.to(device)
            compute_and_save_heatmap(model, feat1_device, feat2_device, heatmap_path)
            heatmap_count += 1

            if len(feature_cache) > 200:
                feature_cache.clear()
                gc.collect()
        pbar.close()

        stage4_time = time.time() - stage4_start
        print(f"> 阶段4完成，生成热图 {heatmap_count} 张，耗时 {stage4_time:.2f}s")

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"全部完成，总耗时: {total_time:.2f}s ({total_time/60:.1f}分钟)")
    print("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n> 用户中断")
    except Exception as e:
        print(f"\n> 未捕获异常: {e}")
        traceback.print_exc()