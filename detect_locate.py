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
    get_sorted_frames_generator,
    get_sorted_frames_count,
    extract_full_dense_frames
)
from datasets.generators import VideoGenerator


# 数据集预设配置
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
    parser.add_argument('--threshold', type=float, default=-0.4,
                        help='第二阶段相似度矩阵筛选阈值')
    parser.add_argument('--sim_type', type=str, default='chamfer',
                        choices=['cos', 'chamfer'],
                        help='第二阶段相似度矩阵计算方式: cos(余弦相似度), chamfer(帧级Chamfer相似度)')
    parser.add_argument('--detection_threshold', type=float, default=0.1,
                        help='若提供，将在阶段四后输出二分类检测结果并计算评估指标')
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
                        choices=['1', '2', '3', '4', '5', 'all'])
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

    sim_type_str = args.sim_type
    threshold_str = f"th{args.threshold}".replace('.', 'p').replace('-', 'm')
    threshold_str_dir = f"{method_suffix}{model_suffix}_{threshold_str}"

    # [修改点] 第二阶段矩阵目录简化：直接放在方法+模型目录下，无子目录
    if args.sim_matrices_dir is None:
        args.sim_matrices_dir = os.path.join(args.output_dir, f'sim_matrices_{sim_type_str}_rough', method_suffix + model_suffix)
    # [修改点] 密集帧目录结构
    if args.dense_frames_dir is None:
        args.dense_frames_dir = os.path.join(args.output_dir, 'dense_frames')
    # intervals_meta 放在 dense_frames 根目录下
    if args.intervals_meta is None:
        args.intervals_meta = os.path.join(args.dense_frames_dir, f'intervals_meta_{threshold_str_dir}.json')
    if args.heatmaps_dir is None:
        args.heatmaps_dir = os.path.join(args.output_dir, 'heatmaps', threshold_str_dir)
    if args.dense_features_dir is None:
        args.dense_features_dir = os.path.join(args.output_dir, 'dense_features')
    if args.sim_matrices_dense_dir is None:
        args.sim_matrices_dense_dir = os.path.join(args.output_dir, 'sim_matrices_chamfer_dense', threshold_str_dir)

    # [修改点] 创建必要的子目录结构
    dense_frames_queries_dir = os.path.join(args.dense_frames_dir, 'queries')
    dense_frames_db_threshold_dir = os.path.join(args.dense_frames_dir, f'database_{threshold_str_dir}')
    dense_features_queries_dir = os.path.join(args.dense_features_dir, 'queries')
    dense_features_db_threshold_dir = os.path.join(args.dense_features_dir, f'database_{threshold_str_dir}')

    for d in [args.sim_matrices_dir, args.dense_frames_dir, args.heatmaps_dir,
              args.dense_features_dir, args.sim_matrices_dense_dir,
              dense_frames_queries_dir, dense_frames_db_threshold_dir,
              dense_features_queries_dir, dense_features_db_threshold_dir]:
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

    # VCDB特殊处理：读取清洗后的查询/数据库文件，并建立事件名到核心视频ID的映射
    vcdb_core_map = {}
    if args.dataset == 'VCDB':
        from datasets import VCDB
        vcdb_root = "datasets/VCDB/core_dataset"
        vcdb = VCDB(root_dir=vcdb_root, pickle_path=os.path.join(vcdb_root, 'vcdb_cleaned.pickle'))
        print(f"> VCDB 数据集加载成功")

        if args.query_file and os.path.exists(args.query_file):
            with open(args.query_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        query_id = parts[0]
                        video_path = ' '.join(parts[1:])
                        id_to_path[query_id] = video_path
                        query_ids.append(query_id)
                        core_video = vcdb.query_to_core.get(query_id)
                        if core_video:
                            vcdb_core_map[query_id] = core_video
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

    # [修改点] 流式特征提取函数
    def _extract_features_from_generator(model, frame_generator, total_frames, video_id, features_dir, device,
                                         batch_size, expected_dims):
        feat_path = os.path.join(features_dir, video_id, f"{video_id}.npy")
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)

        if os.path.exists(feat_path):
            try:
                feat_np = np.load(feat_path)
                feat = torch.from_numpy(feat_np).float()
                if feat.dim() == 3 and feat.shape[1] == 9 and feat.shape[2] == expected_dims:
                    return feat
            except:
                pass

        if total_frames == 0:
            return None

        try:
            import psutil
            def get_safe_batch_size(desired_bs):
                mem = psutil.virtual_memory()
                if mem.available < 2 * 1024 ** 3:
                    return max(1, desired_bs // 2)
                return desired_bs
        except ImportError:
            def get_safe_batch_size(desired_bs):
                return desired_bs

        features_list = []
        batch_frames = []
        model.eval()

        current_bs = get_safe_batch_size(batch_size)
        processed = 0

        with torch.no_grad():
            for idx, img in frame_generator:
                if img.shape[0] != 224 or img.shape[1] != 224:
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
                batch_frames.append(img)

                if len(batch_frames) >= current_bs:
                    batch_np = np.stack(batch_frames, axis=0)
                    batch_tensor = torch.from_numpy(batch_np).float().to(device)
                    batch_feat = model.extract_features(batch_tensor)
                    features_list.append(batch_feat.cpu())
                    processed += len(batch_frames)
                    del batch_tensor, batch_feat, batch_np
                    batch_frames.clear()
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    current_bs = get_safe_batch_size(batch_size)

            if batch_frames:
                batch_np = np.stack(batch_frames, axis=0)
                batch_tensor = torch.from_numpy(batch_np).float().to(device)
                batch_feat = model.extract_features(batch_tensor)
                features_list.append(batch_feat.cpu())
                del batch_tensor, batch_feat, batch_np
                batch_frames.clear()

        if not features_list:
            return None

        features = torch.cat(features_list, dim=0)
        if features.shape[1] != 9 or features.shape[2] != expected_dims:
            return None

        np.save(feat_path, features.cpu().numpy())
        return features

    def _extract_features_from_list(model, frames_list, video_id, features_dir, device, batch_size, expected_dims):
        feat_path = os.path.join(features_dir, video_id, f"{video_id}.npy")
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)

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
                batch_frames_np = np.stack(frames_list[i:end], axis=0)
                batch_tensor = torch.from_numpy(batch_frames_np).float().to(device)
                batch_feat = model.extract_features(batch_tensor)
                features_list.append(batch_feat.cpu())
                del batch_tensor, batch_feat
                if device.type == 'cpu':
                    gc.collect()

        if not features_list:
            return None

        features = torch.cat(features_list, dim=0)
        if features.shape[1] != 9 or features.shape[2] != expected_dims:
            return None

        np.save(feat_path, features.cpu().numpy())
        return features

    # 阶段1
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
                fail_log.write(f"[{video_type}] {video_id}: 视频文件不存在或路径无效 ({video_path})\n")
                return False, False

            frames_ready = False
            feat_ready = False

            if feature_exists(video_id):
                return True, True

            frames_np_list = None
            try:
                if check_frames_exist(video_id):
                    indices, frames_np_list = load_frames_from_disk(video_id)
                    if indices is not None and frames_np_list is not None:
                        frames_ready = True
                else:
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

        print("\n[子阶段1] 帧提取/加载...")
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

    # 阶段2
    def run_stage2():
        print("\n" + "=" * 60)
        print(f"第二阶段：关键帧相似度矩阵计算 ({args.sim_type}，阈值 {args.threshold})")
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
                # [修改点] 矩阵直接存放在方法目录下，无查询子目录
                matrix_path = os.path.join(args.sim_matrices_dir, f"{q_id}_{db_id}.npy")
                if os.path.exists(matrix_path):
                    processed_pairs.add(pair_key)
                    pbar.update(1)
                    cached_pairs += 1
                    continue

                if args.sim_type == 'chamfer':
                    with torch.no_grad():
                        q_feat_device = q_feat.to(device)
                        db_feat_device = db_feat.to(device)
                        sim_tensor = model.calculate_f2f_matrix(q_feat_device.unsqueeze(0), db_feat_device.unsqueeze(0))
                        sim_matrix = sim_tensor.squeeze(0).cpu().numpy()
                else:
                    sim_matrix = compute_keyframe_similarity_matrix(q_feat, db_feat)

                # [修改点] 直接保存到 args.sim_matrices_dir
                np.save(matrix_path, sim_matrix)
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
    stage5_time = 0.0

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
    if args.stage in ['3', '4', 'all']:
        if 'model' not in locals() or model is None:
            try:
                model = ViSiL(pretrained=True, symmetric=is_symmetric).to(device)
                model.eval()
                print("> 模型加载成功")
            except Exception as e:
                print(f"错误: 模型加载失败 - {e}")
                return

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

        if need_stage1 and args.stage != '4':
            print("> 检测到阶段1特征缺失，自动执行阶段1...")
            stage1_time = run_stage1(model, device)

        # 检查第二阶段矩阵是否存在
        matrix_files = glob.glob(os.path.join(args.sim_matrices_dir, '*.npy'))
        need_stage2 = len(matrix_files) == 0

        if need_stage2 and args.stage != '4':
            print("> 检测到阶段2矩阵缺失，自动执行阶段2...")
            stage2_time = run_stage2()

        # 阶段3
        if args.stage in ['3', 'all']:
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

                # [修改点] 查询密集帧保存到 queries/{q_id}/ 下，无 full 子目录
                query_dense_dir = os.path.join(dense_frames_queries_dir, q_id)
                if q_id not in query_full_frames_extracted:
                    num_frames = extract_full_dense_frames(id_to_path[q_id], q_id, dense_frames_queries_dir, args.dense_fps)
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

                    # [修改点] 加载矩阵路径
                    matrix_path = os.path.join(args.sim_matrices_dir, f"{q_id}_{db_id}.npy")
                    if not os.path.exists(matrix_path):
                        pbar.update(1)
                        continue
                    sim_matrix = np.load(matrix_path)

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
                        db_save_dir = os.path.join(dense_frames_db_threshold_dir, db_id, interval_id)

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
                            'query_full_dir': query_dense_dir   # [修改点] 无 full
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

        # 阶段4
        if args.stage in ['4', 'all']:
            if args.stage == '4' and not intervals_meta:
                if os.path.exists(args.intervals_meta):
                    with open(args.intervals_meta, 'r') as f:
                        intervals_meta = json.load(f)
                else:
                    print("> 错误：阶段4需要区间元数据，但未找到。请先运行阶段3。")
                    return

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

                query_feat_key = f"query_full_{q_id}"
                if query_feat_key in feature_cache:
                    q_feat = feature_cache[query_feat_key]
                else:
                    query_full_dir = intervals_list[0]['query_full_dir']
                    # [修改点] 查询密集特征路径无 full
                    query_feat_path = os.path.join(dense_features_queries_dir, q_id, f"{q_id}_dense.npy")
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
                        num_q_frames = get_sorted_frames_count(query_full_dir)
                        if num_q_frames == 0:
                            pbar.update(1)
                            continue
                        q_gen = get_sorted_frames_generator(query_full_dir)
                        if q_gen is None:
                            pbar.update(1)
                            continue
                        q_feat = _extract_features_from_generator(
                            model, q_gen, num_q_frames, f"{q_id}_dense", dense_features_queries_dir,
                            device, args.batch_sz, expected_dims
                        )
                        if q_feat is None:
                            pbar.update(1)
                            continue
                        feature_cache[query_feat_key] = q_feat

                db_feat_key = f"db_{db_id}_" + "_".join(sorted([meta['db_save_dir'] for meta in intervals_list]))
                if db_feat_key in feature_cache:
                    db_feat = feature_cache[db_feat_key]
                else:
                    db_merged_feat_path = os.path.join(dense_features_db_threshold_dir, db_id, f"{db_id}_merged.npy")
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
                        import heapq
                        generators = []
                        for meta in intervals_list:
                            gen = get_sorted_frames_generator(meta['db_save_dir'])
                            if gen is not None:
                                generators.append(gen)
                        if not generators:
                            pbar.update(1)
                            continue

                        heap = []
                        for i, gen in enumerate(generators):
                            try:
                                idx, img = next(gen)
                                heapq.heappush(heap, (idx, i, img, gen))
                            except StopIteration:
                                pass

                        def merged_gen():
                            seen = set()
                            while heap:
                                idx, i, img, gen = heapq.heappop(heap)
                                if idx not in seen:
                                    seen.add(idx)
                                    yield idx, img
                                try:
                                    next_idx, next_img = next(gen)
                                    heapq.heappush(heap, (next_idx, i, next_img, gen))
                                except StopIteration:
                                    pass

                        total_db_frames = sum(get_sorted_frames_count(meta['db_save_dir']) for meta in intervals_list)
                        merge_gen = merged_gen()
                        db_feat = _extract_features_from_generator(
                            model, merge_gen, total_db_frames, f"{db_id}_merged", dense_features_db_threshold_dir,
                            device, args.batch_sz, expected_dims
                        )
                        if db_feat is None:
                            pbar.update(1)
                            continue
                        feature_cache[db_feat_key] = db_feat

                sim_val = compute_dense_similarity(model, q_feat, db_feat)

                os.makedirs(args.heatmaps_dir, exist_ok=True)
                heatmap_path = os.path.join(args.heatmaps_dir, f"{q_id}_{db_id}_merged.png")
                matrix_save_path = os.path.join(args.sim_matrices_dense_dir, f"{q_id}_{db_id}_merged.npy")
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

            final_sim_file = os.path.join(args.output_dir, f"similarities_{threshold_str_dir}.json")
            with open(final_sim_file, 'w') as f:
                json.dump(similarities, f, separators=(',', ':'))
            print(f"> 最终相似度保存至: {final_sim_file}")

            if args.detection_threshold is not None:
                detection_results = {}
                for q_id, db_scores in similarities.items():
                    detection_results[q_id] = {}
                    for db_id, score in db_scores.items():
                        detection_results[q_id][db_id] = 1 if score >= args.detection_threshold else 0
                detection_file = os.path.join(args.output_dir, f"detection_predictions_th{args.detection_threshold}.json")
                with open(detection_file, 'w') as f:
                    json.dump(detection_results, f, separators=(',', ':'))
                print(f"> 二分类预测结果已保存至: {detection_file}")

            if os.path.exists(stage4_checkpoint):
                os.remove(stage4_checkpoint)

            stage4_time = time.time() - stage4_start
            print(f"> 第四阶段完成，耗时 {stage4_time:.2f}s")

        total_time = time.time() - total_start

        if args.stage in ['4', 'all'] and similarities:
            print("\n" + "=" * 60)
            print("检索排序评估 (mAP)")
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
                eval_file = os.path.join(args.output_dir, f"evaluation_{threshold_str_dir}.json")
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
                'sim_type': args.sim_type,
                'dense_fps': args.dense_fps,
                'stage1_time': stage1_time,
                'stage2_time': stage2_time,
                'stage3_time': stage3_time,
                'stage4_time': stage4_time,
                'total_time': total_time,
                'device': 'GPU' if not args.cpu_only and torch.cuda.is_available() else 'CPU'
            }
            summary_file = os.path.join(args.output_dir, f"summary_{threshold_str_dir}.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"> 总结保存: {summary_file}")

    if args.stage in ['5', 'all']:
        print("\n" + "=" * 60)
        print("第五阶段：二分类检测评估")
        print("=" * 60)
        stage5_start = time.time()

        sim_file = os.path.join(args.output_dir, f"similarities_{threshold_str_dir}.json")
        if not os.path.exists(sim_file):
            print(f"> 错误: 相似度文件不存在 {sim_file}")
            return
        with open(sim_file, 'r') as f:
            similarities = json.load(f)

        print("> 补全缺失视频对相似度分数 (-1.0)...")
        for q_id in query_ids:
            if q_id not in similarities:
                similarities[q_id] = {}
            for db_id in database_ids:
                if db_id not in similarities[q_id]:
                    similarities[q_id][db_id] = -1.0

        print("> 构建真实标签...")
        y_true, raw_scores = [], []
        if args.dataset == 'VCDB':
            positive_set = {}
            for q_id in query_ids:
                core_video = vcdb.query_to_core.get(q_id)
                if core_video is None:
                    continue
                copies = vcdb.core_to_copies.get(core_video, [])
                positive_set[q_id] = {os.path.splitext(v)[0] for v in copies}
            for q_id in query_ids:
                for db_id in database_ids:
                    true_label = 1 if (q_id in positive_set and db_id in positive_set[q_id]) else 0
                    score = similarities[q_id][db_id]
                    y_true.append(true_label)
                    raw_scores.append(score)
        elif 'FIVR' in args.dataset:
            from datasets import FIVR
            version = args.dataset.split('-')[1].lower() if '-' in args.dataset else '5k'
            fivr_dataset = FIVR(version=version)
            positive_set = {}
            for q_id in query_ids:
                relevant_dict = fivr_dataset.annotation.get(q_id, {})
                pos_list = sum([relevant_dict.get(label, []) for label in ['ND', 'DS']], [])
                positive_set[q_id] = set(pos_list)
            for q_id in query_ids:
                for db_id in database_ids:
                    true_label = 1 if db_id in positive_set.get(q_id, set()) else 0
                    score = similarities[q_id][db_id]
                    y_true.append(true_label)
                    raw_scores.append(score)
        else:
            print(f"> 警告: 数据集 {args.dataset} 暂不支持二分类评估")
            return

        y_true = np.array(y_true)
        raw_scores = np.array(raw_scores)

        print("> 训练 Platt Scaling 校准器...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        X = raw_scores.reshape(-1, 1)
        y = y_true
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        calib = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
        calib.fit(X_train, y_train)

        proba = calib.predict_proba(X)[:, 1]
        pred_labels = (proba >= args.detection_threshold).astype(int)

        acc = accuracy_score(y_true, pred_labels)
        prec = precision_score(y_true, pred_labels, zero_division=0)
        rec = recall_score(y_true, pred_labels, zero_division=0)
        f1 = f1_score(y_true, pred_labels, zero_division=0)
        cm = confusion_matrix(y_true, pred_labels)

        print("\n" + "=" * 50)
        print(f"二分类检测评估 (Platt校准后, 决策阈值={args.detection_threshold})")
        print(f"总样本数: {len(y_true)}")
        print(f"正样本数: {np.sum(y_true)}")
        print("-" * 30)
        print(f"准确率:  {acc:.4f}")
        print(f"精确率:  {prec:.4f}")
        print(f"召回率:  {rec:.4f}")
        print(f"F1 分数: {f1:.4f}")
        print("-" * 30)
        print("混淆矩阵:")
        print(f"           预测负类  预测正类")
        print(f"真实负类:  {cm[0,0]:6d}    {cm[0,1]:6d}")
        print(f"真实正类:  {cm[1,0]:6d}    {cm[1,1]:6d}")
        print("=" * 50)

        import joblib
        calib_file = os.path.join(args.output_dir, f"platt_calibrator_{threshold_str_dir}.pkl")
        joblib.dump(calib, calib_file)
        print(f"> Platt校准器已保存至: {calib_file}")

        metrics_file = os.path.join(args.output_dir, f"detection_metrics_calibrated_{threshold_str_dir}.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                'threshold': args.detection_threshold,
                'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                'confusion_matrix': cm.tolist()
            }, f, indent=2)
        print(f"> 评估指标已保存至: {metrics_file}")

        stage5_time = time.time() - stage5_start
        print(f"> 第五阶段完成，耗时 {stage5_time:.2f}s")

    if args.stage == 'all':
        print("\n" + "=" * 60)
        print("全部阶段完成")
        print("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n> 用户中断")
    except Exception as e:
        print(f"\n> 未捕获异常: {e}")
        traceback.print_exc()