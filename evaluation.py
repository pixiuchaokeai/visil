import torch
import argparse
import os
import json
import gc
import numpy as np
from tqdm import tqdm
import time
import traceback
import subprocess
import tempfile
import shutil
import cv2
from PIL import Image  # [修改点] 用于读取 JPG

from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator


# [修改点] ffprobe 检查与帧类型获取函数
def check_ffprobe(ffprobe_path='ffprobe'):
    """检查 ffprobe 是否可用"""
    try:
        result = subprocess.run([ffprobe_path, '-version'],
                                capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def get_frame_types(video_path, ffprobe_path='ffprobe'):
    """
    使用 ffprobe 获取视频每一帧的类型（I/P/B）
    返回：帧类型列表，例如 ['I','P','B','I',...]
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        cmd = [
            ffprobe_path,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_frames',
            '-show_entries', 'frame=pict_type',
            '-print_format', 'json',
            '-o', tmp_path,
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []

        with open(tmp_path, 'r') as f:
            data = json.load(f)
        os.unlink(tmp_path)

        if 'frames' not in data:
            return []

        frame_types = []
        for frame in data['frames']:
            ftype = frame.get('pict_type', '?')
            if ftype in ['I', 'P', 'B']:
                frame_types.append(ftype)
            else:
                frame_types.append('?')
        return frame_types
    except Exception as e:
        print(f"> ffprobe 分析失败: {e}")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return []


# [修改点] 视频帧读取器（流式，支持随机访问）
class VideoFrameReader:
    """封装 Decord 和 OpenCV 的视频帧读取器，支持随机访问和批量读取"""
    def __init__(self, video_path):
        self.video_path = video_path
        self.reader = None
        self.total_frames = 0
        self.use_decord = False
        self._init_reader()

    def _init_reader(self):
        # 尝试 Decord
        try:
            import decord
            from decord import VideoReader, cpu
            self.reader = VideoReader(self.video_path, ctx=cpu(0))
            self.total_frames = len(self.reader)
            self.use_decord = True
            print(f"信息: Decord成功打开视频 {os.path.basename(self.video_path)} ({self.total_frames} 帧)")
            return
        except Exception as e:
            print(f"Decord打开失败: {e}")

        # 尝试 OpenCV
        try:
            self.reader = cv2.VideoCapture(self.video_path)
            if not self.reader.isOpened():
                raise ValueError("无法打开视频")
            self.total_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
            self.use_decord = False
            print(f"信息: OpenCV成功打开视频 {os.path.basename(self.video_path)} ({self.total_frames} 帧)")
            return
        except Exception as e:
            print(f"OpenCV打开失败: {e}")

        raise RuntimeError(f"无法打开视频: {self.video_path}")

    def __len__(self):
        return self.total_frames

    def get_frames(self, indices):
        """
        获取指定索引的帧，返回 numpy 数组 [len(indices), H, W, 3] (RGB, uint8)
        """
        if self.use_decord:
            # Decord 批量读取
            frames = self.reader.get_batch(indices).asnumpy()  # [N, H, W, 3]
            return frames
        else:
            # OpenCV 逐帧读取
            frames = []
            for idx in indices:
                self.reader.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.reader.read()
                if not ret:
                    # 若读取失败，返回空白帧（但一般不应发生）
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    # BGR -> RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            return np.array(frames)

    def close(self):
        if not self.use_decord and self.reader is not None:
            self.reader.release()


# [修改点] 保存关键帧为 JPG 文件（路径调整为 frames_dir/视频名/jpg/）
def save_frames_as_jpg(frames, video_id, base_dir, indices):
    """
    将选定的帧保存为 JPG 图像
    frames: numpy 数组 [N, H, W, 3] (RGB, uint8)
    base_dir: 视频专属目录，例如 frames_dir/视频名
    indices: 选定的帧索引列表（对应原始帧号）
    """
    jpg_dir = os.path.join(base_dir, 'jpg')
    os.makedirs(jpg_dir, exist_ok=True)
    saved = 0
    for i, frame_idx in enumerate(indices):
        frame = frames[i]
        jpg_path = os.path.join(jpg_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(jpg_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved += 1
    print(f"    已保存 {saved} 个 JPG 关键帧到 {jpg_dir}")


# [修改点] 从 JPG 缓存加载帧张量
def load_frames_from_jpg(video_subdir, indices):
    """
    从 jpg 目录读取指定索引的帧，返回 torch.Tensor [N, C, H, W] (值范围 0-255)
    """
    jpg_dir = os.path.join(video_subdir, 'jpg')
    frames = []
    for idx in indices:
        jpg_path = os.path.join(jpg_dir, f"frame_{idx:06d}.jpg")
        if not os.path.exists(jpg_path):
            print(f"> 警告: JPG 文件缺失 {jpg_path}")
            return None
        img = Image.open(jpg_path).convert('RGB')
        img_np = np.array(img)  # [H, W, 3]
        frames.append(img_np)
    if not frames:
        return None
    frames_np = np.stack(frames, axis=0)  # [N, H, W, 3]
    # 缩放到 224x224（假设 JPG 已经是 224x224，但为了保险）
    import torch.nn.functional as F
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
    if frames_tensor.shape[2] != 224 or frames_tensor.shape[3] != 224:
        frames_tensor = F.interpolate(frames_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    return frames_tensor


def extract_frames_from_video(video_path: str, video_id: str, frames_dir: str,
                              method: str, max_frames: int = 75,
                              lm_threshold: float = 0.6, ffprobe_path='ffprobe',
                              save_jpg: bool = True,
                              cache_format: str = 'npy'):  # [修改点] 新增 cache_format 参数
    """
    从视频文件提取帧并保存为指定格式的缓存，支持多种帧提取方法。
    方法：
        default       : 每秒1帧（均匀采样）
        2s            : 每2秒1帧（均匀采样）
        local_maxima  : 基于帧间差异的局部最大值关键帧（需要 lm_threshold）
        iframe        : 基于 I 帧抽取（需 ffprobe）
    返回: [T, C, H, W] 的 torch.Tensor 或 None
    """
    video_subdir = os.path.join(frames_dir, video_id)
    os.makedirs(video_subdir, exist_ok=True)

    # [修改点] 根据缓存格式检查是否存在有效缓存
    indices_file = os.path.join(video_subdir, 'indices.json')
    if cache_format == 'npy':
        frames_file = os.path.join(video_subdir, f"{video_id}.npy")
        if os.path.exists(frames_file):
            try:
                frames_np = np.load(frames_file)
                frames = torch.from_numpy(frames_np).float()
                if frames.dim() == 4 and frames.shape[1] == 3 and frames.shape[2] == frames.shape[3] == 224:
                    print(f"> 使用 npy 缓存: {frames_file}")
                    return frames
                else:
                    print(f"> npy 缓存维度异常，重新提取")
                    os.remove(frames_file)
            except Exception:
                try:
                    os.remove(frames_file)
                except:
                    pass
    else:  # cache_format == 'jpg'
        if os.path.exists(indices_file):
            try:
                with open(indices_file, 'r') as f:
                    indices = json.load(f)
                # [修改点] 确保索引为 int 类型（JSON 默认 int，但可能被读为 float？保险转换）
                indices = [int(idx) for idx in indices]
                # 检查所有 jpg 文件是否存在
                jpg_dir = os.path.join(video_subdir, 'jpg')
                all_exist = True
                for idx in indices:
                    jpg_path = os.path.join(jpg_dir, f"frame_{idx:06d}.jpg")
                    if not os.path.exists(jpg_path):
                        all_exist = False
                        break
                if all_exist:
                    print(f"> 使用 jpg 缓存: {jpg_dir}")
                    frames = load_frames_from_jpg(video_subdir, indices)
                    if frames is not None:
                        return frames
            except Exception as e:
                print(f"> jpg 缓存加载失败: {e}，重新提取")

    if not os.path.exists(video_path):
        print(f"> 视频文件不存在: {video_path}")
        return None

    # 创建帧读取器
    try:
        reader = VideoFrameReader(video_path)
    except Exception as e:
        print(f"> 无法打开视频 {video_id}: {e}")
        return None

    total_frames = len(reader)
    if total_frames < 4:
        print(f"> 视频 {video_id} 帧数不足 {total_frames}")
        reader.close()
        return None

    selected_indices = []

    # 根据方法确定关键帧索引
    if method == 'default':
        step = max(1, total_frames // max_frames)
        selected_indices = list(range(0, total_frames, step))[:max_frames]

    elif method == '2s':
        target_count = min(max_frames, total_frames // 2)
        if target_count < 1:
            target_count = 1
        step = max(1, total_frames // target_count)
        selected_indices = list(range(0, total_frames, step))[:target_count]

    elif method == 'local_maxima':
        print(f"  计算帧间差异...")
        diff = []
        prev_frame = None
        for i in range(total_frames):
            frames_batch = reader.get_frames([i])
            curr_frame = frames_batch[0]
            if prev_frame is not None:
                delta = np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32)).mean()
                diff.append(delta)
            prev_frame = curr_frame
            if (i+1) % 1000 == 0:
                print(f"    已处理 {i+1}/{total_frames} 帧")
        diff = np.array(diff)

        from scipy.ndimage import gaussian_filter1d
        diff_smooth = gaussian_filter1d(diff, sigma=1.0)

        from scipy.signal import argrelextrema
        local_max = argrelextrema(diff_smooth, np.greater)[0] + 1
        diff_mean = diff_smooth.mean()
        candidates = [0]
        for idx in local_max:
            if idx < total_frames and diff_smooth[idx-1] > diff_mean * lm_threshold:
                candidates.append(idx)
        if len(candidates) == 1:
            sorted_idx = np.argsort(diff_smooth)[::-1][:min(5, len(diff_smooth))]
            candidates = [0] + (sorted_idx+1).tolist()

        candidates = sorted(set(candidates))
        if len(candidates) > max_frames:
            step = len(candidates) // max_frames
            candidates = candidates[::step][:max_frames]
        selected_indices = [idx for idx in candidates if 0 <= idx < total_frames]
        print(f"  选定 {len(selected_indices)} 个关键帧")

    elif method == 'iframe':
        if not check_ffprobe(ffprobe_path):
            raise RuntimeError("ffprobe 不可用，无法使用 iframe 方法")
        frame_types = get_frame_types(video_path, ffprobe_path)
        if not frame_types:
            print(f"> 无法获取帧类型，回退到 default 方法")
            step = max(1, total_frames // max_frames)
            selected_indices = list(range(0, total_frames, step))[:max_frames]
        else:
            i_frames = [i for i, ft in enumerate(frame_types) if ft == 'I']
            if not i_frames:
                print(f"> 视频无 I 帧，回退到 default")
                step = max(1, total_frames // max_frames)
                selected_indices = list(range(0, total_frames, step))[:max_frames]
            else:
                if len(i_frames) > max_frames:
                    step = len(i_frames) // max_frames
                    i_frames = i_frames[::step][:max_frames]
                selected_indices = i_frames
    else:
        raise ValueError(f"未知的帧提取方法: {method}")

    if not selected_indices:
        selected_indices = [0]

    # 读取选中的帧
    frames_np = reader.get_frames(selected_indices)
    reader.close()

    # 缩放到 224x224
    import torch.nn.functional as F
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
    if frames_tensor.shape[2] != 224 or frames_tensor.shape[3] != 224:
        frames_tensor = F.interpolate(frames_tensor, size=(224, 224), mode='bilinear', align_corners=False)

    # [修改点] 根据缓存格式保存
    if cache_format == 'npy':
        frames_file = os.path.join(video_subdir, f"{video_id}.npy")
        np.save(frames_file, frames_tensor.cpu().numpy())
        print(f"> 保存 npy 缓存: {frames_file}")
    else:  # jpg
        # 保存 JPG（即使 cache_format 是 jpg，也调用 save_jpg 保存，但 save_jpg 可能为 False，这里强制保存）
        jpg_dir = os.path.join(video_subdir, 'jpg')
        os.makedirs(jpg_dir, exist_ok=True)
        for i, idx in enumerate(selected_indices):
            frame = frames_np[i]
            jpg_path = os.path.join(jpg_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(jpg_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"> 保存 jpg 缓存: {jpg_dir}")

    # [修改点] 保存索引列表，并确保索引为 Python int 类型
    # 将 selected_indices 中的 numpy 类型转换为 Python int
    selected_indices_py = [int(idx) for idx in selected_indices]
    with open(indices_file, 'w') as f:
        json.dump(selected_indices_py, f)

    # 如果 save_jpg 为 True 且缓存格式不是 jpg，额外保存一份 JPG 用于可视化
    if save_jpg and cache_format != 'jpg':
        save_frames_as_jpg(frames_np, video_id, video_subdir, selected_indices_py)

    return frames_tensor


def extract_features_from_frames(model, frames: torch.Tensor, video_id: str, features_dir: str,
                                 device: torch.device, batch_size: int = 4):
    """从帧数据提取特征并保存为npy，优先使用缓存，并验证特征维度（第二维必须为9）"""
    video_subdir = os.path.join(features_dir, video_id)
    os.makedirs(video_subdir, exist_ok=True)
    features_file = os.path.join(video_subdir, f"{video_id}.npy")

    if os.path.exists(features_file):
        try:
            features_np = np.load(features_file)
            features = torch.from_numpy(features_np).float()
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
        if frames.dim() == 4 and frames.shape[1] != 3 and frames.shape[3] == 3:
            frames = frames.permute(0, 3, 1, 2).contiguous()
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)

        features_list = []
        total_frames = frames.shape[0]
        for i in range(0, total_frames, batch_size):
            end = min(i + batch_size, total_frames)
            batch = frames[i:end].to(device).float()

            # 转换为模型期望的 [B, H, W, C]
            if batch.dim() == 4:
                if batch.shape[1] == 3:
                    batch = batch.permute(0, 2, 3, 1).contiguous()
                elif batch.shape[3] != 3:
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
                         frames_dir, features_dir, frame_method, lm_threshold,
                         batch_size=4, max_frames=75, ffprobe_path='ffprobe',
                         save_jpg=True, cache_format='npy'):  # [修改点] 传递 cache_format
    """处理查询视频，返回特征字典和失败列表"""
    query_features = {}
    failed_queries = []

    print(f"> 处理查询视频 ({len(query_ids)}个) 方法: {frame_method}, 缓存格式: {cache_format}")
    for idx, query_id in enumerate(tqdm(query_ids, desc="查询视频")):
        try:
            if query_id in id_to_path:
                video_path = id_to_path[query_id]
            else:
                failed_queries.append(query_id)
                continue

            if not os.path.exists(video_path):
                print(f"> 视频文件不存在: {video_path}")
                failed_queries.append(query_id)
                continue

            frames = extract_frames_from_video(video_path, query_id, frames_dir,
                                               frame_method, max_frames, lm_threshold,
                                               ffprobe_path, save_jpg, cache_format)
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

        except KeyboardInterrupt:
            raise
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
    parser.add_argument('--query_file', type=str, default="datasets/fivr-5k-queries-filtered.txt",
                        help='查询视频列表文件（每行：id 路径）')
    parser.add_argument('--database_file', type=str, default="datasets/fivr-5k-database-filtered.txt",
                        help='数据库视频列表文件（每行：id 路径）')
    parser.add_argument('--max_frames', type=int, default=75,
                        help='每个视频最大帧数，默认75')
    parser.add_argument('--batch_sz', type=int, default=4,
                        help='特征提取批次大小')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--cpu_only', action='store_true',

    help='强制使用CPU')
    parser.add_argument('--frame_method', type=str, default='local_maxima',
                        choices=['default', '2s', 'local_maxima', 'iframe'],
                        help='帧提取方法: default(每秒1帧), 2s(每2秒1帧), local_maxima(局部极大值), iframe(I帧)')
    parser.add_argument('--lm_threshold', type=float, default=0.6,
                        help='local_maxima 方法的差异阈值（默认0.6）')
    parser.add_argument('--similarity_function', type=str, default='symmetric_chamfer',
                        choices=["chamfer", "symmetric_chamfer"],
                        help='相似度函数')
    parser.add_argument('--frames_dir', type=str, default='output/frames1',
                        help='基础帧文件目录，实际会根据方法附加后缀')
    parser.add_argument('--features_dir', type=str, default='output/features1',
                        help='基础特征文件目录，实际会根据方法附加后缀')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出目录')
    parser.add_argument('--ffprobe_path', type=str, default='ffprobe',
                        help='ffprobe 可执行文件路径（用于 iframe 方法）')
    parser.add_argument('--save_jpg', action='store_true', default=True,
                        help='是否额外保存 JPG 图像用于可视化（即使缓存格式为 npy）')
    # [修改点] 新增缓存格式参数
    parser.add_argument('--cache_format', type=str, default='jpg',
                        choices=['npy', 'jpg'],
                        help='帧缓存格式: npy (快速加载) 或 jpg (节省空间，但加载较慢)')
    args = parser.parse_args()

    # 根据帧提取方法和阈值确定实际目录
    method_suffix = args.frame_method
    if args.frame_method == 'local_maxima':
        method_suffix = f'local_maxima_{args.lm_threshold}'
    actual_frames_dir = os.path.join(args.frames_dir, method_suffix)
    actual_features_dir = os.path.join(args.features_dir, method_suffix)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(actual_frames_dir, exist_ok=True)
    os.makedirs(actual_features_dir, exist_ok=True)

    if args.cpu_only or not torch.cuda.is_available():
        device = torch.device('cpu')
        args.batch_sz = min(args.batch_sz, 4)
        print("> 使用CPU")
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"> 使用GPU: {device}")

    total_start = time.time()
    stage_times = {}

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

    for vid in query_ids + database_ids:
        if vid not in id_to_path:
            path_candidate = os.path.join(args.video_dir, args.pattern.format(id=vid))
            if not os.path.exists(path_candidate):
                base = os.path.splitext(path_candidate)[0]
                found = False
                for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
                    alt_path = base + ext
                    if os.path.exists(alt_path):
                        path_candidate = alt_path
                        found = True
                        break
            id_to_path[vid] = path_candidate

    # ========== 第一阶段：帧提取 ==========
    print("\n" + "=" * 60)
    print(f"第一阶段：帧提取 (方法: {args.frame_method}, 缓存格式: {args.cache_format})")
    print("=" * 60)
    frame_start = time.time()
    query_features, q_failed = process_query_videos(
        query_ids, id_to_path, model, device,
        actual_frames_dir, actual_features_dir, args.frame_method, args.lm_threshold,
        args.batch_sz, args.max_frames, args.ffprobe_path, args.save_jpg, args.cache_format
    )
    frame_time = time.time() - frame_start
    stage_times['frame_extraction'] = frame_time
    if not query_features:
        print("> 错误: 无查询特征")
        return

    # ========== 第二阶段：特征提取 ==========
    print("\n" + "=" * 60)
    print("第二阶段：数据库视频特征提取")
    print("=" * 60)
    feature_start = time.time()
    checkpoint_file = os.path.join(args.output_dir, f"checkpoint_{method_suffix}.json")
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
                    failed_db.add(db_id)
                    processed_db.add(db_id)
                    pbar.update(1)
                    continue

                frames = extract_frames_from_video(video_path, db_id, actual_frames_dir,
                                                   args.frame_method, args.max_frames,
                                                   args.lm_threshold, args.ffprobe_path,
                                                   args.save_jpg, args.cache_format)
                if frames is None or frames.shape[0] < 4:
                    failed_db.add(db_id)
                    processed_db.add(db_id)
                    pbar.update(1)
                    continue

                db_features = extract_features_from_frames(model, frames, db_id, actual_features_dir,
                                                           device, args.batch_sz)
                if db_features is None:
                    failed_db.add(db_id)
                    processed_db.add(db_id)
                    pbar.update(1)
                    continue

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

            except KeyboardInterrupt:
                checkpoint = {
                    'similarities': similarities,
                    'processed_db': list(processed_db)
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f)
                print("\n> 用户中断，检查点已保存")
                return
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
        print(f"> 数据库视频特征提取完成: 成功 {len(processed_db) - len(failed_db)}/{len(database_ids)}, 失败 {len(failed_db)}")

    feature_time = time.time() - feature_start
    stage_times['feature_extraction'] = feature_time

    # ========== 第三阶段：相似度计算 ==========
    print("\n" + "=" * 60)
    print("第三阶段：相似度计算")
    print("=" * 60)
    sim_start = time.time()
    similarities = {}
    for query_id, q_feat in query_features.items():
        q_feat_device = q_feat.to(device)
        similarities[query_id] = {}
        for db_id in tqdm(processed_db - failed_db, desc=f"查询 {query_id}"):
            db_feat_file = os.path.join(actual_features_dir, db_id, f"{db_id}.npy")
            if not os.path.exists(db_feat_file):
                continue
            try:
                db_feat_np = np.load(db_feat_file)
                db_feat = torch.from_numpy(db_feat_np).float().to(device)
                with torch.no_grad():
                    if q_feat_device.dim() == 2:
                        q_input = q_feat_device.unsqueeze(0)
                    else:
                        q_input = q_feat_device
                    if db_feat.dim() == 2:
                        db_input = db_feat.unsqueeze(0)
                    else:
                        db_input = db_feat
                    sim = model.calculate_video_similarity(q_input, db_input)
                similarities[query_id][db_id] = float(sim.item())
                del db_feat
            except Exception as e:
                print(f"> 计算 {query_id}-{db_id} 相似度失败: {e}")
        del q_feat_device
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sim_time = time.time() - sim_start
    stage_times['similarity'] = sim_time
    total_time = time.time() - total_start
    stage_times['total'] = total_time

    # ========== 保存结果 ==========
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    result_file = os.path.join(args.output_dir, f"similarities_{method_suffix}.json")
    with open(result_file, 'w') as f:
        json.dump(similarities, f, separators=(',', ':'))
    print(f"> 相似度保存: {result_file}")

    sorted_file = os.path.join(args.output_dir, f"sorted_results_{method_suffix}.json")
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

    # ========== 评估 ==========
    print("\n" + "=" * 60)
    print("第四阶段：评估")
    print("=" * 60)
    eval_start = time.time()
    try:
        if 'dataset' not in locals():
            if 'FIVR' in args.dataset:
                from datasets import FIVR
                version = args.dataset.split('-')[1].lower() if '-' in args.dataset else '5k'
                dataset = FIVR(version=version)
            else:
                raise ValueError(f"未知数据集: {args.dataset}")
        eval_results = dataset.evaluate(similarities)
        eval_file = os.path.join(args.output_dir, f"evaluation_results_{method_suffix}.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"> 评估结果保存: {eval_file}")
        dsvr = eval_results.get('DSVR', 0.0)
        csvr = eval_results.get('CSVR', 0.0)
        isvr = eval_results.get('ISVR', 0.0)
        print(f"\n> DSVR: {dsvr:.4f}")
        print(f"> CSVR: {csvr:.4f}")
        print(f"> ISVR: {isvr:.4f}")
    except Exception as e:
        print(f"> 评估失败: {e}")
        dsvr = csvr = isvr = 0.0
    eval_time = time.time() - eval_start

    # ========== 生成总结文件 ==========
    summary_file = os.path.join(args.output_dir, f"summary_{method_suffix}.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ViSiL 评估总结\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"数据集: {args.dataset}\n")
        f.write(f"模型类型: {'visil_sym' if 'symmetric' in args.similarity_function else 'visil_v'}\n")
        f.write(f"帧提取方法: {args.frame_method}\n")
        if args.frame_method == 'local_maxima':
            f.write(f"  lm_threshold: {args.lm_threshold}\n")
        f.write(f"缓存格式: {args.cache_format}\n")
        f.write(f"最大帧数: {args.max_frames}\n")
        f.write(f"设备: {'GPU' if not args.cpu_only and torch.cuda.is_available() else 'CPU'}\n\n")
        f.write("各阶段耗时:\n")
        f.write(f"  帧提取: {stage_times['frame_extraction']:.2f} 秒\n")
        f.write(f"  特征提取: {stage_times['feature_extraction']:.2f} 秒\n")
        f.write(f"  相似度计算: {stage_times['similarity']:.2f} 秒\n")
        f.write(f"  评估: {eval_time:.2f} 秒\n")
        f.write(f"  总耗时: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)\n\n")
        f.write("评估结果:\n")
        f.write(f"  DSVR: {dsvr:.4f}\n")
        f.write(f"  CSVR: {csvr:.4f}\n")
        f.write(f"  ISVR: {isvr:.4f}\n")
    print(f"> 总结保存: {summary_file}")

    print("\n" + "=" * 60)
    print("所有阶段完成")
    print("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n> 用户中断")
    except Exception as e:
        print(f"\n> 未捕获的异常: {e}")
        traceback.print_exc()