import os
import json
import torch
import numpy as np
import gc
import cv2
from PIL import Image
import torch.nn.functional as F
import subprocess
import tempfile
import sys

# [修改点] 设置 OpenCV 日志级别为 0（仅错误）
try:
    cv2.setLogLevel(0)
except:
    pass

# [修改点] 上下文管理器：临时抑制 stderr 输出
class suppress_stderr:
    """抑制标准错误输出"""
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_WRONLY)
        self.stderr_fd = os.dup(2)
        os.dup2(self.null_fd, 2)
        return self

    def __exit__(self, *args):
        os.dup2(self.stderr_fd, 2)
        os.close(self.null_fd)
        os.close(self.stderr_fd)


# [修改点] 帧读取器（强制只使用 Decord，且读取时抑制 stderr）
class VideoFrameReader:
    """封装 Decord 的视频帧读取器，若 Decord 不可用或失败则抛出异常。"""
    def __init__(self, video_path):
        with suppress_stderr():
            self.video_path = video_path
            self.reader = None
            self.total_frames = 0
            self.fps = 30.0
            self.use_decord = False

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            if os.path.getsize(video_path) < 1024:
                raise ValueError(f"视频文件过小，可能已损坏: {video_path}")

            self._init_reader()

    def _init_reader(self):
        try:
            import decord
            from decord import VideoReader, cpu
            self.reader = VideoReader(self.video_path, ctx=cpu(0))
            self.total_frames = len(self.reader)
            try:
                self.fps = self.reader.get_avg_fps()
            except:
                self.fps = 30.0
            self.use_decord = True
            return
        except Exception as e:
            raise RuntimeError(f"Decord 无法打开视频: {self.video_path}") from e

    def __len__(self):
        return self.total_frames

    def get_frames(self, indices):
        if not self.use_decord:
            raise RuntimeError("没有可用的读取器")
        with suppress_stderr():
            frames = self.reader.get_batch(indices).asnumpy()
        return frames

    def close(self):
        pass


# [修改点] ffprobe 相关
def check_ffprobe(ffprobe_path='ffprobe'):
    try:
        result = subprocess.run([ffprobe_path, '-version'],
                                capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def get_frame_types(video_path, ffprobe_path='ffprobe'):
    try:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        cmd = [
            ffprobe_path, '-v', 'error', '-select_streams', 'v:0',
            '-show_frames', '-show_entries', 'frame=pict_type',
            '-print_format', 'json', '-o', tmp_path, video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                stderr=subprocess.DEVNULL)
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
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return []


# [修改点] 关键帧提取函数（原 extract_keyframes 及相关内部函数）
def extract_keyframes(video_path, video_id, frames_dir, method,
                      max_frames=0, lm_threshold=0.6, ffprobe_path='ffprobe'):
    """提取关键帧，返回 (indices列表, frames_tensor [T, C, H, W], fps, total_frames)"""
    reader = None
    try:
        video_subdir = os.path.join(frames_dir, video_id)
        os.makedirs(video_subdir, exist_ok=True)

        indices_file = os.path.join(video_subdir, 'indices.json')
        info_file = os.path.join(video_subdir, 'info.json')
        jpg_dir = os.path.join(video_subdir, 'jpg')

        # 尝试从缓存加载
        if os.path.exists(indices_file) and os.path.exists(info_file):
            try:
                with open(indices_file, 'r') as f:
                    indices = json.load(f)
                indices = [int(idx) for idx in indices]
                with open(info_file, 'r') as f:
                    info = json.load(f)
                fps = info.get('fps', 30.0)
                total_frames = info.get('total_frames', 0)
                all_exist = True
                for idx in indices:
                    jpg_path = os.path.join(jpg_dir, f"frame_{idx:06d}.jpg")
                    if not os.path.exists(jpg_path):
                        all_exist = False
                        break
                if all_exist:
                    frames = _load_frames_from_jpg(video_subdir, indices)
                    if frames is not None:
                        return indices, frames, fps, total_frames
            except Exception:
                pass

        if not os.path.exists(video_path):
            return None, None, None, None

        reader = VideoFrameReader(video_path)
        total_frames = len(reader)
        fps = reader.fps
        if total_frames < 4:
            return None, None, None, None

        selected_indices = _select_keyframes_by_method(
            video_path, reader, method, max_frames, lm_threshold, ffprobe_path, total_frames
        )
        if not selected_indices:
            selected_indices = [0]

        frames_np = reader.get_frames(selected_indices)
        if frames_np.size == 0:
            return None, None, None, None

        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
        if frames_tensor.shape[2] != 224 or frames_tensor.shape[3] != 224:
            frames_tensor = F.interpolate(frames_tensor, size=(224, 224), mode='bilinear', align_corners=False)

        os.makedirs(jpg_dir, exist_ok=True)
        for i, idx in enumerate(selected_indices):
            jpg_path = os.path.join(jpg_dir, f"frame_{idx:06d}.jpg")
            Image.fromarray(frames_np[i]).save(jpg_path, quality=90)

        with open(indices_file, 'w') as f:
            json.dump(selected_indices, f)
        with open(info_file, 'w') as f:
            json.dump({'fps': fps, 'total_frames': total_frames}, f)

        return selected_indices, frames_tensor, fps, total_frames
    except Exception as e:
        return None, None, None, None
    finally:
        if reader is not None:
            reader.close()


def _select_keyframes_by_method(video_path, reader, method, max_frames, lm_threshold, ffprobe_path, total_frames):
    """内部函数：根据方法选择关键帧索引"""
    if method == 'default':
        step = max(1, total_frames // max_frames) if max_frames > 0 else 1
        indices = list(range(0, total_frames, step))
        if max_frames > 0:
            indices = indices[:max_frames]
        return indices

    elif method == '2s':
        fps = reader.fps
        interval = int(round(fps * 2))
        if interval < 1:
            interval = 1
        indices = list(range(0, total_frames, interval))
        if max_frames > 0 and len(indices) > max_frames:
            step = len(indices) / max_frames
            indices = [indices[int(i * step)] for i in range(max_frames)]
        return indices

    elif method == 'local_maxima':
        diff = []
        prev = None
        for i in range(total_frames):
            frames = reader.get_frames([i])
            if frames.size == 0:
                continue
            curr = frames[0]
            if prev is not None:
                delta = np.abs(curr.astype(np.float32) - prev.astype(np.float32)).mean()
                diff.append(delta)
            prev = curr
        diff = np.array(diff)
        from scipy.ndimage import gaussian_filter1d
        diff_smooth = gaussian_filter1d(diff, sigma=1.0)
        from scipy.signal import argrelextrema
        local_max = argrelextrema(diff_smooth, np.greater)[0] + 1
        mean_val = diff_smooth.mean()
        candidates = [0]
        for idx in local_max:
            if idx < total_frames and diff_smooth[idx-1] > mean_val * lm_threshold:
                candidates.append(idx)
        candidates = sorted(set(candidates))
        if max_frames > 0 and len(candidates) > max_frames:
            step = len(candidates) / max_frames
            candidates = [candidates[int(i * step)] for i in range(max_frames)]
        return [idx for idx in candidates if 0 <= idx < total_frames]

    elif method == 'iframe':
        if not check_ffprobe(ffprobe_path):
            raise RuntimeError("ffprobe不可用")
        frame_types = get_frame_types(video_path, ffprobe_path)
        if not frame_types:
            print(f"警告: 无法获取帧类型，iframe 方法返回空列表")
            return []
        i_frames = [i for i, ft in enumerate(frame_types) if ft == 'I']
        if not i_frames:
            return []
        return i_frames

    elif method == 'i_p_mixed':
        if not check_ffprobe(ffprobe_path):
            raise RuntimeError("ffprobe不可用")
        frame_types = get_frame_types(video_path, ffprobe_path)
        if not frame_types:
            print(f"警告: 无法获取帧类型，i_p_mixed 方法返回空列表")
            return []
        # 筛选I/P帧（无重叠，无需去重）
        i_frames = [i for i, ft in enumerate(frame_types) if ft == 'I']
        p_frames = [i for i, ft in enumerate(frame_types) if ft == 'P']
        # 无I/P帧时明确提示
        if not i_frames and not p_frames:
            print(f"警告: 视频无I/P帧，i_p_mixed 方法返回空列表")
            return []
        # 数量控制逻辑优化
        if max_frames > 0:
            if len(i_frames) >= max_frames:
                # I帧足够，返回前max_frames个I帧
                return i_frames[:max_frames]
            else:
                # I帧不足，补充P帧（均匀采样，避免越界）
                need_p = max_frames - len(i_frames)
                if len(p_frames) <= need_p:
                    # P帧足够，直接合并排序
                    combined = i_frames + p_frames
                else:
                    # 均匀采样P帧（用linspace保证覆盖全范围）
                    sample_indices = np.linspace(0, len(p_frames) - 1, need_p, dtype=int)
                    sampled_p = [p_frames[idx] for idx in sample_indices]
                    combined = i_frames + sampled_p
                # 排序（保证帧索引按时间顺序）
                return sorted(combined)
        else:
            # max_frames≤0，返回所有I/P帧（直接合并排序）
            combined = i_frames + p_frames
            return sorted(combined)

    else:
        raise ValueError(f"未知关键帧方法: {method}")


def _load_frames_from_jpg(video_subdir, indices):
    """从jpg缓存加载帧张量"""
    jpg_dir = os.path.join(video_subdir, 'jpg')
    frames = []
    for idx in indices:
        jpg_path = os.path.join(jpg_dir, f"frame_{idx:06d}.jpg")
        try:
            if not os.path.exists(jpg_path):
                return None
            img = Image.open(jpg_path).convert('RGB')
            img_np = np.array(img)
            frames.append(img_np)
        except Exception:
            return None
    frames_np = np.stack(frames, axis=0)
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
    if frames_tensor.shape[2] != 224 or frames_tensor.shape[3] != 224:
        frames_tensor = F.interpolate(frames_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    return frames_tensor


# [修改点] 密集帧提取与保存函数
def extract_dense_frames_range_and_save(video_path, start_idx, end_idx, step,
                                        save_dir, max_size=None):
    """
    从视频中提取指定范围内的帧，按步长采样，并保存为 JPG 到 save_dir。
    返回保存的帧数，若失败返回 0。
    """
    reader = None
    try:
        reader = VideoFrameReader(video_path)
        total = reader.total_frames
        if start_idx < 0 or end_idx >= total or start_idx > end_idx:
            return 0
        indices = list(range(start_idx, end_idx + 1, step))
        if not indices:
            return 0
        frames_np = reader.get_frames(indices)  # [T, H, W, 3] RGB
        if frames_np.size == 0:
            return 0

        os.makedirs(save_dir, exist_ok=True)
        for i, frame_idx in enumerate(indices):
            # 可选择缩放
            img = frames_np[i]
            if max_size is not None and (img.shape[0] > max_size or img.shape[1] > max_size):
                # 简单缩放（保持宽高比）
                h, w = img.shape[:2]
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            jpg_path = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
            Image.fromarray(img).save(jpg_path, quality=90)
        return len(indices)
    except Exception:
        return 0
    finally:
        if reader is not None:
            reader.close()


def load_dense_frames_from_dir(frame_dir):
    """
    从 JPG 目录加载所有帧，返回 torch.Tensor [T, H, W, 3] (RGB, 0-255)
    """
    import glob
    jpg_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if not jpg_files:
        return None
    frames = []
    for f in jpg_files:
        img = Image.open(f).convert('RGB')
        frames.append(np.array(img))
    frames_np = np.stack(frames, axis=0)
    # 缩放到 224x224 以匹配模型输入
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
    if frames_tensor.shape[2] != 224 or frames_tensor.shape[3] != 224:
        frames_tensor = F.interpolate(frames_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    return frames_tensor.permute(0, 2, 3, 1)  # 转回 [T, H, W, C] 以便后续处理


def get_sorted_frames_from_dir(frame_dir):
    """
    从密集帧目录加载所有帧，返回按帧索引排序的列表 [(frame_idx, img_array), ...]
    若目录不存在或无帧，返回 None。
    """
    import glob
    jpg_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if not jpg_files:
        return None
    frames = []
    for f in jpg_files:
        # 从文件名提取帧索引
        base = os.path.basename(f)
        idx_str = base.replace('frame_', '').replace('.jpg', '')
        try:
            frame_idx = int(idx_str)
        except:
            continue
        img = Image.open(f).convert('RGB')
        img_np = np.array(img)
        frames.append((frame_idx, img_np))
    if not frames:
        return None
    return frames


def save_interval_metadata(metadata, filepath):
    """保存区间元数据到 JSON"""
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_interval_metadata(filepath):
    """加载区间元数据"""
    with open(filepath, 'r') as f:
        return json.load(f)


def merge_interval_pairs(pairs):
    """
    合并重叠的区间对。
    pairs: 列表，每个元素为字典 {'q_start':int, 'q_end':int, 'db_start':int, 'db_end':int, ...}
    返回合并后的列表，格式相同。
    合并规则：如果两个区间对的查询区间重叠且数据库区间也重叠，则合并为一个大区间对。
    """
    if not pairs:
        return []
    # 按查询起始排序
    sorted_pairs = sorted(pairs, key=lambda x: (x['q_start'], x['db_start']))
    merged = []
    current = sorted_pairs[0].copy()
    for next_pair in sorted_pairs[1:]:
        # 检查查询区间是否重叠或相邻（允许相接）
        q_overlap = (next_pair['q_start'] <= current['q_end'] + 1)
        # 检查数据库区间是否重叠或相邻
        db_overlap = (next_pair['db_start'] <= current['db_end'] + 1)
        if q_overlap and db_overlap:
            # 合并
            current['q_end'] = max(current['q_end'], next_pair['q_end'])
            current['db_end'] = max(current['db_end'], next_pair['db_end'])
            # 步长取最小（更密集的采样）? 这里简单保留原步长，但合并后步长应一致，取第一个的步长或取较小值。我们假设所有区间步长相同，直接保留当前步长。
        else:
            merged.append(current)
            current = next_pair.copy()
    merged.append(current)
    return merged


# [修改点] 特征提取函数
def extract_features_from_frames(model, frames, video_id, features_dir, device,
                                 batch_size=4, expected_dims=3840, use_cache=True):
    """从帧数据提取特征并保存为npy（如果use_cache=True），返回特征张量 [T, N, D]"""
    if use_cache and features_dir is not None:
        video_subdir = os.path.join(features_dir, video_id)
        os.makedirs(video_subdir, exist_ok=True)
        features_file = os.path.join(video_subdir, f"{video_id}.npy")
        if os.path.exists(features_file):
            try:
                features_np = np.load(features_file)
                features = torch.from_numpy(features_np).float()
                if features.dim() == 3 and features.shape[1] == 9 and features.shape[2] == expected_dims:
                    return features
                else:
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
            return None
        features = torch.cat(features_list, dim=0)  # [T, N, D]
        if features.dim() != 3 or features.shape[1] != 9:
            return None
        if features.shape[2] != expected_dims:
            return None

        if use_cache and features_dir is not None:
            np.save(features_file, features.cpu().numpy())
        return features
    except Exception:
        return None