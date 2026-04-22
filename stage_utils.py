import os
import json
import torch
import numpy as np
import gc
import cv2
from PIL import Image
import torch.nn.functional as F
import subprocess
import sys
from tqdm import tqdm

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


# -------------------- 帧读取器 --------------------
class VideoFrameReader:
    """封装 Decord 的视频帧读取器"""
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
                raise ValueError(f"视频文件过小: {video_path}")

            self._init_reader()

    # [修改点] 在 VideoFrameReader._init_reader 中抑制 Decord 的错误输出
    def _init_reader(self):
        try:
            import decord
            from decord import VideoReader, cpu
            # 临时重定向 stderr 以抑制 Decord 内部错误打印
            stderr_fd = sys.stderr.fileno()
            saved_stderr = os.dup(stderr_fd)
            null_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null_fd, stderr_fd)
            os.close(null_fd)
            try:
                self.reader = VideoReader(self.video_path, ctx=cpu(0))
            finally:
                os.dup2(saved_stderr, stderr_fd)
                os.close(saved_stderr)
            self.total_frames = len(self.reader)
            try:
                self.fps = self.reader.get_avg_fps()
            except:
                self.fps = 30.0
            self.use_decord = True
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


# -------------------- ffprobe 相关 --------------------
def check_ffprobe(ffprobe_path='ffprobe'):
    try:
        subprocess.run([ffprobe_path, '-version'], capture_output=True, timeout=5, check=True)
        return True
    except:
        return False


# [修改点] 完全重写 get_frame_types，使用 Popen 逐行读取，精准提取 I/P/B，增强错误处理
def get_frame_types(video_path, ffprobe_path='ffprobe'):
    """
    使用 ffprobe 获取视频每一帧的类型（I/P/B）。
    返回帧类型列表，若失败返回 []。
    """
    try:
        cmd = [
            ffprobe_path,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_frames',
            '-show_entries', 'frame=pict_type',
            '-of', 'csv=p=0',
            video_path
        ]
        # [修改点] 使用 Popen 以便逐行处理输出流，避免编码问题
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_types = []
        # 以二进制模式逐行读取，手动处理 \r 和 \n
        for line_bytes in iter(proc.stdout.readline, b''):
            if not line_bytes:
                continue
            # 去除行尾换行符并解码
            line = line_bytes.rstrip(b'\r\n').decode('utf-8', errors='ignore')
            if not line:
                continue
            # [修改点] 在行中寻找第一个 I、P 或 B 字符（不依赖位置，防止逗号干扰）
            found = None
            for ch in line:
                if ch in ('I', 'P', 'B'):
                    found = ch
                    break
            if found:
                frame_types.append(found)
            else:
                frame_types.append('?')
        proc.stdout.close()
        proc.wait(timeout=60)
        return frame_types
    except Exception as e:
        # [修改点] 发生异常时返回空列表，不抛出异常
        sys.stderr.write(f"[ffprobe] 错误: {e}\n")
        return []


# -------------------- 第一阶段：关键帧提取 --------------------
def extract_keyframes(video_path, video_id, frames_dir, method='iframe',
                      max_frames=0, lm_threshold=0.6, ffprobe_path='ffprobe'):
    """
    从视频中提取关键帧，返回 (indices, frames, fps, total_frames)
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        sys.stderr.write(f"[{video_id}] 文件不存在\n")
        return None, None, None, None

    # [修改点] 获取视频信息，优先使用 Decord
    reader = None
    use_decord = True
    try:
        reader = VideoFrameReader(video_path)
        total_frames = len(reader)
        fps = reader.fps
    except Exception:
        use_decord = False
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            sys.stderr.write(f"[{video_id}] OpenCV无法打开\n")
            return None, None, None, None
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or total_frames <= 0:
            sys.stderr.write(f"[{video_id}] 视频信息无效\n")
            return None, None, None, None

    frame_indices = []

    if method == 'iframe':
        frame_types = get_frame_types(video_path, ffprobe_path)
        if not frame_types:
            sys.stderr.write(f"[{video_id}] ffprobe获取帧类型失败或返回空\n")
            if reader: reader.close()
            return None, None, None, None

        i_frame_indices = [i for i, ft in enumerate(frame_types) if ft == 'I']
        if not i_frame_indices:
            sys.stderr.write(f"[{video_id}] 未检测到 I 帧\n")
            if reader: reader.close()
            return None, None, None, None

        # 过滤超出实际帧数的索引
        valid_indices = [idx for idx in i_frame_indices if idx < total_frames]
        if not valid_indices:
            sys.stderr.write(f"[{video_id}] I帧索引超出范围\n")
            if reader: reader.close()
            return None, None, None, None
        i_frame_indices = valid_indices

        if max_frames > 0 and len(i_frame_indices) > max_frames:
            step = len(i_frame_indices) / max_frames
            sampled = []
            for i in range(max_frames):
                idx = int(i * step)
                if idx < len(i_frame_indices):
                    sampled.append(i_frame_indices[idx])
            i_frame_indices = sampled

        frame_indices = i_frame_indices

    elif method == 'shot':
        from shot_based_keyframes import extract_keyframes as shot_extract
        # 参数设置：镜头检测阈值、每镜头最大关键帧数
        shot_threshold = 0.5  # 可后续作为参数传入
        max_per_shot = 2
        save_dir = os.path.join(frames_dir, video_id)
        os.makedirs(save_dir, exist_ok=True)

        try:
            # 调用镜头检测函数
            indices = shot_extract(video_path, save_dir,
                                   threshold=shot_threshold,
                                   max_keyframes_per_shot=max_per_shot)
        except Exception as e:
            sys.stderr.write(f"[{video_id}] shot_extract 失败: {e}\n")
            if reader: reader.close()
            return None, None, None, None

        # 验证索引
        valid_indices = [idx for idx in indices if idx < total_frames]
        if not valid_indices:
            sys.stderr.write(f"[{video_id}] shot_extract 无有效帧\n")
            if reader: reader.close()
            return None, None, None, None
        frame_indices = valid_indices

        # 读取帧（沿用已有逻辑）
        try:
            if use_decord:
                frames_np = reader.get_frames(frame_indices)
                if frames_np.size == 0:
                    reader.close()
                    return None, None, None, None
                for i, idx in enumerate(frame_indices):
                    img = frames_np[i]
                    img_path = os.path.join(save_dir, f"{idx:06d}.jpg")
                    Image.fromarray(img).save(img_path, quality=90)
                frames = [frames_np[i] for i in range(len(frame_indices))]
                saved_indices = frame_indices
            else:
                # OpenCV 读取逻辑...
                pass
            # 保存元数据
            with open(os.path.join(save_dir, 'indices.json'), 'w') as f:
                json.dump(saved_indices, f)
            with open(os.path.join(save_dir, 'info.json'), 'w') as f:
                json.dump({'fps': fps, 'total_frames': total_frames}, f)
            return saved_indices, frames, fps, total_frames
        except Exception as e:
            sys.stderr.write(f"[{video_id}] 读取帧异常: {e}\n")
            return None, None, None, None

    elif method == '2s':
        step = max(1, int(2 * fps))
        frame_indices = list(range(0, total_frames, step))
        if max_frames > 0 and len(frame_indices) > max_frames:
            frame_indices = frame_indices[:max_frames]

    elif method == 'default':
        step = max(1, int(fps))
        frame_indices = list(range(0, total_frames, step))
        if max_frames > 0 and len(frame_indices) > max_frames:
            frame_indices = frame_indices[:max_frames]

    else:
        sys.stderr.write(f"[{video_id}] 未知方法: {method}\n")
        if reader: reader.close()
        return None, None, None, None

    if not frame_indices:
        sys.stderr.write(f"[{video_id}] 无关键帧索引\n")
        if reader: reader.close()
        return None, None, None, None

    frame_save_dir = os.path.join(frames_dir, video_id)
    os.makedirs(frame_save_dir, exist_ok=True)

    # [修改点] 读取帧并保存
    try:
        if not use_decord:
            cap = cv2.VideoCapture(video_path)
            frames = []
            saved_indices = []
            current_idx = 0
            target_set = set(frame_indices)
            max_target = max(frame_indices)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if current_idx in target_set:
                    img_path = os.path.join(frame_save_dir, f"{current_idx:06d}.jpg")
                    cv2.imwrite(img_path, frame)
                    frames.append(frame)
                    saved_indices.append(current_idx)
                current_idx += 1
                if current_idx > max_target:
                    break
            cap.release()
            if len(frames) != len(frame_indices):
                sys.stderr.write(f"[{video_id}] OpenCV读取帧数量不符\n")
                return None, None, None, None
        else:
            frames_np = reader.get_frames(frame_indices)
            if frames_np.size == 0:
                sys.stderr.write(f"[{video_id}] Decord读取帧返回空\n")
                reader.close()
                return None, None, None, None

            for i, idx in enumerate(frame_indices):
                img = frames_np[i]
                img_path = os.path.join(frame_save_dir, f"{idx:06d}.jpg")
                Image.fromarray(img).save(img_path, quality=90)
            frames = [frames_np[i] for i in range(len(frame_indices))]
            saved_indices = frame_indices
            reader.close()

        # 保存元数据
        with open(os.path.join(frame_save_dir, 'indices.json'), 'w') as f:
            json.dump(saved_indices, f)
        with open(os.path.join(frame_save_dir, 'info.json'), 'w') as f:
            json.dump({'fps': fps, 'total_frames': total_frames}, f)

        return saved_indices, frames, fps, total_frames

    except Exception as e:
        sys.stderr.write(f"[{video_id}] 提取异常: {e}\n")
        if use_decord and reader:
            reader.close()
        return None, None, None, None


# ---------- 以下函数保持不变，未作修改 ----------
def _select_keyframes_by_method(video_path, reader, method, max_frames, lm_threshold, ffprobe_path, total_frames):
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
            return []
        i_frames = [i for i, ft in enumerate(frame_types) if ft == 'I']
        return i_frames

    elif method == 'i_p_mixed':
        if not check_ffprobe(ffprobe_path):
            raise RuntimeError("ffprobe不可用")
        frame_types = get_frame_types(video_path, ffprobe_path)
        if not frame_types:
            return []
        i_frames = [i for i, ft in enumerate(frame_types) if ft == 'I']
        p_frames = [i for i, ft in enumerate(frame_types) if ft == 'P']
        if not i_frames and not p_frames:
            return []
        if max_frames > 0:
            if len(i_frames) >= max_frames:
                return i_frames[:max_frames]
            else:
                need_p = max_frames - len(i_frames)
                if len(p_frames) <= need_p:
                    combined = i_frames + p_frames
                else:
                    sample_indices = np.linspace(0, len(p_frames) - 1, need_p, dtype=int)
                    sampled_p = [p_frames[idx] for idx in sample_indices]
                    combined = i_frames + sampled_p
                return sorted(combined)
        else:
            combined = i_frames + p_frames
            return sorted(combined)

    else:
        raise ValueError(f"未知关键帧方法: {method}")


def _load_frames_from_jpg(video_subdir, indices):
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


def extract_features_from_frames(model, frames, video_id, features_dir, device,
                                 batch_size=4, expected_dims=3840, use_cache=True):
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
        if isinstance(frames, list):
            frames = np.stack(frames, axis=0)
            frames = torch.from_numpy(frames).float()
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
        features = torch.cat(features_list, dim=0)
        if features.dim() != 3 or features.shape[1] != 9:
            return None
        if features.shape[2] != expected_dims:
            return None

        if use_cache and features_dir is not None:
            np.save(features_file, features.cpu().numpy())
        return features
    except Exception:
        return None


def compute_keyframe_similarity_matrix(q_feat, db_feat):
    q_frame = q_feat.mean(dim=1)
    db_frame = db_feat.mean(dim=1)
    q_frame = F.normalize(q_frame, p=2, dim=-1)
    db_frame = F.normalize(db_frame, p=2, dim=-1)
    sim = torch.mm(q_frame, db_frame.t())
    return sim.cpu().numpy()


def find_candidate_pairs(sim_matrix, threshold):
    pairs = np.argwhere(sim_matrix >= threshold)
    return [(int(i), int(j)) for i, j in pairs]


def save_similarity_matrix(matrix, q_id, db_id, sim_dir):
    subdir = os.path.join(sim_dir, q_id)
    os.makedirs(subdir, exist_ok=True)
    filepath = os.path.join(subdir, f"{q_id}_{db_id}.npy")
    np.save(filepath, matrix)


def load_similarity_matrix(q_id, db_id, sim_dir):
    filepath = os.path.join(sim_dir, q_id, f"{q_id}_{db_id}.npy")
    try:
        return np.load(filepath)
    except Exception:
        return None


def load_features(video_id, features_dir, expected_dims):
    feat_path = os.path.join(features_dir, video_id, f"{video_id}.npy")
    try:
        feat_np = np.load(feat_path)
        feat = torch.from_numpy(feat_np).float()
        if feat.dim() == 3 and feat.shape[1] == 9 and feat.shape[2] == expected_dims:
            return feat
        else:
            return None
    except Exception:
        return None


def extract_full_dense_frames(video_path, video_id, dense_frames_base_dir, dense_fps):
    save_dir = os.path.join(dense_frames_base_dir, 'queries', video_id, 'full')
    os.makedirs(save_dir, exist_ok=True)
    existing = [f for f in os.listdir(save_dir) if f.startswith('frame_') and f.endswith('.jpg')]
    if existing:
        return len(existing)

    reader = None
    try:
        reader = VideoFrameReader(video_path)
        total_frames = len(reader)
        fps = reader.fps
        step = max(1, int(round(fps / dense_fps)))
        indices = list(range(0, total_frames, step))
        if not indices:
            return 0
        frames_np = reader.get_frames(indices)
        if frames_np.size == 0:
            return 0
        for i, frame_idx in enumerate(indices):
            img = frames_np[i]
            jpg_path = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
            Image.fromarray(img).save(jpg_path, quality=90)
        return len(indices)
    except Exception:
        return 0
    finally:
        if reader is not None:
            reader.close()


def extract_dense_frames_range_and_save(video_path, start_idx, end_idx, step,
                                        save_dir, max_size=None):
    reader = None
    try:
        reader = VideoFrameReader(video_path)
        total = reader.total_frames
        if start_idx < 0 or end_idx >= total or start_idx > end_idx:
            return 0
        indices = list(range(start_idx, end_idx + 1, step))
        if not indices:
            return 0
        frames_np = reader.get_frames(indices)
        if frames_np.size == 0:
            return 0

        os.makedirs(save_dir, exist_ok=True)
        for i, frame_idx in enumerate(indices):
            img = frames_np[i]
            if max_size is not None and (img.shape[0] > max_size or img.shape[1] > max_size):
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
    import glob
    jpg_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if not jpg_files:
        return None
    frames = []
    for f in jpg_files:
        img = Image.open(f).convert('RGB')
        frames.append(np.array(img))
    frames_np = np.stack(frames, axis=0)
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
    if frames_tensor.shape[2] != 224 or frames_tensor.shape[3] != 224:
        frames_tensor = F.interpolate(frames_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    return frames_tensor.permute(0, 2, 3, 1)


# stage_utils.py (完整文件中仅修改此函数)
# [修改点] 新增：流式读取帧，避免内存溢出
def get_sorted_frames_generator(frame_dir):
    """
    流式读取目录下的 frame_*.jpg 文件，按帧索引排序后逐个 yield。
    返回生成器，每次 yield (frame_idx, img_np)。
    若目录无有效帧，返回 None。
    """
    import glob
    jpg_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if not jpg_files:
        return None

    # 先收集所有 (idx, path) 对并排序
    frame_items = []
    for f in jpg_files:
        base = os.path.basename(f)
        idx_str = base.replace('frame_', '').replace('.jpg', '')
        try:
            frame_idx = int(idx_str)
            frame_items.append((frame_idx, f))
        except ValueError:
            continue
    if not frame_items:
        return None
    frame_items.sort(key=lambda x: x[0])

    def generator():
        for idx, path in frame_items:
            try:
                img = Image.open(path).convert('RGB')
                img_np = np.array(img)
                yield idx, img_np
            except Exception:
                continue

    return generator()


def get_sorted_frames_count(frame_dir):
    """返回目录下有效帧的数量，不加载图像数据"""
    import glob
    jpg_files = glob.glob(os.path.join(frame_dir, "frame_*.jpg"))
    return len(jpg_files)


# [修改点] 原 get_sorted_frames_from_dir 保留但标记为可能内存危险，建议使用生成器版本
def get_sorted_frames_from_dir(frame_dir):
    """
    注意：此函数可能因帧数过多导致内存溢出，建议使用 get_sorted_frames_generator。
    """
    import glob
    jpg_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if not jpg_files:
        return None
    frames = []
    for f in jpg_files:
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
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_interval_metadata(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def merge_db_intervals(intervals):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x['db_start'])
    merged = []
    current = sorted_intervals[0].copy()
    for next_int in sorted_intervals[1:]:
        if next_int['db_start'] <= current['db_end'] + 1:
            current['db_end'] = max(current['db_end'], next_int['db_end'])
        else:
            merged.append(current)
            current = next_int.copy()
    merged.append(current)
    return merged


def compute_dense_similarity(model, feat_q, feat_db):
    if feat_q.size(0) < 4 or feat_db.size(0) < 4:
        q_avg = feat_q.mean(dim=(0, 1))
        db_avg = feat_db.mean(dim=(0, 1))
        q_avg = F.normalize(q_avg, p=2, dim=-1)
        db_avg = F.normalize(db_avg, p=2, dim=-1)
        sim_val = torch.dot(q_avg, db_avg).item()
        return sim_val

    if feat_q.dim() == 3:
        feat_q = feat_q.unsqueeze(0)
    if feat_db.dim() == 3:
        feat_db = feat_db.unsqueeze(0)
    with torch.no_grad():
        sim = model.calculate_video_similarity(feat_q, feat_db)
    return sim.item()


def compute_and_save_heatmap(model, feat_q, feat_db, save_path, matrix_save_path=None, vmin=None, vmax=None, do_plot=True):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    with torch.no_grad():
        f2f_sim = model.calculate_f2f_matrix(feat_q.unsqueeze(0), feat_db.unsqueeze(0))
        if f2f_sim.dim() == 5:
            matrix = f2f_sim[0, :, :, 0, 0].cpu().numpy()
        elif f2f_sim.dim() == 4:
            matrix = f2f_sim[0, :, :, 0].cpu().numpy()
        elif f2f_sim.dim() == 3:
            matrix = f2f_sim[0, :, :].cpu().numpy()
        elif f2f_sim.dim() == 2:
            matrix = f2f_sim.cpu().numpy()
        else:
            return

        if matrix.size == 0 or matrix.ndim != 2:
            return

    if matrix_save_path is not None:
        os.makedirs(os.path.dirname(matrix_save_path), exist_ok=True)
        np.save(matrix_save_path, matrix)

    if not do_plot:
        return

    cmap = 'RdBu' if matrix.min() < 0 else 'Reds'
    if vmin is None:
        vmin = matrix.min()
    if vmax is None:
        vmax = matrix.max()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Similarity')
    plt.xlabel('Database frame')
    plt.ylabel('Query frame')
    plt.title('Frame-to-Frame Similarity Matrix')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("stage_utils 模块已加载。请在其他脚本中导入使用。")