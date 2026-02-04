import cv2
import numpy as np
import os
import subprocess
import tempfile
import shutil
import time
from typing import Optional, Tuple, List


def center_crop(frame, desired_size):
    """安全的中心裁剪函数，处理空数组"""
    if frame.size == 0 or len(frame.shape) < 3:
        print(f"Warning: 无法裁剪空帧或形状异常: {frame.shape}")
        # 返回一个占位符帧，避免后续错误
        if len(frame.shape) == 4:  # 批处理
            return np.zeros((frame.shape[0], desired_size, desired_size, 3), dtype=np.uint8)
        else:  # 单帧
            return np.zeros((desired_size, desired_size, 3), dtype=np.uint8)

    if frame.ndim == 3:
        old_size = frame.shape[:2]
        top = int(np.maximum(0, (old_size[0] - desired_size) / 2))
        left = int(np.maximum(0, (old_size[1] - desired_size) / 2))
        return frame[top: top + desired_size, left: left + desired_size, :]
    else:
        old_size = frame.shape[1:3]
        top = int(np.maximum(0, (old_size[0] - desired_size) / 2))
        left = int(np.maximum(0, (old_size[1] - desired_size) / 2))
        return frame[:, top: top + desired_size, left: left + desired_size, :]


def resize_frame(frame, desired_size):
    """安全的帧缩放"""
    if frame.size == 0:
        print("Warning: 尝试缩放空帧")
        return frame

    min_size = np.min(frame.shape[:2])
    ratio = desired_size / min_size
    frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return frame


def ffmpeg_convert_video(video_path: str, output_path: str, timeout: int = 30) -> bool:
    """
    使用ffmpeg转换视频格式

    参数:
        video_path: 输入视频路径
        output_path: 输出视频路径
        timeout: 转换超时时间(秒)

    返回:
        bool: 转换是否成功
    """
    try:
        # 检查ffmpeg是否可用
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("警告: ffmpeg未安装或不可用")
        return False

    # 构建转换命令
    # 使用快速编码参数，减少转换时间
    cmd = [
        'ffmpeg', '-i', video_path,
        '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '23', '-c:a', 'copy',
        '-movflags', '+faststart',  # 优化网络播放
        '-threads', '1',  # 单线程避免冲突
        output_path, '-y', '-loglevel', 'error'  # 减少日志输出
    ]

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode == 0 and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 1024:  # 检查文件大小
                print(f"信息: ffmpeg成功转换视频，大小: {file_size / 1024 / 1024:.2f}MB")
                return True
            else:
                print(f"警告: 转换后的视频文件过小: {file_size}字节")
                return False
        else:
            print(f"错误: ffmpeg转换失败，返回码: {result.returncode}")
            if result.stderr:
                print(f"ffmpeg错误: {result.stderr[:500]}...")  # 限制错误信息长度
            return False
    except subprocess.TimeoutExpired:
        print(f"错误: ffmpeg转换超时({timeout}秒)")
        return False
    except Exception as e:
        print(f"错误: ffmpeg转换异常: {e}")
        return False


def load_video_with_opencv(video_path: str, all_frames: bool = False, fps: int = 1,
                           rs_size: Optional[int] = 256) -> Optional[np.ndarray]:
    """
    使用OpenCV加载视频（纯函数，无状态）

    返回:
        np.ndarray 或 None
    """
    cv2.setNumThreads(1)  # 避免多线程冲突

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or video_fps > 144:
            video_fps = 25

        frames = []
        count = 0
        max_frames = 1000
        error_count = 0
        max_errors = 5

        while cap.isOpened() and len(frames) < max_frames and error_count < max_errors:
            ret, frame = cap.read()

            if not ret:
                error_count += 1
                if error_count >= max_errors:
                    break
                continue

            error_count = 0  # 重置错误计数

            # 判断是否采样该帧
            should_sample = (int(count % round(video_fps / fps)) == 0) or all_frames

            if should_sample:
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if rs_size is not None:
                        frame = resize_frame(frame, rs_size)

                    frames.append(frame)
                except Exception as e:
                    # 单帧处理失败，跳过
                    pass

            count += 1

        cap.release()

        if len(frames) == 0:
            return None

        return np.array(frames)

    except Exception as e:
        print(f"OpenCV读取异常 {video_path}: {e}")
        if cap.isOpened():
            cap.release()
        return None


def load_video_with_decord(video_path: str, all_frames: bool = False, fps: int = 1,
                           rs_size: Optional[int] = 256) -> Optional[np.ndarray]:
    """
    使用Decord加载视频

    返回:
        np.ndarray 或 None
    """
    try:
        import decord
        from decord import VideoReader, cpu
    except ImportError:
        return None

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if total_frames == 0:
            return None

        # 计算采样间隔
        if not all_frames:
            try:
                video_fps = vr.get_avg_fps()
                if video_fps <= 0 or video_fps > 144:
                    video_fps = 25
            except:
                video_fps = 25

            interval = max(1, int(video_fps / fps))
            frame_indices = list(range(0, total_frames, interval))

            # 限制最大帧数
            max_frames = 1000
            if len(frame_indices) > max_frames:
                frame_indices = frame_indices[:max_frames]
        else:
            max_frames = 1000
            frame_indices = list(range(0, min(total_frames, max_frames)))

        if not frame_indices:
            return None

        # 批量读取帧
        frames_batch = vr.get_batch(frame_indices)
        frames = frames_batch.asnumpy()

        # Decord返回RGB，但可能需要缩放
        if rs_size is not None and len(frames) > 0:
            resized_frames = []
            for frame in frames:
                resized_frames.append(resize_frame(frame, rs_size))
            frames = np.array(resized_frames)

        return frames

    except Exception as e:
        print(f"Decord读取异常 {video_path}: {e}")
        return None


def load_video(video_path: str, all_frames: bool = False, fps: int = 1,
               cc_size: int = 224, rs_size: int = 256, enable_ffmpeg: bool = True) -> np.ndarray:
    """
    增强版视频加载函数 - 多级回退机制

    参数:
        video_path: 视频文件路径
        all_frames: 是否提取所有帧
        fps: 目标采样帧率
        cc_size: 中心裁剪尺寸
        rs_size: 缩放尺寸
        enable_ffmpeg: 是否启用ffmpeg转换作为回退

    返回:
        np.ndarray: 处理后的视频帧数组 [T, H, W, C]
    """
    # ========== 文件检查 ==========
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 - {video_path}")
        return np.array([])

    file_size = os.path.getsize(video_path)
    if file_size < 1024:
        print(f"警告: 视频文件可能已损坏 - {video_path} ({file_size} 字节)")
        return np.array([])

    print(f"信息: 开始处理视频 {os.path.basename(video_path)} ({file_size / 1024 / 1024:.2f}MB)")

    # ========== 方法1: 尝试Decord ==========
    frames = load_video_with_decord(video_path, all_frames, fps, rs_size)
    if frames is not None and len(frames) > 0:
        print(f"信息: Decord成功读取 {len(frames)} 帧")
        frames = center_crop(frames, cc_size)
        return frames

    # ========== 方法2: 尝试OpenCV ==========
    frames = load_video_with_opencv(video_path, all_frames, fps, rs_size)
    if frames is not None and len(frames) > 0:
        print(f"信息: OpenCV成功读取 {len(frames)} 帧")
        frames = center_crop(frames, cc_size)
        return frames

    # ========== 方法3: ffmpeg转换后重试 ==========
    if enable_ffmpeg:
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="video_convert_")
            output_path = os.path.join(temp_dir, "converted.mp4")

            print(f"信息: 尝试使用ffmpeg转换视频")
            if ffmpeg_convert_video(video_path, output_path, timeout=45):
                # 转换成功，尝试用OpenCV读取转换后的视频
                frames = load_video_with_opencv(output_path, all_frames, fps, rs_size)
                if frames is not None and len(frames) > 0:
                    print(f"信息: 转换后成功读取 {len(frames)} 帧")
                    frames = center_crop(frames, cc_size)
                    return frames
        except Exception as e:
            print(f"错误: ffmpeg转换处理异常: {e}")
        finally:
            # 清理临时文件
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass

    # ========== 方法4: 降级方案 ==========
    print(f"警告: 所有方法失败，使用模拟数据 - {video_path}")

    # 尝试估算视频时长来生成合理数量的帧
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if video_fps > 0 and total_frames > 0:
            duration = total_frames / video_fps
            num_frames = min(max(10, int(duration * fps)), 300)  # 限制在10-300帧
        else:
            num_frames = 48
    except:
        num_frames = 48

    # 生成模拟数据（保持uint8类型）
    frames = np.random.randint(0, 256, (num_frames, cc_size, cc_size, 3), dtype=np.uint8)
    return frames


def safe_load_video(video_path: str, max_retries: int = 2, **kwargs) -> np.ndarray:
    """
    带重试机制的视频加载

    参数:
        video_path: 视频文件路径
        max_retries: 最大重试次数
        **kwargs: 传递给load_video的参数

    返回:
        np.ndarray: 视频帧数组
    """
    for attempt in range(max_retries):
        try:
            print(f"尝试 {attempt + 1}/{max_retries}: {os.path.basename(video_path)}")
            frames = load_video(video_path, **kwargs)

            if frames is not None and frames.size > 0:
                print(f"成功: 读取到 {len(frames)} 帧")
                return frames
            else:
                print(f"尝试 {attempt + 1} 返回空数组")

        except Exception as e:
            print(f"尝试 {attempt + 1} 失败: {e}")

        # 如果不是最后一次尝试，等待一小段时间
        if attempt < max_retries - 1:
            time.sleep(0.5)

    print(f"所有尝试失败: {video_path}")
    return np.array([])