import cv2
import numpy as np
import os
import warnings


def normalize_path_input(path):
    """
    标准化输入路径，兼容所有相对路径格式
    """
    if not path:
        return path

    # 统一斜杠方向
    path = path.replace('\\', '/')

    # 移除开头的./（如果存在）
    if path.startswith('./'):
        path = path[2:]

    return path


def find_file_path(file_path):
    """
    查找文件的实际路径，尝试多种可能性
    """
    # 尝试直接路径
    if os.path.exists(file_path):
        return os.path.abspath(file_path)

    # 尝试添加./前缀
    test_path = f"./{file_path}"
    if os.path.exists(test_path):
        return os.path.abspath(test_path)

    # 尝试在当前目录下查找
    base_name = os.path.basename(file_path)
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == base_name:
                return os.path.abspath(os.path.join(root, file))

    return None


def center_crop(frame, desired_size):
    """安全的中心裁剪函数，处理空数组"""
    if frame.size == 0:
        warnings.warn(f"无法裁剪空帧")
        return np.zeros((1, desired_size, desired_size, 3), dtype=np.uint8)

    # 确保是 4D 数组 [T, H, W, C]
    if frame.ndim == 3:
        frame = np.expand_dims(frame, axis=0)

    # 现在统一按 4D 处理
    if frame.ndim == 4:
        old_size = frame.shape[1:3]  # (H, W)
        top = int(np.maximum(0, (old_size[0] - desired_size) / 2))
        left = int(np.maximum(0, (old_size[1] - desired_size) / 2))

        # 确保不超出边界
        top = min(top, max(0, frame.shape[1] - desired_size))
        left = min(left, max(0, frame.shape[2] - desired_size))

        cropped = frame[:, top: top + desired_size, left: left + desired_size, :]

        # 如果裁剪后尺寸不对，进行resize
        if cropped.shape[1] != desired_size or cropped.shape[2] != desired_size:
            # 使用OpenCV逐帧resize
            resized_frames = []
            for f in cropped:
                resized = cv2.resize(f, (desired_size, desired_size), interpolation=cv2.INTER_CUBIC)
                resized_frames.append(resized)
            cropped = np.array(resized_frames)

        return cropped

    return frame


def resize_frame(frame, desired_size):
    """安全的帧缩放"""
    if frame.size == 0:
        warnings.warn("尝试缩放空帧")
        return frame

    min_size = np.min(frame.shape[:2])
    ratio = desired_size / min_size
    frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return frame


def load_video(video_path, all_frames=False, fps=1, cc_size=224, rs_size=256, max_attempts=3):
    """
    更健壮的视频加载函数，处理损坏的视频文件和不同路径格式
    支持重试机制
    """
    cv2.setNumThreads(1)

    # 标准化输入路径
    original_path = video_path
    video_path = normalize_path_input(video_path)

    # 查找文件的实际路径
    actual_path = find_file_path(video_path)

    if actual_path is None:
        warnings.warn(f"视频文件不存在 - 原始路径: {original_path}, 标准化后: {video_path}")
        return np.array([])

    # 尝试多次加载
    for attempt in range(max_attempts):
        try:
            frames = _load_video_internal(actual_path, all_frames, fps, cc_size, rs_size)
            if frames.size > 0:
                # 最终验证维度
                if len(frames.shape) != 4 or frames.shape[-1] != 3:
                    warnings.warn(f"视频 {actual_path} 维度异常: {frames.shape}")
                    return np.array([])
                return frames
            else:
                warnings.warn(f"第{attempt + 1}次尝试: 视频 {actual_path} 返回空数组")
        except Exception as e:
            warnings.warn(f"第{attempt + 1}次尝试加载视频 {actual_path} 时出错: {e}")

    warnings.warn(f"所有 {max_attempts} 次尝试都失败 - {actual_path}, 返回空数组")
    return np.array([])


def _load_video_internal(video_path, all_frames=False, fps=1, cc_size=224, rs_size=256):
    """内部视频加载函数"""
    # 检查文件大小
    try:
        file_size = os.path.getsize(video_path)
        if file_size < 1024:  # 小于1KB的文件可能损坏
            warnings.warn(f"视频文件过小可能损坏 - {video_path} ({file_size} bytes)")
            return np.array([])
    except Exception as e:
        warnings.warn(f"无法获取文件大小 - {video_path}: {e}")

    # 尝试不同的解码器
    codecs_to_try = [
        cv2.CAP_ANY,  # 自动检测
        cv2.CAP_FFMPEG,
        cv2.CAP_MSMF,  # Windows Media Foundation
    ]

    for codec in codecs_to_try:
        cap = cv2.VideoCapture(video_path, codec)
        if cap.isOpened():
            break

    if not cap.isOpened():
        warnings.warn(f"无法打开视频文件 - {video_path}")
        return np.array([])

    fps_div = fps
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps > 144 or video_fps is None or video_fps <= 0:
        video_fps = 25

    frames = []
    count = 0
    max_frames = 1000  # 限制最大帧数，避免内存问题
    consecutive_errors = 0
    max_consecutive_errors = 10

    try:
        while cap.isOpened() and len(frames) < max_frames:
            ret = cap.grab()
            if not ret:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    warnings.warn(f"连续读取失败，停止读取视频 - {video_path}")
                    break
                continue

            consecutive_errors = 0  # 重置错误计数

            if int(count % round(video_fps / fps_div)) == 0 or all_frames:
                ret, frame = cap.retrieve()
                if ret and isinstance(frame, np.ndarray) and frame.size > 0:
                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if rs_size is not None:
                            frame = resize_frame(frame, rs_size)
                        frames.append(frame)
                    except Exception as e:
                        warnings.warn(f"处理帧时出错 - {video_path}, 错误: {e}")
                else:
                    # 跳过无效帧
                    pass

            count += 1
    except Exception as e:
        warnings.warn(f"读取视频时发生异常 - {video_path}, 错误: {e}")
    finally:
        cap.release()

    if len(frames) == 0:
        warnings.warn(f"视频没有读取到任何有效帧 - {video_path}")
        return np.array([])

    try:
        frames = np.array(frames)

        # 确保是 4D 数组 [T, H, W, C]
        if frames.ndim == 3:
            frames = np.expand_dims(frames, axis=0)

        if cc_size is not None:
            frames = center_crop(frames, cc_size)

        # 最终验证：必须是4D且通道在最后
        if len(frames.shape) != 4 or frames.shape[-1] != 3:
            warnings.warn(f"处理后的帧数组维度异常: {frames.shape}，期望 (T, H, W, 3)")
            return np.array([])

        return frames
    except Exception as e:
        warnings.warn(f"处理帧数组时出错 - {video_path}, 错误: {e}")
        return np.array([])


def safe_load_video(video_path, max_retries=3):
    """带重试机制的视频加载（兼容性函数）"""
    return load_video(video_path, max_attempts=max_retries)