import cv2
import numpy as np
import os

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


def load_video(video_path, all_frames=False, fps=1, cc_size=224, rs_size=256):
    """
    更健壮的视频加载函数，处理损坏的视频文件
    """
    cv2.setNumThreads(1)

    # 检查文件是否存在

    if not os.path.exists(video_path):
        print(f"Error: 视频文件不存在 - {video_path}")
        return np.array([])

    # 检查文件大小
    file_size = os.path.getsize(video_path)
    if file_size < 1024:  # 小于1KB的文件可能损坏
        print(f"Warning: 视频文件过小可能损坏 - {video_path} ({file_size} bytes)")
        return np.array([])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 - {video_path}")
        return np.array([])

    fps_div = fps
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps > 144 or video_fps is None or video_fps <= 0:
        video_fps = 25

    frames = []
    count = 0
    max_frames = 1000  # 限制最大帧数，避免内存问题
    consecutive_errors = 0
    max_consecutive_errors = 30

    try:
        while cap.isOpened() and len(frames) < max_frames:
            ret = cap.grab()
            if not ret:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Warning: 连续读取失败，停止读取视频 - {video_path}")
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
                        print(f"Warning: 处理帧时出错 - {video_path}, 错误: {e}")
                else:
                    # 跳过无效帧
                    pass

            count += 1
    except Exception as e:
        print(f"Error: 读取视频时发生异常 - {video_path}, 错误: {e}")
    finally:
        cap.release()

    if len(frames) == 0:
        print(f"Warning: 视频没有读取到任何有效帧 - {video_path}")
        return np.array([])

    try:
        frames = np.array(frames)
        if cc_size is not None:
            frames = center_crop(frames, cc_size)
        return frames
    except Exception as e:
        print(f"Error: 处理帧数组时出错 - {video_path}, 错误: {e}")
        return np.array([])


# 添加一个专门处理损坏视频的函数
def safe_load_video(video_path, max_retries=2):
    """带重试机制的视频加载"""
    for attempt in range(max_retries):
        try:
            frames = load_video(video_path)
            if frames.size > 0:
                return frames
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {video_path}: {e}")

    print(f"All attempts failed for {video_path}, returning empty array")
    return np.array([])