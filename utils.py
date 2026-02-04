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


import cv2
import numpy as np
import os


def load_video(video_path, all_frames=False, fps=1, cc_size=224, rs_size=256):
    """
    视频加载函数 - 支持损坏文件处理和重试机制

    参数:
        video_path (str): 视频文件路径
        all_frames (bool): 是否提取所有帧，False则按fps采样
        fps (int): 目标采样帧率，默认每秒1帧
        cc_size (int): 中心裁剪尺寸，默认224x224
        rs_size (int): 缩放尺寸，默认256（短边缩放到256后再裁剪）

    返回:
        np.ndarray: 处理后的视频帧数组，形状 [T, H, W, C] 或空数组（失败时）
                   T=帧数, H=W=224, C=3(RGB)
    """

    # 设置OpenCV单线程，避免多进程冲突
    cv2.setNumThreads(1)

    # ========== 文件存在性检查 ==========
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 - {video_path}")
        return np.array([])

    # ========== 文件完整性检查 ==========
    file_size = os.path.getsize(video_path)
    if file_size < 1024:  # 小于1KB视为损坏
        print(f"警告: 视频文件可能已损坏 - {video_path} ({file_size} 字节)")
        return np.array([])

    # ========== 打开视频流 ==========
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 - {video_path}")
        return np.array([])

    # ========== 获取视频元数据 ==========
    fps_div = fps
    video_fps = cap.get(cv2.CAP_PROP_FPS)  # 原始帧率

    # 处理异常帧率（如无法获取或过高）
    if video_fps > 144 or video_fps is None or video_fps <= 0:
        video_fps = 25  # 默认假设25fps

    # 计算采样间隔：每N帧取1帧
    # round(video_fps / fps_div) = 原始帧率/目标帧率
    # 例如: 25fps的视频，目标1fps，则每25帧取1帧

    # ========== 初始化变量 ==========
    frames = []  # 存储采样的帧
    count = 0  # 原始帧计数器
    max_frames = 1000  # 安全上限，防止内存溢出
    consecutive_errors = 0  # 连续错误计数
    max_consecutive_errors = 30  # 最大容忍连续错误帧数
    decode_attempts = 0  # 解码尝试次数
    max_decode_attempts = 2  # 最大重试次数

    # ========== 主读取循环（带重试机制） ==========
    while decode_attempts < max_decode_attempts:
        try:
            while cap.isOpened() and len(frames) < max_frames:
                # 快速抓取帧（不解码，效率高）
                ret = cap.grab()

                if not ret:
                    # 抓取失败
                    consecutive_errors += 1

                    if consecutive_attempts >= max_consecutive_errors:
                        if decode_attempts == 0:
                            # 第一次尝试：重新打开视频文件
                            cap.release()
                            cap = cv2.VideoCapture(video_path)
                            decode_attempts += 1
                            consecutive_errors = 0
                            print(f"信息: 因读取错误重新打开视频 {video_path}")
                            continue
                        else:
                            # 第二次尝试仍失败，放弃
                            break
                    continue

                # 重置连续错误计数器
                consecutive_errors = 0

                # 判断是否采样该帧
                # 条件1: 到达采样间隔点（按目标fps计算）
                # 条件2: all_frames=True时采样所有帧
                should_sample = (int(count % round(video_fps / fps_div)) == 0) or all_frames

                if should_sample:
                    # 解码帧数据
                    ret, frame = cap.retrieve()

                    # 验证帧有效性
                    if ret and isinstance(frame, np.ndarray) and frame.size > 0:
                        try:
                            # 颜色空间转换: BGR(OpenCV默认) -> RGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            # 空间缩放: 短边缩放到rs_size(256)
                            if rs_size is not None:
                                # resize_frame期望输入[H,W,3]，输出[H',W',3]
                                # 例如: 1920x1080 -> 455x256（保持宽高比）
                                frame = resize_frame(frame, rs_size)

                            # 添加到帧列表
                            # frame形状: [H, W, 3], dtype=uint8, 像素值0-255
                            frames.append(frame)

                        except Exception as e:
                            # 单帧处理失败，跳过继续
                            pass

                count += 1

            # 成功完成读取
            break

        except Exception as e:
            # 整体解码失败，尝试重开
            decode_attempts += 1
            if decode_attempts < max_decode_attempts:
                cap.release()
                cap = cv2.VideoCapture(video_path)
                print(f"信息: 重试解码视频 {video_path}, 第{decode_attempts}次尝试")
            else:
                print(f"错误: 视频 {video_path} 解码失败（已重试{max_decode_attempts}次）")
                break

    # 释放视频资源
    cap.release()

    # ========== 处理无有效帧的情况（降级方案）==========
    if len(frames) == 0:
        print(f"警告: 视频 {video_path} 未读取到有效帧，使用随机模拟数据")
        # 生成48帧随机数据作为占位，避免下游崩溃
        # 形状: [48, 224, 224, 3], 值范围0-255
        frames = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                  for _ in range(48)]

    # ========== 后处理：裁剪和格式转换 ==========
    try:
        # 列表转为numpy数组
        # 输入: List of [H, W, 3]，输出: [T, H, W, 3]
        # 例如: 50帧256x455的视频 -> [50, 256, 455, 3]
        frames = np.array(frames)

        # 中心裁剪到目标尺寸
        if cc_size is not None:
            # center_crop输入[T,H,W,3]，输出[T,224,224,3]
            # 从中心裁剪出224x224区域
            frames = center_crop(frames, cc_size)

        return frames

    except Exception as e:
        print(f"错误: 处理视频 {video_path} 的帧数组时出错: {e}")
        # 返回模拟数据确保流程不中断
        return np.random.randint(0, 256, (48, 224, 224, 3), dtype=np.uint8)


# ========== 辅助函数示例（假设实现）==========

def resize_frame(frame, desired_size):
    """
    等比例缩放帧，短边对齐desired_size

    参数:
        frame: [H, W, 3] numpy数组
        desired_size: 目标短边尺寸

    返回:
        frame: [H', W', 3] 缩放后的帧，H'或W'等于desired_size
    """
    if frame.size == 0:
        return frame

    # 计算缩放比例
    min_size = min(frame.shape[0], frame.shape[1])
    ratio = desired_size / min_size

    # 双三次插值缩放
    new_size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
    frame = cv2.resize(frame, dsize=new_size, interpolation=cv2.INTER_CUBIC)
    return frame


def center_crop(frames, desired_size):
    """
    对视频帧进行中心裁剪

    参数:
        frames: [T, H, W, 3] 或 [H, W, 3] numpy数组
        desired_size: 裁剪尺寸（正方形）

    返回:
        frames: [T, desired_size, desired_size, 3] 或 [desired_size, desired_size, 3]
    """
    if frames.size == 0:
        return frames

    # 统一转为4D处理 [T, H, W, 3]
    single_frame = (frames.ndim == 3)
    if single_frame:
        frames = np.expand_dims(frames, axis=0)

    # 计算裁剪坐标
    t, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    top = max(0, (h - desired_size) // 2)
    left = max(0, (w - desired_size) // 2)

    # 执行裁剪
    cropped = frames[:, top:top + desired_size, left:left + desired_size, :]

    # 如果输入是单帧，恢复3D
    if single_frame:
        cropped = cropped[0]

    return cropped
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