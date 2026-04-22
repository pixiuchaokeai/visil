# -*- coding: utf-8 -*-
"""
基于镜头边界检测的关键帧抽取算法
原理：
1. 使用颜色直方图差异检测镜头边界（切变）
2. 对于渐变镜头，使用更复杂的模型检测
3. 从每个镜头中选取关键帧（第一帧、中间帧、最后一帧）
"""

import cv2
import os
import numpy as np
import sys


def compute_histogram(frame, bins=64):
    """
    计算帧的颜色直方图

    参数:
        frame: 输入帧 (BGR格式)
        bins: 直方图的bin数量

    返回:
        hist: 归一化的颜色直方图
    """
    # 转换到HSV颜色空间（对光照变化更鲁棒）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 计算HSV直方图
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    # 合并三个通道的直方图
    hist = np.concatenate([hist_h, hist_s, hist_v])

    # 归一化直方图
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist.flatten()


def detect_shot_boundaries(video_path, threshold=0.5, min_shot_length=12):
    """
    检测镜头边界

    参数:
        video_path: 视频文件路径
        threshold: 直方图差异阈值，用于检测镜头切换
        min_shot_length: 最小镜头长度（帧数）

    返回:
        shot_boundaries: 镜头边界列表，每个元素为[开始帧, 结束帧]
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25  # 默认帧率

    shot_boundaries = []
    frame_count = 0
    prev_hist = None
    shot_start = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 计算当前帧的直方图
        curr_hist = compute_histogram(frame)

        if prev_hist is not None:
            # 计算直方图差异（使用巴氏距离）
            diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)

            # 如果差异超过阈值，检测到镜头边界
            if diff > threshold:
                # 确保镜头长度至少为min_shot_length
                if frame_count - shot_start >= min_shot_length:
                    shot_boundaries.append([shot_start, frame_count - 1])
                    shot_start = frame_count

        prev_hist = curr_hist
        frame_count += 1

    cap.release()

    # 添加最后一个镜头
    if shot_start < frame_count:
        shot_boundaries.append([shot_start, frame_count - 1])

    return shot_boundaries


def extract_keyframes_from_shot(video_path, shot_boundary, max_keyframes=3):
    """
    从单个镜头中抽取关键帧

    参数:
        video_path: 视频文件路径
        shot_boundary: 镜头边界 [开始帧, 结束帧]
        max_keyframes: 每个镜头最多抽取的关键帧数

    返回:
        keyframe_indices: 关键帧索引列表
    """
    start_frame, end_frame = shot_boundary
    shot_length = end_frame - start_frame + 1

    # 选择关键帧的策略
    keyframe_indices = []

    if shot_length <= max_keyframes:
        # 如果镜头长度小于等于最大关键帧数，取所有帧
        for i in range(start_frame, end_frame + 1):
            keyframe_indices.append(i)
    else:
        # 1. 第一帧
        keyframe_indices.append(start_frame)

        # 2. 中间帧（如果镜头足够长）
        if shot_length >= 3:
            middle_frame = start_frame + shot_length // 2
            keyframe_indices.append(middle_frame)

        # 3. 最后一帧（如果镜头足够长）
        if shot_length >= 2:
            keyframe_indices.append(end_frame)

    # 如果关键帧数超过最大值，均匀采样
    if len(keyframe_indices) > max_keyframes:
        step = shot_length // max_keyframes
        keyframe_indices = [start_frame + i * step for i in range(max_keyframes)]
        # 确保不超过结束帧
        keyframe_indices = [min(idx, end_frame) for idx in keyframe_indices]

    return sorted(set(keyframe_indices))  # 去重并排序


def extract_keyframes(video_path, output_dir, threshold=0.5, min_shot_length=12, max_keyframes_per_shot=3):
    """
    主函数：抽取视频的关键帧

    参数:
        video_path: 视频文件路径
        output_dir: 关键帧输出目录
        threshold: 镜头边界检测阈值
        min_shot_length: 最小镜头长度
        max_keyframes_per_shot: 每个镜头最多抽取的关键帧数

    返回:
        all_keyframes: 所有关键帧的索引列表
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 检测镜头边界
    print(f"检测镜头边界: {video_path}")
    shot_boundaries = detect_shot_boundaries(video_path, threshold, min_shot_length)
    print(f"检测到 {len(shot_boundaries)} 个镜头")

    # 2. 从每个镜头中抽取关键帧
    all_keyframes = []
    for i, shot in enumerate(shot_boundaries):
        shot_keyframes = extract_keyframes_from_shot(video_path, shot, max_keyframes_per_shot)
        all_keyframes.extend(shot_keyframes)
        print(f"镜头 {i + 1}: 开始帧={shot[0]}, 结束帧={shot[1]}, 关键帧={shot_keyframes}")

    # 去重并排序
    all_keyframes = sorted(set(all_keyframes))

    # 3. 保存关键帧
    cap = cv2.VideoCapture(video_path)
    saved_frames = []

    for frame_idx in all_keyframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # 生成文件名
            frame_filename = f"keyframe_{frame_idx:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            # 保存帧
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_idx)
            print(f"保存关键帧: {frame_path}")

    cap.release()

    return saved_frames


def file_sort(name):
    """
    读取标签文件并排序视频名

    参数:
        name: 标签文件路径

    返回:
        index_list: 排序后的视频名列表
    """
    index_list = []
    with open(name, "r", encoding="utf-8") as f1:
        for line in f1:
            items = line.split(',')
            query_video_name = items[0]
            index_list.append(query_video_name)
        index_list = sorted(set(index_list))
    f1.close()
    print(f"找到 {len(index_list)} 个唯一视频")
    return index_list


def smooth(x, window_len=13, window='hanning'):
    """
    平滑信号

    参数:
        x: 输入信号
        window_len: 平滑窗口长度
        window: 窗口类型

    返回:
        y: 平滑后的信号
    """
    if len(x) < window_len:
        return x

    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]

    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)

    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


if __name__ == "__main__":
    # 参数设置
    THRESHOLD = 0.5  # 镜头边界检测阈值（巴氏距离）
    MIN_SHOT_LENGTH = 12  # 最小镜头长度（帧）
    MAX_KEYFRAMES_PER_SHOT = 3  # 每个镜头最多关键帧数

    # 文件路径
    label_txt = './vcdb_index.txt'
    txt_index_path = 'vcdb_shot_based_index_all.txt'
    video_dir = "./core_dataset/"

    # 0. 排序视频文件
    print("开始排序视频文件...")
    index_list = file_sort(label_txt)

    # 创建总的关键帧目录
    if not os.path.exists('./shot_based_keyframes/'):
        os.mkdir('./shot_based_keyframes/')

    # 1. 为每个视频抽取关键帧
    with open(txt_index_path, 'w', encoding='UTF-8') as f2:
        index = 0
        for video in index_list:
            videopath = os.path.join(video_dir, video)
            basename = video.split(".")[0]
            dir = f'./shot_based_keyframes/{basename}'

            print(f"\n处理视频 {index + 1}/{len(index_list)}: {video}")
            print(f"视频路径: {videopath}")
            print(f"帧保存目录: {dir}")

            try:
                # 抽取关键帧
                keyframe_list = extract_keyframes(
                    videopath,
                    dir,
                    threshold=THRESHOLD,
                    min_shot_length=MIN_SHOT_LENGTH,
                    max_keyframes_per_shot=MAX_KEYFRAMES_PER_SHOT
                )

                # 获取视频帧率
                cap = cv2.VideoCapture(videopath)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

                # 写入索引文件
                f2.writelines(f"{video},{fps},{keyframe_list}\n")
                print(f"视频 {video} 抽取了 {len(keyframe_list)} 个关键帧")

            except Exception as e:
                print(f"处理视频 {video} 时出错: {str(e)}")
                # 写入错误信息
                f2.writelines(f"{video},0,[]\n")

            index += 1
            print("=" * 50)

    f2.close()
    # print(f"\n关键帧抽取完成！结果保存在: {vcdb_shot_based_index_all.txt}")