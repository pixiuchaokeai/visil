import os
import glob
import utils
import torch
import numpy as np
import warnings
from torch.utils.data import Dataset


class VideoGenerator(Dataset):
    """
    视频数据集生成器，从视频文件中加载视频数据
    继承自PyTorch的Dataset类，用于创建可迭代的数据集
    """

    def __init__(self, video_file, fps=1, cc_size=224, rs_size=256):
        """
        初始化视频生成器

        参数:
            video_file (str): 视频文件列表的文本文件路径
            fps (int): 视频的帧率，默认每秒1帧
            cc_size (int): 中心裁剪的尺寸，默认为224
            rs_size (int): 调整大小后的尺寸，默认为256
        """
        super(VideoGenerator, self).__init__()

        print(f"加载视频列表文件: {video_file}")

        # 使用更安全的方式加载文件，处理格式问题
        videos = []

        try:
            with open(video_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # 跳过空行和注释行
                    if not line or line.startswith('#'):
                        continue

                    # 分割行，支持制表符和多个空格
                    parts = line.split()

                    if len(parts) >= 2:
                        video_id = parts[0]
                        video_path = ' '.join(parts[1:])  # 处理路径中可能包含空格的情况

                        # 检查路径是否有效
                        if not os.path.exists(video_path):
                            warnings.warn(f"行 {line_num}: 视频文件不存在 - {video_path}")
                            # 仍然添加，让后续处理决定如何处理

                        videos.append([video_id, video_path])
                    else:
                        warnings.warn(f"行 {line_num}: 格式错误，跳过 - {line}")

            if not videos:
                raise ValueError(f"文件 {video_file} 中没有有效的视频条目")

            # 转换为numpy数组
            self.videos = np.array(videos, dtype=str)

            # 确保是二维数组
            if self.videos.ndim == 1:
                self.videos = np.expand_dims(self.videos, axis=0)

            print(f"成功加载 {len(self.videos)} 个视频")

        except Exception as e:
            raise RuntimeError(f"读取文件 {video_file} 时出错: {e}")

        self.fps = fps
        self.cc_size = cc_size
        self.rs_size = rs_size

        # 记录失败的视频
        self.failed_videos = []

    def __len__(self):
        """返回数据集中视频的数量"""
        return len(self.videos)

    def __getitem__(self, index):
        """
        根据索引获取视频数据和对应的标签

        参数:
            index (int): 视频索引

        返回:
            tuple: (视频张量, 视频标签)
                视频张量: 形状为[T, C, H, W]的视频数据，T是时间维度，C是通道数，H和W是高度和宽度
                视频标签: 视频的标签或类别
        """
        video_id = self.videos[index][0]
        video_path = self.videos[index][1]

        # 检查文件是否存在
        if not os.path.exists(video_path):
            warnings.warn(f"视频文件不存在: {video_path}")
            # 返回空张量
            empty_video = torch.zeros((1, 3, self.cc_size, self.cc_size), dtype=torch.float32)
            return empty_video, video_id

        try:
            # 加载视频数据
            video = utils.load_video(video_path, fps=self.fps, cc_size=self.cc_size, rs_size=self.rs_size)

            # 检查是否成功加载
            if video.size == 0:
                warnings.warn(f"视频加载失败或为空: {video_path}")
                self.failed_videos.append(video_id)
                # 返回空张量
                empty_video = torch.zeros((1, 3, self.cc_size, self.cc_size), dtype=torch.float32)
                return empty_video, video_id

            # 将numpy数组转换为PyTorch张量
            # 注意: utils.load_video返回的形状是[T, H, W, C]
            # 需要转换为[T, C, H, W]
            video = torch.from_numpy(video).permute(0, 3, 1, 2).float()

            return video, video_id

        except Exception as e:
            warnings.warn(f"处理视频 {video_id} ({video_path}) 时出错: {e}")
            self.failed_videos.append(video_id)
            # 返回空张量
            empty_video = torch.zeros((1, 3, self.cc_size, self.cc_size), dtype=torch.float32)
            return empty_video, video_id

    def get_failed_videos(self):
        """获取加载失败的视频列表"""
        return self.failed_videos


class DatasetGenerator(Dataset):
    """
    通用数据集生成器，根据视频ID从指定目录加载视频数据
    支持通过模式匹配查找视频文件
    """

    def __init__(self, rootDir, videos, pattern, fps=1, cc_size=224, rs_size=256):
        """
        初始化数据集生成器

        参数:
            rootDir (str): 视频文件所在的根目录路径
            videos (list): 视频ID列表
            pattern (str): 视频文件名模式，使用'{id}'作为占位符
            fps (int): 视频的帧率，默认每秒1帧
            cc_size (int): 中心裁剪的尺寸，默认为224
            rs_size (int): 调整大小后的尺寸，默认为256
        """
        super(DatasetGenerator, self).__init__()
        self.rootDir = rootDir
        self.videos = videos
        self.pattern = pattern
        self.fps = fps
        self.cc_size = cc_size
        self.rs_size = rs_size

        # 记录失败的视频
        self.failed_videos = []

    def __len__(self):
        """返回数据集中视频的数量"""
        return len(self.videos)

    def __getitem__(self, idx):
        """
        根据索引获取视频数据和对应的视频ID

        参数:
            idx (int): 视频索引

        返回:
            tuple: (视频张量, 视频ID)
                视频张量: 加载并预处理后的视频数据
                视频ID: 视频的唯一标识符
        """
        video_id = self.videos[idx]

        try:
            # 使用glob根据模式和视频ID查找匹配的视频文件路径
            pattern_with_id = self.pattern.replace('{id}', video_id)
            video_path = glob.glob(os.path.join(self.rootDir, pattern_with_id))

            if not video_path:
                warnings.warn(f"未找到匹配的视频文件: {pattern_with_id}")
                self.failed_videos.append(video_id)
                # 返回空张量
                empty_video = torch.zeros((1, 3, self.cc_size, self.cc_size), dtype=torch.float32)
                return empty_video, video_id

            # 加载第一个匹配的视频文件
            video = utils.load_video(video_path[0], fps=self.fps, cc_size=self.cc_size, rs_size=self.rs_size)

            # 检查是否成功加载
            if video.size == 0:
                warnings.warn(f"视频加载失败或为空: {video_path[0]}")
                self.failed_videos.append(video_id)
                # 返回空张量
                empty_video = torch.zeros((1, 3, self.cc_size, self.cc_size), dtype=torch.float32)
                return empty_video, video_id

            # 转换为PyTorch张量
            video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
            return video, video_id

        except Exception as e:
            warnings.warn(f"处理视频 {video_id} 时出错: {e}")
            self.failed_videos.append(video_id)
            # 返回空张量
            empty_video = torch.zeros((1, 3, self.cc_size, self.cc_size), dtype=torch.float32)
            return empty_video, video_id

    def get_failed_videos(self):
        """获取加载失败的视频列表"""
        return self.failed_videos