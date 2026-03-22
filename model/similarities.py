import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F


# [修改点] 帧级余弦相似度矩阵计算
def compute_keyframe_similarity_matrix(q_feat, db_feat):
    """计算帧级平均特征的余弦相似度矩阵 [Tq, Tdb]"""
    q_frame = q_feat.mean(dim=1)  # [Tq, D]
    db_frame = db_feat.mean(dim=1)  # [Tdb, D]
    q_frame = F.normalize(q_frame, p=2, dim=-1)
    db_frame = F.normalize(db_frame, p=2, dim=-1)
    sim = torch.mm(q_frame, db_frame.t())  # [Tq, Tdb]
    return sim.cpu().numpy()


def find_candidate_pairs(sim_matrix, threshold):
    """返回所有满足 sim >= threshold 的 (i, j) 对列表"""
    pairs = np.argwhere(sim_matrix >= threshold)
    return [(int(i), int(j)) for i, j in pairs]


def save_similarity_matrix(matrix, q_id, db_id, sim_dir):
    """保存相似度矩阵到 .npy 文件"""
    subdir = os.path.join(sim_dir, q_id)
    os.makedirs(subdir, exist_ok=True)
    filepath = os.path.join(subdir, f"{q_id}_{db_id}.npy")
    np.save(filepath, matrix)


def load_similarity_matrix(q_id, db_id, sim_dir):
    """加载相似度矩阵，若失败返回 None"""
    filepath = os.path.join(sim_dir, q_id, f"{q_id}_{db_id}.npy")
    try:
        return np.load(filepath)
    except Exception:
        return None


def load_features(video_id, features_dir, expected_dims):
    """从磁盘加载特征，返回 torch.Tensor [T,9,D] 或 None"""
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


# [修改点] 新增：密集帧余弦相似度计算（视频级平均特征）
def compute_dense_cosine_similarity(q_feat, db_feat):
    """
    计算两个密集特征序列的视频级余弦相似度。
    输入: q_feat [Tq, 9, D], db_feat [Tdb, 9, D]
    输出: 标量相似度
    """
    # 对区域维求平均得到帧级特征 [T, D]
    q_frame = q_feat.mean(dim=1)  # [Tq, D]
    db_frame = db_feat.mean(dim=1)  # [Tdb, D]
    # 对整个视频求平均得到视频级特征 [D]
    q_video = q_frame.mean(dim=0)  # [D]
    db_video = db_frame.mean(dim=0)  # [D]
    # 归一化
    q_video = F.normalize(q_video, p=2, dim=-1)
    db_video = F.normalize(db_video, p=2, dim=-1)
    # 点积
    sim = torch.dot(q_video, db_video)
    return sim.item()




class TensorDot(nn.Module):

    def __init__(self, pattern='iak,jbk->iabj', metric='cosine'):
        super(TensorDot, self).__init__()
        self.pattern = pattern
        self.metric = metric

    def forward(self, query, target):
        if self.metric == 'cosine':
            sim = torch.einsum(self.pattern, [query, target])
        elif self.metric == 'euclidean':
            sim = 1 - 2 * torch.einsum(self.pattern, [query, target])
        elif self.metric == 'hamming':
            sim = torch.einsum(self.pattern, query, target) / target.shape[-1]
        return sim


class ChamferSimilarity(nn.Module):
    """
    Chamfer相似度计算模块
    核心思想：通过最大池化和平均池化的组合计算两组特征之间的相似度
    """
    def __init__(self, symmetric=False, axes=[1, 0]):
        super(ChamferSimilarity, self).__init__()  # 调用父类的初始化函数
        if symmetric:  # 如果启用对称模式
            # 绑定对称Chamfer相似度计算函数，使用指定的轴参数
            self.sim_fun = lambda x: self.symmetric_chamfer_similarity(x, axes=axes)
        else:  # 非对称模式
            # 绑定普通Chamfer相似度计算函数，拆分axes参数分别作为max轴和mean轴
            self.sim_fun = lambda x: self.chamfer_similarity(x, max_axis=axes[0], mean_axis=axes[1])

    def chamfer_similarity(self, s, max_axis=1, mean_axis=0):
        """
        基础Chamfer相似度计算
        Args:
            s: 输入相似度矩阵
            max_axis: 执行max池化的维度
            mean_axis: 执行mean池化的维度
        Returns:
            计算后的标量相似度值
        """
        s = torch.max(s, max_axis, keepdim=True)[0]  # 在指定维度上执行最大池化，keepdim=True保持维度数量不变，[0]取池化后的值（torch.max返回(values, indices)）
        s = torch.mean(s, mean_axis, keepdim=True)  # 在指定维度上执行平均池化
        # 连续两次squeeze操作移除多余的维度：先移除较大的轴，再移除较小的轴，最终得到标量
        return s.squeeze(max(max_axis, mean_axis)).squeeze(min(max_axis, mean_axis))

    def symmetric_chamfer_similarity(self, s, axes=[0, 1]):
        """
        对称Chamfer相似度计算
        计算方式：交换max和mean轴计算两次相似度后取平均，保证相似度的对称性
        """
        return (self.chamfer_similarity(s, max_axis=axes[0], mean_axis=axes[1]) +  # 第一次计算（原轴顺序）
                self.chamfer_similarity(s, max_axis=axes[1], mean_axis=axes[0])) / 2  # 第二次计算（轴顺序交换），然后取平均

    def forward(self, s):
        """
        前向传播函数（PyTorch必需）
        Args:
            s: 输入相似度矩阵
        Returns:
            Chamfer相似度值
        """
        return self.sim_fun(s)  # 调用初始化时绑定的相似度计算函数


class VideoComperator(nn.Module):

    def __init__(self):
        super(VideoComperator, self).__init__()
        self.rpad1 = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(1, 32, 3)

        self.pool1 = nn.MaxPool2d((2, 2), 2)

        self.rpad2 = nn.ReplicationPad2d(1)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d((2, 2), 2)

        self.rpad3 = nn.ReplicationPad2d(1)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.fconv = nn.Conv2d(128, 1, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sim_matrix):
        sim = self.rpad1(sim_matrix)
        sim = F.relu(self.conv1(sim))
        sim = self.pool1(sim)

        sim = self.rpad2(sim)
        sim = F.relu(self.conv2(sim))
        sim = self.pool2(sim)

        sim = self.rpad3(sim)
        sim = F.relu(self.conv3(sim))
        sim = self.fconv(sim)
        return sim
