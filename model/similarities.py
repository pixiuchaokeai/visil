import torch
import torch.nn as nn
import torch.nn.functional as F


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
