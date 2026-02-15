import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class VideoNormalizer(nn.Module):

    def __init__(self):
        super(VideoNormalizer, self).__init__()
        self.scale = nn.Parameter(torch.Tensor([255.]), requires_grad=False)
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]), requires_grad=False)

    def forward(self, video):
        video = ((video / self.scale) - self.mean) / self.std
        return video.permute(0, 3, 1, 2)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMAC(nn.Module):
    """
    R-MAC (Regional Maximum Activations of Convolutions)

    一种图像/视频帧的特征提取方法，通过在不同尺度（L层）上提取多个区域的特征，
    并做最大池化，最后拼接成固定长度的特征向量。

    相比全局MAC（Maximum Activations of Convolutions），R-MAC保留了空间信息，
    通过多区域池化获得更好的判别性和位置鲁棒性。

    参数:
        L: 区域层级列表，默认[3]表示使用3x3=9个区域
           可以是[1,2,3]表示多尺度融合
    """

    def __init__(self, L=[3]):
        """
        初始化R-MAC模块

        参数:
            L: 列表，指定区域划分的层级
               L=1: 1x1=1个区域（全局）
               L=2: 2x2=4个区域
               L=3: 3x3=9个区域（默认）
        """
        super(RMAC, self).__init__()
        self.L = L  # 存储区域层级配置

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征图，形状为 [B, C, H, W]
               B: batch size
               C: 通道数
               H: 高度
               W: 宽度

        返回:
            区域池化后的特征，形状为 [B, C, N, 1, 1] 或展平后 [B, C*N]
        """
        return self.region_pooling(x, L=self.L)

    def region_pooling(self, x, L=[3]):
        """
        核心方法：多尺度区域池化

        算法步骤：
        1. 根据输入尺寸计算最优的区域大小和步长
        2. 对于每个层级l，生成l×l个均匀分布的区域中心
        3. 在每个区域做最大池化
        4. 拼接所有区域的特征

        参数:
            x: 输入特征图 [B, C, H, W]
            L: 区域层级列表

        返回:
            拼接后的区域特征 [B, C, 总区域数, 1, 1]
        """
        # 重叠率参数：相邻区域期望的重叠比例
        ovr = 0.4  # desired overlap of neighboring regions

        # 可能的区域划分步数（用于长边的区域数）
        steps = torch.Tensor([2, 3, 4, 5, 6, 7])

        # 获取输入特征图的尺寸
        W = x.shape[3]  # 宽度
        H = x.shape[2]  # 高度

        # 计算短边长度（以短边为基准计算区域大小）
        w = min(W, H)
        # 区域大小的一半（向下取整）
        w2 = math.floor(w / 2.0 - 1)

        # 计算不同steps下的实际重叠率，选择最接近ovr=0.4的step
        # b: 相邻区域中心之间的距离（步长）
        b = (max(H, W) - w) / (steps - 1)

        # 计算实际重叠率与期望重叠率的差距
        # 公式: (w^2 - w*b) / w^2 = 1 - b/w，表示非重叠区域比例
        actual_ovr = ((w ** 2 - w * b) / w ** 2)

        # 找到最接近目标重叠率ovr的step索引
        (tmp, idx) = torch.min(torch.abs(actual_ovr - ovr), 0)
        # steps[idx] 就是长边方向的最优区域数

        # 判断长边方向，确定在哪个维度增加额外区域
        Wd = 0  # 宽度方向额外区域数
        Hd = 0  # 高度方向额外区域数

        # [修改点] 强制输出固定 9 个区域 (3x3 网格)，忽略动态区域数
        # if H < W:
        #     # 如果宽度>高度，宽度方向需要更多区域
        #     Wd = idx.item() + 1
        # elif H > W:
        #     # 如果高度>宽度，高度方向需要更多区域
        #     Hd = idx.item() + 1
        # # 如果H==W，不增加额外区域

        # 存储所有区域的池化特征
        vecs = []

        # 遍历每个指定的层级l
        for l in L:
            # 计算当前层级的区域边长
            # 公式: wl = floor(2*w / (l+1))
            # 例如w=30, l=3: wl = floor(60/4) = 15
            wl = math.floor(2 * w / (l + 1))
            # 区域边长的一半
            wl2 = math.floor(wl / 2 - 1)

            # ========== 计算宽度方向的区域中心坐标 ==========
            if l + Wd == 1:
                # 特殊情况：只有一个区域，步长为0
                b = 0
            else:
                # 计算相邻区域中心的水平距离
                b = (W - wl) / (l + Wd - 1)

            # 生成l + Wd个区域的中心x坐标
            # range(l - 1 + Wd + 1) = range(l + Wd)
            # 乘以b得到偏移量，加上wl2得到中心位置，再减去wl2对齐
            cenW = torch.floor(wl2 + torch.tensor(range(l + Wd)) * b) - wl2

            # ========== 计算高度方向的区域中心坐标 ==========
            if l + Hd == 1:
                b = 0
            else:
                b = (H - wl) / (l + Hd - 1)

            # 生成l + Hd个区域的中心y坐标
            cenH = torch.floor(wl2 + torch.tensor(range(l + Hd)) * b) - wl2

            # ========== 遍历所有区域，提取特征 ==========
            for i in cenH.tolist():  # 遍历高度方向中心
                for j in cenW.tolist():  # 遍历宽度方向中心
                    # 跳过无效区域（wl=0时）
                    if wl == 0:
                        continue

                    # [修改点] 修正变量名 i_ -> i, j_ -> j，避免 NameError

                    # 步骤1：在高度方向切片
                    # 生成从i开始的wl个连续索引
                    h_indices = (int(i) + torch.Tensor(range(wl)).long()).tolist()
                    R = x[:, :, h_indices, :]  # 形状: [B, C, wl, W]

                    # 步骤2：在宽度方向切片
                    w_indices = (int(j) + torch.Tensor(range(wl)).long()).tolist()
                    R = R[:, :, :, w_indices]  # 形状: [B, C, wl, wl]

                    # 步骤3：对该区域做全局最大池化，得到1x1的特征
                    # max_pool2d将[H, W]池化为[1, 1]
                    pooled = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                    # pooled形状: [B, C, 1, 1]

                    vecs.append(pooled)

        # 拼接所有区域的特征
        # 沿着空间维度（dim=2）拼接，最终形状: [B, C, 总区域数, 1, 1]
        return torch.cat(vecs, dim=2)


class PCA(nn.Module):

    def __init__(self, n_components=None):
        super(PCA, self).__init__()
        pretrained_url = 'http://ndd.iti.gr/visil/pca_resnet50_vcdb_1M.pth'
        white = torch.hub.load_state_dict_from_url(pretrained_url)
        idx = torch.argsort(white['d'], descending=True)[: n_components]
        d = white['d'][idx]
        V = white['V'][:, idx]
        D = torch.diag(1. / torch.sqrt(d + 1e-7))
        self.mean = nn.Parameter(white['mean'], requires_grad=False)
        self.DVt = nn.Parameter(torch.mm(D, V.T).T, requires_grad=False)

    def forward(self, logits):
        logits -= self.mean.expand_as(logits)
        logits = torch.matmul(logits, self.DVt)
        logits = F.normalize(logits, p=2, dim=-1)
        return logits


class L2Constrain(object):

    def __init__(self, axis=-1, eps=1e-6):
        self.axis = axis
        self.eps = eps

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            module.weight.data = F.normalize(w, p=2, dim=self.axis, eps=self.eps)


class Attention(nn.Module):

    def __init__(self, dims, norm=False):
        super(Attention, self).__init__()
        self.norm = norm
        if self.norm:
            self.constrain = L2Constrain()
        else:
            self.transform = nn.Linear(dims, dims)
        self.context_vector = nn.Linear(dims, 1, bias=False)
        self.reset_parameters()

    def forward(self, x):
        if self.norm:
            weights = self.context_vector(x)
            weights = torch.add(torch.div(weights, 2.), .5)
        else:
            x_tr = torch.tanh(self.transform(x))
            weights = self.context_vector(x_tr)
            weights = torch.sigmoid(weights)
        x = x * weights
        return x, weights

    def reset_parameters(self):
        if self.norm:
            nn.init.normal_(self.context_vector.weight)
            self.constrain(self.context_vector)
        else:
            nn.init.xavier_uniform_(self.context_vector.weight)
            nn.init.xavier_uniform_(self.transform.weight)
            nn.init.zeros_(self.transform.bias)

    def apply_contraint(self):
        if self.norm:
            self.constrain(self.context_vector)