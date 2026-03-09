import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.layers import *
from model.similarities import *


class Feature_Extractor(nn.Module):

    def __init__(self, network='resnet50', whiteninig=False, dims=3840):
        super(Feature_Extractor, self).__init__()
        self.normalizer = VideoNormalizer()

        self.cnn = models.resnet50(pretrained=True)

        self.rpool = RMAC()
        self.layers = {'layer1': 28, 'layer2': 14, 'layer3': 6, 'layer4': 3}
        if whiteninig or dims != 3840:
            self.pca = PCA(dims)

    def extract_region_vectors(self, x):
        tensors = []
        for nm, module in self.cnn._modules.items():
            if nm not in {'avgpool', 'fc', 'classifier'}:
                x = module(x).contiguous()
                if nm in self.layers:
                    # region_vectors = self.rpool(x)
                    s = self.layers[nm]
                    region_vectors = F.max_pool2d(x, [s, s], int(np.ceil(s / 2)))
                    region_vectors = F.normalize(region_vectors, p=2, dim=1)
                    tensors.append(region_vectors)
        x = torch.cat(tensors, 1)
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def forward(self, x):
        x = self.normalizer(x)
        x = self.extract_region_vectors(x)
        if hasattr(self, 'pca'):
            x = self.pca(x)
            # [修改点] 根据 apply_extra_norm 标志决定是否在 PCA 后重新归一化
            if hasattr(self, 'apply_extra_norm') and self.apply_extra_norm:
                x = F.normalize(x, p=2, dim=-1)
        # [修改点] 根据 apply_extra_norm 标志决定是否进行最终强制归一化
        if hasattr(self, 'apply_extra_norm') and self.apply_extra_norm:
            final_norm = torch.norm(x, dim=-1).mean()
            if abs(final_norm - 1.0) > 0.01:
                x = F.normalize(x, p=2, dim=-1)
        return x


class ViSiLHead(nn.Module):
    """
    ViSiL模型头部模块
    负责：帧级相似度计算 -> 视频级相似度聚合 -> 注意力加权（可选）
    """

    def __init__(self, dims=3840, attention=True, video_comperator=True, symmetric=False):
        super(ViSiLHead, self).__init__()  # 调用父类的初始化函数
        if attention:  # 如果启用注意力机制
            self.attention = Attention(dims, norm=True)  # 初始化注意力层（假设Attention类已定义）
        if video_comperator:  # 如果启用视频比较器
            self.video_comperator = VideoComperator()  # 初始化视频比较器（假设VideoComperator类已定义）
        self.tensor_dot = TensorDot("biok,bjpk->biopj")  # 初始化张量点积层（自定义维度映射）
        # 初始化帧到帧的Chamfer相似度计算层，指定轴参数[3,2]，对称模式由参数控制
        self.f2f_sim = ChamferSimilarity(axes=[3, 2], symmetric=symmetric)
        # 初始化视频到视频的Chamfer相似度计算层，指定轴参数[2,1]，对称模式由参数控制
        self.v2v_sim = ChamferSimilarity(axes=[2, 1], symmetric=symmetric)
        self.htanh = nn.Hardtanh()  # 初始化Hardtanh激活函数（限制输出范围在[-1,1]）

    def frame_to_frame_similarity(self, query, target):
        """
        计算帧到帧的相似度矩阵
        Args:
            query: 查询视频特征 [batch, n_frames, n_proposals, dim]
            target: 目标视频特征 [batch, m_frames, n_proposals, dim]
        Returns:
            帧级相似度矩阵
        """
        sim = self.tensor_dot(query, target)  # 计算两个张量的点积，得到原始相似度矩阵
        return self.f2f_sim(sim)  # 应用Chamfer相似度计算，得到帧级相似度

    def visil_output(self, sim):
        """
        ViSiL输出处理：通过视频比较器聚合帧级相似度
        Args:
            sim: 帧级相似度矩阵
        Returns:
            聚合后的相似度特征
        """
        sim = sim.unsqueeze(1)  # 在维度1增加一个维度，适配video_comperator的输入格式
        return self.video_comperator(sim).squeeze(1)  # 视频比较器处理后移除新增的维度

    def video_to_video_similarity(self, query, target):
        """
        计算视频到视频的最终相似度
        Args:
            query: 查询视频特征（注意力加权后）
            target: 目标视频特征（注意力加权后）
        Returns:
            视频级相似度值
        """
        # [修改点] 根据 apply_input_norm 标志决定是否对输入特征进行归一化检查
        if hasattr(self, 'apply_input_norm') and self.apply_input_norm:  # 检查是否有归一化标志且为True
            query_norm = torch.norm(query, dim=-1).mean()  # 计算查询特征的L2范数均值（最后一维是特征维度）
            target_norm = torch.norm(target, dim=-1).mean()  # 计算目标特征的L2范数均值
            # 如果任一特征的范数偏离1.0超过0.1，则进行L2归一化（保证特征在单位球面上）
            if abs(query_norm - 1.0) > 0.1 or abs(target_norm - 1.0) > 0.1:
                query = F.normalize(query, p=2, dim=-1)  # 对查询特征做L2归一化（p=2表示L2范数）
                target = F.normalize(target, p=2, dim=-1)  # 对目标特征做L2归一化

        sim = self.frame_to_frame_similarity(query, target)  # 计算帧级相似度矩阵
        if hasattr(self, 'video_comperator'):  # 如果启用了视频比较器
            sim = self.visil_output(sim)  # 通过视频比较器聚合相似度
            sim = self.htanh(sim)  # 应用Hardtanh激活，限制输出范围

        return self.v2v_sim(sim)  # 应用视频级Chamfer相似度计算，得到最终视频相似度

    def attention_weights(self, x):
        """
        计算注意力权重并加权特征
        Args:
            x: 输入特征 [batch, frames, dim]
        Returns:
            加权后的特征 + 注意力权重
        """
        x, weights = self.attention(x)  # 计算注意力加权后的特征和权重
        # [修改点] 根据 apply_att_norm 标志决定是否在注意力加权后重新归一化
        if hasattr(self, 'apply_att_norm') and self.apply_att_norm:  # 检查是否有注意力后归一化标志且为True
            x = F.normalize(x, p=2, dim=-1)  # 对注意力加权后的特征重新做L2归一化

        return x, weights  # 返回加权特征和权重

    def prepare_tensor(self, x):
        """
        预处理输入张量（主要应用注意力机制）
        Args:
            x: 原始输入特征
        Returns:
            预处理后的特征（注意力加权后）
        """
        if hasattr(self, 'attention'):  # 如果启用了注意力机制
            x, _ = self.attention_weights(x)  # 应用注意力加权，忽略权重输出
        return x  # 返回预处理后的特征

    def apply_constrain(self):
        """
        应用约束条件（可选）
        通常用于对注意力层等模块施加正则化约束
        """
        if hasattr(self, 'att'):  # 检查是否有att属性（注意力层）
            self.att.apply_contraint()  # 调用注意力层的约束函数（假设已定义）

    def forward(self, query, target):
        """
        ViSiLHead主前向传播函数
        Args:
            query: 查询视频特征 [batch, frames, dim] 或 [frames, dim]
            target: 目标视频特征 [batch, frames, dim] 或 [frames, dim]
        Returns:
            视频到视频的相似度值
        """
        if query.ndim == 3:  # 如果查询特征是3维（缺少batch维度）
            query = query.unsqueeze(0)  # 在第0维增加batch维度（batch_size=1）
        if target.ndim == 3:  # 如果目标特征是3维（缺少batch维度）
            target = target.unsqueeze(0)  # 在第0维增加batch维度（batch_size=1）

        # 预处理查询和目标特征（应用注意力机制）
        query = self.prepare_tensor(query)  # 预处理查询特征
        target = self.prepare_tensor(target)  # 预处理目标特征

        return self.video_to_video_similarity(query, target)  # 计算并返回视频级相似度
class ViSiL(nn.Module):

    def __init__(self, network='resnet50', pretrained=False, dims=3840,
                 whiteninig=True, attention=True, video_comperator=True, symmetric=False):
        super(ViSiL, self).__init__()
        print("=" * 60)
        print(f"初始化 ViSiL")
        print(f"  网络: {network}")
        print(f"  预训练: {pretrained}")
        print(f"  特征维度: {dims}")
        print(f"  区域数量: 9 (L3级别)")
        print(f"  总维度: 9 × {dims} = {9 * dims}")
        print(f"  白化: {whiteninig}")
        print(f"  注意力: {attention}")
        print(f"  视频比较器: {video_comperator}")
        print(f"  对称: {symmetric}")
        print("=" * 60)
        if pretrained and not symmetric:
            self.cnn = Feature_Extractor('resnet50', True, 3840)
            self.visil_head = ViSiLHead(3840, True, True, False)
            self.visil_head.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    'http://ndd.iti.gr/visil/visil.pth'))
        elif pretrained and symmetric:
            self.cnn = Feature_Extractor('resnet50', True, 512)
            self.visil_head = ViSiLHead(512, True, True, True)
            self.visil_head.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    'http://ndd.iti.gr/visil/visil_symmetric.pth'))
        else:
            self.cnn = Feature_Extractor(network, whiteninig, dims)
            self.visil_head = ViSiLHead(dims, attention, video_comperator, symmetric)

        # [修改点] 根据对称性设置归一化标志
        if symmetric:
            self.cnn.apply_extra_norm = True
            self.visil_head.apply_input_norm = True
            self.visil_head.apply_att_norm = True
        # 对于非对称模型，不设置这些属性，保持原有行为

    def calculate_video_similarity(self, query, target):
        return self.visil_head(query, target)

    def calculate_f2f_matrix(self, query, target):
        return self.visil_head.frame_to_frame_similarity(query, target)

    def calculate_visil_output(self, query, target):
        sim = self.visil_head.frame_to_frame_similarity(query, target)
        return self.visil_head.visil_output(sim)

    def extract_features(self, video_tensor):
        features = self.cnn(video_tensor)
        features = self.visil_head.prepare_tensor(features)
        return features