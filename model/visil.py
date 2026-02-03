import time  # 时间相关操作（代码中未实际使用）
import torch  # PyTorch深度学习框架核心库
import numpy as np  # 数值计算库
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数库（激活函数、池化等）
import torchvision.models as models  # PyTorch预训练模型库

# 从model包导入自定义层和相似度计算模块
from model.layers import *  # 自定义层（VideoNormalizer, RMAC, PCA, Attention等）
from model.similarities import *  # 相似度计算函数（ChamferSimilarity等）


class Feature_Extractor(nn.Module):
    """
    特征提取器类
    功能：从输入视频帧中提取多尺度区域特征向量
    基于预训练的ResNet50骨干网络，融合多个层的特征
    """

    def __init__(self, network='resnet50', whiteninig=False, dims=3840):
        """
        初始化特征提取器

        参数:
            network (str): 骨干网络架构，默认resnet50
            whiteninig (bool): 是否使用白化（PCA）降维
            dims (int): 输出特征维度（PCA降维后）
        """
        super(Feature_Extractor, self).__init__()
        # 视频标准化层，对输入帧进行归一化处理
        self.normalizer = VideoNormalizer()

        # 加载预训练的ResNet50模型作为特征提取骨干网络
        self.cnn = models.resnet50(pretrained=True)

        # RMAC（Regional Maximum Activation of Convolutions）池化层
        # 用于从卷积特征图中提取区域特征
        self.rpool = RMAC()

        # 定义要提取的ResNet层及其对应的特征图尺寸
        # key: 层名, value: 该层特征图尺寸（用于自适应池化）
        # 以输入224x224为例，经过各层后的特征图大小：
        # layer1: 56x56, layer2: 28x28, layer3: 14x14, layer4: 7x7
        self.layers = {'layer1': 28, 'layer2': 14, 'layer3': 6, 'layer4': 3}

        # 如果需要白化或降维，添加PCA层
        if whiteninig or dims != 3840:
            self.pca = PCA(dims)

    def extract_region_vectors(self, x):
        """
        从ResNet的不同层提取多尺度区域特征向量

        参数:
            x (tensor): 输入视频帧，形状 [B, C, H, W]
                       B-批次大小, C-通道数(3), H/W-高/宽

        返回:
            x (tensor): 融合后的区域特征向量，形状 [B, N, D]
                       B-批次大小, N-空间位置数, D-特征维度
        """
        tensors = []  # 存储各层的特征

        # 遍历ResNet的所有模块（conv1, bn1, relu, maxpool, layer1-4等）
        for nm, module in self.cnn._modules.items():
            # 跳过最后的全局池化和全连接层（用于分类）
            if nm not in {'avgpool', 'fc', 'classifier'}:
                # 前向传播通过当前模块
                x = module(x).contiguous()

                # 如果是我们关注的层（layer1-4），提取区域特征
                if nm in self.layers:
                    # 原代码注释掉的RMAC池化方式
                    # region_vectors = self.rpool(x)

                    # 使用自适应最大池化提取区域特征
                    s = self.layers[nm]  # 获取该层的池化窗口大小
                    # 执行最大池化，输出尺寸自动计算
                    # 例如layer4特征图7x7，池化窗口3，步长2，输出3x3的区域特征
                    region_vectors = F.max_pool2d(x, [s, s], int(np.ceil(s / 2)))

                    # L2归一化（按通道维度）
                    region_vectors = F.normalize(region_vectors, p=2, dim=1)

                    # 收集该层特征
                    tensors.append(region_vectors)

        # 将所有层的特征在通道维度上拼接
        # 例如4层，每层256通道 -> 总通道数 = 256*4 = 1024
        x = torch.cat(tensors, 1)

        # 重塑张量形状：将空间维度展平，并调整维度顺序
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        # 最终每个空间位置对应一个特征向量
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        # 对特征向量进行L2归一化（最后一个维度）
        x = F.normalize(x, p=2, dim=-1)
        return x

    def forward(self, x):
        """
        完整的前向传播过程

        参数:
            x (tensor): 输入视频帧，形状 [B, C, H, W]

        返回:
            x (tensor): 最终的特征表示
        """
        # 第一步：标准化输入帧
        x = self.normalizer(x)

        # 第二步：提取多尺度区域特征向量
        x = self.extract_region_vectors(x)

        # 第三步：如果定义了PCA层，应用降维/白化
        if hasattr(self, 'pca'):
            x = self.pca(x)

        return x


class ViSiLHead(nn.Module):
    """
    ViSiL模型头
    功能：计算视频间的细粒度相似度（帧到帧 + 视频到视频）
    包含注意力机制、视频比较器和Chamfer相似度计算
    """

    def __init__(self, dims=3840, attention=True, video_comperator=True, symmetric=False):
        """
        初始化ViSiL头

        参数:
            dims (int): 输入特征维度
            attention (bool): 是否使用帧级注意力机制
            video_comperator (bool): 是否使用视频比较器网络
            symmetric (bool): 是否使用对称的Chamfer相似度
        """
        super(ViSiLHead, self).__init__()

        # 如果启用注意力机制，创建注意力层（带归一化）
        if attention:
            self.attention = Attention(dims, norm=True)

        # 如果启用视频比较器，创建视频比较器网络
        # 用于将帧级相似度矩阵转换为视频级相似度
        if video_comperator:
            self.video_comperator = VideoComperator()

        # TensorDot层：计算查询帧和目标帧之间的点积相似度
        # 输入形状：查询 [B, I, O, K] 和目标 [B, J, P, K]
        # 输出形状：[B, I, O, P, J] （广播机制）
        self.tensor_dot = TensorDot("biok,bjpk->biopj")

        # 帧到帧相似度计算（Chamfer距离）
        # axes=[3, 2]：在指定维度上计算Chamfer相似度
        self.f2f_sim = ChamferSimilarity(axes=[3, 2], symmetric=symmetric)

        # 视频到视频相似度计算（Chamfer距离）
        # axes=[2, 1]：在时序维度上聚合帧级相似度
        self.v2v_sim = ChamferSimilarity(axes=[2, 1], symmetric=symmetric)

        # Hardtanh激活函数，将输出限制在[-1, 1]范围内
        self.htanh = nn.Hardtanh()

    def frame_to_frame_similarity(self, query, target):
        """
        计算帧到帧相似度矩阵

        参数:
            query (tensor): 查询视频特征，形状 [B1, T1, D]
            target (tensor): 目标视频特征，形状 [B2, T2, D]

        返回:
            sim (tensor): 帧级相似度矩阵
        """
        # 计算所有查询帧与所有目标帧之间的点积相似度
        sim = self.tensor_dot(query, target)

        # 应用Chamfer相似度计算，得到帧级对齐的相似度
        return self.f2f_sim(sim)

    def visil_output(self, sim):
        """
        通过视频比较器处理帧级相似度矩阵

        参数:
            sim (tensor): 帧级相似度矩阵

        返回:
            sim (tensor): 视频级相似度
        """
        # 增加一个维度以匹配视频比较器的输入要求
        sim = sim.unsqueeze(1)

        # 通过视频比较器网络（通常是CNN）
        return self.video_comperator(sim).squeeze(1)

    def video_to_video_similarity(self, query, target):
        """
        完整的视频到视频相似度计算流程

        参数:
            query (tensor): 查询视频特征
            target (tensor): 目标视频特征

        返回:
            sim (tensor): 最终的视频相似度分数
        """
        # 第一步：计算帧到帧相似度矩阵
        sim = self.frame_to_frame_similarity(query, target)

        # 第二步：如果启用了视频比较器，通过它聚合帧级相似度
        if hasattr(self, 'video_comperator'):
            sim = self.visil_output(sim)
            # 应用Hardtanh激活函数约束输出范围
            sim = self.htanh(sim)

        # 第三步：在时序维度上应用Chamfer相似度，得到最终视频相似度
        return self.v2v_sim(sim)

    def attention_weights(self, x):
        """
        获取注意力权重

        参数:
            x (tensor): 输入特征

        返回:
            x (tensor): 加权后的特征
            weights (tensor): 注意力权重
        """
        # 通过注意力层，得到加权特征和权重
        x, weights = self.attention(x)
        return x, weights

    def prepare_tensor(self, x):
        """
        准备张量：应用注意力机制

        参数:
            x (tensor): 输入特征

        返回:
            x (tensor): 处理后的特征
        """
        # 如果启用了注意力，应用注意力加权
        if hasattr(self, 'attention'):
            x, _ = self.attention_weights(x)
        return x

    def apply_constrain(self):
        """
        应用约束（修正方法名拼写错误）
        对注意力权重施加约束
        """
        # 原代码有拼写错误：self.att 应为 self.attention
        if hasattr(self, 'att'):
            self.att.apply_contraint()

    def forward(self, query, target):
        """
        ViSiL头的前向传播

        参数:
            query (tensor): 查询视频特征，形状 [T1, D] 或 [B, T1, D]
            target (tensor): 目标视频特征，形状 [T2, D] 或 [B, T2, D]

        返回:
            sim (tensor): 视频相似度
        """
        # 如果输入是3D张量（单视频），增加批次维度
        if query.ndim == 3:
            query = query.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)

        # 计算视频到视频相似度
        return self.video_to_video_similarity(query, target)


class ViSiL(nn.Module):
    """
    ViSiL主模型类
    整合特征提取器和ViSiL头，提供完整的视频相似度计算功能
    """

    def __init__(self, network='resnet50', pretrained=False, dims=3840,
                 whiteninig=True, attention=True, video_comperator=True, symmetric=False):
        """
        初始化ViSiL模型

        参数:
            network (str): 骨干网络
            pretrained (bool): 是否加载预训练权重
            dims (int): 特征维度
            whiteninig (bool): 是否使用PCA白化
            attention (bool): 是否使用注意力
            video_comperator (bool): 是否使用视频比较器
            symmetric (bool): 是否使用对称相似度
        """
        super(ViSiL, self).__init__()

        # ===== 加载预训练模型配置 =====
        # 非对称版本（标准ViSiL）
        if pretrained and not symmetric:
            # 创建特征提取器（带白化，输出3840维）
            self.cnn = Feature_Extractor('resnet50', True, 3840)
            # 创建ViSiL头（非对称模式）
            self.visil_head = ViSiLHead(3840, True, True, False)
            # 从官方URL加载预训练权重
            self.visil_head.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    'http://ndd.iti.gr/visil/visil.pth  '))

        # 对称版本（使用对称Chamfer距离）
        elif pretrained and symmetric:
            # 创建特征提取器（带白化，降维到512维）
            self.cnn = Feature_Extractor('resnet50', True, 512)
            # 创建ViSiL头（对称模式）
            self.visil_head = ViSiLHead(512, True, True, True)
            # 加载对称版本的预训练权重
            self.visil_head.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    'http://ndd.iti.gr/visil/visil_symmetric.pth  '))

        # ===== 从头训练配置 =====
        else:
            # 创建自定义的特征提取器
            self.cnn = Feature_Extractor(network, whiteninig, dims)
            # 创建自定义的ViSiL头
            self.visil_head = ViSiLHead(dims, attention, video_comperator, symmetric)

    # ===== 对外接口方法 =====
    def calculate_video_similarity(self, query, target):
        """
        计算两个视频之间的相似度

        参数:
            query (tensor): 查询视频特征
            target (tensor): 目标视频特征

        返回:
            similarity (tensor): 相似度分数 [0,1]
        """
        return self.visil_head(query, target)

    def calculate_f2f_matrix(self, query, target):
        """
        仅计算帧到帧相似度矩阵（中间表示）

        参数:
            query (tensor): 查询视频特征
            target (tensor): 目标视频特征

        返回:
            f2f_sim (tensor): 帧级相似度矩阵
        """
        return self.visil_head.frame_to_frame_similarity(query, target)

    def calculate_visil_output(self, query, target):
        """
        计算ViSiL输出（视频比较器后的相似度）

        参数:
            query (tensor): 查询视频特征
            target (tensor): 目标视频特征

        返回:
            sim (tensor): 视频比较器输出的相似度
        """
        # 先计算帧到帧相似度矩阵
        sim = self.visil_head.frame_to_frame_similarity(query, target)
        # 再通过视频比较器处理
        return self.visil_head.visil_output(sim)

    def extract_features(self, video_tensor):
        """
        提取视频的ViSiL特征（用于检索或缓存）

        参数:
            video_tensor (tensor): 输入视频帧，形状 [T, C, H, W] 或 [B, T, C, H, W]

        返回:
            features (tensor): 提取并处理后的特征
                              形状 [T, D] 或 [B, T, D]（取决于输入）
        """
        # 第一步：通过CNN提取多尺度区域特征
        features = self.cnn(video_tensor)

        # 第二步：通过ViSiL头准备特征（应用注意力等）
        features = self.visil_head.prepare_tensor(features)

        return features