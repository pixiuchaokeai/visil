# visil.py - 修正为完整维度版本
"""
visil.py - 修正为完整维度版本
ViSiL_v使用完整的L3-iMAC9x特征（9×3840维），不降维
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import warnings

warnings.filterwarnings("ignore")

from model.layers import VideoNormalizer, RMAC, PCA, Attention, L2Constrain
from model.similarities import TensorDot, ChamferSimilarity, VideoComperator


class Feature_Extractor(nn.Module):
    def __init__(self, network='resnet50', whiteninig=False, dims=3840, use_full_dim=True):
        """
        特征提取器 - 修正版本
        Args:
            dims: 特征维度（3840为完整维度）
            use_full_dim: 是否使用完整维度（True表示不降维）
        """
        super(Feature_Extractor, self).__init__()
        
        self.normalizer = VideoNormalizer()
        
        # 使用ResNet50
        self.cnn = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])  # 移除最后的全连接层和平均池化层
        
        self.layers = {'layer1': 28, 'layer2': 14, 'layer3': 6, 'layer4': 3}
        
        print(f"> Feature_Extractor 初始化:")
        print(f"  网络: {network}")
        print(f"  白化: {whiteninig}")
        print(f"  特征维度: {dims}")
        print(f"  使用完整维度: {use_full_dim}")
        
        if whiteninig:
            # 重要：根据论文，ViSiL_v使用PCA白化但不降维
            # 预训练模型已经包含了适当的PCA白化层
            if use_full_dim:
                # 不降维，保持3840维
                print(f"> 启用PCA白化但不降维，保持{dims}维")
                self.pca = PCA(n_components=dims)
            else:
                # 降维版本（用于存储受限情况）
                print(f"> 启用PCA白化并降维到{dims}维")
                self.pca = PCA(n_components=dims)
        else:
            print(f"> 未启用PCA白化")
            self.pca = None

        self.output_dim = dims



    def extract_region_vectors(self, x):
        """提取区域特征向量"""
        tensors = []
        
        # 通过ResNet50的各层
        x = self.cnn[0](x)  # conv1
        x = self.cnn[1](x)  # bn1
        x = self.cnn[2](x)  # relu
        x = self.cnn[3](x)  # maxpool
        
        # layer1: 256通道 -> 最大池化到28x28
        x = self.cnn[4](x)
        s = self.layers['layer1']
        region_vectors = F.max_pool2d(x, [s, s], int(torch.ceil(torch.tensor(s / 2)).item()))
        region_vectors = F.normalize(region_vectors, p=2, dim=1)
        tensors.append(region_vectors)
        
        # layer2: 512通道 -> 最大池化到14x14
        x = self.cnn[5](x)
        s = self.layers['layer2']
        region_vectors = F.max_pool2d(x, [s, s], int(torch.ceil(torch.tensor(s / 2)).item()))
        region_vectors = F.normalize(region_vectors, p=2, dim=1)
        tensors.append(region_vectors)
        
        # layer3: 1024通道 -> 最大池化到6x6
        x = self.cnn[6](x)
        s = self.layers['layer3']
        region_vectors = F.max_pool2d(x, [s, s], int(torch.ceil(torch.tensor(s / 2)).item()))
        region_vectors = F.normalize(region_vectors, p=2, dim=1)
        tensors.append(region_vectors)
        
        # layer4: 2048通道 -> 最大池化到3x3
        x = self.cnn[7](x)
        s = self.layers['layer4']
        region_vectors = F.max_pool2d(x, [s, s], int(torch.ceil(torch.tensor(s / 2)).item()))
        region_vectors = F.normalize(region_vectors, p=2, dim=1)
        tensors.append(region_vectors)
        
        # 连接所有层的特征: 256+512+1024+2048 = 3840维
        x = torch.cat(tensors, 1)  # [B, 3840, H, W]
        
        # L3级别: 3x3网格划分，得到9个区域
        # 重新形状: [B, C, H, W] -> [B, regions, C]
        B, C, H, W = x.shape
        
        # 如果特征图不是3x3，需要调整
        if H != 3 or W != 3:
            # 使用自适应平均池化调整到3x3
            x = F.adaptive_avg_pool2d(x, (3, 3))
            H, W = 3, 3
        
        # 重塑为区域向量: [B, C, H, W] -> [B, H*W, C]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = x.view(B, H * W, C)  # [B, 9, 3840]
        
        # L2归一化
        x = F.normalize(x, p=2, dim=-1)
        
        return x

    def forward(self, x):
        # 标准化
        x = self.normalizer(x)
        
        # 提取区域特征
        x = self.extract_region_vectors(x)
        
        # 应用PCA白化（如果启用）
        if self.pca is not None:
            x = self.pca(x)

        if x.shape[-1] != self.output_dim:
            print(f"> 警告: 特征维度不匹配! 期望: {self.output_dim}, 实际: {x.shape[-1]}")
            print(f"> 将特征调整为期望维度")
            # 使用线性投影调整维度
            if not hasattr(self, 'projection'):
                self.projection = nn.Linear(x.shape[-1], self.output_dim).to(x.device)
            x = self.projection(x)

        return x


class ViSiLHead(nn.Module):
    def __init__(self, dims=3840, attention=True, video_comperator=True, symmetric=False):
        super(ViSiLHead, self).__init__()
        
        self.dims = dims
        self.symmetric = symmetric
        
        # 根据论文，ViSiLv使用注意力机制
        if attention:
            self.attention = Attention(dims, norm=True)
            print(f"> ViSiLHead 启用注意力机制")
        else:
            self.attention = None
        
        # 根据论文，只有ViSiLv使用视频比较器
        if video_comperator:
            self.video_comperator = VideoComperator()
            print(f"> ViSiLHead 启用视频比较器")
        else:
            self.video_comperator = None
        
        # 使用余弦相似度的TensorDot
        self.tensor_dot = TensorDot(pattern="biok,bjpk->biopj", metric='cosine')
        
        # Chamfer相似度
        self.f2f_sim = ChamferSimilarity(symmetric=False, axes=[3, 2])  # 帧到帧
        self.v2v_sim = ChamferSimilarity(symmetric=symmetric, axes=[2, 1])  # 视频到视频
        
        # Hardtanh将相似度限制在[-1, 1]范围内
        self.htanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)

    def frame_to_frame_similarity(self, query, target):
        """计算帧到帧相似度矩阵"""
        # query: [B, query_frames, regions, dims]
        # target: [B, target_frames, regions, dims]
        
        # 计算相似度矩阵
        sim = self.tensor_dot(query, target)  # [B, query_frames, regions, target_frames, regions]
        
        # 应用Chamfer相似度
        result = self.f2f_sim(sim)  # [B, query_frames, target_frames]
        
        return result

    def visil_output(self, sim):
        """通过视频比较器处理相似度矩阵"""
        # sim: [B, query_frames, target_frames]
        
        if self.video_comperator is not None:
            # 添加通道维度 [B, 1, H, W]
            sim = sim.unsqueeze(1)
            sim = self.video_comperator(sim)
            sim = sim.squeeze(1)
        
        return sim

    def video_to_video_similarity(self, query, target):
        """计算视频到视频相似度"""
        # 计算帧到帧相似度矩阵
        sim = self.frame_to_frame_similarity(query, target)  # [B, query_frames, target_frames]
        
        # 如果有视频比较器，则应用
        if self.video_comperator is not None:
            sim = self.visil_output(sim)
            
            # 应用Hardtanh限制范围
            sim = self.htanh(sim)
        
        # 计算最终的视频相似度
        result = self.v2v_sim(sim)  # [B] 或标量
        
        return result

    def attention_weights(self, x):
        if self.attention is not None:
            x, weights = self.attention(x)
            return x, weights
        return x, None

    def prepare_tensor(self, x):
        if self.attention is not None:
            x, _ = self.attention_weights(x)
        return x

    def forward(self, query, target):
        # 确保有批次维度
        if query.dim() == 3:
            query = query.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        
        # 计算视频相似度
        return self.video_to_video_similarity(query, target)


class ViSiL(nn.Module):
    def __init__(self, network='resnet50', pretrained=False, dims=3840,
                 whiteninig=True, attention=True, video_comperator=True, symmetric=False):
        super(ViSiL, self).__init__()
        
        print("="*60)
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
        print("="*60)
        
        # 根据论文，ViSiL_v使用完整维度，不降维
        if dims != 3840 and pretrained:
            print(f"> 警告: 预训练ViSiL_v模型使用3840维特征")
            print(f"> 将维度从{dims}调整为3840")
            dims = 3840
        
        # 使用完整维度
        use_full_dim = (dims == 3840)
        
        # 创建特征提取器
        self.cnn = Feature_Extractor(
            network=network,
            whiteninig=whiteninig,
            dims=dims,
            use_full_dim=use_full_dim
        )
        
        # 创建ViSiL头部
        self.visil_head = ViSiLHead(
            dims=dims,
            attention=attention,
            video_comperator=video_comperator,
            symmetric=symmetric
        )
        
        # 加载预训练权重
        if pretrained:
            self.load_pretrained_weights(symmetric=symmetric, dims=dims)
        
        print("="*60)

    def load_pretrained_weights(self, symmetric=False, dims=3840):
        """加载预训练权重"""
        print("> 加载预训练权重...")
        
        try:
            if symmetric:
                # 对称版本
                print("> 加载ViSiLsym权重...")
                weight_url = 'http://ndd.iti.gr/visil/visil_symmetric.pth'
            else:
                # 非对称版本 (ViSiLv)
                print("> 加载ViSiLv权重...")
                weight_url = 'http://ndd.iti.gr/visil/visil.pth'
            
            # 加载权重
            state_dict = torch.hub.load_state_dict_from_url(
                weight_url,
                map_location='cpu',
                progress=True
            )
            
            if state_dict:
                # 加载权重
                self.visil_head.load_state_dict(state_dict)
                print(f"> ✓ 预训练权重加载成功 (维度: {dims})")
            else:
                print("> ⚠️ 预训练权重加载失败，使用随机初始化")
        
        except Exception as e:
            print(f"> ✗ 预训练权重加载失败: {e}")
            print("> 使用随机初始化")

    def calculate_video_similarity(self, query, target):
        return self.visil_head(query, target)

    def calculate_f2f_matrix(self, query, target):
        return self.visil_head.frame_to_frame_similarity(query, target)

    def extract_features(self, video_tensor):
        """提取视频特征"""
        # 提取特征
        features = self.cnn(video_tensor)
        
        # 准备特征（应用注意力）
        features = self.visil_head.prepare_tensor(features)
        
        return features

    def forward(self, query_video, target_video=None):
        """
        前向传播
        如果只提供一个视频，返回特征
        如果提供两个视频，返回相似度
        """
        if target_video is None:
            # 提取特征模式
            return self.extract_features(query_video)
        else:
            # 相似度计算模式
            # 提取特征
            query_features = self.extract_features(query_video)
            target_features = self.extract_features(target_video)
            
            # 计算相似度
            return self.calculate_video_similarity(query_features, target_features)