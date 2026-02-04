"""
debug_model.py 修复补丁
修复两个主要问题：
1. 视频加载警告阈值调整
2. 梯度张量转换为numpy的问题
"""

import json
import torch
import argparse
import os
import numpy as np
import warnings

from tqdm import tqdm
from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator
from evaluation import extract_features, calculate_similarities_to_queries


class DebugVideoGenerator:
    """调试用的视频生成器，只处理少量视频"""

    def __init__(self, video_list_file, max_videos=5):
        print(f"\n[DEBUG] 初始化DebugVideoGenerator，最多处理 {max_videos} 个视频")

        # 读取视频列表
        with open(video_list_file, 'r') as f:
            lines = f.readlines()

        # 只取前max_videos个视频
        self.videos = []
        for i, line in enumerate(lines[:max_videos]):
            parts = line.strip().split()
            if len(parts) >= 2:
                video_id = parts[0]
                video_path = parts[1]
                self.videos.append((video_id, video_path))
                print(f"[DEBUG] 视频 {i+1}: ID={video_id}, 路径={video_path}")
            else:
                print(f"[WARNING] 跳过格式错误的行: {line}")

        print(f"[DEBUG] 总共加载 {len(self.videos)} 个视频")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_id, video_path = self.videos[idx]

        # 导入utils模块
        from utils import load_video

        print(f"\n[DEBUG] 加载视频 {idx+1}/{len(self.videos)}: {video_id}")
        print(f"[DEBUG] 视频路径: {video_path}")

        # 检查文件是否存在
        if not os.path.exists(video_path):
            print(f"[ERROR] 视频文件不存在: {video_path}")
            # 创建模拟数据用于调试
            frames = np.random.randn(48, 224, 224, 3).astype(np.float32)
            print(f"[DEBUG] 使用模拟数据: 形状={frames.shape}")
        else:
            # 加载真实视频，临时修改警告阈值
            import utils
            original_warn = warnings.warn

            def custom_warn(message, category=None, stacklevel=1, source=None):
                # 过滤掉连续读取失败的警告
                if "连续读取失败，停止读取视频" in str(message):
                    print(f"[INFO] 视频读取警告: {message}")
                else:
                    original_warn(message, category, stacklevel, source)

            warnings.warn = custom_warn

            try:
                frames = load_video(video_path)
            finally:
                warnings.warn = original_warn

            if frames.size == 0:
                print(f"[WARNING] 视频 {video_id} 加载失败，使用模拟数据")
                frames = np.random.randn(48, 224, 224, 3).astype(np.float32)
            else:
                print(f"[DEBUG] 视频加载成功: 帧数={frames.shape[0]}, 形状={frames.shape}")

        # 转换为PyTorch张量
        frames_tensor = torch.from_numpy(frames).float()
        print(f"[DEBUG] 转换为张量: 形状={frames_tensor.shape}, dtype={frames_tensor.dtype}")

        return frames_tensor, video_id


def debug_extract_features(model, frames, args):
    """
    调试版本的特征提取函数，打印维度变化
    修复梯度问题
    """
    print(f"\n[DEBUG] 开始特征提取")
    print(f"[DEBUG] 输入帧形状: {frames.shape}")

    features = []
    batch_sz = args.batch_sz

    # 使用torch.no_grad()确保不计算梯度
    with torch.no_grad():
        # 分批处理
        for i in range(0, frames.shape[0], batch_sz):
            batch = frames[i:i + batch_sz]
            print(f"[DEBUG] 处理批次 {i//batch_sz + 1}: 批次大小={batch.shape[0]}")

            if batch.shape[0] > 0:
                # 修正：使用args.device而不是args.gpu_id
                batch = batch.to(args.device).float()
                print(f"[DEBUG] 批次移动到设备后形状: {batch.shape}, 设备={batch.device}")

                # 提取特征
                try:
                    batch_features = model.extract_features(batch)
                    print(f"[DEBUG] 批次特征提取成功: 形状={batch_features.shape}")
                    features.append(batch_features)
                except Exception as e:
                    print(f"[ERROR] 特征提取失败: {e}")
                    # 创建模拟特征用于继续调试
                    batch_features = torch.randn(batch.shape[0], 9, 3840).to(args.device)
                    features.append(batch_features)
                    print(f"[DEBUG] 使用模拟特征: 形状={batch_features.shape}")

    # 合并特征
    if features:
        features = torch.cat(features, 0)
        print(f"[DEBUG] 合并所有批次特征: 最终形状={features.shape}")
    else:
        print(f"[WARNING] 没有提取到任何特征，创建默认特征")
        features = torch.randn(frames.shape[0], 9, 3840).to(args.device)

    # 确保至少有4帧
    while features.shape[0] < 4:
        print(f"[DEBUG] 特征帧数不足4帧 ({features.shape[0]})，复制特征")
        features = torch.cat([features, features], 0)
        print(f"[DEBUG] 复制后特征形状: {features.shape}")

    print(f"[DEBUG] 特征提取完成，返回形状: {features.shape}")
    return features


def debug_calculate_similarities_to_queries(model, queries, target, args):
    """
    调试版本的相似度计算函数，打印维度变化
    修复梯度转换numpy的问题
    """
    print(f"\n[DEBUG] 开始计算相似度")
    print(f"[DEBUG] 查询数量: {len(queries)}")
    print(f"[DEBUG] 目标特征形状: {target.shape}")

    similarities = []

    for i, query in enumerate(queries):
        print(f"\n[DEBUG] 计算查询 {i+1}/{len(queries)} 与目标的相似度")
        print(f"[DEBUG] 查询特征形状: {query.shape}")

        # 修正：使用args.device而不是args.gpu_id
        if query.device != args.device:
            query = query.to(args.device)
            print(f"[DEBUG] 查询特征移动到设备: {query.device}")

        sim_values = []

        # 使用torch.no_grad()确保不计算梯度
        with torch.no_grad():
            # 分批处理目标特征
            for b in range(0, target.shape[0], args.batch_sz_sim):
                batch = target[b:b + args.batch_sz_sim]
                print(f"[DEBUG] 处理目标批次 {b//args.batch_sz_sim + 1}: 批次大小={batch.shape[0]}")

                if batch.shape[0] >= 4:
                    # 修正：使用args.device而不是args.gpu_id
                    if batch.device != args.device:
                        batch = batch.to(args.device)
                        print(f"[DEBUG] 目标批次移动到设备: {batch.device}")

                    # 计算相似度
                    try:
                        print(f"[DEBUG] 调用calculate_video_similarity...")
                        print(f"[DEBUG] 查询输入形状: {query.shape} (维度: {query.dim()})")
                        print(f"[DEBUG] 目标输入形状: {batch.shape} (维度: {batch.dim()})")

                        # 临时调试：查看模型内部结构
                        print(f"[DEBUG] 模型类型: {type(model)}")
                        print(f"[DEBUG] 模型是否有visil_head: {hasattr(model, 'visil_head')}")

                        # 尝试计算相似度
                        batch_sim = model.calculate_video_similarity(query, batch)
                        print(f"[DEBUG] 相似度计算成功: 结果={batch_sim}, 形状={batch_sim.shape}")
                        sim_values.append(batch_sim)

                    except Exception as e:
                        print(f"[ERROR] 相似度计算失败: {e}")
                        import traceback
                        traceback.print_exc()

                        # 尝试手动计算相似度用于调试
                        print(f"[DEBUG] 尝试手动计算相似度...")
                        try:
                            # 简单点积相似度
                            if query.dim() == 3 and batch.dim() == 3:
                                # [frames, regions, dim] -> [frames*regions, dim]
                                q_flat = query.view(-1, query.shape[-1])
                                t_flat = batch.view(-1, batch.shape[-1])

                                # 归一化
                                q_norm = torch.nn.functional.normalize(q_flat, p=2, dim=1)
                                t_norm = torch.nn.functional.normalize(t_flat, p=2, dim=1)

                                # 计算相似度
                                manual_sim = torch.mean(torch.matmul(q_norm, t_norm.T))
                                print(f"[DEBUG] 手动计算相似度: {manual_sim}")
                                sim_values.append(manual_sim)
                            else:
                                # 使用随机相似度
                                manual_sim = torch.tensor(0.5).to(args.device)
                                sim_values.append(manual_sim)
                        except:
                            # 添加默认相似度
                            default_sim = torch.tensor(0.0).to(args.device)
                            sim_values.append(default_sim)

        if sim_values:
            # 计算平均相似度
            sim_tensor = torch.stack(sim_values, 0)
            sim_value = torch.mean(sim_tensor)
            print(f"[DEBUG] 平均相似度: {sim_value.item():.4f}")
            # 修复：使用detach()分离梯度
            similarities.append(sim_value.cpu().detach().numpy())
        else:
            print(f"[WARNING] 没有相似度值，使用默认值0.0")
            similarities.append(0.0)

    print(f"[DEBUG] 相似度计算完成，返回 {len(similarities)} 个值")
    return similarities


def print_model_structure(model):
    """打印模型结构信息"""
    print(f"\n[DEBUG] ====== 模型结构信息 ======")
    print(f"[DEBUG] 模型类型: {type(model)}")
    print(f"[DEBUG] 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 打印特征提取器
    if hasattr(model, 'cnn'):
        print(f"[DEBUG] 特征提取器: {type(model.cnn)}")
        print(f"[DEBUG]   - 有normalizer: {hasattr(model.cnn, 'normalizer')}")
        print(f"[DEBUG]   - 有pca: {hasattr(model.cnn, 'pca')}")

    # 打印ViSiL头
    if hasattr(model, 'visil_head'):
        print(f"[DEBUG] ViSiL头: {type(model.visil_head)}")
        print(f"[DEBUG]   - 有attention: {hasattr(model.visil_head, 'attention')}")
        print(f"[DEBUG]   - 有video_comperator: {hasattr(model.visil_head, 'video_comperator')}")
        print(f"[DEBUG]   - 有tensor_dot: {hasattr(model.visil_head, 'tensor_dot')}")
        if hasattr(model.visil_head, 'tensor_dot'):
            print(f"[DEBUG]   - tensor_dot模式: {model.visil_head.tensor_dot.pattern}")
        print(f"[DEBUG]   - 有f2f_sim: {hasattr(model.visil_head, 'f2f_sim')}")
        print(f"[DEBUG]   - 有v2v_sim: {hasattr(model.visil_head, 'v2v_sim')}")
        print(f"[DEBUG]   - 有htanh: {hasattr(model.visil_head, 'htanh')}")

    print(f"[DEBUG] ============================\n")


if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='调试版的视频相似度计算，只处理少量视频并打印维度信息', formatter_class=formatter)

    # ===== 输入输出文件参数 =====
    parser.add_argument('--query_file', type=str,
                        default='datasets/fivr-5k-queries-filtered.txt',
                        help='包含查询视频列表的文件路径（文本格式，每行：视频ID 视频路径），默认：datasets/fivr-5k-queries-filtered.txt')
    parser.add_argument('--database_file', type=str,
                        default='datasets/fivr-5k-database-filtered.txt',
                        help='包含数据库视频列表的文件路径（文本格式，每行：视频ID 视频路径），默认：datasets/fivr-5k-database-filtered.txt')
    parser.add_argument('--output_file', type=str, default='debug_results.json',
                        help='输出文件名，保存查询视频与数据库视频的相似度结果（JSON格式）')

    # ===== 批处理大小参数 =====
    parser.add_argument('--batch_sz', type=int, default=32,
                        help='特征提取时每批包含的帧数。调试模式下使用较小值')
    parser.add_argument('--batch_sz_sim', type=int, default=512,
                        help='相似度计算时每批包含的特征张量数。调试模式下使用较小值')

    # ===== 设备参数 =====
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='使用的GPU设备ID。默认：0（使用第一块GPU）')

    # ===== 相似度计算参数 =====
    parser.add_argument('--similarity_function', type=str, default='symmetric_chamfer',
                        choices=["chamfer", "symmetric_chamfer"],
                        help='用于计算查询-目标帧和视频之间相似度的函数。')

    # ===== 数据加载参数 =====
    parser.add_argument('--workers', type=int, default=2,
                        help='视频加载时使用的工作进程数。调试模式下使用较少进程')

    # ===== 调试参数 =====
    parser.add_argument('--num_debug_videos', type=int, default=5,
                        help='调试时处理的视频数量（查询和数据库各取前N个）')

    args = parser.parse_args()

    print("[DEBUG] =========================================")
    print("[DEBUG] 开始调试ViSiL模型，处理少量视频")
    print("[DEBUG] =========================================")

    # 设备选择
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print(f"[DEBUG] CUDA不可用，使用CPU进行计算")
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"[DEBUG] 使用GPU设备: {device}")

    # 将设备信息添加到args中 - 关键修复
    args.device = device

    # 初始化ViSiL模型
    print(f"\n[DEBUG] 初始化ViSiL模型...")
    symmetric_flag = 'symmetric' in args.similarity_function
    print(f"[DEBUG] 使用对称版本: {symmetric_flag}")

    try:
        model = ViSiL(pretrained=True, symmetric=symmetric_flag).to(device)
        model.eval()
        print(f"[DEBUG] 模型初始化成功")

        # 打印模型结构
        print_model_structure(model)

    except Exception as e:
        print(f"[ERROR] 初始化模型失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # ===== 第一阶段：提取查询视频特征 =====
    print('\n[DEBUG] ===== 第一阶段：提取查询视频特征 =====')

    # 使用调试版视频生成器（只取前5个视频）
    print(f"[DEBUG] 创建查询视频生成器，最多处理 {args.num_debug_videos} 个视频")
    query_generator = DebugVideoGenerator(args.query_file, max_videos=args.num_debug_videos)

    queries, queries_ids = [], []
    print(f'\n[DEBUG] 开始提取查询视频的特征')

    for idx in range(len(query_generator)):
        print(f"\n[DEBUG] --- 处理查询视频 {idx+1}/{len(query_generator)} ---")

        # 获取视频帧和ID
        frames, video_id = query_generator[idx]
        print(f"[DEBUG] 视频ID: {video_id}")
        print(f"[DEBUG] 原始帧形状: {frames.shape}")

        # 检查帧数是否足够
        if frames.shape[0] < 4:
            print(f"[WARNING] 查询视频 {video_id} 帧数过少 ({frames.shape[0]} < 4)，跳过")
            continue

        try:
            # 使用调试版特征提取函数
            features = debug_extract_features(model, frames, args)

            # 确保特征是3D: [frames, regions, dim]
            print(f"[DEBUG] 检查特征维度...")
            if features.dim() == 2:
                print(f"[DEBUG] 特征维度为2D ({features.shape})，添加区域维度")
                features = features.unsqueeze(1)  # [frames, dim] -> [frames, 1, dim]
            elif features.dim() == 4:
                print(f"[DEBUG] 特征维度为4D ({features.shape})，去除批次维度")
                features = features.squeeze(0)  # [batch, frames, regions, dim] -> [frames, regions, dim]

            print(f"[DEBUG] 最终特征形状: {features.shape}")

            # 根据参数决定特征存储位置
            if device.type == 'cpu':
                features = features.cpu()
                print(f"[DEBUG] 特征存储在CPU")
            else:
                print(f"[DEBUG] 特征存储在GPU")

            queries.append(features)
            queries_ids.append(video_id)

            print(f"[DEBUG] 成功添加查询视频 {video_id} 的特征")

        except Exception as e:
            print(f"[ERROR] 处理查询视频 {video_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 报告查询视频处理结果
    print(f"\n[DEBUG] 成功提取 {len(queries)} 个查询视频的特征")
    if not queries:
        print("[ERROR] 没有成功提取任何查询视频特征，程序退出")
        exit(1)

    # ===== 第二阶段：计算与数据库视频的相似度 =====
    print('\n[DEBUG] ===== 第二阶段：计算与数据库视频的相似度 =====')

    # 使用调试版视频生成器（只取前5个视频）
    print(f"[DEBUG] 创建数据库视频生成器，最多处理 {args.num_debug_videos} 个视频")
    database_generator = DebugVideoGenerator(args.database_file, max_videos=args.num_debug_videos)

    similarities = dict({query: dict() for query in queries_ids})
    print(f"[DEBUG] 初始化相似度字典，查询数量: {len(queries_ids)}")

    for idx in range(len(database_generator)):
        print(f"\n[DEBUG] --- 处理数据库视频 {idx+1}/{len(database_generator)} ---")

        # 获取视频帧和ID
        frames, video_id = database_generator[idx]
        print(f"[DEBUG] 数据库视频ID: {video_id}")
        print(f"[DEBUG] 原始帧形状: {frames.shape}")

        # 检查帧数是否足够
        if frames.shape[0] < 4:
            print(f"[WARNING] 数据库视频 {video_id} 帧数过少 ({frames.shape[0]} < 4)，跳过")
            continue

        try:
            # 使用调试版特征提取函数
            features = debug_extract_features(model, frames, args)

            # 确保特征是3D: [frames, regions, dim]
            if features.dim() == 2:
                features = features.unsqueeze(1)
            elif features.dim() == 4:
                features = features.squeeze(0)

            print(f"[DEBUG] 数据库视频特征最终形状: {features.shape}")

            # 使用调试版相似度计算函数
            sims = debug_calculate_similarities_to_queries(model, queries, features, args)

            # 存储相似度结果
            for i, s in enumerate(sims):
                similarities[queries_ids[i]][video_id] = float(s)
                print(f"[DEBUG] 查询 {queries_ids[i]} 与数据库 {video_id} 的相似度: {s:.4f}")

            print(f"[DEBUG] 成功处理数据库视频 {video_id}")

        except Exception as e:
            print(f"[ERROR] 处理数据库视频 {video_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 报告数据库视频处理结果
    if similarities and len(similarities) > 0:
        first_key = list(similarities.keys())[0]
        print(f"\n[DEBUG] 成功处理 {len(similarities[first_key])} 个数据库视频")
    else:
        print(f"\n[DEBUG] 未成功处理任何数据库视频")

    # ===== 第三阶段：保存结果 =====
    print('\n[DEBUG] ===== 第三阶段：保存结果 =====')
    print('[DEBUG] 保存相似度结果到JSON文件')

    with open(args.output_file, 'w') as f:
        json.dump(similarities, f, indent=1)

    print(f'[DEBUG] 完成！结果已保存到 {args.output_file}')

    # 打印最终的相似度矩阵
    print(f"\n[DEBUG] ===== 最终的相似度矩阵 ======")
    print("[DEBUG] 查询视频: ", queries_ids)

    if similarities:
        for query_id in similarities:
            print(f"\n[DEBUG] 查询视频: {query_id}")
            for db_id, sim in similarities[query_id].items():
                print(f"[DEBUG]  与 {db_id} 的相似度: {sim:.4f}")

    print(f"\n[DEBUG] =========================================")
    print("[DEBUG] 调试脚本执行完成")
    print("[DEBUG] =========================================")