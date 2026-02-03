import torch
import argparse
import gc
import json  # 添加json导入
import sys  # 添加sys导入

from tqdm import tqdm
from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import DatasetGenerator


@torch.no_grad()
def extract_features(model, frames, args):
    """
    从视频帧中提取特征，支持分批处理以避免GPU内存溢出
    改进：支持CPU/GPU混合环境
    """
    print(f"Input frames shape: {frames.shape}")  # 应该是 (N, C, H, W)

    features = []

    # 动态批次大小调整
    batch_sz = args.batch_sz
    if hasattr(args, 'device') and args.device.type == 'cpu':
        # CPU环境下使用更小的批次大小
        batch_sz = min(batch_sz, 32)

    for i in range(frames.shape[0] // batch_sz + 1):
        batch = frames[i * batch_sz: (i + 1) * batch_sz]

        if batch.shape[0] > 0:
            # 移动到设备并计算特征
            if hasattr(args, 'device'):
                batch = batch.to(args.device).float()
            else:
                batch = batch.to(args.gpu_id).float()
            batch_features = model.extract_features(batch)
            features.append(batch_features)

    if features:
        features = torch.cat(features, 0)
    else:
        # 如果没有任何特征，创建一个空张量
        features = torch.zeros((4, 512), device=batch.device if hasattr(args, 'device') else args.gpu_id)

    # 确保最少4帧
    while features.shape[0] < 4:
        features = torch.cat([features, features], 0)

    return features


@torch.no_grad()
def calculate_similarities_to_queries(model, queries, target, args):
    """
    计算目标视频与所有查询视频的相似度
    改进：支持CPU/GPU混合环境
    """
    similarities = []

    # 动态批次大小调整
    batch_sz_sim = args.batch_sz_sim
    if hasattr(args, 'device') and args.device.type == 'cpu':
        batch_sz_sim = min(batch_sz_sim, 512)

    for i, query in enumerate(queries):
        # 确保查询特征在正确的设备上
        if hasattr(args, 'device'):
            if query.device != args.device:
                query = query.to(args.device)
        elif query.device.type == 'cpu':
            query = query.to(args.gpu_id)

        sim = []

        # 分批处理目标视频特征
        for b in range(target.shape[0] // batch_sz_sim + 1):
            batch = target[b * batch_sz_sim: (b + 1) * batch_sz_sim]

            if batch.shape[0] >= 4:
                # 确保批次在正确的设备上
                if hasattr(args, 'device') and batch.device != args.device:
                    batch = batch.to(args.device)
                elif not hasattr(args, 'device') and batch.device.type == 'cpu':
                    batch = batch.to(args.gpu_id)

                batch_sim = model.calculate_video_similarity(query, batch)
                sim.append(batch_sim)

        if sim:
            sim = torch.mean(torch.cat(sim, 0))
            similarities.append(sim.cpu().numpy())
        else:
            similarities.append(0.0)

    return similarities


def query_vs_target(model, dataset, args):
    """
    主评估函数：处理查询视频和数据库视频，计算相似度并评估
    改进：添加内存管理
    """
    # 检查查询和数据库视频列表
    queries_list = dataset.get_queries()
    database_list = dataset.get_database()

    print(f"> 查询视频数量: {len(queries_list)}")
    print(f"> 数据库视频数量: {len(database_list)}")

    if not queries_list:
        print("错误: 没有查询视频")
        return

    if not database_list:
        print("错误: 没有数据库视频")
        return

    generator = DatasetGenerator(args.video_dir, queries_list, args.pattern)
    loader = DataLoader(generator, num_workers=args.workers)

    all_db, queries, queries_ids = set(), [], []
    print('> 提取查询视频的特征')

    for idx, video in enumerate(tqdm(loader, desc='查询视频')):
        frames = video[0][0]
        video_id = video[1][0]

        if frames.shape[0] > 0:
            features = extract_features(model, frames, args)

            if not args.load_queries or (hasattr(args, 'device') and args.device.type == 'cpu'):
                features = features.cpu()

            all_db.add(video_id)
            queries.append(features)
            queries_ids.append(video_id)

            # 内存管理
            if idx % 50 == 0 and hasattr(args, 'enable_garbage_collection') and args.enable_garbage_collection:
                gc.collect()
        else:
            print(f"警告: 查询视频 {video_id} 没有有效帧，跳过")

    if not queries:
        print("错误: 没有成功提取任何查询视频特征")
        return

    generator = DatasetGenerator(args.video_dir, database_list, args.pattern)
    loader = DataLoader(generator, num_workers=args.workers)

    similarities = dict({query: dict() for query in queries_ids})
    print('\n> 计算查询-目标视频相似度')

    processed_count = 0
    for idx, video in enumerate(tqdm(loader, desc='数据库视频')):
        frames = video[0][0]
        video_id = video[1][0]

        if frames.shape[0] > 0:
            features = extract_features(model, frames, args)
            sims = calculate_similarities_to_queries(model, queries, features, args)

            all_db.add(video_id)
            for i, s in enumerate(sims):
                similarities[queries_ids[i]][video_id] = float(s)

            processed_count += 1

            # 内存管理
            if idx % 50 == 0:
                if hasattr(args, 'enable_garbage_collection') and args.enable_garbage_collection:
                    gc.collect()
                if hasattr(args, 'checkpoint_interval') and args.checkpoint_interval > 0 and (
                        idx + 1) % args.checkpoint_interval == 0:
                    checkpoint_file = f"checkpoint_{dataset.name}_{idx + 1}.json"
                    with open(checkpoint_file, 'w') as f:
                        json.dump(similarities, f, indent=1)
        else:
            print(f"警告: 数据库视频 {video_id} 没有有效帧，跳过")

    print(f'> 成功处理 {processed_count} 个数据库视频')

    print('\n> 在{}数据集上评估'.format(dataset.name))

    # 确保数据集有evaluate方法
    if hasattr(dataset, 'evaluate'):
        try:
            dataset.evaluate(similarities, all_db)
        except Exception as e:
            print(f"评估过程中出错: {e}")
            # 保存相似度结果供后续分析
            output_file = f"{dataset.name}_similarities.json"
            with open(output_file, 'w') as f:
                json.dump(similarities, f, indent=1)
            print(f"相似度结果已保存到: {output_file}")
    else:
        print(f"警告: 数据集 {dataset.name} 没有evaluate方法")
        # 保存相似度结果
        output_file = f"{dataset.name}_similarities.json"
        with open(output_file, 'w') as f:
            json.dump(similarities, f, indent=1)
        print(f"相似度结果已保存到: {output_file}")


if __name__ == '__main__':
    # 设置参数解析器格式
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='在五个数据集上评估ViSiL网络的代码', formatter_class=formatter)

    # ===== 核心参数 =====
    parser.add_argument('--dataset', type=str, required=True,
                        choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE"],
                        help='评估数据集名称')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='包含数据库视频的目录路径')
    parser.add_argument('--pattern', type=str, required=True,
                        help='视频在目录中的存储模式，例如：\"{id}/video.*\"，其中\"{id}\"会被视频ID替换。'
                             '也支持Unix风格的路径名模式扩展（如*.mp4）')

    # ===== 批处理参数 =====
    parser.add_argument('--batch_sz', type=int, default=128,
                        help='特征提取时每批包含的帧数，默认128')
    parser.add_argument('--batch_sz_sim', type=int, default=2048,
                        help='相似度计算时每批包含的特征张量数，默认2048')

    # ===== 设备参数 =====
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='使用的GPU设备ID，默认0（第一个GPU）')
    parser.add_argument('--load_queries', action='store_true',
                        help='是否将查询视频特征加载到GPU内存中的标志位（查询视频多时可能显存不足）')

    # ===== 相似度函数参数 =====
    parser.add_argument('--similarity_function', type=str, default='chamfer',
                        choices=["chamfer", "symmetric_chamfer"],
                        help='用于计算查询-目标帧和视频之间相似度的函数。'
                             'chamfer: 单向Chamfer距离；symmetric_chamfer: 对称Chamfer距离')

    # ===== 性能参数 =====
    parser.add_argument('--workers', type=int, default=8,
                        help='视频加载时使用的工作进程数，默认8（多进程加速）')

    # ===== 新增内存管理参数 =====
    parser.add_argument('--cpu_only', action='store_true',
                        help='强制使用CPU，即使GPU可用')
    parser.add_argument('--adaptive_batch', action='store_true',
                        help='启用自适应批次大小')
    parser.add_argument('--enable_garbage_collection', action='store_true',
                        help='启用垃圾回收')
    parser.add_argument('--max_cpu_memory_gb', type=float, default=16.0,
                        help='CPU最大内存使用')
    parser.add_argument('--checkpoint_interval', type=int, default=0,
                        help='检查点保存间隔（0表示不保存）')

    args = parser.parse_args()

    # ===== 加载数据集 =====
    dataset = None
    try:
        if 'CC_WEB' in args.dataset:
            from datasets import CC_WEB_VIDEO

            dataset = CC_WEB_VIDEO()
        elif 'FIVR' in args.dataset:
            from datasets import FIVR

            # 根据数据集版本创建对象（200K或5K）
            version = args.dataset.split('-')[1].lower()
            dataset = FIVR(version=version)
        elif 'EVVE' in args.dataset:
            from datasets import EVVE

            dataset = EVVE()
        elif 'SVD' in args.dataset:
            from datasets import SVD

            dataset = SVD()
    except ImportError as e:
        print(f"错误: 无法导入数据集模块 - {e}")
        print("请确保已安装所需的数据集模块")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 加载数据集时出错 - {e}")
        sys.exit(1)

    if dataset is None:
        print(f"错误: 无法加载数据集 {args.dataset}")
        sys.exit(1)

    print(f"> 已加载数据集: {dataset.name}")

    # ===== 设备选择 =====
    if args.cpu_only or not torch.cuda.is_available():
        args.device = torch.device('cpu')
        print("> 使用CPU进行计算")
    else:
        args.device = torch.device(f'cuda:{args.gpu_id}')
        print(f"> 使用GPU设备: {args.device}")

    # ===== 初始化模型 =====
    try:
        model = ViSiL(pretrained=True, symmetric='symmetric' in args.similarity_function).to(args.device)
        model.eval()  # 设置为评估模式（禁用dropout等）
        print("> 模型初始化成功")
    except Exception as e:
        print(f"错误: 初始化模型失败 - {e}")
        sys.exit(1)

    # ===== 执行评估 =====
    try:
        query_vs_target(model, dataset, args)
        print("> 评估完成")
    except KeyboardInterrupt:
        print("\n> 评估被用户中断")
    except Exception as e:
        print(f"> 评估过程中出错: {e}")
        import traceback

        traceback.print_exc()