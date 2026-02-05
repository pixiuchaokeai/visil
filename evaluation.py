import torch
import argparse

from tqdm import tqdm
from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import DatasetGenerator


@torch.no_grad()
def extract_features(model, frames, args):
    """
    从视频帧中提取特征

    参数:
        model: ViSiL模型，用于提取特征
        frames: 输入的视频帧，形状为 [帧数, 高, 宽, 通道] 或 [高, 宽, 通道]
        args: 命令行参数，包含batch_sz、device等配置

    返回:
        features: 提取的特征张量，形状经过处理后确保至少有4个片段
    """
    features = []
    batch_sz = args.batch_sz  # 每次处理的帧数，默认128帧

    # 确保输入维度正确：如果是单帧 [H, W, C]，增加batch维度变成 [1, H, W, C]
    if frames.dim() == 3:
        frames = frames.unsqueeze(0)

    # 分批处理视频帧，避免显存溢出
    for i in range(0, frames.shape[0], batch_sz):
        batch = frames[i:i + batch_sz]  # 获取当前批次的帧
        if batch.shape[0] > 0:
            # 将数据移动到指定设备（GPU或CPU）
            if hasattr(args, 'device'):
                batch = batch.to(args.device).float()
            else:
                batch = batch.to(args.gpu_id).float()

            # 确保输入格式为 [N, H, W, C]（N=batch, H=高, W=宽, C=3通道RGB）
            if batch.shape[-1] != 3:
                # 如果通道在第二维 [N, C, H, W]，需要置换为 [N, H, W, C]
                if batch.shape[1] == 3:
                    batch = batch.permute(0, 2, 3, 1)

            # 使用模型提取当前批次的特征并保存
            features.append(model.extract_features(batch))

    # 合并所有批次的特征
    if features:
        features = torch.cat(features, 0)
    else:
        # 如果没有提取到特征（空视频），创建随机默认特征
        if hasattr(args, 'device'):
            device = args.device
        else:
            device = torch.device(f'cuda:{args.gpu_id}')
        features = torch.randn(4, 9, 512).to(device)  # 形状 [4, 9, 512] 是ViSiL的标准输出格式

    # 确保特征至少有4个片段（ViSiL模型的最小要求）
    # 如果不足4个，通过重复自身来扩充
    while features.shape[0] < 4:
        features = torch.cat([features, features], 0)
    return features


@torch.no_grad()
def calculate_similarities_to_queries(model, queries, target, args):
    """
    计算目标视频与所有查询视频的相似度

    参数:
        model: ViSiL模型，用于计算相似度
        queries: 查询视频的特征列表，每个元素是一个特征张量
        target: 目标视频的特征张量
        args: 命令行参数，包含batch_sz_sim（相似度计算的批次大小）

    返回:
        similarities: 目标视频与每个查询视频的相似度分数列表
    """
    similarities = []
    for i, query in enumerate(queries):
        # 确定计算设备（GPU或CPU）
        if hasattr(args, 'device'):
            device = args.device
        else:
            device = torch.device(f'cuda:{args.gpu_id}')

        # 确保查询特征在正确的设备上
        if query.device != device:
            query = query.to(device)

        sim = []
        # 分批计算相似度，避免显存不足
        # 计算需要多少个批次：总片段数 // 批次大小 + 1
        for b in range(target.shape[0] // args.batch_sz_sim + 1):
            batch = target[b * args.batch_sz_sim: (b + 1) * args.batch_sz_sim]
            # 只处理至少有4个片段的批次（ViSiL的要求）
            if batch.shape[0] >= 4:
                sim.append(model.calculate_video_similarity(query, batch))

        # 计算该查询视频与目标视频的平均相似度
        sim = torch.mean(torch.cat(sim, 0))
        # 将结果分离计算图，转到CPU，转为numpy数组
        similarities.append(sim.cpu().detach().numpy())
    return similarities


def query_vs_target(model, dataset, args):
    """
    执行查询视频与目标视频库的相似度计算和评估

    这是主流程函数，包括：
    1. 提取查询视频的特征
    2. 提取数据库视频的特征
    3. 计算查询视频与每个数据库视频的相似度
    4. 评估结果

    参数:
        model: ViSiL模型
        dataset: 数据集对象（如FIVR、CC_WEB_VIDEO等），提供查询视频和数据库视频列表
        args: 命令行参数
    """
    # ========== 第一步：提取查询视频的特征 ==========
    # 创建数据集生成器，用于加载查询视频
    generator = DatasetGenerator(args.video_dir, dataset.get_queries(), args.pattern)
    # 创建DataLoader，使用多进程加载视频（workers指定进程数）
    loader = DataLoader(generator, num_workers=args.workers)

    all_db, queries, queries_ids = set(), [], []
    print('> 正在提取查询视频的特征')

    # 遍历所有查询视频
    for video in tqdm(loader):
        frames = video[0][0]  # 视频帧数据
        video_id = video[1][0]  # 视频ID

        if frames.shape[0] > 0:  # 确保视频不为空
            features = extract_features(model, frames, args)
            # 如果不加载到GPU，将特征移到CPU节省显存
            if not args.load_queries:
                features = features.cpu()

            all_db.add(video_id)  # 记录已处理的视频ID
            queries.append(features)  # 保存特征
            queries_ids.append(video_id)  # 保存ID

    # ========== 第二步：提取数据库视频并计算相似度 ==========
    # 创建数据集生成器，用于加载数据库视频
    generator = DatasetGenerator(args.video_dir, dataset.get_database(), args.pattern)
    loader = DataLoader(generator, num_workers=args.workers)

    # 初始化相似度字典：每个查询视频对应一个字典，存储与所有目标视频的相似度
    similarities = dict({query: dict() for query in queries_ids})

    print('\n> 正在计算查询视频与目标视频的相似度')

    # 遍历所有数据库视频
    for video in tqdm(loader):
        frames = video[0][0]  # 视频帧数据
        video_id = video[1][0]  # 视频ID

        if frames.shape[0] > 0:  # 确保视频不为空
            # 提取当前数据库视频的特征
            features = extract_features(model, frames, args)
            # 计算该视频与所有查询视频的相似度
            sims = calculate_similarities_to_queries(model, queries, features, args)

            all_db.add(video_id)
            # 保存相似度结果到字典
            for i, s in enumerate(sims):
                similarities[queries_ids[i]][video_id] = float(s)

    # ========== 第三步：评估结果 ==========
    print('\n> 在 {} 数据集上进行评估'.format(dataset.name))
    # 调用数据集的评估方法，传入相似度矩阵和所有视频ID集合
    dataset.evaluate(similarities, all_db)


if __name__ == '__main__':
    # 创建参数解析器，使用默认格式化类显示默认值
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(
        description='ViSiL网络在五个数据集上的评估代码', formatter_class=formatter)

    # 数据集选择参数
    parser.add_argument('--dataset', type=str, default="FIVR-5K",
                        choices=["FIVR-200K", "FIVR-5K", "CC_WEB_VIDEO", "SVD", "EVVE"],
                        help='评估数据集的名称')

    # 视频目录路径参数（已改为默认参数）
    parser.add_argument('--video_dir', type=str, default='datasets/FIVR-200K',
                        help='包含数据库视频的文件路径 (默认: datasets/FIVR-200K)')

    # 视频文件命名模式参数（已改为默认参数）
    parser.add_argument('--pattern', type=str, default='{id}.mp4',
                        help='视频在视频目录中的存储模式，例如 "{id}.mp4"，其中 "{id}" 会被替换为视频ID。'
                             '同时支持 Unix 风格的路径名模式扩展 (默认: {id}.mp4)')

    # 特征提取的批次大小
    parser.add_argument('--batch_sz', type=int, default=128,
                        help='特征提取时每个批次包含的帧数 (默认: 128)')

    # 相似度计算的批次大小
    parser.add_argument('--batch_sz_sim', type=int, default=2048,
                        help='相似度计算时每个批次包含的特征张量数 (默认: 2048)')

    # GPU设备ID
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='使用的GPU设备ID (默认: 0)')

    # 是否将查询视频加载到GPU内存
    parser.add_argument('--load_queries', action='store_true',
                        help='是否将查询视频特征加载到GPU内存中（默认False，即存于CPU节省显存）')

    # 相似度计算函数选择
    parser.add_argument('--similarity_function', type=str, default='chamfer',
                        choices=["chamfer", "symmetric_chamfer"],
                        help='用于计算查询-目标帧和视频相似度的函数 (默认: chamfer)')

    # 数据加载的工作进程数
    parser.add_argument('--workers', type=int, default=8,
                        help='用于视频加载的工作进程数 (默认: 8)')

    args = parser.parse_args()

    # ========== 根据数据集名称加载对应的数据集类 ==========
    if 'CC_WEB' in args.dataset:
        from datasets import CC_WEB_VIDEO

        dataset = CC_WEB_VIDEO()
    elif 'FIVR' in args.dataset:
        from datasets import FIVR

        # FIVR有两个版本：200K（20万视频）和5K（5千视频）
        dataset = FIVR(version=args.dataset.split('-')[1].lower())
    elif 'EVVE' in args.dataset:
        from datasets import EVVE

        dataset = EVVE()
    elif 'SVD' in args.dataset:
        from datasets import SVD

        dataset = SVD()

    # ========== 设备选择逻辑 ==========
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print("> 使用CPU进行计算")
    else:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"> 使用GPU设备: {device}")

    # 将设备信息添加到args，方便后续函数使用
    args.device = device

    # ========== 初始化ViSiL模型 ==========
    # pretrained=True 加载预训练权重
    # symmetric 根据similarity_function参数决定是否使用对称Chamfer距离
    model = ViSiL(pretrained=True, symmetric='symmetric' in args.similarity_function).to(device)
    model.eval()  # 设置为评估模式（关闭Dropout等）

    # ========== 执行查询与目标视频的相似度计算 ==========
    query_vs_target(model, dataset, args)