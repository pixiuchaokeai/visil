import argparse
import os
import json
import numpy as np
import shutil
import math


# 加载视频的帧索引文件
def load_frame_indices(video_id, frames_base_dir, method):
    """
    从第一阶段帧目录加载指定视频的帧索引列表。
    返回列表 [帧号0, 帧号1, ...] 或 None
    """
    indices_path = os.path.join(frames_base_dir, method, video_id, 'indices.json')
    if not os.path.isfile(indices_path):
        return None
    try:
        with open(indices_path, 'r') as f:
            indices = json.load(f)
        return indices  # 已经是列表
    except Exception as e:
        print(f"  加载帧索引失败 {indices_path}: {e}")
        return None


def load_matrix(q_id, db_id, base_dir):
    """加载第二阶段矩阵，如果存在返回矩阵，否则返回None"""
    filepath = os.path.join(base_dir, q_id, f"{q_id}_{db_id}.npy")
    if os.path.isfile(filepath):
        try:
            mat = np.load(filepath)
            return mat
        except Exception as e:
            print(f"  加载失败 {filepath}: {e}")
            return None
    else:
        return None


def print_matrix_with_indices(matrix, q_indices, db_indices, chunk_size=50, max_rows=None, max_cols=None):
    """
    打印矩阵（可限制最大行列数），并在行首和列首显示帧号。
    为避免单次打印内容过多，按 chunk_size 分段打印列（每段最多显示 chunk_size 列）

    Args:
        matrix: 相似度矩阵 (二维数组/tensor)
        q_indices: 查询视频帧索引列表
        db_indices: 数据库视频帧索引列表
        chunk_size: 每段打印的列数（默认50列，可根据终端宽度调整）
        max_rows: 打印的最大行数（None表示打印所有行）
        max_cols: 打印的最大列数（None表示打印所有列）
    """
    # 兼容torch tensor和numpy数组
    try:
        import torch
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.cpu().numpy()
    except ImportError:
        pass

    # 应用最大行列数限制
    rows, cols = matrix.shape
    use_rows = rows if max_rows is None else min(rows, max_rows)
    use_cols = cols if max_cols is None else min(cols, max_cols)

    # 截取矩阵到指定的最大行列数
    matrix = matrix[:use_rows, :use_cols]
    q_indices = q_indices[:use_rows]
    db_indices = db_indices[:use_cols]

    print(f"矩阵形状: {rows} × {cols} (实际打印: {use_rows} × {use_cols})")
    print("=" * 80)

    # 计算需要分多少段打印列
    num_chunks = math.ceil(use_cols / chunk_size)

    for chunk_idx in range(num_chunks):
        # 计算当前段的列范围
        start_col = chunk_idx * chunk_size
        end_col = min((chunk_idx + 1) * chunk_size, use_cols)
        current_cols = end_col - start_col

        print(f"\n【第 {chunk_idx + 1}/{num_chunks} 段】列范围: {start_col} ~ {end_col - 1}")
        print("-" * 80)

        # 准备表头：数据库帧号
        header = "查询\\数据库"
        for j in range(start_col, end_col):
            header += f" | {db_indices[j]:6d}"
        print(header)
        print("-" * len(header))  # 打印分隔线

        # 打印所有行的当前段列数据
        for i in range(use_rows):
            row_str = f"{q_indices[i]:6d}"
            for j in range(start_col, end_col):
                val = matrix[i, j]
                row_str += f" | {val:6.4f}"
            print(row_str)

        print("-" * 80)

    print(f"\n✅ 矩阵打印完成，共打印 {use_rows} 行 × {use_cols} 列")


def main():
    parser = argparse.ArgumentParser(description='获取指定查询-数据库对的第二阶段相似矩阵并显示具体数值及帧号')
    parser.add_argument('--json', type=str, default="pairs.json",
                        help='包含查询和数据库列表的JSON文件路径')
    parser.add_argument('--sim_dir', type=str, default=None,
                        help='第二阶段矩阵基础目录，默认自动从 output/sim_matrices_cos_rough 下找最新子目录')
    parser.add_argument('--frames_dir', type=str, default='output/frames1',
                        help='第一阶段帧保存的基础目录（默认 output/frames1）')
    parser.add_argument('--method', type=str, default='iframe',
                        help='第一阶段方法名称 (default: iframe)')
    parser.add_argument('--symmetric', action='store_true',
                        help='是否使用对称模型 (默认非对称，仅影响矩阵目录查找，不影响帧目录)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='如果指定，将矩阵复制到此目录，保持原目录结构')
    parser.add_argument('--list_only', action='store_true',
                        help='只列出矩阵路径，不加载矩阵内容')
    parser.add_argument('--max_rows', type=int, default=100,
                        help='打印矩阵的最大行数（默认100）')
    parser.add_argument('--max_cols', type=int, default=100,
                        help='打印矩阵的最大列数（默认100）')
    args = parser.parse_args()

    # 确定模型后缀（仅用于矩阵目录查找）
    model_suffix = '_sym' if args.symmetric else '_v'
    method_suffix = args.method + model_suffix

    # 确定矩阵基础目录
    if args.sim_dir is None:
        base_cos = os.path.join('output', 'sim_matrices_cos_rough')
        if not os.path.isdir(base_cos):
            print(f"错误: 基础目录 {base_cos} 不存在")
            return
        # 优先精确匹配方法后缀的目录
        candidate = os.path.join(base_cos, method_suffix)
        if os.path.isdir(candidate):
            base_dir = candidate
        else:
            # 否则取最新子目录（按修改时间）
            subdirs = [d for d in os.listdir(base_cos)
                       if os.path.isdir(os.path.join(base_cos, d))]
            if not subdirs:
                print(f"错误: {base_cos} 下没有子目录")
                return
            latest = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(base_cos, d)))
            base_dir = os.path.join(base_cos, latest)
            print(f"警告: 未找到精确目录 {method_suffix}，使用最新目录 {latest}")
    else:
        base_dir = args.sim_dir

    if not os.path.isdir(base_dir):
        print(f"错误: 目录不存在 {base_dir}")
        return

    print(f"使用矩阵目录: {base_dir}")

    # 加载 JSON
    if not os.path.isfile(args.json):
        print(f"错误: JSON 文件不存在 {args.json}")
        return
    try:
        with open(args.json, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载 JSON 失败: {e}")
        return

    if not isinstance(data, dict):
        print("JSON 应为字典格式")
        return

    total_pairs = 0
    found_pairs = 0
    missing_pairs = []

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # 遍历每个查询及其数据库列表
    for q_id, db_list in data.items():
        if not isinstance(db_list, list):
            print(f"查询 {q_id} 的值不是列表，跳过")
            continue

        # 加载查询视频的帧索引
        q_indices = load_frame_indices(q_id, args.frames_dir, args.method)
        if q_indices is None and not args.list_only:
            print(f"警告: 无法加载查询视频 {q_id} 的帧索引，将使用索引0,1,...")

        for db_id in db_list:
            total_pairs += 1
            if args.list_only:
                filepath = os.path.join(base_dir, q_id, f"{q_id}_{db_id}.npy")
                if os.path.isfile(filepath):
                    print(f"存在: {filepath}")
                    found_pairs += 1
                else:
                    print(f"缺失: {q_id} - {db_id}")
                    missing_pairs.append((q_id, db_id))
            else:
                mat = load_matrix(q_id, db_id, base_dir)
                if mat is not None:
                    found_pairs += 1
                    print(f"\n成功: {q_id} - {db_id}, 形状 {mat.shape}")

                    # 加载数据库视频的帧索引
                    db_indices = load_frame_indices(db_id, args.frames_dir, args.method)
                    if db_indices is None:
                        print(f"  警告: 无法加载数据库视频 {db_id} 的帧索引，将使用索引0,1,...")
                        db_indices = list(range(mat.shape[1]))

                    # 使用查询帧索引，如果缺失则用0,1,...
                    if q_indices is None:
                        q_use = list(range(mat.shape[0]))
                    else:
                        q_use = q_indices

                    # 打印矩阵数值和帧号（修复参数传递）
                    print_matrix_with_indices(
                        matrix=mat,
                        q_indices=q_use,
                        db_indices=db_indices,
                        chunk_size=50,  # 分段列数
                        max_rows=args.max_rows,  # 最大行数
                        max_cols=args.max_cols  # 最大列数
                    )

                    if args.output_dir:
                        src = os.path.join(base_dir, q_id, f"{q_id}_{db_id}.npy")
                        dst_dir = os.path.join(args.output_dir, q_id)
                        os.makedirs(dst_dir, exist_ok=True)
                        dst = os.path.join(dst_dir, f"{q_id}_{db_id}.npy")
                        shutil.copy2(src, dst)
                else:
                    print(f"缺失: {q_id} - {db_id}")
                    missing_pairs.append((q_id, db_id))

    print(f"\n总计 {total_pairs} 对，找到 {found_pairs} 对，缺失 {len(missing_pairs)} 对")
    if missing_pairs:
        print("缺失列表:")
        for q, d in missing_pairs:
            print(f"  {q} - {d}")


if __name__ == '__main__':
    main()