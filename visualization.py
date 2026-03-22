import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# [修改点] 热图生成函数，从原 stage_utils 移入
def compute_and_save_heatmap(model, feat_q, feat_db, save_path, matrix_save_path=None, vmin=None, vmax=None, do_plot=True):
    """
    计算帧间相似度矩阵并保存为热图，可选择不绘制热图（仅保存矩阵）。
    参数:
        matrix_save_path: 如果提供，将矩阵保存为此路径（.npy）
        vmin, vmax: 颜色范围，若为 None 则使用矩阵的实际最小最大值
        do_plot: 是否生成热图图片，若为 False 则只保存矩阵
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    with torch.no_grad():
        f2f_sim = model.calculate_f2f_matrix(feat_q.unsqueeze(0), feat_db.unsqueeze(0))
        # 根据维度动态提取二维矩阵
        if f2f_sim.dim() == 5:
            matrix = f2f_sim[0, :, :, 0, 0].cpu().numpy()
        elif f2f_sim.dim() == 4:
            matrix = f2f_sim[0, :, :, 0].cpu().numpy()
        elif f2f_sim.dim() == 3:
            matrix = f2f_sim[0, :, :].cpu().numpy()
        elif f2f_sim.dim() == 2:
            matrix = f2f_sim.cpu().numpy()
        else:
            print(f"> 警告: 相似度矩阵维度异常 {f2f_sim.shape}，跳过热图")
            return

        if matrix.size == 0 or matrix.ndim != 2:
            return

    # 保存矩阵文件
    if matrix_save_path is not None:
        os.makedirs(os.path.dirname(matrix_save_path), exist_ok=True)
        np.save(matrix_save_path, matrix)

    if not do_plot:
        return

    # 自动选择颜色映射：有负值用 RdBu，否则 Reds
    cmap = 'RdBu' if matrix.min() < 0 else 'Reds'
    if vmin is None:
        vmin = matrix.min()
    if vmax is None:
        vmax = matrix.max()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Similarity')
    plt.xlabel('Database frame')
    plt.ylabel('Query frame')
    plt.title('Frame-to-Frame Similarity Matrix')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()