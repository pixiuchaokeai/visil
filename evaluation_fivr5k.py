# evaluation_fivr5k.py - FIVR-5K评估器（简化参数传递）
"""
FIVR-5K数据集评估器
修复：简化参数传递，使用config配置
"""

import os
import argparse
import time
from config import EvaluationConfig, FEATURE_CONFIGS, ABLATION_CONFIGS
from base_evaluator import BaseEvaluator


class FIVR5KEvaluator(BaseEvaluator):
    """FIVR-5K数据集评估器 - 简化版本"""

    def __init__(self, config: EvaluationConfig):
        # ==================== 修复点1：确保使用FIVR-5K数据集 ====================
        if not hasattr(config, 'dataset') or config.dataset != "FIVR-5K":
            print(f"> 注意：将数据集设置为FIVR-5K (原为: {getattr(config, 'dataset', '未设置')})")
            config.dataset = "FIVR-5K"

        # ==================== 修复点2：为实验添加时间戳 ====================
        if hasattr(config, 'experiment_id'):
            timestamp = str(int(time.time()))[-6:]
            config.experiment_id = f"{config.feature_config.name}_{timestamp}"

        super().__init__(config)


def evaluate_fivr5k():
    """主评估函数 - 简化命令行参数"""
    parser = argparse.ArgumentParser(description='FIVR-5K数据集评估')

    # 基本参数
    parser.add_argument('--base_output_dir', type=str, default='output',
                        help='基础输出目录')

    # 特征配置参数
    parser.add_argument('--feature_config', type=str, default='L3-iMAC9x',
                        choices=list(FEATURE_CONFIGS.keys()) + list(ABLATION_CONFIGS.keys()),
                        help='特征配置名称')

    # 设备参数
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU设备ID')
    parser.add_argument('--cpu_only', action='store_true',
                        help='强制使用CPU')

    # 实验参数
    parser.add_argument('--experiment_id', type=str, default=None,
                        help='实验标识符')
    parser.add_argument('--clear_cache', action='store_true',
                        help='清理缓存文件')

    args = parser.parse_args()

    print("=" * 60)
    print("FIVR-5K 评估")
    print("=" * 60)

    # ==================== 修复点3：根据参数选择特征配置 ====================
    if args.feature_config in FEATURE_CONFIGS:
        feature_config = FEATURE_CONFIGS[args.feature_config]
    elif args.feature_config in ABLATION_CONFIGS:
        feature_config = ABLATION_CONFIGS[args.feature_config]
    else:
        raise ValueError(f"未知的特征配置: {args.feature_config}")

    # ==================== 修复点4：正确创建配置 ====================
    config = EvaluationConfig(
        base_output_dir=args.base_output_dir,
        feature_config=feature_config,
        gpu_id=args.gpu_id,
        cpu_only=args.cpu_only,
        experiment_id=args.experiment_id or f"{args.feature_config}_{int(time.time())}",
        clear_cache=args.clear_cache
    )

    # 运行评估
    evaluator = FIVR5KEvaluator(config)
    results = evaluator.evaluate()

    return results


if __name__ == '__main__':
    evaluate_fivr5k()