# feature_comparison.py - 特征比较实验（修复参数传递）
"""
运行论文中的特征比较实验
比较L2-iMAC4x和L3-iMAC9x
修复点：正确配置特征比较实验
"""

import os
import json
import time
import random
from datetime import datetime
from tqdm import tqdm

from config import FEATURE_CONFIGS, EvaluationConfig
from evaluation_fivr5k import FIVR5KEvaluator


class FeatureComparisonExperiment:
    """特征比较实验 - 修复版本"""

    def __init__(self, base_output_dir="paper_experiments/feature_comparison"):
        # ==================== 修复点1：使用正确的输出目录 ====================
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)

        # 实验记录
        self.results = []
        self.start_time = time.time()

        print("=" * 80)
        print("论文特征比较实验（表2）")
        print("=" * 80)
        print(f"基础输出目录: {base_output_dir}")
        print(f"比较方法: {list(FEATURE_CONFIGS.keys())}")

    def generate_experiment_id(self, feature_name):
        """生成唯一的实验ID"""
        timestamp = str(int(time.time()))[-6:]
        random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
        return f"{feature_name}_{timestamp}_{random_suffix}"

    def run_single_experiment(self, feature_name: str, feature_config) -> dict:
        """运行单个特征配置的实验"""
        print(f"\n{'=' * 60}")
        print(f"特征比较实验: {feature_name}")
        print(f"{'=' * 60}")

        # ==================== 修复点2：生成唯一的实验ID ====================
        experiment_id = self.generate_experiment_id(feature_name)

        # 创建配置
        config = EvaluationConfig(
            base_output_dir="output",  # 使用统一的output目录
            experiment_id=experiment_id,
            feature_config=feature_config,
            clear_cache=False  # 不清除缓存，可以复用帧数据
        )

        try:
            # 运行评估
            evaluator = FIVR5KEvaluator(config)
            result = evaluator.evaluate()

            # 记录结果
            experiment_result = {
                "feature_name": feature_name,
                "experiment_id": experiment_id,
                "feature_config": config.feature_config.to_dict(),
                "evaluation_results": result["evaluation"],
                "total_time": result["total_time"],
                "output_dir": result["output_dir"],
                "timestamp": datetime.now().isoformat()
            }

            print(f"> 实验完成: {feature_name}")
            print(f"  实验ID: {experiment_id}")
            print(f"  DSVR: {experiment_result['evaluation_results'].get('DSVR', 0):.4f}")
            print(f"  CSVR: {experiment_result['evaluation_results'].get('CSVR', 0):.4f}")
            print(f"  ISVR: {experiment_result['evaluation_results'].get('ISVR', 0):.4f}")
            print(f"  耗时: {experiment_result['total_time']:.1f}秒")

            return experiment_result

        except Exception as e:
            print(f"> 实验失败 {feature_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "feature_name": feature_name,
                "experiment_id": experiment_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def run_all_experiments(self, feature_names=None):
        """运行所有特征配置的实验"""
        if feature_names is None:
            feature_names = list(FEATURE_CONFIGS.keys())

        print(f"\n开始运行 {len(feature_names)} 个特征配置实验...")

        for feature_name in tqdm(feature_names, desc="特征比较实验"):
            if feature_name in FEATURE_CONFIGS:
                feature_config = FEATURE_CONFIGS[feature_name]
                result = self.run_single_experiment(feature_name, feature_config)
                self.results.append(result)

                # 每完成一个实验保存一次进度
                self.save_progress()
            else:
                print(f"> 跳过未知的特征配置: {feature_name}")

        # 生成报告
        self.generate_report()

        total_time = time.time() - self.start_time
        print(f"\n{'=' * 80}")
        print(f"特征比较实验完成!")
        print(f"总耗时: {total_time:.2f} 秒 ({total_time / 60:.1f} 分钟)")
        print(f"结果保存到: {self.base_output_dir}")
        print(f"{'=' * 80}")

    def save_progress(self):
        """保存实验进度"""
        progress_file = os.path.join(self.base_output_dir, "experiment_progress.json")
        try:
            with open(progress_file, 'w') as f:
                json.dump({
                    "results": self.results,
                    "total_experiments": len(self.results),
                    "elapsed_time": time.time() - self.start_time
                }, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"> 保存进度失败: {e}")

    def generate_report(self):
        """生成实验报告 - 与论文表2对比"""
        # 过滤成功的实验
        successful_results = [r for r in self.results if "evaluation_results" in r]

        if not successful_results:
            print("> 没有成功的实验结果")
            return

        # 生成文本报告
        report_file = os.path.join(self.base_output_dir, "feature_comparison_report.txt")
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ViSiL 论文特征比较实验报告（表2）\n")
                f.write("比较L2-iMAC4x和L3-iMAC9x\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总实验数: {len(self.results)}\n")
                f.write(f"成功实验数: {len(successful_results)}\n")
                f.write(f"总耗时: {time.time() - self.start_time:.1f}秒\n\n")

                f.write("=" * 80 + "\n")
                f.write("实验结果汇总（论文表2对比）\n")
                f.write("=" * 80 + "\n\n")

                # 论文表2中的结果（来自论文）
                paper_results_table2 = {
                    "L2-iMAC4x": {"DSVR": 0.773, "CSVR": 0.770, "ISVR": 0.670},  # 论文中的值
                    "L3-iMAC9x": {"DSVR": 0.838, "CSVR": 0.832, "ISVR": 0.739}
                }

                f.write(
                    f"{'特征方法':<15} {'DSVR(论文)':<12} {'DSVR(实验)':<12} {'CSVR(论文)':<12} {'CSVR(实验)':<12} {'ISVR(论文)':<12} {'ISVR(实验)':<12}\n")
                f.write("-" * 105 + "\n")

                for result in successful_results:
                    feature_name = result.get('feature_name', '')
                    eval_results = result.get("evaluation_results", {})
                    exp_dsvr = eval_results.get("DSVR", 0)
                    exp_csvr = eval_results.get("CSVR", 0)
                    exp_isvr = eval_results.get("ISVR", 0)

                    paper_result = paper_results_table2.get(feature_name, {})
                    paper_dsvr = paper_result.get("DSVR", 0)
                    paper_csvr = paper_result.get("CSVR", 0)
                    paper_isvr = paper_result.get("ISVR", 0)

                    f.write(f"{feature_name:<15} ")
                    f.write(f"{paper_dsvr:.4f}        {exp_dsvr:.4f}        ")
                    f.write(f"{paper_csvr:.4f}        {exp_csvr:.4f}        ")
                    f.write(f"{paper_isvr:.4f}        {exp_isvr:.4f}\n")

                # 添加性能分析
                f.write("\n" + "=" * 80 + "\n")
                f.write("性能分析\n")
                f.write("=" * 80 + "\n\n")

                for result in successful_results:
                    feature_name = result.get('feature_name', '')
                    eval_results = result.get("evaluation_results", {})
                    exp_dsvr = eval_results.get("DSVR", 0)
                    paper_dsvr = paper_results_table2.get(feature_name, {}).get("DSVR", 0)

                    if paper_dsvr > 0:
                        diff = exp_dsvr - paper_dsvr
                        diff_percent = (diff / paper_dsvr) * 100

                        if abs(diff) < 0.02:
                            status = "✓ 通过"
                        elif diff < -0.05:
                            status = "✗ 失败"
                        else:
                            status = "⚠ 接近"

                        f.write(f"{feature_name}: 差异 = {diff:+.4f} ({diff_percent:+.1f}%) - {status}\n")

            print(f"> 实验报告已生成: {report_file}")

        except Exception as e:
            print(f"> 生成报告失败: {e}")


def run_feature_comparison():
    """运行特征比较实验"""
    import argparse

    parser = argparse.ArgumentParser(description='论文特征比较实验')
    parser.add_argument('--base_output_dir', type=str, default='paper_experiments/feature_comparison',
                        help='基础输出目录')
    parser.add_argument('--features', type=str, nargs='+',
                        default=list(FEATURE_CONFIGS.keys()),
                        help='要比较的特征方法列表')

    args = parser.parse_args()

    print("=" * 60)
    print("论文特征比较实验（表2）")
    print("=" * 60)
    print(f"比较方法: {args.features}")

    experiment = FeatureComparisonExperiment(args.base_output_dir)
    experiment.run_all_experiments(args.features)

    return experiment.results


if __name__ == '__main__':
    run_feature_comparison()