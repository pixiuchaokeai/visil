# run_all_experiments.py - 运行所有论文实验（修复参数传递问题）
"""
运行所有论文实验
1. 特征比较实验（论文表2）
2. 消融实验（按论文顺序）
修复点：修复参数传递错误
修复点：统一输出目录结构
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import traceback

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

print(f"> Python路径: {sys.path}")
print(f"> 当前目录: {current_dir}")

# ==================== 修复点1：确保正确导入模块 ====================
try:
    from config import FEATURE_CONFIGS, ABLATION_CONFIGS, EXPERIMENT_PHASES, EvaluationConfig

    print("> 成功导入 config 模块")
except ImportError as e:
    print(f"> 导入 config 失败: {e}")
    traceback.print_exc()

try:
    from evaluation_fivr5k import FIVR5KEvaluator

    print("> 成功导入 evaluation_fivr5k")
except ImportError as e:
    print(f"> 导入 evaluation_fivr5k 失败: {e}")
    traceback.print_exc()

try:
    from feature_comparison import FeatureComparisonExperiment

    print("> 成功导入 feature_comparison")
except ImportError as e:
    print(f"> 导入 feature_comparison 失败: {e}")
    traceback.print_exc()

try:
    from ablation_study import AblationStudyExperiment

    print("> 成功导入 ablation_study")
except ImportError as e:
    print(f"> 导入 ablation_study 失败: {e}")
    traceback.print_exc()


class ExperimentRunner:
    """实验运行器 - 修复参数传递版本"""

    def __init__(self, output_base="paper_experiments"):
        # ==================== 修复点2：统一输出目录结构 ====================
        self.output_base = output_base
        self.start_time = time.time()
        self.experiment_log = []

        # 创建输出目录
        os.makedirs(output_base, exist_ok=True)

        # 日志目录
        self.log_dir = os.path.join(output_base, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # 日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"experiment_log_{timestamp}.json")

        print("\n" + "=" * 80)
        print("ViSiL 论文实验套件 - 修复版本")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输出基础目录: {output_base}")
        print(f"日志目录: {self.log_dir}")

    def log_experiment(self, name: str, status: str, results=None, error=None, duration=None):
        """记录实验日志"""
        log_entry = {
            "name": name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "duration": duration or 0.0
        }

        if results:
            # 确保结果可以被JSON序列化
            if isinstance(results, dict):
                log_entry["results"] = results
            else:
                log_entry["results"] = str(results)[:500]

        if error:
            log_entry["error"] = str(error)[:500]

        self.experiment_log.append(log_entry)
        self.save_log()

    def save_log(self):
        """保存日志文件"""
        log_data = {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "total_duration": time.time() - self.start_time,
            "experiments": self.experiment_log
        }

        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"> 保存日志失败: {e}")

    def run_feature_comparison(self, **kwargs):
        """运行特征比较实验（论文表2）"""
        print("\n" + "=" * 80)
        print("实验1: 论文表2 - 特征比较 (L2-iMAC4x vs L3-iMAC9x)")
        print("=" * 80)

        experiment_start = time.time()

        try:
            # 获取参数
            base_output_dir = kwargs.get('base_output_dir', os.path.join(self.output_base, "feature_comparison"))
            features = kwargs.get('features', list(FEATURE_CONFIGS.keys()))

            print(f"> 特征比较实验:")
            print(f"  基础输出目录: {base_output_dir}")
            print(f"  比较方法: {len(features)} 种")
            print(f"  方法列表: {', '.join(features)}")

            # 创建实验对象
            experiment = FeatureComparisonExperiment(base_output_dir)

            # 运行实验
            experiment.run_all_experiments(features)

            duration = time.time() - experiment_start
            self.log_experiment(
                "Feature Comparison (Table 2)",
                "success",
                {"base_output_dir": base_output_dir, "features": features},
                duration=duration
            )

            print(f"\n> 特征比较实验完成!")
            print(f"> 耗时: {duration:.1f}秒")
            print(f"> 结果目录: {base_output_dir}")

            return experiment.results

        except Exception as e:
            duration = time.time() - experiment_start
            error_msg = str(e)
            self.log_experiment(
                "Feature Comparison (Table 2)",
                "failed",
                error=error_msg,
                duration=duration
            )
            print(f"> 特征比较实验失败: {error_msg}")
            traceback.print_exc()
            return None

    def run_ablation_study(self, **kwargs):
        print("\n" + "=" * 80)
        print("实验2: 消融实验")
        print("=" * 80)

        start = time.time()
        try:
            base_output_dir = kwargs.get('base_output_dir', os.path.join(self.output_base, "ablation_study"))
            num_workers = kwargs.get('num_workers', 1)
            feature_batch_size = kwargs.get('feature_batch_size', 16)
            cpu_only = kwargs.get('cpu_only', False)
            # 自动继续标志，避免阻塞
            auto_continue = kwargs.get('auto_continue', True)

            print(f"> 并行数: {num_workers}, 特征批次大小: {feature_batch_size}, CPU模式: {cpu_only}")
            print(f"> 自动继续模式: 开启")

            experiment = AblationStudyExperiment(base_output_dir)
            experiment.run_all_experiments(
                num_workers=num_workers,
                feature_batch_size=feature_batch_size,
                cpu_only=cpu_only,
                auto_continue=auto_continue  # 传递自动继续标志
            )

            duration = time.time() - start
            self.log_experiment("Ablation Study", "success", {"base_output_dir": base_output_dir}, duration=duration)
            return experiment.results
        except Exception as e:
            duration = time.time() - start
            self.log_experiment("Ablation Study", "failed", error=str(e), duration=duration)
            traceback.print_exc()
            return None


    def run_all_experiments(self, experiments_config=None):
        """运行所有论文实验"""
        print("\n" + "=" * 80)
        print("开始运行所有论文实验")
        print("=" * 80)

        if experiments_config is None:
            experiments_config = {
                "feature_comparison": {
                    "base_output_dir": os.path.join(self.output_base, "feature_comparison"),
                    "features": list(FEATURE_CONFIGS.keys())
                },
                "ablation_study": {
                    "base_output_dir": os.path.join(self.output_base, "ablation_study")
                }
            }

        all_results = {}

        try:
            # ==================== 步骤1：特征比较实验（论文表2） ====================
            print(f"\n{'=' * 60}")
            print("步骤1: 论文表2 - 特征比较 (L2-iMAC4x vs L3-iMAC9x)")
            print(f"{'=' * 60}")
            all_results["feature_comparison"] = self.run_feature_comparison(
                **experiments_config.get("feature_comparison", {}))

            # ==================== 步骤2：消融实验 ====================
            print(f"\n{'=' * 60}")
            print("步骤2: 消融实验 (按论文顺序)")
            print(f"{'=' * 60}")
            all_results["ablation_study"] = self.run_ablation_study(**experiments_config.get("ablation_study", {}))

        except KeyboardInterrupt:
            print("\n> 实验被用户中断")
        except Exception as e:
            print(f"\n> 实验过程中出现错误: {e}")
            traceback.print_exc()

        # 生成总结报告
        self.generate_summary(all_results)

        total_time = time.time() - self.start_time
        print("\n" + "=" * 80)
        print("所有论文实验完成!")
        print(f"总耗时: {total_time:.2f} 秒 ({total_time / 60:.1f} 分钟)")
        print(f"日志文件: {self.log_file}")
        print("=" * 80)

        return all_results

    def generate_summary(self, results):
        """生成实验总结报告"""
        summary_file = os.path.join(self.output_base, "paper_experiments_summary.txt")

        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ViSiL 论文实验总结报告\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总耗时: {time.time() - self.start_time:.1f}秒\n\n")

                f.write("实验概览:\n")
                f.write("-" * 60 + "\n")

                success_count = 0
                fail_count = 0

                for exp in self.experiment_log:
                    status_icon = "✓" if exp["status"] == "success" else "✗"
                    exp_name = exp.get("name", "未知实验")
                    exp_duration = exp.get("duration", 0.0)
                    f.write(f"  {status_icon} {exp_name:<35} {exp_duration:.1f}秒\n")

                    if exp["status"] == "success":
                        success_count += 1
                    else:
                        fail_count += 1

                f.write(f"\n总计: {success_count} 成功, {fail_count} 失败\n\n")

                # 特征比较结果
                feature_results = results.get("feature_comparison", [])
                if feature_results and isinstance(feature_results, list):
                    successful_features = [r for r in feature_results if "evaluation_results" in r]
                    if successful_features:
                        f.write("论文表2结果 (特征比较):\n")
                        f.write("-" * 50 + "\n")
                        f.write(f"{'特征方法':<15} {'DSVR':<10} {'CSVR':<10} {'ISVR':<10}\n")
                        f.write("-" * 50 + "\n")

                        for result in successful_features:
                            feature_name = result.get('feature_name', 'Unknown')
                            eval_results = result.get('evaluation_results', {})
                            f.write(f"{feature_name:<15} ")
                            f.write(f"{eval_results.get('DSVR', 0):.4f}    ")
                            f.write(f"{eval_results.get('CSVR', 0):.4f}    ")
                            f.write(f"{eval_results.get('ISVR', 0):.4f}\n")

                # 消融实验结果
                ablation_results = results.get("ablation_study", [])
                if ablation_results and isinstance(ablation_results, list):
                    successful_ablation = [r for r in ablation_results if "evaluation_results" in r]
                    if successful_ablation:
                        f.write("\n消融实验结果:\n")
                        f.write("-" * 50 + "\n")
                        f.write(f"{'配置':<15} {'DSVR':<10} {'PCA':<8} {'注意力':<8} {'比较器':<8} {'对称':<8}\n")
                        f.write("-" * 50 + "\n")

                        # 按配置名称排序
                        config_order = ["visil_v", "visil_f_w_a", "visil_f_w", "visil_f", "visil_sym"]
                        sorted_results = sorted(
                            successful_ablation,
                            key=lambda x: config_order.index(x.get("config_name", ""))
                            if x.get("config_name") in config_order else 100
                        )

                        for result in sorted_results:
                            config_name = result.get('config_name', 'Unknown')
                            eval_results = result.get('evaluation_results', {})
                            feature_cfg = result.get('feature_config', {})

                            f.write(f"{config_name:<15} ")
                            f.write(f"{eval_results.get('DSVR', 0):.4f}    ")
                            f.write(f"{feature_cfg.get('use_pca', False)!s:<8} ")
                            f.write(f"{feature_cfg.get('use_attention', False)!s:<8} ")
                            f.write(f"{feature_cfg.get('use_video_comparator', False)!s:<8} ")
                            f.write(f"{feature_cfg.get('symmetric', False)!s:<8}\n")

            print(f"> 实验总结已生成: {summary_file}")

        except Exception as e:
            print(f"> 生成总结报告失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='运行ViSiL论文实验')
    parser.add_argument('--output_base', type=str, default='paper_experiments',
                        help='基础输出目录')
    parser.add_argument('--experiments', type=str, nargs='+',
                        choices=['all', 'feature_comparison', 'ablation_study'],
                        default=['all'],
                        help='要运行的实验列表')

    args = parser.parse_args()

    print("=" * 80)
    print("ViSiL 论文实验运行器")
    print("=" * 80)
    print(f"参数:")
    print(f"  基础输出目录: {args.output_base}")
    print(f"  实验列表: {args.experiments}")
    print("=" * 80)

    runner = ExperimentRunner(args.output_base)

    try:
        if 'all' in args.experiments:
            runner.run_all_experiments()
        else:
            if 'feature_comparison' in args.experiments:
                runner.run_feature_comparison(
                    base_output_dir=os.path.join(args.output_base, "feature_comparison")
                )

            if 'ablation_study' in args.experiments:
                runner.run_ablation_study(
                    base_output_dir=os.path.join(args.output_base, "ablation_study")
                )

        print("\n" + "=" * 80)
        print("实验运行完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n> 运行实验时发生错误: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)