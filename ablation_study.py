# ablation_study.py - 消融实验（修复参数传递）
"""
ViSiL模型消融实验 - 修复参数传递
修复点：确保使用正确的参数名称
修复点：按照论文顺序运行实验
"""

import os
import json
import time
import random
from datetime import datetime
from tqdm import tqdm

from config import ABLATION_CONFIGS, EXPERIMENT_PHASES, EvaluationConfig
from evaluation_fivr5k import FIVR5KEvaluator


class AblationStudyExperiment:
    """消融实验 - 修复参数传递版本"""

    def __init__(self, base_output_dir="paper_experiments/ablation_study"):
        # ==================== 修复点1：使用base_output_dir ====================
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)

        # 实验记录
        self.results = []
        self.start_time = time.time()

        # 论文结果参考
        self.paper_results = {
            "visil_v": {"DSVR": 0.882, "CSVR": 0.869, "ISVR": 0.777},
            "visil_f_w_a": {"DSVR": 0.869, "CSVR": 0.852, "ISVR": 0.755},
            "visil_f_w": {"DSVR": 0.851, "CSVR": 0.833, "ISVR": 0.738},
            "visil_f": {"DSVR": 0.838, "CSVR": 0.832, "ISVR": 0.739},
            "visil_sym": {"DSVR": 0.882, "CSVR": 0.869, "ISVR": 0.777},
        }

        print("=" * 80)
        print("ViSiL 消融实验（修复参数传递）")
        print("=" * 80)
        print(f"基础输出目录: {base_output_dir}")
        print(f"消融配置数: {len(ABLATION_CONFIGS)}")
        print(f"实验顺序: {EXPERIMENT_PHASES}")

    def generate_experiment_id(self, config_name):
        """生成唯一的实验ID"""
        timestamp = str(int(time.time()))[-6:]
        random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
        return f"{config_name}_{timestamp}_{random_suffix}"

    def check_baseline_performance(self, baseline_result):
        """检查baseline性能是否达到论文要求"""
        if baseline_result is None:
            return False, "没有baseline结果"

        exp_dsvr = baseline_result.get("evaluation_results", {}).get("DSVR", 0)
        paper_dsvr = self.paper_results.get("visil_v", {}).get("DSVR", 0.882)

        diff = abs(exp_dsvr - paper_dsvr)

        if diff < 0.02:  # 差异小于2%
            return True, f"通过 (实验: {exp_dsvr:.4f}, 论文: {paper_dsvr:.4f}, 差异: {diff:.4f})"
        elif diff < 0.05:  # 差异小于5%
            return True, f"接近 (实验: {exp_dsvr:.4f}, 论文: {paper_dsvr:.4f}, 差异: {diff:.4f})"
        else:
            return False, f"未达标 (实验: {exp_dsvr:.4f}, 论文: {paper_dsvr:.4f}, 差异: {diff:.4f})"

    def run_single_experiment(self, config_name: str, feature_config) -> dict:
        """运行单个消融配置的实验"""
        print(f"\n{'=' * 60}")
        print(f"消融实验: {config_name}")
        print(f"{'=' * 60}")

        # ==================== 修复点2：为每个实验生成唯一ID ====================
        experiment_id = self.generate_experiment_id(config_name)

        # ==================== 修复点3：根据实验阶段设置清理策略 ====================
        # 只有第一个实验清理缓存，后续实验复用缓存但不清理
        clear_cache = False
        if config_name == "visil_v":
            clear_cache = True
            print(f"> 清理缓存: 是 (baseline实验，清理特征缓存)")
        else:
            print(f"> 清理缓存: 否 (复用缓存)")

        # 创建配置
        config = EvaluationConfig(
            base_output_dir="output",  # 使用统一的output目录
            experiment_id=experiment_id,
            feature_config=feature_config,
            clear_cache=clear_cache
        )

        try:
            # 运行评估
            evaluator = FIVR5KEvaluator(config)
            result = evaluator.evaluate()

            # 记录结果
            experiment_result = {
                "config_name": config_name,
                "experiment_id": experiment_id,
                "feature_config": config.feature_config.to_dict(),
                "evaluation_results": result["evaluation"],
                "total_time": result["total_time"],
                "output_dir": result["output_dir"],
                "timestamp": datetime.now().isoformat()
            }

            print(f"> 实验完成: {config_name}")
            print(f"  实验ID: {experiment_id}")
            print(f"  DSVR: {experiment_result['evaluation_results'].get('DSVR', 0):.4f}")

            # 打印消融配置
            feature_cfg = config.feature_config
            print(f"  配置详情:")
            print(f"    注意力: {feature_cfg.use_attention}")
            print(f"    视频比较器: {feature_cfg.use_video_comparator}")
            print(f"    PCA: {feature_cfg.use_pca}")
            print(f"    对称: {feature_cfg.symmetric}")

            return experiment_result

        except Exception as e:
            print(f"> 实验失败 {config_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "config_name": config_name,
                "experiment_id": experiment_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def run_phase_experiments(self, phase_name, config_names):
        """运行指定阶段的实验"""
        print(f"\n{'=' * 80}")
        print(f"阶段: {phase_name}")
        print(f"配置: {config_names}")
        print(f"{'=' * 80}")

        phase_results = []

        for config_name in tqdm(config_names, desc=f"阶段 {phase_name}"):
            if config_name in ABLATION_CONFIGS:
                feature_config = ABLATION_CONFIGS[config_name]
                result = self.run_single_experiment(config_name, feature_config)
                phase_results.append(result)
                self.results.append(result)

                # 每完成一个实验保存一次进度
                self.save_progress()
            else:
                print(f"> 跳过未知的消融配置: {config_name}")

        return phase_results

    def run_all_experiments(self, phases=None, num_workers=1, feature_batch_size=16,
                            cpu_only=False, auto_continue=False):
        """按阶段运行所有消融实验"""
        if phases is None:
            phases = EXPERIMENT_PHASES

        print(f"\n开始按论文顺序进行消融实验...")
        print(f"并行数: {num_workers}, 特征批次大小: {feature_batch_size}, CPU模式: {cpu_only}")
        print(f"自动继续模式: {'开启' if auto_continue else '关闭'}")

        # 阶段1：ViSiL_v（baseline）
        phase1_results = self.run_phase_experiments(
            "阶段1: 验证ViSiL_v", phases.get("phase1", []),
            num_workers=num_workers, feature_batch_size=feature_batch_size,
            cpu_only=cpu_only, clear_cache=True
        )

        # 检查baseline性能
        baseline_result = next((r for r in phase1_results if r.get("config_name") == "visil_v"), None)
        if baseline_result and "evaluation_results" in baseline_result:
            exp_dsvr = baseline_result["evaluation_results"].get("DSVR", 0)
            paper_dsvr = self.paper_results["visil_v"]["DSVR"]
            diff = abs(exp_dsvr - paper_dsvr)
            print(f"\n> Baseline检查: 实验={exp_dsvr:.4f}, 论文={paper_dsvr:.4f}, 差异={diff:.4f}")

            if diff >= 0.05:
                print("> 警告: Baseline性能与论文差距较大")
                if not auto_continue:
                    response = input("> 是否继续后续实验？(y/n): ").strip().lower()
                    if response != 'y':
                        print("> 用户终止实验")
                        return
                else:
                    print("> 自动继续模式开启，继续执行...")
        else:
            print("> 警告: 未获取到有效的baseline结果")
            if not auto_continue:
                response = input("> 是否继续？(y/n): ").strip().lower()
                if response != 'y':
                    print("> 用户终止实验")
                    return
            else:
                print("> 自动继续模式开启，继续执行...")

        # 阶段2：消融实验
        phase2_results = self.run_phase_experiments(
            "阶段2: 消融实验", phases.get("phase2", []),
            num_workers=num_workers, feature_batch_size=feature_batch_size,
            cpu_only=cpu_only, clear_cache=False
        )
        # 生成报告
        self.generate_report()
        total_time = time.time() - self.start_time
        print(f"\n{'=' * 80}")
        print(f"消融实验完成!")
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
        """生成消融实验报告"""
        # 过滤成功的实验
        successful_results = [r for r in self.results if "evaluation_results" in r]

        if not successful_results:
            print("> 没有成功的实验结果")
            return

        # 生成文本报告
        report_file = os.path.join(self.base_output_dir, "ablation_study_report.txt")
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ViSiL 消融实验报告\n")
                f.write("修复参数传递版本\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总实验数: {len(self.results)}\n")
                f.write(f"成功实验数: {len(successful_results)}\n")
                f.write(f"总耗时: {time.time() - self.start_time:.1f}秒\n\n")

                # 按实验阶段组织结果
                phases = EXPERIMENT_PHASES

                # ==================== 阶段1结果 ====================
                f.write("=" * 80 + "\n")
                f.write("阶段1: ViSiL_v (Baseline)\n")
                f.write("=" * 80 + "\n\n")

                phase1_results = [r for r in successful_results if r.get("config_name") in phases["phase1"]]
                for result in phase1_results:
                    eval_results = result.get("evaluation_results", {})
                    f.write(f"{result.get('config_name', 'Unknown')} (ID: {result.get('experiment_id', 'N/A')}):\n")
                    f.write(
                        f"  DSVR: {eval_results.get('DSVR', 0):.4f} (论文: {self.paper_results.get(result.get('config_name'), {}).get('DSVR', 0):.4f})\n")
                    f.write(
                        f"  CSVR: {eval_results.get('CSVR', 0):.4f} (论文: {self.paper_results.get(result.get('config_name'), {}).get('CSVR', 0):.4f})\n")
                    f.write(
                        f"  ISVR: {eval_results.get('ISVR', 0):.4f} (论文: {self.paper_results.get(result.get('config_name'), {}).get('ISVR', 0):.4f})\n\n")

                # ==================== 阶段2结果 ====================
                f.write("\n" + "=" * 80 + "\n")
                f.write("阶段2: 消融实验\n")
                f.write("=" * 80 + "\n\n")

                phase2_results = [r for r in successful_results if r.get("config_name") in phases["phase2"]]

                # 获取baseline结果
                baseline = next((r for r in phase1_results if r.get("config_name") == "visil_v"), None)
                baseline_dsvr = baseline.get("evaluation_results", {}).get("DSVR", 0) if baseline else 0

                f.write(f"Baseline DSVR: {baseline_dsvr:.4f}\n\n")
                f.write(
                    f"{'配置':<15} {'实验DSVR':<12} {'论文DSVR':<12} {'差异':<12} {'PCA':<8} {'注意力':<10} {'比较器':<10}\n")
                f.write("-" * 80 + "\n")

                for result in phase2_results:
                    config_name = result.get("config_name", "")
                    eval_results = result.get("evaluation_results", {})
                    exp_dsvr = eval_results.get("DSVR", 0)
                    paper_dsvr = self.paper_results.get(config_name, {}).get("DSVR", 0)
                    feature_cfg = result.get("feature_config", {})

                    # 计算相对于baseline的变化
                    change = exp_dsvr - baseline_dsvr
                    change_percent = (change / baseline_dsvr) * 100 if baseline_dsvr > 0 else 0

                    f.write(
                        f"{config_name:<15} {exp_dsvr:.4f}        {paper_dsvr:.4f}        {change:+.4f} ({change_percent:+.1f}%)  ")
                    f.write(f"{feature_cfg.get('use_pca', False)!s:<8} ")
                    f.write(f"{feature_cfg.get('use_attention', False)!s:<10} ")
                    f.write(f"{feature_cfg.get('use_video_comparator', False)!s:<10}\n")

                # ==================== 阶段3结果 ====================
                f.write("\n" + "=" * 80 + "\n")
                f.write("阶段3: 对称版本\n")
                f.write("=" * 80 + "\n\n")

                phase3_results = [r for r in successful_results if r.get("config_name") in phases["phase3"]]
                for result in phase3_results:
                    eval_results = result.get("evaluation_results", {})
                    config_name = result.get("config_name", "")
                    paper_result = self.paper_results.get(config_name, {})

                    f.write(f"{config_name} (ID: {result.get('experiment_id', 'N/A')}):\n")
                    f.write(
                        f"  实验DSVR: {eval_results.get('DSVR', 0):.4f} (论文: {paper_result.get('DSVR', 0):.4f})\n")
                    f.write(
                        f"  实验CSVR: {eval_results.get('CSVR', 0):.4f} (论文: {paper_result.get('CSVR', 0):.4f})\n")
                    f.write(
                        f"  实验ISVR: {eval_results.get('ISVR', 0):.4f} (论文: {paper_result.get('ISVR', 0):.4f})\n\n")

            print(f"> 消融实验报告已生成: {report_file}")

        except Exception as e:
            print(f"> 生成报告失败: {e}")


def run_ablation_study():
    """运行消融实验"""
    import argparse

    parser = argparse.ArgumentParser(description='消融实验')
    parser.add_argument('--base_output_dir', type=str, default='paper_experiments/ablation_study',
                        help='基础输出目录')
    parser.add_argument('--phase', type=str, choices=['all', 'phase1', 'phase2', 'phase3'],
                        default='all', help='运行阶段')
    parser.add_argument('--auto_continue', action='store_true', help='自动继续实验，无需人工确认')

    args = parser.parse_args()

    print("=" * 60)
    print("ViSiL 消融实验（修复参数传递版本）")
    print("=" * 60)

    experiment = AblationStudyExperiment(args.base_output_dir)

    if args.phase == 'all':
        experiment.run_all_experiments()
    else:
        # 只运行指定阶段
        phases = {
            "phase1": EXPERIMENT_PHASES["phase1"],
            "phase2": EXPERIMENT_PHASES["phase2"],
            "phase3": EXPERIMENT_PHASES["phase3"]
        }
        experiment.run_phase_experiments(args.phase, phases.get(args.phase, []))

    return experiment.results


if __name__ == '__main__':
    run_ablation_study()