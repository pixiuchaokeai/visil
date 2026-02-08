# -*- coding: utf-8 -*-
"""
run_all_experiments.py
实验运行器（修复subprocess输出问题版本）
"""

import os
import json
import subprocess
import time
import sys
from datetime import datetime
from typing import List, Dict

# 设置系统编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None


class ExperimentRunner:
    """实验运行器（修复版）"""

    def __init__(self):
        self.experiments = [
            {
                "name": "FIVR-5K 评估",
                "script": "evaluation_fivr5k.py",
                "description": "运行完整的FIVR-5K评估",
                "args": "",
                "output_dir": "reproduce_results/fivr5k_results",
            },
            {
                "name": "消融实验分析",
                "script": "ablation_study.py",
                "description": "分析各组件的影响",
                "args": "",
                "output_dir": "reproduce_results/ablation_study",
            },
            {
                "name": "特征提取对比",
                "script": "feature_comparison.py",
                "description": "对比不同特征提取方法",
                "args": "",
                "output_dir": "reproduce_results/feature_comparison",
            },
        ]

        self.log_dir = "experiment_logs"
        os.makedirs(self.log_dir, exist_ok=True)

        self.start_time = datetime.now()

    def run_experiment(self, experiment: Dict) -> Dict:
        """运行单个实验（修复subprocess输出问题）"""
        print(f"\n{'=' * 80}")
        print(f"开始实验: {experiment['name']}")
        print(f"描述: {experiment['description']}")
        print(f"{'=' * 80}")

        start_time = time.time()

        try:
            # 运行脚本 - 直接使用Python执行
            cmd = [sys.executable, experiment['script']]
            if experiment['args']:
                cmd.extend(experiment['args'].split())

            print(f"执行命令: {' '.join(cmd)}")

            # 设置环境变量
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'

            # 运行命令，输出到控制台
            process = subprocess.run(
                cmd,
                env=env,
                encoding='utf-8',
                errors='replace',
                text=True,
                capture_output=False,  # 不捕获输出，直接显示
                check=False  # 不检查返回码
            )

            # 获取返回码
            returncode = process.returncode
            success = returncode == 0

        except Exception as e:
            print(f"运行异常: {e}")
            success = False
            returncode = 1

        end_time = time.time()
        duration = end_time - start_time

        # 更新实验信息
        experiment.update({
            "success": success,
            "start_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            "end_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
            "duration": f"{duration:.2f}秒",
            "returncode": returncode,
        })

        # 状态指示符
        status_indicator = "[成功]" if success else "[失败]"
        print(f"\n实验完成: {status_indicator}")
        print(f"耗时: {duration:.2f}秒")

        return experiment

    def run_all(self):
        """运行所有实验"""
        print("\n" + "=" * 80)
        print("ViSiL 论文复现实验套件")
        print("=" * 80)
        print(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"实验数量: {len(self.experiments)}")
        print("=" * 80)

        # 运行所有实验
        for exp in self.experiments:
            self.run_experiment(exp)

        # 生成总结报告
        self.generate_summary()

    def generate_summary(self):
        """生成总结报告"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        # 统计
        successful = sum(1 for exp in self.experiments if exp.get("success", False))
        failed = len(self.experiments) - successful

        print("\n" + "=" * 80)
        print("实验总结报告")
        print("=" * 80)

        print(f"总实验数: {len(self.experiments)}")
        print(f"成功: {successful}")
        print(f"失败: {failed}")
        print(f"总耗时: {total_duration:.2f}秒 ({total_duration / 60:.1f}分钟)")
        print(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n" + "-" * 80)
        print("各实验详情:")
        print("-" * 80)

        for i, exp in enumerate(self.experiments, 1):
            status = "成功" if exp.get("success", False) else "失败"
            print(f"{i}. {exp['name']}: {status} ({exp.get('duration', 'N/A')})")

        # 保存日志
        log_file = os.path.join(
            self.log_dir,
            f"experiment_log_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        )

        log_data = {
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration": total_duration,
            "experiments": self.experiments,
            "summary": {
                "total": len(self.experiments),
                "successful": successful,
                "failed": failed,
            }
        }

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"\n详细日志已保存: {log_file}")

        # 生成文本报告
        report_file = os.path.join(self.log_dir, "experiment_summary.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ViSiL 论文复现实验总结报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"生成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总耗时: {total_duration:.2f}秒\n\n")

            f.write("实验统计:\n")
            f.write(f"  总实验数: {len(self.experiments)}\n")
            f.write(f"  成功: {successful}\n")
            f.write(f"  失败: {failed}\n\n")

            f.write("各实验结果:\n")
            for exp in self.experiments:
                status = "成功" if exp.get("success", False) else "失败"
                f.write(f"  - {exp['name']}: {status} ({exp.get('duration', 'N/A')})\n")

        print(f"总结报告已保存: {report_file}")


def main():
    """主函数"""
    runner = ExperimentRunner()
    runner.run_all()


if __name__ == '__main__':
    main()