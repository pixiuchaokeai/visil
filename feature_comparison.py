# -*- coding: utf-8 -*-
"""
feature_comparison.py
特征提取方法对比脚本（修复编码问题版本）
对应论文表2
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List

from config import experiment_config


class FeatureComparison:
    """特征提取对比类（修复编码问题版本）"""

    def __init__(self):
        self.config = experiment_config
        self.results_dir = os.path.join(self.config.output_dir, "feature_comparison")
        os.makedirs(self.results_dir, exist_ok=True)

        # 论文结果
        self.paper_results = self.config.paper_results

        # 您的当前结果
        self.your_results = self.load_your_results()

        # 特征提取方法
        self.methods = [
            {
                "name": "MAC",
                "desc": "Maximum Activations of Convolutions [33]",
                "dims": 2048,
            },
            {
                "name": "SPoC",
                "desc": "Sum-Pooled Convolutional features [1]",
                "dims": 2048,
            },
            {
                "name": "R-MAC",
                "desc": "Regional Maximum Activation of Convolutions [33]",
                "dims": 2048,
            },
            {
                "name": "GeM",
                "desc": "Generalized Mean pooling [27]",
                "dims": 2048,
            },
            {
                "name": "iMAC",
                "desc": "intermediate Maximum Activation of Convolutions [20]",
                "dims": 3840,
            },
            {
                "name": "L2-iMAC4x",
                "desc": "Level 2, 4 regions (3840 dim)",
                "dims": 3840,
            },
            {
                "name": "L3-iMAC9x",
                "desc": "Level 3, 9 regions (3840 dim) - Proposed",
                "dims": 3840,
            },
        ]

    def load_your_results(self) -> Dict:
        """加载您的结果"""
        # 尝试加载您的FIVR-5K结果
        result_file = os.path.join(self.config.output_dir, "fivr5k_results", "task_results_ViSiLv.json")

        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass

        # 使用您提供的结果
        return {
            "DSVR": 0.6247751074765724,
            "CSVR": 0.6416741577861441,
            "ISVR": 0.5930094676455455
        }

    def run_comparison(self):
        """运行特征提取对比"""
        print("\n" + "="*80)
        print("特征提取方法对比 (对应论文表2)")
        print("="*80)

        # 1. 展示论文结果
        self.display_paper_results()

        # 2. 与您的结果对比
        self.compare_with_your_results()

        # 3. 分析不同维度的效果
        self.analyze_dimension_impact()

        # 4. 生成报告
        self.generate_report()

        print("\n" + "="*80)
        print("特征提取对比完成!")
        print("="*80)

    def display_paper_results(self):
        """展示论文结果"""
        print("\n" + "="*50)
        print("1. 论文特征提取结果 (表2)")
        print("="*50)

        print(f"{'方法':<15} {'维度':<8} {'DSVR':<10} {'CSVR':<10} {'ISVR':<10} {'描述':<40}")
        print("-" * 93)

        for method_info in self.methods:
            name = method_info["name"]
            if name in self.paper_results["table2"]:
                results = self.paper_results["table2"][name]
                print(f"{name:<15} {method_info['dims']:<8} "
                      f"{results['DSVR']:<10.3f} {results['CSVR']:<10.3f} "
                      f"{results['ISVR']:<10.3f} {method_info['desc']}")

    def compare_with_your_results(self):
        """与您的结果对比"""
        print("\n" + "="*50)
        print("2. 与您的结果对比")
        print("="*50)

        paper_best = self.paper_results["table2"]["L3-iMAC9x"]

        print(f"{'任务':<8} {'论文(L3-iMAC9x)':<15} {'您的结果':<12} {'差异':<12} {'相对差距':<12}")
        print("-" * 60)

        for task in ["DSVR", "CSVR", "ISVR"]:
            paper_score = paper_best[task]
            your_score = self.your_results.get(task, 0.0)
            diff = your_score - paper_score
            rel_diff = diff / paper_score * 100 if paper_score > 0 else 0

            print(f"{task:<8} {paper_score:<15.4f} {your_score:<12.4f} "
                  f"{diff:+.4f} ({rel_diff:+.1f}%)")

        # 性能差距分析
        print("\n" + "="*50)
        print("3. 性能差距分析")
        print("="*50)

        gaps = {
            task: paper_best[task] - self.your_results.get(task, 0.0)
            for task in ["DSVR", "CSVR", "ISVR"]
        }

        avg_gap = np.mean(list(gaps.values()))

        print(f"平均绝对差距: {avg_gap:.4f}")
        print(f"DSVR差距: {gaps['DSVR']:.4f}")
        print(f"CSVR差距: {gaps['CSVR']:.4f}")
        print(f"ISVR差距: {gaps['ISVR']:.4f}")

        # 差距等级
        if avg_gap > 0.2:
            print("\n[严重] 差距较大 - 需要检查特征提取配置")
        elif avg_gap > 0.1:
            print("\n[中等] 中等差距 - 建议优化特征提取")
        elif avg_gap > 0.05:
            print("\n[良好] 差距较小 - 接近最佳特征")
        else:
            print("\n[优秀] 优秀 - 达到或超过论文最佳特征")

    def analyze_dimension_impact(self):
        """分析不同维度的效果"""
        print("\n" + "="*50)
        print("4. 特征维度影响分析")
        print("="*50)

        # 按维度分组
        dim_groups = {}
        for method_info in self.methods:
            dim = method_info["dims"]
            name = method_info["name"]

            if name in self.paper_results["table2"]:
                if dim not in dim_groups:
                    dim_groups[dim] = []

                results = self.paper_results["table2"][name]
                dim_groups[dim].append({
                    "name": name,
                    "DSVR": results["DSVR"],
                    "CSVR": results["CSVR"],
                    "ISVR": results["ISVR"],
                })

        # 分析每个维度的最佳性能
        print(f"{'维度':<8} {'方法数':<8} {'最佳DSVR':<12} {'最佳方法':<15}")
        print("-" * 45)

        for dim in sorted(dim_groups.keys()):
            methods = dim_groups[dim]
            best_method = max(methods, key=lambda x: x["DSVR"])

            print(f"{dim:<8} {len(methods):<8} {best_method['DSVR']:<12.3f} {best_method['name']:<15}")

        # 维度与性能关系
        print("\n" + "="*50)
        print("5. 维度与性能关系总结")
        print("="*50)

        print("2048维方法:")
        for method in self.methods:
            if method["dims"] == 2048 and method["name"] in self.paper_results["table2"]:
                results = self.paper_results["table2"][method["name"]]
                print(f"  {method['name']}: DSVR={results['DSVR']:.3f}")

        print("\n3840维方法:")
        for method in self.methods:
            if method["dims"] == 3840 and method["name"] in self.paper_results["table2"]:
                results = self.paper_results["table2"][method["name"]]
                print(f"  {method['name']}: DSVR={results['DSVR']:.3f}")

        # L3-iMAC9x的优势
        print(f"\nL3-iMAC9x (最佳方法) 优势:")
        baseline = self.paper_results["table2"]["MAC"]["DSVR"]
        best = self.paper_results["table2"]["L3-iMAC9x"]["DSVR"]
        improvement = best - baseline
        relative = improvement / baseline * 100

        print(f"  相比MAC提升: {improvement:.3f} ({relative:.1f}%)")
        print(f"  相比R-MAC提升: {best - self.paper_results['table2']['R-MAC']['DSVR']:.3f}")
        print(f"  相比iMAC提升: {best - self.paper_results['table2']['iMAC']['DSVR']:.3f}")

    def generate_report(self):
        """生成报告"""
        report_file = os.path.join(self.results_dir, "feature_comparison_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("特征提取方法对比报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 1. 论文结果
            f.write("一、论文特征提取结果 (表2):\n")
            f.write("-" * 60 + "\n")
            for method_info in self.methods:
                name = method_info["name"]
                if name in self.paper_results["table2"]:
                    results = self.paper_results["table2"][name]
                    f.write(f"{name:<15} (dim={method_info['dims']}): "
                           f"DSVR={results['DSVR']:.3f}, "
                           f"CSVR={results['CSVR']:.3f}, "
                           f"ISVR={results['ISVR']:.3f}\n")

            # 2. 与您的结果对比
            f.write("\n二、与您的结果对比:\n")
            f.write("-" * 60 + "\n")
            paper_best = self.paper_results["table2"]["L3-iMAC9x"]
            for task in ["DSVR", "CSVR", "ISVR"]:
                diff = self.your_results.get(task, 0.0) - paper_best[task]
                f.write(f"{task}: 论文={paper_best[task]:.3f}, "
                       f"您的={self.your_results.get(task, 0.0):.4f}, "
                       f"差异={diff:+.4f}\n")

            # 3. 维度分析
            f.write("\n三、特征维度分析:\n")
            f.write("-" * 60 + "\n")

            # 按维度分组
            dim_results = {}
            for method_info in self.methods:
                name = method_info["name"]
                dim = method_info["dims"]

                if name in self.paper_results["table2"]:
                    if dim not in dim_results:
                        dim_results[dim] = []

                    dim_results[dim].append({
                        "name": name,
                        "DSVR": self.paper_results["table2"][name]["DSVR"]
                    })

            for dim in sorted(dim_results.keys()):
                f.write(f"\n{dim}维方法:\n")
                for method in dim_results[dim]:
                    f.write(f"  {method['name']}: DSVR={method['DSVR']:.3f}\n")

            # 4. 总结
            f.write("\n四、总结:\n")
            f.write("-" * 60 + "\n")

            avg_gap = np.mean([
                paper_best[task] - self.your_results.get(task, 0.0)
                for task in ["DSVR", "CSVR", "ISVR"]
            ])

            if avg_gap > 0.2:
                f.write("结论: [严重] 特征提取可能存在较大问题\n")
                f.write("建议:\n")
                f.write("  1. 检查是否使用L3-iMAC9x特征\n")
                f.write("  2. 验证特征维度是否为3840\n")
                f.write("  3. 检查区域划分是否正确 (9 regions)\n")
            elif avg_gap > 0.1:
                f.write("结论: [中等] 特征提取有改进空间\n")
                f.write("建议:\n")
                f.write("  1. 尝试使用PCA降维\n")
                f.write("  2. 检查特征归一化\n")
                f.write("  3. 验证特征提取网络\n")
            elif avg_gap > 0.05:
                f.write("结论: [良好] 特征提取效果良好\n")
                f.write("建议:\n")
                f.write("  1. 可以尝试其他池化方法\n")
                f.write("  2. 考虑特征融合\n")
            else:
                f.write("结论: [优秀] 特征提取达到优秀水平\n")

            f.write(f"\n平均差距: {avg_gap:.4f}\n")

            # 5. 论文关键发现
            f.write("\n五、论文关键发现:\n")
            f.write("-" * 60 + "\n")
            f.write("1. L3-iMAC9x (9 regions) 效果最好\n")
            f.write("2. 3840维特征优于2048维特征\n")
            f.write("3. 区域特征比全局特征更有效\n")
            f.write("4. 在存储有限时，降维到256维仍有不错效果\n")

        # 保存JSON数据
        json_file = os.path.join(self.results_dir, "feature_comparison_data.json")
        data = {
            "paper_results": self.paper_results["table2"],
            "your_results": self.your_results,
            "methods": self.methods,
            "analysis": {
                "best_method": "L3-iMAC9x",
                "your_gap": avg_gap,
            }
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n> 特征提取对比报告已保存到: {self.results_dir}/")
        print(f"  - {report_file}")
        print(f"  - {json_file}")


def main():
    """主函数"""
    comparison = FeatureComparison()
    comparison.run_comparison()


if __name__ == '__main__':
    main()