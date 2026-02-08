# -*- coding: utf-8 -*-
"""
ablation_study.py
消融实验脚本（修复编码问题版本）
对应论文表3和表4
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List
import sys

# 设置系统编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

from config import experiment_config


class AblationStudy:
    """消融实验类（修复编码问题版本）"""

    def __init__(self):
        self.config = experiment_config
        self.results_dir = os.path.join(self.config.output_dir, "ablation_study")
        os.makedirs(self.results_dir, exist_ok=True)

        # 论文结果
        self.paper_results = self.config.paper_results

        # 您的当前结果
        self.your_results = self.load_your_results()

        # 实验配置
        self.experiments = [
            {
                "name": "ViSiLf",
                "desc": "基础特征提取 (L3-iMAC9x)",
                "components": ["L3-iMAC9x"],
            },
            {
                "name": "ViSiLf+W",
                "desc": "基础 + 白化",
                "components": ["L3-iMAC9x", "Whitening"],
            },
            {
                "name": "ViSiLf+W+A",
                "desc": "基础 + 白化 + 注意力",
                "components": ["L3-iMAC9x", "Whitening", "Attention"],
            },
            {
                "name": "ViSiLsym",
                "desc": "对称 Chamfer 相似度",
                "components": ["L3-iMAC9x", "Whitening", "Attention", "Symmetric"],
            },
            {
                "name": "ViSiLv",
                "desc": "完整模型 (ViSiLv)",
                "components": ["L3-iMAC9x", "Whitening", "Attention", "Video Comparator"],
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

    def run_study(self):
        """运行消融实验分析"""
        print("\n" + "="*80)
        print("消融实验分析")
        print("="*80)

        # 1. 展示论文结果
        self.display_paper_results()

        # 2. 与您的结果对比
        self.compare_with_your_results()

        # 3. 分析各组件贡献
        self.analyze_component_contributions()

        # 4. 正则化影响分析
        self.analyze_regularization_impact()

        # 5. 生成报告
        self.generate_report()

        print("\n" + "="*80)
        print("消融实验分析完成!")
        print("="*80)

    def display_paper_results(self):
        """展示论文结果"""
        print("\n" + "="*50)
        print("1. 论文消融实验结果 (表3)")
        print("="*50)

        print(f"{'模型':<20} {'DSVR':<10} {'CSVR':<10} {'ISVR':<10} {'描述':<30}")
        print("-" * 80)

        for exp in self.experiments:
            name = exp["name"]
            if name in self.paper_results["table3"]:
                results = self.paper_results["table3"][name]
                print(f"{name:<20} {results['DSVR']:<10.3f} {results['CSVR']:<10.3f} "
                      f"{results['ISVR']:<10.3f} {exp['desc']}")

    def compare_with_your_results(self):
        """与您的结果对比"""
        print("\n" + "="*50)
        print("2. 与您的结果对比")
        print("="*50)

        paper_visilv = self.paper_results["table3"]["ViSiLv"]

        print(f"{'任务':<8} {'论文(ViSiLv)':<12} {'您的结果':<12} {'差异':<12} {'相对差距':<12}")
        print("-" * 60)

        for task in ["DSVR", "CSVR", "ISVR"]:
            paper_score = paper_visilv[task]
            your_score = self.your_results.get(task, 0.0)
            diff = your_score - paper_score
            rel_diff = diff / paper_score * 100 if paper_score > 0 else 0

            print(f"{task:<8} {paper_score:<12.4f} {your_score:<12.4f} "
                  f"{diff:+.4f} ({rel_diff:+.1f}%)")

    def analyze_component_contributions(self):
        """分析各组件贡献"""
        print("\n" + "="*50)
        print("3. 各组件对DSVR任务的贡献")
        print("="*50)

        table3 = self.paper_results["table3"]

        contributions = [
            ("L3-iMAC9x", "基础特征", table3["ViSiLf"]["DSVR"]),
            ("+ Whitening", "白化", table3["ViSiLf+W"]["DSVR"] - table3["ViSiLf"]["DSVR"]),
            ("+ Attention", "注意力", table3["ViSiLf+W+A"]["DSVR"] - table3["ViSiLf+W"]["DSVR"]),
            ("+ Video Comparator", "视频比较器", table3["ViSiLv"]["DSVR"] - table3["ViSiLf+W+A"]["DSVR"]),
        ]

        print(f"{'组件':<25} {'提升':<12} {'累计':<12} {'贡献说明':<20}")
        print("-" * 70)

        cumulative = 0.0
        for component, desc, improvement in contributions:
            cumulative += improvement
            print(f"{component:<25} {improvement:<12.4f} {cumulative:<12.4f} {desc:<20}")

        print(f"\n总提升: {cumulative:.4f} (从 {table3['ViSiLf']['DSVR']:.3f} 到 {table3['ViSiLv']['DSVR']:.3f})")

        # 计算相对提升百分比
        baseline = table3["ViSiLf"]["DSVR"]
        final = table3["ViSiLv"]["DSVR"]
        total_relative = (final - baseline) / baseline * 100
        print(f"相对提升: {total_relative:.1f}%")

    def analyze_regularization_impact(self):
        """分析正则化影响"""
        print("\n" + "="*50)
        print("4. 正则化影响分析 (表4)")
        print("="*50)

        table4 = self.paper_results["table4"]

        print(f"{'配置':<20} {'DSVR':<10} {'CSVR':<10} {'ISVR':<10} {'提升':<10}")
        print("-" * 60)

        for config_name, config_desc in [("without_reg", "无正则化"), ("with_reg", "有正则化")]:
            results = table4[config_name]
            print(f"{config_desc:<20} {results['DSVR']:<10.3f} {results['CSVR']:<10.3f} "
                  f"{results['ISVR']:<10.3f}")

        # 计算正则化带来的提升
        reg_improvement = {
            "DSVR": table4["with_reg"]["DSVR"] - table4["without_reg"]["DSVR"],
            "CSVR": table4["with_reg"]["CSVR"] - table4["without_reg"]["CSVR"],
            "ISVR": table4["with_reg"]["ISVR"] - table4["without_reg"]["ISVR"],
        }

        print(f"\n正则化带来的提升:")
        for task in ["DSVR", "CSVR", "ISVR"]:
            print(f"  {task}: +{reg_improvement[task]:.4f} "
                  f"({reg_improvement[task]/table4['without_reg'][task]*100:.1f}%)")

    def generate_report(self):
        """生成报告"""
        report_file = os.path.join(self.results_dir, "ablation_study_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("消融实验分析报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 1. 论文结果
            f.write("一、论文消融实验结果 (表3):\n")
            f.write("-" * 60 + "\n")
            for exp in self.experiments:
                name = exp["name"]
                if name in self.paper_results["table3"]:
                    results = self.paper_results["table3"][name]
                    f.write(f"{name:<20} DSVR={results['DSVR']:.3f}, "
                           f"CSVR={results['CSVR']:.3f}, ISVR={results['ISVR']:.3f}\n")

            # 2. 与您的结果对比
            f.write("\n二、与您的结果对比:\n")
            f.write("-" * 60 + "\n")
            paper_visilv = self.paper_results["table3"]["ViSiLv"]
            for task in ["DSVR", "CSVR", "ISVR"]:
                diff = self.your_results.get(task, 0.0) - paper_visilv[task]
                f.write(f"{task}: 论文={paper_visilv[task]:.3f}, "
                       f"您的={self.your_results.get(task, 0.0):.4f}, "
                       f"差异={diff:+.4f}\n")

            # 3. 组件贡献
            f.write("\n三、各组件贡献分析:\n")
            f.write("-" * 60 + "\n")
            table3 = self.paper_results["table3"]
            contributions = [
                ("L3-iMAC9x", table3["ViSiLf"]["DSVR"]),
                ("+ Whitening", table3["ViSiLf+W"]["DSVR"] - table3["ViSiLf"]["DSVR"]),
                ("+ Attention", table3["ViSiLf+W+A"]["DSVR"] - table3["ViSiLf+W"]["DSVR"]),
                ("+ Video Comparator", table3["ViSiLv"]["DSVR"] - table3["ViSiLf+W+A"]["DSVR"]),
            ]

            cumulative = 0.0
            for component, improvement in contributions:
                cumulative += improvement
                f.write(f"{component:<25}: {improvement:+.4f} (累计: {cumulative:.4f})\n")

            # 4. 正则化影响
            f.write("\n四、正则化影响 (表4):\n")
            f.write("-" * 60 + "\n")
            table4 = self.paper_results["table4"]
            for config_name, config_desc in [("without_reg", "无正则化"), ("with_reg", "有正则化")]:
                results = table4[config_name]
                f.write(f"{config_desc}: DSVR={results['DSVR']:.3f}, "
                       f"CSVR={results['CSVR']:.3f}, ISVR={results['ISVR']:.3f}\n")

            # 5. 总结
            f.write("\n五、总结:\n")
            f.write("-" * 60 + "\n")
            avg_gap = np.mean([
                paper_visilv[task] - self.your_results.get(task, 0.0)
                for task in ["DSVR", "CSVR", "ISVR"]
            ])

            if avg_gap > 0.2:
                f.write("结论: [严重] 差距较大，需要检查核心实现\n")
            elif avg_gap > 0.1:
                f.write("结论: [中等] 中等差距，需要优化参数\n")
            elif avg_gap > 0.05:
                f.write("结论: [良好] 差距较小，接近论文结果\n")
            else:
                f.write("结论: [优秀] 复现成功，结果优秀\n")

            f.write(f"平均差距: {avg_gap:.4f}\n")

            # 6. 建议
            f.write("\n六、改进建议:\n")
            f.write("-" * 60 + "\n")
            suggestions = [
                "1. 检查是否使用L3-iMAC9x特征 (9 regions, 3840 dim)",
                "2. 验证白化权重是否正确加载",
                "3. 确保使用注意力机制",
                "4. 检查视频比较器CNN是否启用",
                "5. 确认使用正确的相似度函数 (Chamfer vs Symmetric Chamfer)",
                "6. 检查训练参数: lr=10e-5, gamma=0.5, r=0.1",
                "7. 验证数据预处理: 1 fps, center crop 224x224",
            ]

            for suggestion in suggestions:
                f.write(suggestion + "\n")

        # 保存JSON数据
        json_file = os.path.join(self.results_dir, "ablation_study_data.json")
        data = {
            "paper_results": self.paper_results,
            "your_results": self.your_results,
            "analysis": {
                "avg_gap": avg_gap,
                "status": "needs_improvement" if avg_gap > 0.1 else "close"
            }
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n> 消融实验报告已保存到: {self.results_dir}/")
        print(f"  - {report_file}")
        print(f"  - {json_file}")


def main():
    """主函数"""
    study = AblationStudy()
    study.run_study()


if __name__ == '__main__':
    main()