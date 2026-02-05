import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set


class FIVR:
    """FIVR 数据集类，用于加载查询和数据库视频列表"""

    def __init__(self, version='5k'):
        """
        初始化 FIVR 数据集

        参数:
            version: 数据集版本，'5k' 或 '200k'
        """
        self.version = version.lower()
        self.name = f"FIVR-{version.upper()}"

        # 根据版本设置默认文件路径
        if self.version == '5k':
            self.query_file = 'datasets/fivr-5k-queries.txt'
            self.database_file = 'datasets/fivr-5k-database.txt'
            self.annotation_file = 'datasets/fivr-5k-annotations.json'
        else:  # 200k
            self.query_file = 'datasets/fivr-200k-queries.txt'
            self.database_file = 'datasets/fivr-200k-database.txt'
            self.annotation_file = 'datasets/fivr-200k-annotations.json'

        # 加载数据
        self.queries = self._load_list(self.query_file)
        self.database = self._load_list(self.database_file)
        self.annotations = self._load_annotations(self.annotation_file)

        print(f"> 加载 {self.name} 数据集")
        print(f"  查询视频数: {len(self.queries)}")
        print(f"  数据库视频数: {len(self.database)}")
        print(f"  标注查询数: {len(self.annotations)}")

    def _load_list(self, file_path: str) -> List[str]:
        """从文件加载视频ID列表"""
        if not os.path.exists(file_path):
            print(f"> 警告: 文件不存在，创建空列表: {file_path}")
            return []

        video_ids = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # 移除可能的扩展名
                        video_id = line.split('.')[0]
                        video_ids.append(video_id)

            # 去重
            video_ids = list(set(video_ids))
            return video_ids
        except Exception as e:
            print(f"> 加载文件失败 {file_path}: {e}")
            return []

    def _load_annotations(self, file_path: str) -> Dict:
        """加载标注文件"""
        if not os.path.exists(file_path):
            print(f"> 警告: 标注文件不存在: {file_path}")
            return {}

        try:
            with open(file_path, 'r') as f:
                annotations = json.load(f)
            return annotations
        except Exception as e:
            print(f"> 加载标注失败 {file_path}: {e}")
            return {}

    def get_queries(self) -> List[str]:
        """获取查询视频ID列表"""
        return self.queries

    def get_database(self) -> List[str]:
        """获取数据库视频ID列表"""
        return self.database

    def get_annotations(self) -> Dict:
        """获取标注数据"""
        return self.annotations

    def get_query_video_path(self, video_id: str, video_dir: str, pattern: str = "{id}.mp4") -> str:
        """获取查询视频文件路径"""
        # 替换模式中的 {id}
        filename = pattern.format(id=video_id)
        return os.path.join(video_dir, filename)

    def get_database_video_path(self, video_id: str, video_dir: str, pattern: str = "{id}.mp4") -> str:
        """获取数据库视频文件路径"""
        # 替换模式中的 {id}
        filename = pattern.format(id=video_id)
        return os.path.join(video_dir, filename)

    def evaluate(self, similarities: Dict, all_db: Set = None) -> Dict:
        """
        评估相似度结果

        参数:
            similarities: 相似度字典 {query_id: {db_id: score}}
            all_db: 所有数据库视频集合（可选）

        返回:
            评估指标字典
        """
        if not self.annotations:
            print("> 警告: 没有标注数据，无法评估")
            return {}

        print(f"\n> 开始评估 {self.name} 数据集")

        # 初始化评估指标
        metrics = {
            'DSVR': {'precision': [], 'recall': [], 'f1': [], 'ap': []},
            'CSVR': {'precision': [], 'recall': [], 'f1': [], 'ap': []},
            'ISVR': {'precision': [], 'recall': [], 'f1': [], 'ap': []}
        }

        total_queries = 0
        evaluated_queries = 0

        for query_id, query_sims in similarities.items():
            total_queries += 1

            # 检查是否有标注
            if query_id not in self.annotations:
                continue

            evaluated_queries += 1

            # 获取相关视频标注
            query_annotations = self.annotations[query_id]

            # 对数据库视频按相似度排序
            sorted_db = sorted(query_sims.items(), key=lambda x: x[1], reverse=True)

            # 为每个任务评估
            for task in ['DSVR', 'CSVR', 'ISVR']:
                if task in query_annotations:
                    relevant_set = set(query_annotations[task])
                    if not relevant_set:
                        continue

                    # 计算AP (Average Precision)
                    ap = self._calculate_ap(sorted_db, relevant_set, all_db)
                    metrics[task]['ap'].append(ap)

        # 计算平均指标
        results = {}
        for task in ['DSVR', 'CSVR', 'ISVR']:
            if metrics[task]['ap']:
                mean_ap = np.mean(metrics[task]['ap'])
                results[f'mAP_{task}'] = mean_ap
                print(f"  {task} mAP: {mean_ap:.4f}")

        print(f"> 评估完成，处理了 {evaluated_queries}/{total_queries} 个查询视频")
        return results

    def _calculate_ap(self, sorted_db: List[Tuple[str, float]],
                      relevant_set: Set[str],
                      all_db: Set = None) -> float:
        """计算 Average Precision"""
        precision_at_k = []
        num_relevant = 0

        for k, (db_id, score) in enumerate(sorted_db, 1):
            # 如果提供了all_db，检查db_id是否有效
            if all_db is not None and db_id not in all_db:
                continue

            if db_id in relevant_set:
                num_relevant += 1
                precision_at_k.append(num_relevant / k)

        if not precision_at_k:
            return 0.0

        # 计算AP (Average Precision)
        ap = sum(precision_at_k) / len(relevant_set)
        return ap

    def filter_videos(self, available_videos: Set[str]) -> Tuple[List[str], List[str]]:
        """
        根据可用视频过滤查询和数据库列表

        参数:
            available_videos: 可用的视频ID集合

        返回:
            (过滤后的查询列表, 过滤后的数据库列表)
        """
        filtered_queries = [vid for vid in self.queries if vid in available_videos]
        filtered_database = [vid for vid in self.database if vid in available_videos]

        print(f"> 视频过滤结果:")
        print(f"  原始查询: {len(self.queries)} -> 可用查询: {len(filtered_queries)}")
        print(f"  原始数据库: {len(self.database)} -> 可用数据库: {len(filtered_database)}")

        return filtered_queries, filtered_database


# 其他数据集类的基类
class BaseDataset:
    """数据集基类"""

    def __init__(self, name: str):
        self.name = name
        self.queries = []
        self.database = []
        self.annotations = {}

    def get_queries(self) -> List[str]:
        return self.queries

    def get_database(self) -> List[str]:
        return self.database

    def get_annotations(self) -> Dict:
        return self.annotations

    def evaluate(self, similarities: Dict, all_db: Set = None) -> Dict:
        raise NotImplementedError("子类必须实现evaluate方法")