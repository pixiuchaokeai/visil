import numpy as np
import pickle as pk

from collections import OrderedDict
from sklearn.metrics import average_precision_score

        
class CC_WEB_VIDEO(object):

    def __init__(self):
        with open('datasets/cc_web_video.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.name = 'CC_WEB_VIDEO'
        self.database = dataset['index']
        self.queries = dataset['queries']
        self.ground_truth = dataset['ground_truth']
        self.excluded = dataset['excluded']

    def get_queries(self):
        return self.queries

    def get_database(self):
        return list(map(str, self.database.keys()))

    def calculate_mAP(self, similarities, all_videos=False, clean=False, positive_labels='ESLMV'):
        mAP = 0.0
        for query_set, labels in enumerate(self.ground_truth):
            query_id = self.queries[query_set]
            i, ri, s = 0.0, 0.0, 0.0
            if query_id in similarities:
                res = similarities[query_id]
                for video_id in sorted(res.keys(), key=lambda x: res[x], reverse=True):
                    video = self.database[video_id]
                    if (all_videos or video in labels) and (not clean or video not in self.excluded[query_set]):
                        ri += 1
                        if video in labels and labels[video] in positive_labels:
                            i += 1.0
                            s += i / ri
                positives = np.sum([1.0 for k, v in labels.items() if
                                    v in positive_labels and (not clean or k not in self.excluded[query_set])])
                mAP += s / positives
        return mAP / len(set(self.queries).intersection(similarities.keys()))

    def evaluate(self, similarities, all_db=None, verbose=True):
        if all_db is None:
            all_db = self.database

        if verbose:
            print('=' * 5, 'CC_WEB_VIDEO Dataset', '=' * 5)
            not_found = len(set(self.queries) - similarities.keys())
            if not_found > 0:
                print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
            print('Queries: {} videos'.format(len(similarities)))
            print('Database: {} videos'.format(len(all_db)))

        mAP = self.calculate_mAP(similarities, all_videos=False, clean=False)
        mAP_star = self.calculate_mAP(similarities, all_videos=True, clean=False)
        if verbose:
            print('-' * 25)
            print('All dataset')
            print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}\n'.format(mAP, mAP_star))

        mAP_c = self.calculate_mAP(similarities, all_videos=False, clean=True)
        mAP_c_star = self.calculate_mAP(similarities, all_videos=True, clean=True)
        if verbose:
            print('Clean dataset')
            print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}'.format(mAP_c, mAP_c_star))
        return {'mAP': mAP, 'mAP_star': mAP_star, 'mAP_c': mAP_c, 'mAP_c_star': mAP_c_star}


class FIVR(object):
    """
    FIVR (Fine-grained Incident Video Retrieval) 数据集类

    用于处理FIVR视频检索数据集，支持200K和5K两个版本。
    提供查询视频列表、数据库视频列表，以及评估检索结果的mAP计算。

    属性:
        version: 数据集版本 ('200k' 或 '5k')
        name: 数据集名称
        annotation: 人工标注的相似性关系
        queries: 查询视频ID列表
        database: 数据库视频ID列表
    """

    def __init__(self, version='200k'):
        """
        初始化FIVR数据集

        参数:
            version: 数据集版本，可选 '200k' 或 '5k'
                    '200k' 包含约20万个视频
                    '5k' 包含约5千个视频（200k的子集）

        加载过程:
            1. 从pickle文件加载数据集元数据
            2. 提取对应版本的查询集和数据库
        """
        self.version = version

        # 从pickle文件加载数据集元数据（包含标注信息、查询集、数据库等）
        with open('datasets/fivr-filtered.pickle', 'rb') as f:
            dataset = pk.load(f)

        self.name = 'FIVR'  # 数据集名称

        # annotation: 字典，key是查询视频ID，value是该视频的相关视频分类标注
        # 相关视频按相似程度分为：ND(复制), DS(相同场景), CS(相同事件), IS(同一事件)
        self.annotation = dataset['annotation']

        # 根据版本选择对应的查询集和数据库
        self.queries = dataset[self.version]['queries']  # 查询视频ID列表
        self.database = dataset[self.version]['database']  # 数据库视频ID列表

    def get_queries(self):
        """
        获取查询视频ID列表

        返回:
            list: 查询视频的ID列表，这些视频将用于作为检索的查询输入
        """
        return self.queries

    def get_database(self):
        """
        获取数据库视频ID列表

        返回:
            list: 数据库视频的ID列表，检索将在这个集合中进行

        注意: 使用list()转换是为了返回一个列表副本，避免外部修改影响原始数据
        """
        return list(self.database)

    def calculate_mAP(self, query, res, all_db, relevant_labels):
        """
        计算单个查询的平均精度均值 (mean Average Precision, mAP)

        这是视频检索的核心评估指标，衡量检索结果的相关性排序质量。

        参数:
            query: 查询视频ID
            res: 检索结果字典，key是视频ID，value是相似度分数
            all_db: 实际检索过的数据库视频集合（可能包含未标注的视频）
            relevant_labels: 考虑哪些相似度标签作为"相关"
                          例如 ['ND', 'DS'] 表示复制和相同场景都算相关

        返回:
            float: 该查询的AP值（0-1之间，越高越好）

        计算逻辑:
            1. 从标注中获取该查询视频的所有相关视频（根据指定的相似度标签）
            2. 只保留实际在数据库中的相关视频
            3. 按相似度分数降序遍历检索结果
            4. 计算Precision@K的累积平均值
        """
        # 从标注中获取该查询视频的相关视频集合
        gt_sets = self.annotation[query]

        # 合并所有指定标签的相关视频（ND=复制, DS=相同场景, CS=相同事件, IS=同一事件）
        # sum()用于合并多个列表，初始值为空列表[]
        query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))

        # 只保留实际在数据库中的相关视频（有些标注视频可能不在当前数据库中）
        query_gt = query_gt.intersection(all_db)

        # i: 当前找到的相关视频数（用于计算precision）
        # ri: 当前遍历到的位置（rank）
        # s: 累积的precision之和（最终除以相关视频总数得到AP）
        i, ri, s = 0.0, 0, 0.0

        # 按相似度分数降序排序检索结果
        for video in sorted(res.keys(), key=lambda x: res[x], reverse=True):
            # 排除查询自身，且只考虑在数据库中的视频
            if video != query and video in all_db:
                ri += 1  # 当前排名+1

                # 如果该视频是相关的
                if video in query_gt:
                    i += 1.0  # 找到的相关视频数+1
                    s += i / ri  # 累加Precision@K（当前找到的相关数/当前排名）

        # 返回AP = 所有Precision@K的平均值 = 总和/相关视频总数
        return s / len(query_gt)

    def evaluate(self, similarities, all_db=None, verbose=True):
        """
        评估整个检索任务的性能

        计算三种不同严格程度的mAP指标：DSVR、CSVR、ISVR

        参数:
            similarities: 相似度矩阵，字典格式
                         {查询ID: {目标ID: 相似度分数, ...}, ...}
            all_db: 实际被检索的数据库视频集合（如果为None则使用全部数据库）
            verbose: 是否打印详细的评估结果

        返回:
            dict: 包含三个指标的字典 {'DSVR': x, 'CSVR': y, 'ISVR': z}

        三种评估指标说明:
            DSVR (Duplicate Scene Video Retrieval): 最严格，只算完全复制(ND)和相同场景(DS)
            CSVR (Complementary Scene Video Retrieval): 中等，包含ND, DS, 相同事件(CS)
            ISVR (Incident Scene Video Retrieval): 最宽松，包含所有相似类型(ND, DS, CS, IS)
        """
        # 如果没有指定数据库集合，使用全部数据库
        if all_db is None:
            all_db = self.database

        # 存储三种指标的AP值列表（每个查询一个值）
        DSVR, CSVR, ISVR = [], [], []

        # 遍历所有查询的检索结果
        for query, res in similarities.items():
            # 只评估在查询集中的视频（防止传入非法查询）
            if query in self.queries:
                # DSVR: 只考虑ND(复制)和DS(相同场景)作为相关
                DSVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS']))

                # CSVR: 考虑ND, DS, CS(相同事件)作为相关
                CSVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS', 'CS']))

                # ISVR: 考虑所有四种类型作为相关
                ISVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS', 'CS', 'IS']))

        # 打印详细的评估报告
        if verbose:
            print('=' * 5, 'FIVR-{} Dataset'.format(self.version.upper()), '=' * 5)

            # 检查是否有查询缺失（有些查询可能没有计算相似度）
            not_found = len(set(self.queries) - similarities.keys())
            if not_found > 0:
                print('[WARNING] {} 个查询在结果中缺失，将被忽略'.format(not_found))

            print('查询集: {} 个视频'.format(len(similarities)))
            print('数据库: {} 个视频'.format(len(all_db)))
            print('-' * 16)

            # 计算并打印三种指标的平均mAP
            print('DSVR mAP: {:.4f}'.format(np.mean(DSVR)))
            print('CSVR mAP: {:.4f}'.format(np.mean(CSVR)))
            print('ISVR mAP: {:.4f}'.format(np.mean(ISVR)))

        # 返回三种指标的平均值
        return {'DSVR': np.mean(DSVR), 'CSVR': np.mean(CSVR), 'ISVR': np.mean(ISVR)}


class EVVE(object):

    def __init__(self):
        with open('datasets/evve.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.name = 'EVVE'
        self.events = dataset['annotation']
        self.queries = dataset['queries']
        self.database = dataset['database']
        self.query_to_event = {qname: evname
                               for evname, (queries, _, _) in self.events.items()
                               for qname in queries}
        
    def get_queries(self):
        return list(self.queries)

    def get_database(self):
        return list(self.database)

    def score_ap_from_ranks_1(self, ranks, nres):
        """ Compute the average precision of one search.
        ranks = ordered list of ranks of true positives (best rank = 0)
        nres  = total number of positives in dataset
        """
        if nres == 0 or ranks == []:
            return 0.0

        ap = 0.0

        # accumulate trapezoids in PR-plot. All have an x-size of:
        recall_step = 1.0 / nres

        for ntp, rank in enumerate(ranks):
            # ntp = nb of true positives so far
            # rank = nb of retrieved items so far

            # y-size on left side of trapezoid:
            if rank == 0:
                precision_0 = 1.0
            else:
                precision_0 = ntp / float(rank)
            # y-size on right side of trapezoid:
            precision_1 = (ntp + 1) / float(rank + 1)
            ap += (precision_1 + precision_0) * recall_step / 2.0
        return ap

    def evaluate(self, similarities, all_db=None, verbose=True):
        results = {e: [] for e in self.events}
        if all_db is None:
            all_db = set(self.database).union(set(self.queries))

        not_found = 0
        for query in self.queries:
            if query not in similarities:
                not_found += 1
            else:
                res = similarities[query]
                evname = self.query_to_event[query]
                _, pos, null = self.events[evname]
                if all_db:
                    pos = pos.intersection(all_db)
                pos_ranks = []

                ri, n_ext = 0.0, 0.0
                for ri, dbname in enumerate(sorted(res.keys(), key=lambda x: res[x], reverse=True)):
                    if dbname in pos:
                        pos_ranks.append(ri - n_ext)
                    if dbname not in all_db:
                        n_ext += 1

                ap = self.score_ap_from_ranks_1(pos_ranks, len(pos))
                results[evname].append(ap)
        if verbose:
            print('=' * 18, 'EVVE Dataset', '=' * 18)
            if not_found > 0:
                print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
            print('Queries: {} videos'.format(len(similarities)))
            print('Database: {} videos\n'.format(len(all_db - set(self.queries))))
            print('-' * 50)
        ap, mAP = [], []
        for evname in sorted(self.events):
            queries, _, _ = self.events[evname]
            nq = len(queries.intersection(all_db))
            ap.extend(results[evname])
            mAP.append(np.sum(results[evname]) / nq)
            if verbose:
                print('{0: <36} '.format(evname), 'mAP = {:.4f}'.format(np.sum(results[evname]) / nq))

        if verbose:
            print('=' * 50)
            print('overall mAP = {:.4f}'.format(np.mean(ap)))
        return {'mAP': np.mean(ap)}


class SVD(object):

    def __init__(self, version='unlabeled'):
        self.name = 'SVD'
        self.ground_truth = self.load_groundtruth('datasets/test_groundtruth')
        self.unlabeled_keys = self.get_unlabeled_keys('datasets/unlabeled-data-id')
        if version == 'labeled':
            self.unlabeled_keys = []
        self.database = []
        for k, v in self.ground_truth.items():
            self.database.extend(list(map(str, v.keys())))
        self.database += self.unlabeled_keys
        self.database_idxs = {d: i for i, d in enumerate(self.database)}

    def load_groundtruth(self, filepath):
        gnds = OrderedDict()
        with open(filepath, 'r') as fp:
            for idx, lines in enumerate(fp):
                tmps = lines.strip().split(' ')
                qid = tmps[0]
                cid = tmps[1]
                gt = int(tmps[-1])
                if qid not in gnds:
                    gnds[qid] = {cid: gt}
                else:
                    gnds[qid][cid] = gt
        return gnds

    def get_unlabeled_keys(self, filepath):
        videos = list()
        with open(filepath, 'r') as fp:
            for tmps in fp:
                videos.append(tmps.strip())
        return videos

    def get_queries(self):
        return list(map(str, self.ground_truth.keys()))

    def get_database(self):
        return self.database

    def evaluate(self, similarities, all_db=None, verbose=True):
        mAP = []
        not_found = len(self.ground_truth.keys() - similarities.keys())
        for query, targets in self.ground_truth.items():
            y_true, y_score = [], []
            for target, label in targets.items():
                if target in all_db:
                    # s = similarities[query][self.database_idxs[target]]
                    s = similarities[query][target]
                    y_true.append(label)
                    y_score.append(s)

            for target in self.unlabeled_keys:
                if target in all_db:
                    # s = similarities[query][self.database_idxs[target]]
                    s = similarities[query][target]
                    y_true.append(0)
                    y_score.append(s)
            mAP.append(average_precision_score(y_true, y_score))
        if verbose:
            print('=' * 5, 'SVD Dataset', '=' * 5)
            if not_found > 0:
                print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
            print('Database: {} videos'.format(len(all_db)))

            print('-' * 16)
            print('mAP: {:.4f}'.format(np.mean(mAP)))
        return {'mAP': np.mean(mAP)}
