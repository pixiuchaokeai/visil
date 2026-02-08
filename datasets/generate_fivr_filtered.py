import pickle
import os


def filter_fivr_pickle(original_pickle_path, query_list_path, database_list_path, output_pickle_path):
    """
    根据过滤后的查询和数据库视频列表，筛选原始的fivr.pickle文件

    Args:
        original_pickle_path: 原始fivr.pickle文件路径
        query_list_path: 查询视频列表文件路径
        database_list_path: 数据库视频列表文件路径
        output_pickle_path: 输出过滤后的pickle文件路径
    """

    # 1. 读取原始pickle文件
    print(f"正在加载原始pickle文件: {original_pickle_path}")
    with open(original_pickle_path, 'rb') as f:
        fivr_data = pickle.load(f)

    print("原始数据结构:")
    print(f"类型: {type(fivr_data)}")
    print(f"字典长度: {len(fivr_data)}")
    print(f"键: {list(fivr_data.keys())}")

    # 检查数据结构
    if '5k' not in fivr_data:
        print("错误: pickle文件中没有找到'5k'键")
        return

    fivr_5k = fivr_data['5k']
    print(f"\n'5k'子字典结构:")
    print(f"键: {list(fivr_5k.keys())}")

    original_queries = fivr_5k.get('queries')
    original_database = fivr_5k.get('database')

    print(f"\n数据类型:")
    print(f"original_queries 类型: {type(original_queries)}, 长度: {len(original_queries) if original_queries else 0}")
    print(
        f"original_database 类型: {type(original_database)}, 长度: {len(original_database) if original_database else 0}")

    # 2. 读取过滤后的查询视频ID列表
    print(f"\n正在读取查询视频列表: {query_list_path}")
    query_video_ids = set()
    with open(query_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 1:
                    video_id = parts[0]
                    query_video_ids.add(video_id)

    print(f"读取到 {len(query_video_ids)} 个查询视频ID")

    # 3. 读取过滤后的数据库视频ID列表
    print(f"\n正在读取数据库视频列表: {database_list_path}")
    database_video_ids = set()
    with open(database_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 1:
                    video_id = parts[0]
                    database_video_ids.add(video_id)

    print(f"读取到 {len(database_video_ids)} 个数据库视频ID")

    # 4. 筛选查询视频数据
    print(f"\n筛选查询视频数据...")

    if isinstance(original_queries, list):
        print(f"原始查询视频数量: {len(original_queries)}")

        # 筛选查询视频ID
        filtered_queries = []
        for video_id in original_queries:
            if video_id in query_video_ids:
                filtered_queries.append(video_id)

        print(f"筛选后查询视频数量: {len(filtered_queries)}")
    else:
        print(f"错误: original_queries 类型不是列表: {type(original_queries)}")
        return

    # 5. 筛选数据库视频数据
    print(f"\n筛选数据库视频数据...")

    if isinstance(original_database, set):
        print(f"原始数据库视频数量: {len(original_database)}")

        # 筛选数据库视频ID
        filtered_database = set()
        for video_id in original_database:
            if video_id in database_video_ids:
                filtered_database.add(video_id)

        print(f"筛选后数据库视频数量: {len(filtered_database)}")
    else:
        print(f"错误: original_database 类型不是集合: {type(original_database)}")
        return

    # 6. 筛选annotation数据 - 修复版本
    print(f"\n筛选annotation数据...")

    original_annotation = fivr_data.get('annotation', {})
    print(f"原始annotation数量: {len(original_annotation)}")

    # 重新构建annotation：只保留过滤后的查询视频的annotation
    filtered_annotation = {}

    for query_id in query_video_ids:
        if query_id in original_annotation:
            query_annotation = original_annotation[query_id]

            if isinstance(query_annotation, dict):
                # 创建新的查询标注
                new_query_annotation = {}

                for label_type, related_videos in query_annotation.items():
                    if isinstance(related_videos, list):
                        # 只保留在过滤数据库中的相关视频
                        filtered_related = [vid for vid in related_videos if vid in database_video_ids]
                        if filtered_related:  # 只添加非空列表
                            new_query_annotation[label_type] = filtered_related

                if new_query_annotation:  # 只添加有相关视频的查询
                    filtered_annotation[query_id] = new_query_annotation

                    # 打印第一个查询的信息
                    if len(filtered_annotation) == 1:
                        print(f"示例查询 {query_id} 的标注:")
                        for label_type, videos in new_query_annotation.items():
                            print(f"  {label_type}: {len(videos)} 个相关视频")
                            if videos:
                                print(f"    示例: {videos[:3]}")
            else:
                print(f"警告: 查询 {query_id} 的标注类型不是字典: {type(query_annotation)}")

    print(f"筛选后annotation数量: {len(filtered_annotation)} 个查询视频")

    # 统计总相关视频数
    total_related_videos = 0
    for query_id, annotation in filtered_annotation.items():
        for label_type, videos in annotation.items():
            total_related_videos += len(videos)

    print(f"总相关视频数: {total_related_videos}")

    # 7. 创建新的数据结构
    print(f"\n创建新的数据结构...")

    # 首先复制原始数据结构
    filtered_data = fivr_data.copy()

    # 更新'5k'部分
    filtered_data['5k'] = {
        'queries': filtered_queries,
        'database': filtered_database
    }

    # 更新'annotation'部分
    filtered_data['annotation'] = filtered_annotation

    # 8. 保存到新的pickle文件
    print(f"\n保存到新的pickle文件: {output_pickle_path}")
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(filtered_data, f)

    print("完成!")

    # 9. 验证保存的文件
    print(f"\n验证保存的文件...")
    with open(output_pickle_path, 'rb') as f:
        loaded_data = pickle.load(f)

    loaded_5k = loaded_data['5k']
    loaded_queries = loaded_5k['queries']
    loaded_database = loaded_5k['database']
    loaded_annotation = loaded_data['annotation']

    print(f"验证结果:")
    print(f"查询视频数量: {len(loaded_queries)}")
    print(f"数据库视频数量: {len(loaded_database)}")
    print(f"annotation数量: {len(loaded_annotation)}")

    # 检查annotation中的查询视频是否都在查询列表中
    annotation_query_ids = set(loaded_annotation.keys())
    loaded_query_ids = set(loaded_queries)

    print(f"annotation中的查询视频: {len(annotation_query_ids)}")
    print(f"查询列表中的视频: {len(loaded_query_ids)}")

    if annotation_query_ids != loaded_query_ids:
        print(f"\n注意: annotation中的查询视频与查询列表不匹配")
        missing_in_annotation = loaded_query_ids - annotation_query_ids
        missing_in_queries = annotation_query_ids - loaded_query_ids

        if missing_in_annotation:
            print(f"查询列表有但annotation中没有 ({len(missing_in_annotation)}): {list(missing_in_annotation)[:5]}")
        if missing_in_queries:
            print(f"annotation中有但查询列表没有 ({len(missing_in_queries)}): {list(missing_in_queries)[:5]}")
    else:
        print("✓ annotation与查询列表匹配")

    # 统计每个查询的相关视频
    print(f"\n每个查询的相关视频统计:")
    for query_id in list(loaded_annotation.keys())[:3]:  # 只显示前3个
        annotation = loaded_annotation[query_id]
        print(f"查询 {query_id}:")
        for label_type, videos in annotation.items():
            print(f"  {label_type}: {len(videos)} 个相关视频")

    return filtered_data


def explore_pickle_structure(filepath):
    """
    探索pickle文件的结构
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    def print_structure(obj, name="root", indent=0, max_items=5):
        prefix = "  " * indent

        if isinstance(obj, dict):
            print(f"{prefix}{name}: 字典 (长度: {len(obj)})")
            keys = list(obj.keys())
            for i, key in enumerate(keys[:max_items]):
                print(f"{prefix}  [{i}] 键: {key} ({type(key)})")
                print_structure(obj[key], f"键 '{key}' 的值", indent + 2, max_items)
            if len(keys) > max_items:
                print(f"{prefix}  ... 还有 {len(keys) - max_items} 个键")

        elif isinstance(obj, list):
            print(f"{prefix}{name}: 列表 (长度: {len(obj)})")
            if obj:
                print(f"{prefix}  第一个元素类型: {type(obj[0])}")
                print_structure(obj[0], "第一个元素", indent + 2, max_items)

        elif isinstance(obj, tuple):
            print(f"{prefix}{name}: 元组 (长度: {len(obj)})")
            if obj:
                print(f"{prefix}  第一个元素类型: {type(obj[0])}")
                print_structure(obj[0], "第一个元素", indent + 2, max_items)

        elif isinstance(obj, set):
            print(f"{prefix}{name}: 集合 (长度: {len(obj)})")
            if obj:
                # 显示集合中的前几个元素
                items = list(obj)[:max_items]
                for i, item in enumerate(items):
                    print(f"{prefix}  [{i}] 元素: {item} ({type(item)})")
                if len(obj) > max_items:
                    print(f"{prefix}  ... 还有 {len(obj) - max_items} 个元素")

        else:
            print(f"{prefix}{name}: {type(obj).__name__}: {str(obj)[:100]}...")

    print(f"\n探索文件: {filepath}")
    print("=" * 50)
    print_structure(data, "root", max_items=3)
    print("=" * 50)

    return data


def main():
    # 配置文件路径 - 基于你当前目录结构
    original_pickle_path = "fivr.pickle"  # 假设与脚本在同一目录
    query_list_path = "fivr-5k-queries-filtered.txt"
    database_list_path = "fivr-5k-database-filtered.txt"
    output_pickle_path = "fivr-filtered.pickle"

    # 检查文件是否存在
    if not os.path.exists(original_pickle_path):
        print(f"错误: 原始pickle文件不存在: {original_pickle_path}")
        # 尝试在datasets目录下查找
        original_pickle_path = os.path.join("datasets", "fivr.pickle")
        if os.path.exists(original_pickle_path):
            print(f"在datasets目录下找到原始pickle文件")

    if not os.path.exists(query_list_path):
        print(f"错误: 查询列表文件不存在: {query_list_path}")
        # 尝试在datasets目录下查找
        query_list_path = os.path.join("datasets", "fivr-5k-queries-filtered.txt")
        if os.path.exists(query_list_path):
            print(f"在datasets目录下找到查询列表文件")

    if not os.path.exists(database_list_path):
        print(f"错误: 数据库列表文件不存在: {database_list_path}")
        # 尝试在datasets目录下查找
        database_list_path = os.path.join("datasets", "fivr-5k-database-filtered.txt")
        if os.path.exists(database_list_path):
            print(f"在datasets目录下找到数据库列表文件")

    # 检查所有文件是否存在
    missing_files = []
    for path, name in [
        (original_pickle_path, "原始pickle文件"),
        (query_list_path, "查询列表文件"),
        (database_list_path, "数据库列表文件")
    ]:
        if not os.path.exists(path):
            missing_files.append((name, path))

    if missing_files:
        print(f"\n以下文件不存在:")
        for name, path in missing_files:
            print(f"  {name}: {path}")
        return

    print(f"所有文件都存在，开始处理...")
    print(f"原始pickle文件: {original_pickle_path}")
    print(f"查询列表文件: {query_list_path}")
    print(f"数据库列表文件: {database_list_path}")
    print(f"输出文件: {output_pickle_path}")

    # 先探索数据结构
    print("\n首先探索原始pickle文件结构...")
    explore_pickle_structure(original_pickle_path)

    # 执行筛选
    filter_fivr_pickle(
        original_pickle_path=original_pickle_path,
        query_list_path=query_list_path,
        database_list_path=database_list_path,
        output_pickle_path=output_pickle_path
    )


if __name__ == "__main__":
    main()