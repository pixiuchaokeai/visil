#!/usr/bin/env python3
"""
修正FIVR数据集文件路径的脚本
使用相对路径和正斜杠格式
"""

import os
import argparse
import re


def normalize_path(path):

    # 如果是绝对路径，转换为相对路径
    if os.path.isabs(path):
        try:
            path = os.path.relpath(path)
        except:
            pass

    # 统一使用正斜杠
    path = path.replace('\\', '/')

    # 移除开头的./（如果存在）
    if path.startswith('./'):
        path = path[2:]

    return path


def fix_file_paths(input_file, video_root="datasets/FIVR-200K"):
    """
    修正文件中的视频路径
    输入格式: video_id video_filename
    输出格式: video_id normalized_video_path
    """
    output_lines = []
    lines_processed = 0
    lines_skipped = 0

    # 标准化视频根目录
    video_root = normalize_path(video_root)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 分割行，处理制表符和多个空格
            parts = re.split(r'\s+', line)
            if len(parts) >= 2:
                video_id = parts[0]
                video_filename = parts[1]

                # 如果文件名已经是完整路径，提取文件名部分
                if '/' in video_filename or '\\' in video_filename:
                    # 提取文件名（不包含路径）
                    video_filename = os.path.basename(video_filename)

                # 构建完整路径
                video_path = f"{video_root}/{video_filename}"
                video_path = normalize_path(video_path)

                # 检查文件是否存在（尝试多种路径格式）
                file_exists = False
                tested_paths = []

                # 尝试直接路径
                if os.path.exists(video_path):
                    file_exists = True
                    actual_path = video_path
                else:
                    # 尝试添加./前缀
                    test_path = f"./{video_path}"
                    if os.path.exists(test_path):
                        file_exists = True
                        actual_path = normalize_path(test_path)
                        tested_paths.append(test_path)

                # 如果还没找到，尝试在video_root目录中搜索
                if not file_exists:
                    search_root = video_root.replace('/', os.sep)
                    if os.path.exists(search_root):
                        for root, dirs, files in os.walk(search_root):
                            for file in files:
                                if file == video_filename or file.startswith(video_id):
                                    # 使用相对路径
                                    rel_path = os.path.relpath(os.path.join(root, file))
                                    actual_path = normalize_path(rel_path)
                                    file_exists = True
                                    print(f"信息 (行{line_num}): 在子目录中找到文件 - {actual_path}")
                                    break
                            if file_exists:
                                break

                if not file_exists:
                    print(f"警告 (行{line_num}): 文件不存在 - {video_path}")
                    # 仍然添加到输出，但标记为可能不存在
                    output_lines.append(f"{video_id} {video_path} # FILE_NOT_FOUND")
                    lines_skipped += 1
                else:
                    # 使用找到的实际路径
                    output_lines.append(f"{video_id} {actual_path}")
                    lines_processed += 1
            else:
                print(f"警告 (行{line_num}): 跳过无效行 - {line}")
                lines_skipped += 1

    return output_lines, lines_processed, lines_skipped


def get_next_version_number(base_filename):
    """获取下一个版本号"""
    # 提取基本名称（不含扩展名）
    base_name = os.path.splitext(base_filename)[0]

    # 查找已存在的版本
    max_version = 0
    dir_name = os.path.dirname(base_filename) or '.'
    file_pattern = re.compile(rf"{re.escape(os.path.basename(base_name))}_(\d+)\.txt")

    for file in os.listdir(dir_name):
        match = file_pattern.match(file)
        if match:
            version = int(match.group(1))
            if version > max_version:
                max_version = version

    return max_version + 1


def create_backup(original_file, version):
    """创建备份文件"""
    base_name = os.path.splitext(original_file)[0]
    backup_file = f"{base_name}_{version}.txt"
    return backup_file


def main():
    parser = argparse.ArgumentParser(description='修正FIVR数据集文件路径')
    parser.add_argument('--queries', type=str, default='datasets/fivr-5k-queries-filtered.txt',
                        help='查询文件路径')
    parser.add_argument('--database', type=str, default='datasets/fivr-5k-database-filtered.txt',
                        help='数据库文件路径')
    parser.add_argument('--video_root', type=str, default='datasets/FIVR-200K',
                        help='视频文件根目录（相对路径）')
    parser.add_argument('--backup_original', action='store_true',
                        help='备份原始文件')

    args = parser.parse_args()

    print(f"当前工作目录: {os.getcwd()}")
    print(f"视频根目录: {args.video_root}")

    # 处理查询文件
    print(f"\n{'=' * 50}")
    print(f"处理查询文件: {args.queries}")

    if os.path.exists(args.queries):
        # 备份原始文件
        if args.backup_original:
            version = get_next_version_number(args.queries)
            backup_file = create_backup(args.queries, version)
            import shutil
            shutil.copy2(args.queries, backup_file)
            print(f"已备份原始文件到: {backup_file}")

        fixed_queries, processed_q, skipped_q = fix_file_paths(args.queries, args.video_root)

        # 覆盖原文件
        with open(args.queries, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_queries))

        print(f"已更新: {args.queries}")
        print(f"处理行数: {processed_q}, 跳过行数: {skipped_q}")

        # 显示前5行示例
        print("\n更新后的前5行示例:")
        for i, line in enumerate(fixed_queries[:5]):
            print(f"  {i + 1}. {line}")
    else:
        print(f"错误: 查询文件不存在 - {args.queries}")

    # 处理数据库文件
    print(f"\n{'=' * 50}")
    print(f"处理数据库文件: {args.database}")

    if os.path.exists(args.database):
        # 备份原始文件
        if args.backup_original:
            version = get_next_version_number(args.database)
            backup_file = create_backup(args.database, version)
            import shutil
            shutil.copy2(args.database, backup_file)
            print(f"已备份原始文件到: {backup_file}")

        fixed_database, processed_d, skipped_d = fix_file_paths(args.database, args.video_root)

        # 覆盖原文件
        with open(args.database, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_database))

        print(f"已更新: {args.database}")
        print(f"处理行数: {processed_d}, 跳过行数: {skipped_d}")

        # 显示前5行示例
        print("\n更新后的前5行示例:")
        for i, line in enumerate(fixed_database[:5]):
            print(f"  {i + 1}. {line}")
    else:
        print(f"错误: 数据库文件不存在 - {args.database}")

    print(f"\n{'=' * 50}")
    print("使用说明:")
    print(f"现在可以直接使用原始文件名:")
    print(f"  --query_file {args.queries}")
    print(f"  --database_file {args.database}")


if __name__ == "__main__":
    main()