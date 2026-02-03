#!/usr/bin/env python3
"""检查数据集文件和目录结构"""

import os
import sys


def check_file_exists(file_path, description=""):
    """检查文件是否存在并显示详细信息"""
    exists = os.path.exists(file_path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {file_path}")

    if exists:
        try:
            size = os.path.getsize(file_path)
            print(f"   大小: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

            # 检查文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"   行数: {len(lines)}")

                # 显示前3行
                if lines:
                    print("   前3行示例:")
                    for i, line in enumerate(lines[:3]):
                        print(f"      {i + 1}. {line.strip()}")
                else:
                    print("   文件为空!")
        except Exception as e:
            print(f"   读取文件时出错: {e}")
    else:
        # 尝试查找类似文件
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)

        if os.path.exists(dir_name):
            print(f"   在目录 {dir_name} 中查找类似文件...")
            files_in_dir = os.listdir(dir_name)
            similar_files = [f for f in files_in_dir if base_name.lower() in f.lower()]
            if similar_files:
                print(f"   找到类似文件: {similar_files}")

    return exists


def check_video_files(video_list_file):
    """检查视频文件列表中的所有视频"""
    if not os.path.exists(video_list_file):
        print(f"视频列表文件不存在: {video_list_file}")
        return

    print(f"\n检查视频列表: {video_list_file}")
    print("=" * 60)

    video_count = 0
    found_count = 0
    missing_count = 0

    with open(video_list_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # 分割行
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print(f"行 {line_num}: 格式错误 - {line}")
                continue

            video_id = parts[0]
            video_path = parts[1]

            video_count += 1

            # 检查视频文件
            if os.path.exists(video_path):
                try:
                    size = os.path.getsize(video_path)
                    found_count += 1
                    print(f"行 {line_num}: ✓ {video_id} - {video_path} ({size:,} bytes)")
                except:
                    print(f"行 {line_num}: ? {video_id} - {video_path} (无法获取大小)")
            else:
                missing_count += 1
                print(f"行 {line_num}: ✗ {video_id} - {video_path} (文件不存在)")

                # 尝试查找文件
                dir_name = os.path.dirname(video_path)
                base_name = os.path.basename(video_path)

                if os.path.exists(dir_name):
                    files_in_dir = os.listdir(dir_name)
                    # 查找可能的匹配
                    possible_matches = [f for f in files_in_dir if video_id in f or base_name in f]
                    if possible_matches:
                        print(f"      可能匹配: {possible_matches}")

    print(f"\n总结:")
    print(f"  总视频数: {video_count}")
    print(f"  找到: {found_count}")
    print(f"  缺失: {missing_count}")
    print(f"  成功率: {found_count / video_count * 100:.1f}%" if video_count > 0 else "N/A")


def main():
    print("检查数据集文件和目录结构")
    print("=" * 60)

    # 当前工作目录
    print(f"当前工作目录: {os.getcwd()}")
    print()

    # 检查必要目录
    dirs_to_check = [
        "datasets",
        "datasets/FIVR-200K",
    ]

    for dir_path in dirs_to_check:
        exists = os.path.exists(dir_path)
        status = "✓" if exists else "✗"
        print(f"{status} 目录: {dir_path}")
        if exists:
            try:
                files_count = len(os.listdir(dir_path))
                print(f"   包含 {files_count} 个文件/目录")
            except:
                print(f"   无法列出目录内容")

    print()

    # 检查文件
    files_to_check = [
        ("datasets/fivr-5k-queries-filtered.txt", "查询文件"),
        ("datasets/fivr-5k-database-filtered.txt", "数据库文件"),
    ]

    for file_path, description in files_to_check:
        check_file_exists(file_path, description)
        print()

    # 检查视频文件
    print("=" * 60)
    print("检查视频文件可用性")
    print("=" * 60)

    # 检查查询视频
    if os.path.exists("datasets/fivr-5k-queries-filtered.txt"):
        check_video_files("datasets/fivr-5k-queries-filtered.txt")

    # 检查数据库视频
    if os.path.exists("datasets/fivr-5k-database-filtered.txt"):
        check_video_files("datasets/fivr-5k-database-filtered.txt")

    print("\n" + "=" * 60)
    print("建议:")
    print("1. 确保所有视频文件已下载到正确位置")
    print("2. 检查文件路径是否正确")
    print("3. 如果视频文件缺失，需要重新下载数据集")


if __name__ == "__main__":
    main()