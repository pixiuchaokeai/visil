#!/usr/bin/env python3
"""
修复或重新编码损坏的视频文件
"""

import os
import subprocess
import argparse


def check_video_with_ffmpeg(video_path):
    """使用ffmpeg检查视频文件"""
    try:
        cmd = ['ffmpeg', '-v', 'error', '-i', video_path, '-f', 'null', '-']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)


def repair_video_with_ffmpeg(input_path, output_path):
    """使用ffmpeg重新编码视频"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)


def process_video_list(video_list_file):
    """处理视频列表中的文件"""
    if not os.path.exists(video_list_file):
        print(f"文件不存在: {video_list_file}")
        return

    print(f"处理视频列表: {video_list_file}")

    repaired_count = 0
    failed_count = 0

    with open(video_list_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 提取视频路径
            parts = line.split()
            if len(parts) < 2:
                continue

            video_path = ' '.join(parts[1:])

            if not os.path.exists(video_path):
                print(f"行 {line_num}: 文件不存在 - {video_path}")
                continue

            print(f"行 {line_num}: 检查 {video_path}")

            # 检查视频
            is_valid, error = check_video_with_ffmpeg(video_path)

            if is_valid:
                print(f"  ✓ 视频正常")
            else:
                print(f"  ✗ 视频可能损坏: {error}")

                # 尝试修复
                backup_path = video_path + '.backup'
                repaired_path = video_path + '.repaired.mp4'

                try:
                    # 备份原文件
                    import shutil
                    shutil.copy2(video_path, backup_path)
                    print(f"    已备份到: {backup_path}")

                    # 修复视频
                    print(f"    正在修复...")
                    success, repair_error = repair_video_with_ffmpeg(video_path, repaired_path)

                    if success and os.path.exists(repaired_path):
                        # 替换原文件
                        os.remove(video_path)
                        shutil.move(repaired_path, video_path)
                        print(f"    ✓ 修复成功")
                        repaired_count += 1
                    else:
                        print(f"    ✗ 修复失败: {repair_error}")
                        failed_count += 1

                except Exception as e:
                    print(f"    ✗ 修复过程中出错: {e}")
                    failed_count += 1

    print(f"\n总结:")
    print(f"  修复成功: {repaired_count}")
    print(f"  修复失败: {failed_count}")


def main():
    parser = argparse.ArgumentParser(description='修复损坏的视频文件')
    parser.add_argument('--queries', type=str, default='datasets/fivr-5k-queries-filtered.txt',
                        help='查询视频列表文件')
    parser.add_argument('--database', type=str, default='datasets/fivr-5k-database-filtered.txt',
                        help='数据库视频列表文件')
    parser.add_argument('--install_ffmpeg', action='store_true',
                        help='尝试安装ffmpeg（需要管理员权限）')

    args = parser.parse_args()

    # 检查ffmpeg是否安装
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✓ ffmpeg已安装")
    except:
        print("✗ ffmpeg未安装")
        if args.install_ffmpeg:
            print("尝试安装ffmpeg...")
            try:
                import sys
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'ffmpeg-python'], check=True)
                print("✓ ffmpeg-python已安装")
            except Exception as e:
                print(f"安装失败: {e}")
                print("请手动安装ffmpeg:")
                print("  Windows: 下载ffmpeg并添加到PATH")
                print("  Linux: sudo apt-get install ffmpeg")
                print("  Mac: brew install ffmpeg")
        else:
            print("请安装ffmpeg以使用视频修复功能")
            return

    # 处理查询视频
    if os.path.exists(args.queries):
        process_video_list(args.queries)
    else:
        print(f"查询文件不存在: {args.queries}")

    # 处理数据库视频（只处理前100个，避免时间太长）
    if os.path.exists(args.database):
        print(f"\n处理数据库视频（前100个）...")

        # 创建临时文件，只包含前100个
        temp_file = 'temp_database_100.txt'
        with open(args.database, 'r', encoding='utf-8') as f_in, \
                open(temp_file, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                if i >= 100:
                    break
                f_out.write(line)

        process_video_list(temp_file)
        os.remove(temp_file)
    else:
        print(f"数据库文件不存在: {args.database}")


if __name__ == "__main__":
    main()