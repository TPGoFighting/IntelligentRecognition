import numpy as np
import os

def analyze_npy_file(path):
    print(f"分析文件: {path}")
    if not os.path.exists(path):
        print("  文件不存在")
        return

    data = np.load(path)
    labels = data[:, 6]

    total = len(labels)
    print(f"  总点数: {total}")

    # 统计各标签数量
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: "其他背景", 1: "隧道壁", 2: "管道"}

    for label, count in zip(unique, counts):
        name = label_names.get(label, f"未知标签{label}")
        percentage = count / total * 100
        print(f"  {name} (标签{label}): {count}点 ({percentage:.1f}%)")

    # 检查比例
    if 2 in unique and 1 in unique and 0 in unique:
        pipe_ratio = counts[np.where(unique == 2)[0][0]] / total
        wall_ratio = counts[np.where(unique == 1)[0][0]] / total
        other_ratio = counts[np.where(unique == 0)[0][0]] / total

        print(f"\n  比例分析:")
        print(f"    管道: {pipe_ratio:.3f}")
        print(f"    隧道壁: {wall_ratio:.3f}")
        print(f"    其他背景: {other_ratio:.3f}")
        print(f"    管道:隧道壁:其他 ≈ {pipe_ratio/wall_ratio:.2f}:1:{other_ratio/wall_ratio:.2f}")

# 分析所有训练文件
files = [
    "test_data/processed/train_small.npy",
    "test_data/processed/train_medium.npy",
    "test_data/processed/train_tiny.npy"
]

for f in files:
    analyze_npy_file(f)
    print()