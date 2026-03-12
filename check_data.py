import numpy as np
import os

def check_npy_file(path):
    print(f"检查文件: {path}")
    if not os.path.exists(path):
        print("  文件不存在")
        return

    data = np.load(path)
    print(f"  形状: {data.shape}")
    print(f"  NaN数量: {np.isnan(data).sum()}")
    print(f"  Inf数量: {np.isinf(data).sum()}")
    print(f"  数值范围: min={data.min():.6f}, max={data.max():.6f}")
    print(f"  均值: {data.mean():.6f}, 标准差: {data.std():.6f}")

    # 检查标签分布（第7列）
    if data.shape[1] >= 7:
        labels = data[:, 6]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  标签分布: {dict(zip(unique, counts))}")
        print(f"  管道点比例: {counts[1]/len(labels):.3f}" if len(counts) > 1 else "  仅一个标签")

# 检查所有训练文件
files = [
    "test_data/processed/train_small.npy",
    "test_data/processed/train_medium.npy",
    "test_data/processed/train_tiny.npy"
]

for f in files:
    check_npy_file(f)
    print()