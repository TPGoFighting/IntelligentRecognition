"""
激进优化训练脚本 - 专注于提高圆柱体检测准确率
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("="*60)
    print("圆柱体检测 - 激进优化训练")
    print("="*60)
    print("\n优化策略:")
    print("1. 大幅减少背景点（3000 vs 8000）")
    print("2. 增加圆柱体权重（20.0 vs 8.0）")
    print("3. 增加学习率（0.003 vs 0.001）")
    print("4. 增加batch size（32 vs 16）")
    print("5. 增加训练轮数（150 vs 100）")
    print("6. 减少噪声（0.01 vs 0.015）")
    print("7. 使用GPU加速")
    print("\n目标准确率: 70%+")
    print("="*60)

    config = "config/cylinder_aggressive_config.yaml"
    data_dir = "data/cylinder_aggressive"
    output_dir = "models/cylinder_aggressive"

    # 步骤1: 生成优化的训练数据
    print("\n[步骤 1/2] 生成优化的训练数据...")
    print("-"*60)

    cmd = [
        sys.executable,
        "scripts/generate_enhanced_data.py",
        "--config", config,
        "--output", data_dir,
        "--num_train", "800",  # 增加到800个场景
        "--num_test", "150",   # 增加到150个场景
        "--objects_min", "3",  # 每个场景至少3个圆柱体
        "--objects_max", "6"   # 最多6个圆柱体
    ]

    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n[错误] 数据生成失败")
        sys.exit(1)

    # 步骤2: 训练模型
    print("\n[步骤 2/2] 训练优化模型...")
    print("-"*60)

    cmd = [
        sys.executable,
        "scripts/train_universal.py",
        "--config", config,
        "--data", data_dir,
        "--output", output_dir
    ]

    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n[错误] 模型训练失败")
        sys.exit(1)

    # 完成
    print("\n" + "="*60)
    print("[成功] 优化训练完成")
    print("="*60)
    print(f"模型保存位置: {output_dir}")
    print(f"最佳模型: {output_dir}/best_model.pth")
    print(f"\n测试模型:")
    print(f"  python test_optimized_model.py")

if __name__ == "__main__":
    main()
