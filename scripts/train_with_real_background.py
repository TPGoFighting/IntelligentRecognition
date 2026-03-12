"""
使用真实背景训练圆柱体检测模型
Train cylinder detection with REAL backgrounds
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("="*60)
    print("使用真实背景训练圆柱体检测")
    print("="*60)
    print("\n策略:")
    print("1. 使用真实隧道点云作为背景")
    print("2. 在真实背景中插入合成圆柱体")
    print("3. 训练模型识别圆柱体")
    print("\n这样模型能够:")
    print("- 学习真实场景的复杂性")
    print("- 区分圆柱体和真实背景")
    print("- 泛化到实际应用场景")
    print("="*60)

    config = "config/cylinder_aggressive_config.yaml"
    data_dir = "data/cylinder_real_bg"
    output_dir = "models/cylinder_real_bg"

    # 步骤1: 使用真实背景生成训练数据
    print("\n[步骤 1/2] 使用真实背景生成训练数据...")
    print("-"*60)

    cmd = [
        sys.executable,
        "scripts/generate_real_background_data.py",
        "--config", config,
        "--background_dir", "test_data",
        "--output", data_dir,
        "--num_train", "500",
        "--num_test", "100"
    ]

    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n[错误] 数据生成失败")
        sys.exit(1)

    # 步骤2: 训练模型
    print("\n[步骤 2/2] 训练模型...")
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
    print("[成功] 训练完成")
    print("="*60)
    print(f"模型保存位置: {output_dir}")
    print(f"最佳模型: {output_dir}/best_model.pth")
    print(f"\n测试模型:")
    print(f"  python test_real_background_model.py")

if __name__ == "__main__":
    main()
