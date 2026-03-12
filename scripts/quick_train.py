"""
快速训练脚本 - 用于训练单一形状识别模型
Quick training script for single shape recognition
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 形状配置映射
SHAPE_CONFIGS = {
    'cylinder': 'config/cylinder_only_config.yaml',
    'cuboid': 'config/cuboid_only_config.yaml',
    'sphere': 'config/sphere_only_config.yaml',
    'plane': 'config/plane_only_config.yaml',
}

def main():
    parser = argparse.ArgumentParser(description='快速训练单一形状识别模型')
    parser.add_argument('--shape', type=str, required=True,
                       choices=['cylinder', 'cuboid', 'sphere', 'plane'],
                       help='要训练的形状类型')
    parser.add_argument('--generate_data', action='store_true',
                       help='是否生成新的训练数据')
    parser.add_argument('--num_train', type=int, default=500,
                       help='训练场景数量 (默认: 500)')
    parser.add_argument('--num_test', type=int, default=100,
                       help='测试场景数量 (默认: 100)')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='预训练权重路径 (可选)')

    args = parser.parse_args()

    # 获取配置文件
    config_path = SHAPE_CONFIGS[args.shape]
    data_dir = f'data/{args.shape}_enhanced'
    output_dir = f'models/{args.shape}_model'

    print(f"\n{'='*60}")
    print(f"训练 {args.shape.upper()} 识别模型")
    print(f"{'='*60}")
    print(f"配置文件: {config_path}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")

    # 生成数据（如果需要）
    if args.generate_data:
        print(f"\n{'='*60}")
        print("步骤 1: 生成训练数据")
        print(f"{'='*60}")

        cmd = [
            sys.executable,
            'scripts/generate_enhanced_data.py',
            '--config', config_path,
            '--output', data_dir,
            '--num_train', str(args.num_train),
            '--num_test', str(args.num_test),
            '--objects_min', '2',
            '--objects_max', '5'
        ]

        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print("\n[错误] 数据生成失败")
            sys.exit(1)
    else:
        print(f"\n跳过数据生成，使用现有数据: {data_dir}")
        if not Path(data_dir).exists():
            print(f"\n[错误] 数据目录不存在: {data_dir}")
            print("请使用 --generate_data 参数生成数据")
            sys.exit(1)

    # 训练模型
    print(f"\n{'='*60}")
    print("步骤 2: 训练模型")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        'scripts/train_universal.py',
        '--config', config_path,
        '--data', data_dir,
        '--output', output_dir
    ]

    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n[错误] 模型训练失败")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("[成功] 训练完成")
    print(f"{'='*60}")
    print(f"模型保存位置: {output_dir}")
    print(f"最佳模型: {output_dir}/best_model.pth")
    print(f"最新模型: {output_dir}/latest_model.pth")


if __name__ == "__main__":
    main()
