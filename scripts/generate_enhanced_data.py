"""
增强的数据生成脚本 - 支持更多场景和更大数据量
Enhanced data generation script with more scenes and larger datasets
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from core.config_loader import load_config
from core.data_generator import UniversalDataGenerator


def main():
    parser = argparse.ArgumentParser(description='生成增强的训练数据集')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径 (例如: config/cylinder_only_config.yaml)')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录 (例如: data/cylinder_enhanced)')
    parser.add_argument('--num_train', type=int, default=500,
                       help='训练场景数量 (默认: 500)')
    parser.add_argument('--num_test', type=int, default=100,
                       help='测试场景数量 (默认: 100)')
    parser.add_argument('--objects_min', type=int, default=2,
                       help='每个场景最少对象数 (默认: 2)')
    parser.add_argument('--objects_max', type=int, default=5,
                       help='每个场景最多对象数 (默认: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    # 加载配置
    print(f"加载配置文件: {args.config}")
    config = load_config(args.config)

    # 初始化生成器
    generator = UniversalDataGenerator(config)

    # 定义每个场景的对象数量范围
    objects_per_scene = {}
    for shape_name in config.shape_names:
        objects_per_scene[shape_name] = (args.objects_min, args.objects_max)

    print(f"\n{'='*60}")
    print("数据生成配置:")
    print(f"{'='*60}")
    print(f"  配置文件: {args.config}")
    print(f"  输出目录: {args.output}")
    print(f"  训练场景: {args.num_train}")
    print(f"  测试场景: {args.num_test}")
    print(f"  形状类型: {config.shape_names}")
    print(f"  每场景对象数: {args.objects_min}-{args.objects_max}")
    print(f"  场景类型: {config.scene['type']}")
    print(f"  场景范围: {config.scene['bounds']}")
    print(f"  背景点数: {config.scene['background_density']}")
    print(f"  噪声水平: {config.scene['noise_level']}")

    # 生成训练数据
    print(f"\n{'='*60}")
    print("生成训练数据")
    print(f"{'='*60}")
    train_dir = Path(args.output) / 'train'
    generator.generate_dataset(args.num_train, objects_per_scene, str(train_dir))

    # 生成测试数据
    print(f"\n{'='*60}")
    print("生成测试数据")
    print(f"{'='*60}")
    test_dir = Path(args.output) / 'test'
    generator.generate_dataset(args.num_test, objects_per_scene, str(test_dir))

    print(f"\n{'='*60}")
    print("[成功] 数据生成完成")
    print(f"{'='*60}")
    print(f"训练数据: {train_dir}")
    print(f"测试数据: {test_dir}")
    print(f"\n训练模型命令:")
    print(f"  python scripts/train_universal.py --config {args.config} --data {args.output}")


if __name__ == "__main__":
    main()
