"""
Script to generate universal training/testing data.
Replaces the tunnel-specific data generation script.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.config_loader import load_config
from core.data_generator import UniversalDataGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate universal shape recognition dataset')
    parser.add_argument('--config', type=str, default='config/shape_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='data/universal',
                       help='Output directory for generated data')
    parser.add_argument('--num_train', type=int, default=100,
                       help='Number of training scenes')
    parser.add_argument('--num_test', type=int, default=20,
                       help='Number of test scenes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    import numpy as np
    np.random.seed(args.seed)

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Initialize generator
    generator = UniversalDataGenerator(config)

    # Define object counts per scene (min, max)
    # Adjust these based on your needs
    objects_per_scene = {}
    for shape_name in config.shape_names:
        objects_per_scene[shape_name] = (1, 3)  # 1-3 objects of each type per scene

    print(f"\nGenerating dataset with:")
    print(f"  Training scenes: {args.num_train}")
    print(f"  Test scenes: {args.num_test}")
    print(f"  Objects per scene: {objects_per_scene}")

    # Generate training data
    print("\n" + "="*60)
    print("GENERATING TRAINING DATA")
    print("="*60)
    train_dir = Path(args.output) / 'train'
    generator.generate_dataset(args.num_train, objects_per_scene, str(train_dir))

    # Generate test data
    print("\n" + "="*60)
    print("GENERATING TEST DATA")
    print("="*60)
    test_dir = Path(args.output) / 'test'
    generator.generate_dataset(args.num_test, objects_per_scene, str(test_dir))

    print("\n" + "="*60)
    print("[SUCCESS] DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Training data: {train_dir}")
    print(f"Test data: {test_dir}")
    print(f"\nTo train the model, run:")
    print(f"  python scripts/train_universal.py --config {args.config} --data {args.output}")


if __name__ == "__main__":
    main()
