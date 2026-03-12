#!/usr/bin/env python3
"""
简单测试脚本，验证核心模块是否能正常导入
"""

import sys
import os

def test_imports():
    """测试基本导入"""
    print("测试模块导入...")

    try:
        import numpy as np
        print("✅ numpy:", np.__version__)
    except ImportError as e:
        print("❌ numpy 导入失败:", e)
        return False

    try:
        import torch
        print("✅ torch:", torch.__version__)
    except ImportError as e:
        print("❌ torch 导入失败:", e)
        return False

    try:
        # 尝试导入项目核心模块
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from core.config_loader import load_config
        print("✅ config_loader 导入成功")
    except ImportError as e:
        print("❌ config_loader 导入失败:", e)
        # 继续测试其他模块

    try:
        from models.pointnet2_sem_seg import get_model
        print("✅ pointnet2_sem_seg 导入成功")
    except ImportError as e:
        print("❌ pointnet2_sem_seg 导入失败:", e)

    return True

def test_config():
    """测试配置文件加载"""
    print("\n测试配置文件加载...")

    config_path = "config/shape_config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False

    try:
        from core.config_loader import load_config
        config = load_config(config_path)
        print(f"✅ 配置文件加载成功")
        print(f"   形状列表: {config.shape_names}")
        print(f"   类别数: {config.num_classes}")
        return True
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False

def test_model_file():
    """测试模型文件"""
    print("\n测试模型文件...")

    model_path = "models/universal/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False

    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅ 模型文件存在: {model_path}")
    print(f"   文件大小: {file_size:.2f} MB")

    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"✅ 模型文件可加载")

        if 'model_state_dict' in checkpoint:
            print(f"   包含 model_state_dict")
            num_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
            print(f"   参数量: {num_params:,}")
        else:
            print(f"   直接包含状态字典")
            num_params = sum(p.numel() for p in checkpoint.values())
            print(f"   参数量: {num_params:,}")

        return True
    except Exception as e:
        print(f"❌ 模型文件加载失败: {e}")
        return False

def test_data_files():
    """测试数据文件"""
    print("\n测试数据文件...")

    test_dir = "test_data"
    if not os.path.exists(test_dir):
        print(f"❌ 测试数据目录不存在: {test_dir}")
        return False

    files = os.listdir(test_dir)
    ply_files = [f for f in files if f.endswith('.ply')]

    print(f"✅ 测试数据目录: {test_dir}")
    print(f"   文件数量: {len(files)}")
    print(f"   PLY文件: {len(ply_files)}")

    for file in sorted(ply_files):
        file_path = os.path.join(test_dir, file)
        size = os.path.getsize(file_path) / 1024
        print(f"     {file}: {size:.1f} KB")

    return len(ply_files) > 0

def main():
    print("=" * 60)
    print("IntelligentRecognition 系统基本测试")
    print("=" * 60)

    all_passed = True

    # 测试导入
    if not test_imports():
        all_passed = False

    # 测试配置
    if not test_config():
        all_passed = False

    # 测试模型
    if not test_model_file():
        all_passed = False

    # 测试数据
    if not test_data_files():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有基本测试通过！")
        print("系统已准备好运行图形化界面测试。")
    else:
        print("⚠️  部分测试失败，请检查依赖项安装。")
        print("运行以下命令安装依赖：")
        print("  pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()