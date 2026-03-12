"""
测试优化后的模型
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))

from core.config_loader import load_config
from core.inference_engine import UniversalInferenceEngine

def test_model():
    print("="*60)
    print("测试优化后的圆柱体检测模型")
    print("="*60)

    # 生成测试场景
    print("\n1. 生成测试场景...")

    # 创建3个明显的圆柱体
    cylinders = []
    for i in range(3):
        radius = 0.5 + i * 0.3
        height = 5.0
        n_points = 800

        theta = np.random.uniform(0, 2*np.pi, n_points)
        z = np.random.uniform(0, height, n_points)
        x = radius * np.cos(theta) + i * 3.0  # 间隔3米
        y = radius * np.sin(theta)

        cyl_points = np.column_stack([x, y, z])
        cylinders.append(cyl_points)

    # 少量背景点
    bg_points = np.random.uniform([-10, -10, 0], [10, 10, 10], (500, 3))

    # 合并
    all_points = np.vstack(cylinders + [bg_points])
    normals = np.zeros_like(all_points)
    normals[:, 2] = 1.0

    print(f"   总点数: {len(all_points)}")
    print(f"   圆柱体点: {sum(len(c) for c in cylinders)}")
    print(f"   背景点: {len(bg_points)}")
    print(f"   圆柱体比例: {sum(len(c) for c in cylinders)/len(all_points)*100:.1f}%")

    # 加载模型
    print("\n2. 加载优化模型...")
    config = load_config("config/cylinder_aggressive_config.yaml")

    model_path = "models/cylinder_aggressive/best_model.pth"
    if not Path(model_path).exists():
        print(f"\n[错误] 模型不存在: {model_path}")
        print("请先运行: python scripts/train_aggressive.py")
        return False

    engine = UniversalInferenceEngine(model_path, config, device='cuda' if torch.cuda.is_available() else 'cpu')

    # 运行推理
    print("\n3. 运行推理...")
    detected = engine.infer(all_points, normals)

    # 显示结果
    print(f"\n4. 检测结果:")
    print(f"   检测到 {len(detected)} 个圆柱体")

    for i, obj in enumerate(detected):
        print(f"   #{i+1}: {obj.shape_type}, 点数={len(obj.points)}, 置信度={obj.confidence:.3f}")

    # 评估
    print("\n5. 评估:")
    if len(detected) >= 3:
        print("   ✓ 成功检测到至少3个圆柱体")
        return True
    else:
        print(f"   ✗ 只检测到{len(detected)}个圆柱体（期望3个）")
        return False

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
