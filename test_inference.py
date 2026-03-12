#!/usr/bin/env python3
"""
推理准确性测试脚本
测试通用形状识别系统在隧道数据上的表现
"""

import numpy as np
import open3d as o3d
import torch
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def load_test_point_cloud(file_path):
    """加载测试点云"""
    print(f"加载点云: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None, None

    pcd = o3d.io.read_point_cloud(file_path)
    if len(pcd.points) == 0:
        print(f"❌ 点云文件为空: {file_path}")
        return None, None

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None

    print(f"✅ 加载成功: {len(points)} 个点")
    print(f"   法线数据: {'有' if normals is not None else '无'}")

    # 如果无法线，估计法线
    if normals is None:
        print("   估计法线...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
        normals = np.asarray(pcd.normals)

    return points, normals

def test_universal_inference(points, normals):
    """测试通用推理引擎"""
    print("\n测试通用推理引擎...")

    try:
        from core.config_loader import load_config
        from core.inference_engine import UniversalInferenceEngine

        # 加载配置
        config_path = "config/shape_config.yaml"
        config = load_config(config_path)
        print(f"✅ 配置加载: {config.shape_names}")

        # 加载模型
        model_path = "models/universal/best_model.pth"
        engine = UniversalInferenceEngine(model_path, config, device='cpu')

        # 运行推理
        print("运行推理...")
        objects = engine.infer(points, normals)

        print(f"\n✅ 推理完成")
        print(f"   检测到的物体数量: {len(objects)}")

        # 统计形状类型
        from collections import Counter
        shape_counts = Counter([obj.shape_type for obj in objects])
        for shape_type, count in shape_counts.items():
            print(f"   {shape_type}: {count}")

        # 显示详细信息
        for i, obj in enumerate(objects, 1):
            print(f"   物体 #{i}: {obj.shape_type}, 置信度: {obj.confidence:.3f}, 点数: {obj.num_points}")
            if obj.shape_type == 'cylinder' and 'radius' in obj.params:
                print(f"       半径: {obj.params['radius']:.3f}m")

        return objects

    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_cylinder_only_inference(points, normals):
    """测试圆柱体专用推理"""
    print("\n测试圆柱体专用推理...")

    try:
        from core.config_loader import load_config
        from core.inference_engine import UniversalInferenceEngine

        # 加载圆柱体专用配置
        config_path = "config/cylinder_only_config.yaml"
        config = load_config(config_path)
        print(f"✅ 配置加载: {config.shape_names}")

        # 加载圆柱体专用模型
        model_path = "models/cylinder_model/best_model.pth"
        engine = UniversalInferenceEngine(model_path, config, device='cpu')

        # 运行推理
        print("运行推理...")
        objects = engine.infer(points, normals)

        print(f"\n✅ 圆柱体推理完成")
        print(f"   检测到的圆柱体数量: {len(objects)}")

        for i, obj in enumerate(objects, 1):
            if 'radius' in obj.params:
                print(f"   圆柱体 #{i}: 半径: {obj.params['radius']:.3f}m, 置信度: {obj.confidence:.3f}, 点数: {obj.num_points}")

        return objects

    except Exception as e:
        print(f"❌ 圆柱体推理失败: {e}")
        return []

def evaluate_accuracy(points, objects):
    """简单准确性评估"""
    print("\n准确性评估...")

    if len(objects) == 0:
        print("❌ 未检测到任何物体")
        return

    # 计算检测点的比例
    total_detected_points = sum(obj.num_points for obj in objects)
    detection_ratio = total_detected_points / len(points)

    print(f"   总点数: {len(points)}")
    print(f"   检测点数: {total_detected_points}")
    print(f"   检测比例: {detection_ratio:.3f}")

    # 检查圆柱体参数的合理性
    cylinder_objects = [obj for obj in objects if obj.shape_type == 'cylinder']
    if cylinder_objects:
        print(f"\n   圆柱体参数检查:")
        for i, obj in enumerate(cylinder_objects, 1):
            if 'radius' in obj.params:
                radius = obj.params['radius']
                status = "合理" if 0.15 <= radius <= 0.8 else "不合理"
                print(f"     圆柱体 #{i}: 半径={radius:.3f}m ({status})")

    # 评估置信度
    avg_confidence = np.mean([obj.confidence for obj in objects])
    print(f"\n   平均置信度: {avg_confidence:.3f}")

    if avg_confidence > 0.5:
        print("   ✅ 置信度良好")
    else:
        print("   ⚠️  置信度较低")

def main():
    print("=" * 70)
    print("IntelligentRecognition 推理准确性测试")
    print("=" * 70)

    # 测试文件列表
    test_files = [
        "test_data/tiny_tunnel.ply",
        "test_data/small_tunnel.ply",
        "test_data/medium_tunnel.ply",
    ]

    for test_file in test_files:
        print(f"\n{'='*70}")
        print(f"测试文件: {test_file}")
        print(f"{'='*70}")

        # 加载点云
        points, normals = load_test_point_cloud(test_file)
        if points is None:
            continue

        # 测试通用推理
        objects = test_universal_inference(points, normals)

        # 评估准确性
        if objects:
            evaluate_accuracy(points, objects)

        # 测试圆柱体专用推理
        cylinder_objects = test_cylinder_only_inference(points, normals)

        print(f"\n测试完成: {test_file}")

    print(f"\n{'='*70}")
    print("所有测试完成")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()