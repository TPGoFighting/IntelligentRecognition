#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("快速测试推理引擎...")

try:
    import numpy as np
    import open3d as o3d

    # 加载小点云
    test_file = "test_data/tiny_tunnel.ply"
    print(f"加载 {test_file}...")
    pcd = o3d.io.read_point_cloud(test_file)
    points = np.asarray(pcd.points)

    if not pcd.has_normals():
        pcd.estimate_normals()
    normals = np.asarray(pcd.normals)

    print(f"点数: {len(points)}")

    # 加载配置和模型
    from core.config_loader import load_config
    from core.inference_engine import UniversalInferenceEngine

    config = load_config("config/shape_config.yaml")
    print(f"配置: {config.shape_names}")

    engine = UniversalInferenceEngine("models/universal/best_model.pth", config, device='cpu')
    print("推理引擎初始化成功")

    # 运行推理（限制点数以加快速度）
    sample_points = points[:10000] if len(points) > 10000 else points
    sample_normals = normals[:10000] if len(normals) > 10000 else normals

    print(f"对 {len(sample_points)} 个点进行推理...")
    objects = engine.infer(sample_points, sample_normals)

    print(f"\n检测结果:")
    print(f"  检测到 {len(objects)} 个物体")

    from collections import Counter
    counts = Counter([obj.shape_type for obj in objects])
    for shape, count in counts.items():
        print(f"  {shape}: {count}")

    print("\n✅ 测试成功完成!")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()