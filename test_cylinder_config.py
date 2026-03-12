#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("测试圆柱体专用配置...")

try:
    import numpy as np
    import open3d as o3d

    # 加载点云
    test_file = "test_data/small_tunnel.ply"
    print(f"加载 {test_file}...")
    pcd = o3d.io.read_point_cloud(test_file)
    points = np.asarray(pcd.points)

    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    normals = np.asarray(pcd.normals)

    print(f"点数: {len(points)}")

    # 加载圆柱体专用配置
    from core.config_loader import load_config
    from core.inference_engine import UniversalInferenceEngine

    config = load_config("config/cylinder_only_config.yaml")
    print(f"配置: {config.shape_names}")
    print(f"圆柱体半径范围: {config.shapes['cylinder'].params['radius_range']}")

    # 使用圆柱体专用模型
    engine = UniversalInferenceEngine("models/cylinder_model/best_model.pth", config, device='cpu')
    print("推理引擎初始化成功")

    # 下采样以加快速度
    if len(points) > 20000:
        pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
        points = np.asarray(pcd_down.points)
        normals = np.asarray(pcd_down.normals) if pcd_down.has_normals() else None
        print(f"下采样后点数: {len(points)}")

    print(f"对 {len(points)} 个点进行推理...")
    objects = engine.infer(points, normals)

    print(f"\n检测结果:")
    print(f"  检测到 {len(objects)} 个圆柱体")

    for i, obj in enumerate(objects, 1):
        if 'radius' in obj.params:
            radius = obj.params['radius']
            status = "合理" if 0.15 <= radius <= 0.8 else "不合理"
            print(f"  圆柱体 #{i}: 半径={radius:.3f}m ({status}), 置信度={obj.confidence:.3f}, 点数={obj.num_points}")

    print("\n✅ 测试完成!")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()