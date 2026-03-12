"""
快速诊断和修复推理问题
"""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.config_loader import load_config
from core.shape_plugins import get_plugin_class

def test_ransac():
    """测试RANSAC是否能检测简单的圆柱体"""
    print("="*60)
    print("测试RANSAC圆柱体检测")
    print("="*60)

    # 生成一个简单的圆柱体
    print("\n1. 生成测试圆柱体...")
    radius = 0.5
    height = 5.0
    n_points = 1000

    # 圆柱体点云（沿Z轴）
    theta = np.random.uniform(0, 2*np.pi, n_points)
    z = np.random.uniform(0, height, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # 添加少量噪声
    noise = np.random.normal(0, 0.02, (n_points, 3))
    points = np.column_stack([x, y, z]) + noise

    print(f"   生成了 {len(points)} 个点")
    print(f"   真实参数: radius={radius}, height={height}")

    # 加载配置
    print("\n2. 加载配置...")
    config = load_config("config/cylinder_only_config.yaml")

    # 获取圆柱体插件
    print("\n3. 初始化圆柱体插件...")
    plugin_class = get_plugin_class('cylinder')
    shape_config = config.shapes['cylinder']

    plugin_config = {
        'params': shape_config.params,
        'fitting': shape_config.fitting,
        'min_points': shape_config.min_points,
        'scene_bounds': config.scene['bounds']
    }
    plugin = plugin_class(plugin_config)

    # 尝试拟合
    print("\n4. 运行RANSAC拟合...")
    print(f"   阈值: {shape_config.fitting['threshold']}")
    print(f"   最大迭代: {shape_config.fitting['max_iterations']}")
    print(f"   最小内点: {shape_config.fitting['min_inliers']}")

    params = plugin.fit(
        points,
        threshold=shape_config.fitting['threshold'],
        max_iterations=shape_config.fitting['max_iterations']
    )

    if params is None:
        print("\n   [FAIL] RANSAC failed - no cylinder found")
        print("\n5. Try relaxed parameters...")

        # 尝试更宽松的参数
        params = plugin.fit(
            points,
            threshold=0.15,  # 增加阈值
            max_iterations=2000  # 增加迭代次数
        )

        if params is None:
            print("   [FAIL] Still failed with relaxed parameters")
            return False
        else:
            print("   [OK] Success with relaxed parameters!")
    else:
        print("\n   [OK] RANSAC success")

    # 验证参数
    print("\n5. 验证拟合参数...")
    print(f"   拟合参数: {params}")

    is_valid = plugin.validate(params)
    print(f"   Validation: {'[OK] Pass' if is_valid else '[FAIL] Failed'}")

    # 计算内点
    print("\n6. Computing inliers...")
    inliers = plugin.compute_inliers(points, params, shape_config.fitting['threshold'])
    print(f"   Inliers: {len(inliers)} / {len(points)} ({len(inliers)/len(points)*100:.1f}%)")

    if len(inliers) >= shape_config.fitting['min_inliers']:
        print("   [OK] Sufficient inliers")
        return True
    else:
        print(f"   [FAIL] Insufficient inliers (need >= {shape_config.fitting['min_inliers']})")
        return False


def test_inference_engine():
    """测试完整的推理引擎"""
    print("\n" + "="*60)
    print("测试推理引擎")
    print("="*60)

    # 生成测试场景
    print("\n1. 生成测试场景...")

    # 圆柱体
    radius = 0.5
    height = 5.0
    n_cyl = 500
    theta = np.random.uniform(0, 2*np.pi, n_cyl)
    z = np.random.uniform(0, height, n_cyl)
    x = radius * np.cos(theta) + 2.0  # 偏移到x=2
    y = radius * np.sin(theta)
    cyl_points = np.column_stack([x, y, z])

    # 背景点
    n_bg = 500
    bg_points = np.random.uniform([-5, -5, 0], [5, 5, 10], (n_bg, 3))

    # 合并
    all_points = np.vstack([cyl_points, bg_points])

    # 计算法向量（简单估计）
    normals = np.zeros_like(all_points)
    normals[:, 2] = 1.0  # 简单的向上法向量

    print(f"   总点数: {len(all_points)}")
    print(f"   圆柱体点: {n_cyl}")
    print(f"   背景点: {n_bg}")

    # 加载推理引擎
    print("\n2. 加载推理引擎...")
    from core.inference_engine import UniversalInferenceEngine

    config = load_config("config/cylinder_only_config.yaml")

    try:
        engine = UniversalInferenceEngine(
            "models/cylinder_model/best_model.pth",
            config,
            device='cpu'
        )

        print("\n3. 运行推理...")
        detected = engine.infer(all_points, normals)

        print(f"\n检测结果: {len(detected)} 个对象")
        for i, obj in enumerate(detected):
            print(f"  #{i+1}: {obj.shape_type}, 点数={len(obj.points)}, 置信度={obj.confidence:.3f}")

        return len(detected) > 0

    except Exception as e:
        print(f"\n❌ 推理引擎失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("开始诊断...\n")

    # 测试1: RANSAC
    ransac_ok = test_ransac()

    # 测试2: 推理引擎
    inference_ok = test_inference_engine()

    print("\n" + "="*60)
    print("Diagnostic Summary")
    print("="*60)
    print(f"RANSAC test: {'[OK] Pass' if ransac_ok else '[FAIL] Failed'}")
    print(f"Inference engine test: {'[OK] Pass' if inference_ok else '[FAIL] Failed'}")

    if not ransac_ok:
        print("\n建议:")
        print("1. 检查shape_plugins中的RANSAC实现")
        print("2. 调整config中的threshold和min_inliers参数")
        print("3. 确保点云数据格式正确")

    if not inference_ok:
        print("\n建议:")
        print("1. 重新训练模型（当前准确率只有17%）")
        print("2. 使用以下命令训练:")
        print("   python scripts/quick_train.py --shape cylinder --generate_data")
