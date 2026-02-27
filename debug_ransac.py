import numpy as np
import pyransac3d as pyrsc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_ransac_cylinder_simple():
    """测试RANSAC圆柱体拟合"""
    print("=== RANSAC圆柱体拟合调试 ===")

    # 生成一个完美的圆柱体
    radius = 0.3
    height = 5.0
    num_points = 2000

    # 生成圆柱体侧面点
    theta = np.linspace(0, 2*np.pi, num_points)
    z = np.linspace(-height/2, height/2, num_points)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    points = np.column_stack([x, y, z])

    print(f"1. 生成完美圆柱体:")
    print(f"   半径: {radius}m")
    print(f"   高度: {height}m")
    print(f"   点数: {len(points)}")
    print(f"   点范围: X[{x.min():.2f}, {x.max():.2f}], Y[{y.min():.2f}, {y.max():.2f}], Z[{z.min():.2f}, {z.max():.2f}]")

    # 添加噪声
    noise_level = 0.01  # 1cm噪声
    points_noisy = points + np.random.randn(*points.shape) * noise_level

    print(f"\n2. 添加噪声 (σ={noise_level}m)")

    # 测试不同RANSAC参数
    print("\n3. 测试不同RANSAC参数:")

    test_cases = [
        {"thresh": 0.01, "maxIteration": 1000, "desc": "严格 (1cm)"},
        {"thresh": 0.02, "maxIteration": 1000, "desc": "中等 (2cm)"},
        {"thresh": 0.05, "maxIteration": 1000, "desc": "宽松 (5cm)"},
        {"thresh": 0.08, "maxIteration": 1000, "desc": "很宽松 (8cm)"},
        {"thresh": 0.10, "maxIteration": 1000, "desc": "非常宽松 (10cm)"},
    ]

    for case in test_cases:
        try:
            cylinder = pyrsc.Cylinder()
            center, axis, radius_fit, inliers = cylinder.fit(
                points_noisy,
                thresh=case["thresh"],
                maxIteration=case["maxIteration"]
            )

            axis_norm = np.linalg.norm(axis)
            axis_unit = axis / axis_norm if axis_norm > 0 else axis

            print(f"\n  参数: {case['desc']}")
            print(f"    拟合半径: {radius_fit:.3f}m (真实: {radius}m)")
            print(f"    拟合中心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
            print(f"    拟合轴方向: [{axis_unit[0]:.3f}, {axis_unit[1]:.3f}, {axis_unit[2]:.3f}]")
            print(f"    内点数: {len(inliers)}/{len(points_noisy)} ({len(inliers)/len(points_noisy)*100:.1f}%)")
            print(f"    半径误差: {abs(radius_fit - radius)/radius*100:.1f}%")

            if 0.2 < radius_fit < 1.5 and len(inliers) > 100:
                print(f"    [OK] 检测成功")
            else:
                print(f"    [FAIL] 检测失败")

        except Exception as e:
            print(f"\n  参数: {case['desc']}")
            print(f"    [ERROR] {e}")

    # 测试点云缩放的影响
    print("\n4. 测试点云缩放:")

    scales = [0.1, 1.0, 10.0, 100.0]
    for scale in scales:
        points_scaled = points_noisy * scale
        try:
            cylinder = pyrsc.Cylinder()
            center, axis, radius_fit, inliers = cylinder.fit(
                points_scaled,
                thresh=0.05 * scale,  # 阈值也按比例缩放
                maxIteration=1000
            )

            # 缩放回原始尺度
            radius_fit_original = radius_fit / scale

            print(f"\n  缩放 {scale}x:")
            print(f"    拟合半径(原始尺度): {radius_fit_original:.3f}m")
            print(f"    内点数: {len(inliers)}/{len(points_scaled)}")

            if 0.2 < radius_fit_original < 1.5:
                print(f"    [OK] 缩放后检测成功")
            else:
                print(f"    [FAIL] 缩放后检测失败")

        except Exception as e:
            print(f"\n  缩放 {scale}x:")
            print(f"    [ERROR] {e}")

    # 可视化
    print("\n5. 可视化点云...")
    try:
        fig = plt.figure(figsize=(12, 8))

        # 3D视图
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(points_noisy[:, 0], points_noisy[:, 1], points_noisy[:, 2],
                   c='b', s=1, alpha=0.5, label='点云')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('圆柱体点云 (带噪声)')
        ax1.legend()

        # XY平面视图
        ax2 = fig.add_subplot(122)
        ax2.scatter(points_noisy[:, 0], points_noisy[:, 1], c='b', s=1, alpha=0.5)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('XY平面投影')
        ax2.axis('equal')

        plt.tight_layout()
        plt.savefig('debug_cylinder.png', dpi=150)
        print(f"  已保存可视化: debug_cylinder.png")
        plt.close()

    except Exception as e:
        print(f"  可视化错误: {e}")

    print("\n=== RANSAC调试完成 ===")

def test_pyransac3d_installation():
    """测试pyransac3d安装"""
    print("\n=== pyransac3d库测试 ===")

    try:
        import pyransac3d as pyrsc
        print(f"1. pyransac3d版本检查: 已导入")

        # 测试基本功能
        test_points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        print(f"2. 测试平面拟合...")
        plane = pyrsc.Plane()
        plane_eq, inliers = plane.fit(test_points, thresh=0.1, maxIteration=1000)
        print(f"   平面方程: {plane_eq}")
        print(f"   内点数: {len(inliers)}")

        print(f"3. 测试圆柱体拟合对象...")
        cylinder = pyrsc.Cylinder()
        print(f"   圆柱体对象创建成功")

        print("[OK] pyransac3d库工作正常")

    except Exception as e:
        print(f"[ERROR] pyransac3d测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pyransac3d_installation()
    test_ransac_cylinder_simple()