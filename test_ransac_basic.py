import numpy as np
import pyransac3d as pyrsc

# 生成一个简单的标准圆柱
def generate_simple_cylinder(radius=0.3, height=5.0, num_points=1000):
    """生成一个沿Z轴的标准圆柱"""
    t = np.random.rand(num_points) * 2 * np.pi
    h = np.random.rand(num_points) * height - height/2

    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = h

    points = np.column_stack([x, y, z])
    return points

# 测试1: 完美圆柱
print("测试1: 完美圆柱 (半径0.3m)")
perfect_cylinder = generate_simple_cylinder(radius=0.3, height=5.0, num_points=1000)
cylinder = pyrsc.Cylinder()
center, axis, radius, inliers = cylinder.fit(perfect_cylinder, thresh=0.05, maxIteration=1000)
print(f"  拟合半径: {radius:.3f}m")
print(f"  内点数: {len(inliers)}/{len(perfect_cylinder)}")
print(f"  结果: {'OK' if 0.25 < radius < 0.35 else 'FAIL'}")

# 测试2: 带噪声的圆柱
print("\n测试2: 带噪声圆柱 (噪声0.02m)")
noisy_cylinder = perfect_cylinder + np.random.randn(*perfect_cylinder.shape) * 0.02
cylinder = pyrsc.Cylinder()
center, axis, radius, inliers = cylinder.fit(noisy_cylinder, thresh=0.05, maxIteration=1000)
print(f"  拟合半径: {radius:.3f}m")
print(f"  内点数: {len(inliers)}/{len(noisy_cylinder)}")
print(f"  结果: {'OK' if 0.25 < radius < 0.35 else 'FAIL'}")

# 测试3: 加载实际生成的数据
print("\n测试3: 实际生成的管道数据")
import open3d as o3d
pcd = o3d.io.read_point_cloud("test_data/small_tunnel.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

pipe_mask = (colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)
pipe_points = points[pipe_mask]

print(f"  管道点数: {len(pipe_points)}")

# 尝试不同的RANSAC参数
thresholds = [0.02, 0.05, 0.08, 0.1, 0.15]
for thresh in thresholds:
    cylinder = pyrsc.Cylinder()
    center, axis, radius, inliers = cylinder.fit(pipe_points, thresh=thresh, maxIteration=5000)
    print(f"  阈值{thresh:.2f}: 半径{radius:.3f}m, 内点{len(inliers)}/{len(pipe_points)} ({len(inliers)/len(pipe_points)*100:.1f}%)")
