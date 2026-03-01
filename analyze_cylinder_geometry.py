import numpy as np
import open3d as o3d

# 生成简单圆柱测试
def generate_simple_cylinder(radius=0.3, height=5.0, num_points=1000):
    t = np.random.rand(num_points) * 2 * np.pi
    h = np.random.rand(num_points) * height - height/2
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = h
    return np.column_stack([x, y, z])

# 测试Open3D的圆柱拟合
print("使用Open3D进行圆柱拟合测试")
points = generate_simple_cylinder(radius=0.3, height=5.0, num_points=1000)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Open3D没有直接的圆柱拟合，但我们可以用其他方法
# 方法1: 检查点的径向距离
center_xy = points[:, :2].mean(axis=0)
radial_distances = np.linalg.norm(points[:, :2] - center_xy, axis=1)

print(f"\n方法1: 径向距离分析")
print(f"  中心XY: {center_xy}")
print(f"  径向距离平均: {radial_distances.mean():.3f}m")
print(f"  径向距离标准差: {radial_distances.std():.3f}m")
print(f"  理论半径: 0.300m")
print(f"  误差: {abs(radial_distances.mean() - 0.3):.3f}m")

# 测试实际数据
print("\n\n测试实际生成的管道数据")
pcd_real = o3d.io.read_point_cloud("test_data/small_tunnel.ply")
points_real = np.asarray(pcd_real.points)
colors_real = np.asarray(pcd_real.colors)

pipe_mask = (colors_real[:, 0] > 0.9) & (colors_real[:, 1] < 0.1) & (colors_real[:, 2] < 0.1)
pipe_points = points_real[pipe_mask]

print(f"管道点数: {len(pipe_points)}")

# 分析管道点的几何特征
center_xy = pipe_points[:, :2].mean(axis=0)
radial_distances = np.linalg.norm(pipe_points[:, :2] - center_xy, axis=1)

print(f"\n径向距离分析:")
print(f"  中心XY: {center_xy}")
print(f"  径向距离平均: {radial_distances.mean():.3f}m")
print(f"  径向距离标准差: {radial_distances.std():.3f}m")
print(f"  径向距离范围: [{radial_distances.min():.3f}, {radial_distances.max():.3f}]")

# 检查Z轴方向的分布
print(f"\nZ轴分布:")
print(f"  Z范围: [{pipe_points[:, 2].min():.2f}, {pipe_points[:, 2].max():.2f}]")
print(f"  Z跨度: {pipe_points[:, 2].max() - pipe_points[:, 2].min():.2f}m")

# 如果标准差很大，说明不是圆柱
if radial_distances.std() > 0.5:
    print(f"\n⚠️ 警告: 径向距离标准差过大 ({radial_distances.std():.3f}m)")
    print("   这些点可能不构成一个标准圆柱体！")
