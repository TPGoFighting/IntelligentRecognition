import numpy as np
import open3d as o3d

# 加载点云
pcd = o3d.io.read_point_cloud("test_data/small_tunnel.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# 提取管道点（红色）
pipe_mask = (colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)
pipe_points = points[pipe_mask]

print(f"管道点数: {len(pipe_points)}")
print(f"管道点范围:")
print(f"  X: [{pipe_points[:, 0].min():.2f}, {pipe_points[:, 0].max():.2f}]")
print(f"  Y: [{pipe_points[:, 1].min():.2f}, {pipe_points[:, 1].max():.2f}]")
print(f"  Z: [{pipe_points[:, 2].min():.2f}, {pipe_points[:, 2].max():.2f}]")

# 检查管道点的分布
center = pipe_points.mean(0)
distances = np.linalg.norm(pipe_points - center, axis=1)
print(f"\n管道点到中心的距离:")
print(f"  平均: {distances.mean():.3f}m")
print(f"  标准差: {distances.std():.3f}m")
print(f"  最小: {distances.min():.3f}m")
print(f"  最大: {distances.max():.3f}m")

# 用真实管道点测试RANSAC
import pyransac3d as pyrsc
cylinder = pyrsc.Cylinder()
try:
    center, axis, radius, inliers = cylinder.fit(pipe_points, thresh=0.08, maxIteration=5000)
    print(f"\n真实管道点的RANSAC拟合:")
    print(f"  半径: {radius:.3f}m")
    print(f"  内点数: {len(inliers)}/{len(pipe_points)}")
    print(f"  内点比例: {len(inliers)/len(pipe_points)*100:.1f}%")
except Exception as e:
    print(f"\nRANSAC失败: {e}")
