import numpy as np
import open3d as o3d

def fit_cylinder_simple(points, axis_direction=np.array([0, 0, 1])):
    """
    简单的圆柱拟合：假设轴向已知（沿Z轴）
    只需要拟合中心XY和半径
    """
    # 投影到XY平面
    center_xy = points[:, :2].mean(axis=0)

    # 计算径向距离
    radial_distances = np.linalg.norm(points[:, :2] - center_xy, axis=1)

    # 半径就是径向距离的中位数（更鲁棒）
    radius = np.median(radial_distances)

    # 中心Z坐标
    center_z = points[:, 2].mean()
    center = np.array([center_xy[0], center_xy[1], center_z])

    # 计算内点（距离圆柱表面小于阈值的点）
    threshold = 0.05
    distances_to_surface = np.abs(radial_distances - radius)
    inliers = np.where(distances_to_surface < threshold)[0]

    return center, axis_direction, radius, inliers

# 测试
print("测试简单圆柱拟合算法")
pcd = o3d.io.read_point_cloud("test_data/small_tunnel.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

pipe_mask = (colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)
pipe_points = points[pipe_mask]

center, axis, radius, inliers = fit_cylinder_simple(pipe_points)

print(f"\n拟合结果:")
print(f"  中心: {center}")
print(f"  半径: {radius:.3f}m")
print(f"  内点数: {len(inliers)}/{len(pipe_points)} ({len(inliers)/len(pipe_points)*100:.1f}%)")
print(f"  物理验证: {'OK' if 0.15 < radius < 0.8 else 'FAIL'}")
