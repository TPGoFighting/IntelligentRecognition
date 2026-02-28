import numpy as np
import open3d as o3d
import os

def generate_cylinder_points(center, axis, radius, height, num_points=5000):
    """生成圆柱体点云"""
    # 生成圆柱侧面的点
    t = np.random.rand(num_points) * 2 * np.pi
    h = np.random.rand(num_points) * height - height/2

    # 圆柱坐标 (局部)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = h

    points_local = np.column_stack([x, y, z])

    # 旋转到目标轴方向
    axis = axis / np.linalg.norm(axis)
    if np.allclose(axis, [0, 0, 1]):
        points = points_local
    else:
        # 计算旋转矩阵
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, axis)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, axis)

        if s < 1e-6:
            if c > 0:
                points = points_local  # 方向相同
            else:
                points = points_local * np.array([1, -1, -1])  # 反转
        else:
            vx = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
            points = points_local @ R.T

    # 平移到中心
    points = points + center
    return points

def generate_tunnel_scene(num_pipes=3, num_background=50000, noise_level=0.1):
    """生成隧道场景点云：包含多个管道和复杂背景
    标签系统：管道=2，隧道壁=1，其他背景=0
    样本比例目标：管道:隧道壁:其他 ≈ 1:1:1
    """
    points_list = []
    normals_list = []
    labels_list = []

    # 计算平衡的点数分配
    # 目标：管道:隧道壁:其他 = 1:1:1
    total_points = num_pipes * 8000 + num_background
    target_per_class = total_points // 3

    # 管道点数保持不变（每根管道8000点）
    total_pipe_points = num_pipes * 8000
    # 隧道壁和其他背景各占剩余点数的一半
    tunnel_wall_points = target_per_class
    other_background_points = target_per_class

    # 生成管道
    for i in range(num_pipes):
        # 随机生成管道参数 - 修复：管道应该在隧道壁附近
        # 管道中心应该在半径1.5-2.5m的圆环内（隧道壁半径3.0m）
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(1.5, 2.5)  # 距离原点
        center_xy = np.array([np.cos(angle) * distance, np.sin(angle) * distance])
        center_z = np.random.uniform(-4, 4)  # 高度方向

        center = np.array([center_xy[0], center_xy[1], center_z])

        # 管道方向：沿隧道轴向（z方向），不倾斜以便RANSAC能正确拟合
        axis = np.array([0, 0, 1])  # 标准Z轴方向

        radius = np.random.uniform(0.25, 0.4)  # 缩小半径范围，更接近真实管道
        height = np.random.uniform(5, 8)  # 增加高度，确保管道足够长

        pipe_points = generate_cylinder_points(center, axis, radius, height, num_points=8000)

        # 计算法向量（近似圆柱法向）
        # 对于沿Z轴的圆柱，法向量就是径向方向
        normals = pipe_points - center
        normals[:, 2] = 0  # Z方向分量为0
        norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norm_lengths > 1e-6, normals / norm_lengths, normals)
        normals = np.nan_to_num(normals)

        points_list.append(pipe_points)
        normals_list.append(normals)
        labels_list.append(np.full(len(pipe_points), 2, dtype=np.int32))  # 管道标签=2

    # 生成复杂背景（隧道壁、支架等）
    # 1. 隧道壁（大圆柱）- 调整为更大半径，使管道在内部
    tunnel_radius = 4.0  # 增加隧道壁半径，使管道完全在内部
    tunnel_height = 15.0
    tunnel_center = np.array([0, 0, 0])
    tunnel_axis = np.array([0, 0, 1])

    t_bg = np.random.rand(tunnel_wall_points) * 2 * np.pi
    z_bg = np.random.rand(tunnel_wall_points) * tunnel_height - tunnel_height/2
    r_bg = tunnel_radius + np.random.randn(tunnel_wall_points) * 0.2

    x_bg = r_bg * np.cos(t_bg)
    y_bg = r_bg * np.sin(t_bg)

    tunnel_points = np.column_stack([x_bg, y_bg, z_bg])

    # 隧道壁法向量（向外）
    tunnel_normals = tunnel_points - tunnel_center
    tunnel_normals[:, 2] = 0  # 垂直方向不向外
    tunnel_normals = tunnel_normals / np.linalg.norm(tunnel_normals, axis=1, keepdims=True)
    tunnel_normals = np.nan_to_num(tunnel_normals)

    # 2. 随机支架和杂点
    random_points = np.random.uniform(-4, 4, size=(other_background_points, 3))
    random_points[:, 2] = np.random.uniform(-tunnel_height/2, tunnel_height/2, size=len(random_points))
    random_normals = np.random.uniform(-1, 1, size=(len(random_points), 3))
    random_normals = random_normals / np.linalg.norm(random_normals, axis=1, keepdims=True)

    # 分别添加隧道壁（标签1）和随机背景（标签0）
    # 1. 隧道壁（大圆柱）
    tunnel_points += np.random.randn(*tunnel_points.shape) * noise_level
    points_list.append(tunnel_points)
    normals_list.append(tunnel_normals)
    labels_list.append(np.full(len(tunnel_points), 1, dtype=np.int32))  # 隧道壁标签=1

    # 2. 随机支架和杂点（标签0）
    random_points += np.random.randn(*random_points.shape) * noise_level
    points_list.append(random_points)
    normals_list.append(random_normals)
    labels_list.append(np.zeros(len(random_points), dtype=np.int32))  # 其他背景标签=0

    # 合并所有点
    all_points = np.vstack(points_list)
    all_normals = np.vstack(normals_list)
    all_labels = np.concatenate(labels_list)

    # 打乱顺序
    indices = np.random.permutation(len(all_points))
    all_points = all_points[indices]
    all_normals = all_normals[indices]
    all_labels = all_labels[indices]

    return all_points, all_normals, all_labels

def save_as_npy(filename, points, normals, labels):
    """保存为训练数据格式 [X, Y, Z, Nx, Ny, Nz, Label]"""
    data = np.hstack([points, normals, labels.reshape(-1, 1)])
    np.save(filename, data)
    print(f"[OK] 保存 {filename}: {data.shape} 个点")
    return data

def save_as_ply(filename, points, normals=None, labels=None):
    """保存为PLY点云文件"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    if labels is not None:
        # 三分类颜色映射：管道=2(红色)，隧道壁=1(蓝色)，其他背景=0(灰色)
        colors = np.zeros((len(points), 3))

        # 管道点 (标签2) - 红色
        pipe_mask = labels == 2
        colors[pipe_mask] = [1, 0, 0]  # 红色

        # 隧道壁点 (标签1) - 蓝色
        wall_mask = labels == 1
        colors[wall_mask] = [0, 0, 1]  # 蓝色

        # 其他背景点 (标签0) - 灰色
        other_mask = labels == 0
        colors[other_mask] = [0.5, 0.5, 0.5]  # 灰色

        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(filename, pcd)
    print(f"[OK] 保存 PLY: {filename}, {len(points)} 个点")

def generate_test_dataset():
    """生成完整的测试数据集"""
    os.makedirs("test_data", exist_ok=True)
    os.makedirs("test_data/processed", exist_ok=True)

    print("生成测试数据集...")

    # 1. 小规模测试点云 (10k点) - 单管道版本
    print("\n1. 生成小规模点云 (10k点) - 单管道...")
    points_small, normals_small, labels_small = generate_tunnel_scene(
        num_pipes=1, num_background=8000, noise_level=0.05
    )
    # 只取前10000个点
    indices = np.random.choice(len(points_small), 10000, replace=False)
    points_small = points_small[indices]
    normals_small = normals_small[indices]
    labels_small = labels_small[indices]

    save_as_ply("test_data/small_tunnel.ply", points_small, normals_small, labels_small)
    save_as_npy("test_data/processed/train_small.npy", points_small, normals_small, labels_small)

    # 2. 中等规模点云 (50k点) - 单管道版本
    print("\n2. 生成中等规模点云 (50k点) - 单管道...")
    points_medium, normals_medium, labels_medium = generate_tunnel_scene(
        num_pipes=1, num_background=48000, noise_level=0.08
    )
    if len(points_medium) >= 50000:
        indices = np.random.choice(len(points_medium), 50000, replace=False)
    else:
        indices = np.random.choice(len(points_medium), 50000, replace=True)
    points_medium = points_medium[indices]
    normals_medium = normals_medium[indices]
    labels_medium = labels_medium[indices]

    save_as_ply("test_data/medium_tunnel.ply", points_medium, normals_medium, labels_medium)
    save_as_npy("test_data/processed/train_medium.npy", points_medium, normals_medium, labels_medium)

    # 3. 大规模点云 (200k点) - 测试大点云处理能力
    print("\n3. 生成大规模点云 (200k点)...")
    points_large, normals_large, labels_large = generate_tunnel_scene(
        num_pipes=5, num_background=190000, noise_level=0.1
    )
    if len(points_large) >= 200000:
        indices = np.random.choice(len(points_large), 200000, replace=False)
    else:
        print(f"警告: 生成的点数 ({len(points_large)}) 少于200k，使用有放回抽样")
        indices = np.random.choice(len(points_large), 200000, replace=True)
    points_large = points_large[indices]
    normals_large = normals_large[indices]
    labels_large = labels_large[indices]

    save_as_ply("test_data/large_tunnel.ply", points_large, normals_large, labels_large)
    # 不保存为npy，因为文件会很大，仅用于测试GUI加载

    # 4. 极简测试数据 (1k点，用于快速调试)
    print("\n4. 生成极简测试数据 (1k点)...")
    points_tiny, normals_tiny, labels_tiny = generate_tunnel_scene(
        num_pipes=1, num_background=900, noise_level=0.02
    )
    indices = np.random.choice(len(points_tiny), 1000, replace=False)
    points_tiny = points_tiny[indices]
    normals_tiny = normals_tiny[indices]
    labels_tiny = labels_tiny[indices]

    save_as_ply("test_data/tiny_tunnel.ply", points_tiny, normals_tiny, labels_tiny)
    save_as_npy("test_data/processed/train_tiny.npy", points_tiny, normals_tiny, labels_tiny)

    print("\n[OK] 测试数据集生成完成！")
    print("文件结构:")
    print("  test_data/")
    print("    ├── small_tunnel.ply     (10k点，用于常规测试)")
    print("    ├── medium_tunnel.ply    (50k点，中等规模)")
    print("    ├── large_tunnel.ply     (200k点，大点云压力测试)")
    print("    ├── tiny_tunnel.ply      (1k点，快速调试)")
    print("    └── processed/")
    print("        ├── train_small.npy  (训练数据)")
    print("        ├── train_medium.npy")
    print("        └── train_tiny.npy")

if __name__ == "__main__":
    generate_test_dataset()