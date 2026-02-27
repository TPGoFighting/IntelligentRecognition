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
    """生成隧道场景点云：包含多个管道和复杂背景"""
    points_list = []
    normals_list = []
    labels_list = []

    # 生成管道
    for i in range(num_pipes):
        # 随机生成管道参数
        center = np.random.uniform(-5, 5, size=3)
        axis = np.random.uniform(-1, 1, size=3)
        axis[2] = 0.8  # 主要沿z方向
        axis = axis / np.linalg.norm(axis)
        radius = np.random.uniform(0.25, 0.5)
        height = np.random.uniform(4, 8)

        pipe_points = generate_cylinder_points(center, axis, radius, height, num_points=8000)

        # 计算法向量（近似圆柱法向）
        normals = pipe_points - center
        normals = normals - np.outer(np.dot(normals, axis), axis)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.nan_to_num(normals)

        points_list.append(pipe_points)
        normals_list.append(normals)
        labels_list.append(np.ones(len(pipe_points), dtype=np.int32))

    # 生成复杂背景（隧道壁、支架等）
    # 1. 隧道壁（大圆柱）
    tunnel_radius = 3.0
    tunnel_height = 15.0
    tunnel_center = np.array([0, 0, 0])
    tunnel_axis = np.array([0, 0, 1])

    t_bg = np.random.rand(num_background // 2) * 2 * np.pi
    z_bg = np.random.rand(num_background // 2) * tunnel_height - tunnel_height/2
    r_bg = tunnel_radius + np.random.randn(num_background // 2) * 0.2

    x_bg = r_bg * np.cos(t_bg)
    y_bg = r_bg * np.sin(t_bg)

    tunnel_points = np.column_stack([x_bg, y_bg, z_bg])

    # 隧道壁法向量（向外）
    tunnel_normals = tunnel_points - tunnel_center
    tunnel_normals[:, 2] = 0  # 垂直方向不向外
    tunnel_normals = tunnel_normals / np.linalg.norm(tunnel_normals, axis=1, keepdims=True)
    tunnel_normals = np.nan_to_num(tunnel_normals)

    # 2. 随机支架和杂点
    random_points = np.random.uniform(-4, 4, size=(num_background // 4, 3))
    random_points[:, 2] = np.random.uniform(-tunnel_height/2, tunnel_height/2, size=len(random_points))
    random_normals = np.random.uniform(-1, 1, size=(len(random_points), 3))
    random_normals = random_normals / np.linalg.norm(random_normals, axis=1, keepdims=True)

    # 合并背景
    bg_points = np.vstack([tunnel_points, random_points])
    bg_normals = np.vstack([tunnel_normals, random_normals])

    # 添加噪声
    bg_points += np.random.randn(*bg_points.shape) * noise_level

    points_list.append(bg_points)
    normals_list.append(bg_normals)
    labels_list.append(np.zeros(len(bg_points), dtype=np.int32))

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
        # 为管道点（标签1）赋予红色，背景点（标签0）赋予灰色
        colors = np.zeros((len(points), 3))
        pipe_mask = labels == 1
        colors[pipe_mask] = [1, 0, 0]  # 红色管道
        colors[~pipe_mask] = [0.5, 0.5, 0.5]  # 灰色背景
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(filename, pcd)
    print(f"[OK] 保存 PLY: {filename}, {len(points)} 个点")

def generate_test_dataset():
    """生成完整的测试数据集"""
    os.makedirs("test_data", exist_ok=True)
    os.makedirs("test_data/processed", exist_ok=True)

    print("生成测试数据集...")

    # 1. 小规模测试点云 (10k点)
    print("\n1. 生成小规模点云 (10k点)...")
    points_small, normals_small, labels_small = generate_tunnel_scene(
        num_pipes=2, num_background=8000, noise_level=0.05
    )
    # 只取前10000个点
    indices = np.random.choice(len(points_small), 10000, replace=False)
    points_small = points_small[indices]
    normals_small = normals_small[indices]
    labels_small = labels_small[indices]

    save_as_ply("test_data/small_tunnel.ply", points_small, normals_small, labels_small)
    save_as_npy("test_data/processed/train_small.npy", points_small, normals_small, labels_small)

    # 2. 中等规模点云 (50k点)
    print("\n2. 生成中等规模点云 (50k点)...")
    points_medium, normals_medium, labels_medium = generate_tunnel_scene(
        num_pipes=3, num_background=45000, noise_level=0.08
    )
    indices = np.random.choice(len(points_medium), 50000, replace=False)
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