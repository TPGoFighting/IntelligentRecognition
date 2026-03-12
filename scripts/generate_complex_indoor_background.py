"""
生成复杂的室内场景背景点云
Generate complex indoor scene background point clouds

包含：墙壁、地板、天花板、家具、管道、门窗开口等元素
Includes: walls, floor, ceiling, furniture, pipes, door/window openings
"""

import argparse
import numpy as np
from pathlib import Path
import sys
from typing import List, Tuple, Optional

# 尝试导入open3d，用于PLY文件生成
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("警告: open3d未安装，将无法生成PLY文件。请运行: pip install open3d")

sys.path.append(str(Path(__file__).parent.parent))

# 复用现有的形状生成函数
from core.data_generator import UniversalDataGenerator
from core.config_loader import load_config


def generate_cuboid_points(center: np.ndarray, size: np.ndarray,
                          rotation: Optional[np.ndarray] = None,
                          num_points: int = 1000) -> np.ndarray:
    """
    生成立方体点云

    Args:
        center: 立方体中心 [x, y, z]
        size: 立方体尺寸 [dx, dy, dz]
        rotation: 旋转矩阵 (3x3)，默认为None（无旋转）
        num_points: 总点数

    Returns:
        points: (N, 3) 点云数组
    """
    # 计算每个面的点数
    points_per_face = num_points // 6
    if points_per_face < 10:
        points_per_face = 10

    all_points = []

    # 生成6个面的点
    # 前后面 (x方向)
    for x_sign in [-1, 1]:
        x = center[0] + x_sign * size[0] / 2
        y = np.random.uniform(center[1] - size[1]/2, center[1] + size[1]/2, points_per_face)
        z = np.random.uniform(center[2] - size[2]/2, center[2] + size[2]/2, points_per_face)
        points = np.column_stack([np.full(points_per_face, x), y, z])
        all_points.append(points)

    # 左右面 (y方向)
    for y_sign in [-1, 1]:
        y = center[1] + y_sign * size[1] / 2
        x = np.random.uniform(center[0] - size[0]/2, center[0] + size[0]/2, points_per_face)
        z = np.random.uniform(center[2] - size[2]/2, center[2] + size[2]/2, points_per_face)
        points = np.column_stack([x, np.full(points_per_face, y), z])
        all_points.append(points)

    # 上下面 (z方向)
    for z_sign in [-1, 1]:
        z = center[2] + z_sign * size[2] / 2
        x = np.random.uniform(center[0] - size[0]/2, center[0] + size[0]/2, points_per_face)
        y = np.random.uniform(center[1] - size[1]/2, center[1] + size[1]/2, points_per_face)
        points = np.column_stack([x, y, np.full(points_per_face, z)])
        all_points.append(points)

    points = np.vstack(all_points)

    # 应用旋转
    if rotation is not None:
        points = (points - center) @ rotation.T + center

    return points


def generate_cylinder_points(center: np.ndarray, radius: float, height: float,
                           axis: np.ndarray = None, num_points: int = 1000) -> np.ndarray:
    """
    生成圆柱体点云

    Args:
        center: 圆柱体中心 [x, y, z]
        radius: 半径
        height: 高度
        axis: 轴向，默认为 [0, 0, 1] (Z轴)
        num_points: 总点数

    Returns:
        points: (N, 3) 点云数组
    """
    if axis is None:
        axis = np.array([0, 0, 1])

    axis = axis / np.linalg.norm(axis)

    # 生成圆柱侧面点
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    h = np.random.uniform(-height/2, height/2, num_points)

    # 局部坐标
    x_local = radius * np.cos(theta)
    y_local = radius * np.sin(theta)
    z_local = h

    points_local = np.column_stack([x_local, y_local, z_local])

    # 旋转到目标轴方向
    if not np.allclose(axis, [0, 0, 1]):
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, axis)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, axis)

        if s < 1e-6:
            if c < 0:
                # 反向
                points_local = points_local * np.array([1, -1, -1])
        else:
            vx = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
            points_local = points_local @ R.T

    # 平移到中心
    points = points_local + center

    return points


def generate_complex_indoor_background(bounds: List[List[float]],
                                      num_points: int = 50000,
                                      noise_level: float = 0.01,
                                      add_furniture: bool = True,
                                      add_pipes: bool = True,
                                      add_openings: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成复杂的室内场景背景点云

    Args:
        bounds: 场景范围 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        num_points: 总点数
        noise_level: 高斯噪声标准差
        add_furniture: 是否添加家具
        add_pipes: 是否添加管道
        add_openings: 是否添加门窗开口

    Returns:
        points: (N, 3) 点坐标
        normals: (N, 3) 法向量
        labels: (N,) 标签 (0=结构, 1=家具, 2=管道, 3=开口, 4=杂点)
    """
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]

    room_width = x_max - x_min
    room_depth = y_max - y_min
    room_height = z_max - z_min

    all_points = []
    all_normals = []
    all_labels = []

    # 1. 基本结构 (地板、天花板、墙壁) - 标签 0
    print("生成基本结构...")
    points_per_surface = num_points // 10  # 10% 用于基本结构

    # 地板
    x_floor = np.random.uniform(x_min, x_max, points_per_surface)
    y_floor = np.random.uniform(y_min, y_max, points_per_surface)
    z_floor = np.full(points_per_surface, z_min)
    floor_points = np.column_stack([x_floor, y_floor, z_floor])
    floor_normals = np.tile([0, 0, 1], (points_per_surface, 1))  # 向上

    # 天花板
    z_ceiling = np.full(points_per_surface, z_max)
    ceiling_points = np.column_stack([x_floor, y_floor, z_ceiling])
    ceiling_normals = np.tile([0, 0, -1], (points_per_surface, 1))  # 向下

    # 四面墙
    walls_points = []
    walls_normals = []

    # x_min 墙
    x_wall = np.full(points_per_surface, x_min)
    y_wall = np.random.uniform(y_min, y_max, points_per_surface)
    z_wall = np.random.uniform(z_min, z_max, points_per_surface)
    walls_points.append(np.column_stack([x_wall, y_wall, z_wall]))
    walls_normals.append(np.tile([1, 0, 0], (points_per_surface, 1)))  # 向内

    # x_max 墙
    x_wall = np.full(points_per_surface, x_max)
    walls_points.append(np.column_stack([x_wall, y_wall, z_wall]))
    walls_normals.append(np.tile([-1, 0, 0], (points_per_surface, 1)))  # 向内

    # y_min 墙
    y_wall = np.full(points_per_surface, y_min)
    x_wall = np.random.uniform(x_min, x_max, points_per_surface)
    walls_points.append(np.column_stack([x_wall, y_wall, z_wall]))
    walls_normals.append(np.tile([0, 1, 0], (points_per_surface, 1)))  # 向内

    # y_max 墙
    y_wall = np.full(points_per_surface, y_max)
    walls_points.append(np.column_stack([x_wall, y_wall, z_wall]))
    walls_normals.append(np.tile([0, -1, 0], (points_per_surface, 1)))  # 向内

    # 合并基本结构
    struct_points = np.vstack([floor_points, ceiling_points] + walls_points)
    struct_normals = np.vstack([floor_normals, ceiling_normals] + walls_normals)
    struct_labels = np.zeros(len(struct_points), dtype=np.int32)

    all_points.append(struct_points)
    all_normals.append(struct_normals)
    all_labels.append(struct_labels)

    current_points = len(struct_points)

    # 2. 家具 (立方体) - 标签 1
    if add_furniture and current_points < num_points * 0.7:
        print("添加家具...")
        furniture_points = []
        furniture_normals = []

        # 随机生成2-5个家具
        n_furniture = np.random.randint(2, 6)
        for i in range(n_furniture):
            # 随机位置（避开墙壁）
            center_x = np.random.uniform(x_min + 1.0, x_max - 1.0)
            center_y = np.random.uniform(y_min + 1.0, y_max - 1.0)
            center_z = z_min + np.random.uniform(0.5, 2.0)  # 家具高度

            # 随机尺寸
            size_x = np.random.uniform(0.5, 2.0)
            size_y = np.random.uniform(0.5, 2.0)
            size_z = np.random.uniform(0.5, 2.0)

            # 生成立方体
            points = generate_cuboid_points(
                center=np.array([center_x, center_y, center_z]),
                size=np.array([size_x, size_y, size_z]),
                num_points=500
            )

            # 简单法向量（实际应该计算每个点的法向量，这里简化）
            normals = np.random.randn(len(points), 3)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

            furniture_points.append(points)
            furniture_normals.append(normals)

        if furniture_points:
            furniture_points = np.vstack(furniture_points)
            furniture_normals = np.vstack(furniture_normals)
            furniture_labels = np.ones(len(furniture_points), dtype=np.int32)

            all_points.append(furniture_points)
            all_normals.append(furniture_normals)
            all_labels.append(furniture_labels)
            current_points += len(furniture_points)

    # 3. 管道系统 - 标签 2
    if add_pipes and current_points < num_points * 0.8:
        print("添加管道系统...")
        pipe_points = []
        pipe_normals = []

        # 随机生成1-3个管道
        n_pipes = np.random.randint(1, 4)
        for i in range(n_pipes):
            # 管道通常靠近墙壁或天花板
            if np.random.rand() < 0.5:
                # 垂直管道
                center_x = np.random.uniform(x_min + 0.5, x_max - 0.5)
                center_y = np.random.uniform(y_min + 0.5, y_max - 0.5)
                center_z = (z_min + z_max) / 2
                axis = np.array([0, 0, 1])  # 垂直
                height = room_height - 1.0
            else:
                # 水平管道（沿墙壁）
                wall_choice = np.random.randint(0, 4)
                if wall_choice == 0:  # x_min墙
                    center_x = x_min + 0.3
                    center_y = np.random.uniform(y_min + 1.0, y_max - 1.0)
                    axis = np.array([0, 1, 0])  # 沿Y轴
                    height = room_depth - 2.0
                elif wall_choice == 1:  # x_max墙
                    center_x = x_max - 0.3
                    center_y = np.random.uniform(y_min + 1.0, y_max - 1.0)
                    axis = np.array([0, 1, 0])
                    height = room_depth - 2.0
                elif wall_choice == 2:  # y_min墙
                    center_x = np.random.uniform(x_min + 1.0, x_max - 1.0)
                    center_y = y_min + 0.3
                    axis = np.array([1, 0, 0])  # 沿X轴
                    height = room_width - 2.0
                else:  # y_max墙
                    center_x = np.random.uniform(x_min + 1.0, x_max - 1.0)
                    center_y = y_max - 0.3
                    axis = np.array([1, 0, 0])
                    height = room_width - 2.0

                center_z = z_min + np.random.uniform(1.0, room_height - 1.0)

            radius = np.random.uniform(0.05, 0.15)

            points = generate_cylinder_points(
                center=np.array([center_x, center_y, center_z]),
                radius=radius,
                height=height,
                axis=axis,
                num_points=300
            )

            # 简单法向量（圆柱径向）
            normals = points - np.array([center_x, center_y, center_z])
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

            pipe_points.append(points)
            pipe_normals.append(normals)

        if pipe_points:
            pipe_points = np.vstack(pipe_points)
            pipe_normals = np.vstack(pipe_normals)
            pipe_labels = np.full(len(pipe_points), 2, dtype=np.int32)

            all_points.append(pipe_points)
            all_normals.append(pipe_normals)
            all_labels.append(pipe_labels)
            current_points += len(pipe_points)

    # 4. 门窗开口 - 标签 3
    if add_openings and current_points < num_points * 0.9:
        print("添加门窗开口...")
        opening_points = []
        opening_normals = []

        # 在墙上随机创建开口
        n_openings = np.random.randint(1, 3)
        for i in range(n_openings):
            wall_idx = np.random.randint(0, 4)  # 选择哪面墙

            # 开口尺寸
            width = np.random.uniform(0.8, 2.0)
            height = np.random.uniform(1.5, 2.2)

            if wall_idx == 0:  # x_min墙
                x = x_min
                y_center = np.random.uniform(y_min + width/2, y_max - width/2)
                z_center = z_min + np.random.uniform(height/2, room_height - height/2)

                # 在开口区域生成点（模拟缺失）
                # 这里我们生成一些点来表示开口边缘
                n_edge = 100
                theta = np.linspace(0, 2*np.pi, n_edge)
                y_edge = y_center + (width/2) * np.cos(theta)
                z_edge = z_center + (height/2) * np.sin(theta)
                x_edge = np.full(n_edge, x)

                points = np.column_stack([x_edge, y_edge, z_edge])
                normals = np.tile([1, 0, 0], (n_edge, 1))  # 向内

            elif wall_idx == 1:  # x_max墙
                x = x_max
                y_center = np.random.uniform(y_min + width/2, y_max - width/2)
                z_center = z_min + np.random.uniform(height/2, room_height - height/2)

                n_edge = 100
                theta = np.linspace(0, 2*np.pi, n_edge)
                y_edge = y_center + (width/2) * np.cos(theta)
                z_edge = z_center + (height/2) * np.sin(theta)
                x_edge = np.full(n_edge, x)

                points = np.column_stack([x_edge, y_edge, z_edge])
                normals = np.tile([-1, 0, 0], (n_edge, 1))

            elif wall_idx == 2:  # y_min墙
                y = y_min
                x_center = np.random.uniform(x_min + width/2, x_max - width/2)
                z_center = z_min + np.random.uniform(height/2, room_height - height/2)

                n_edge = 100
                theta = np.linspace(0, 2*np.pi, n_edge)
                x_edge = x_center + (width/2) * np.cos(theta)
                z_edge = z_center + (height/2) * np.sin(theta)
                y_edge = np.full(n_edge, y)

                points = np.column_stack([x_edge, y_edge, z_edge])
                normals = np.tile([0, 1, 0], (n_edge, 1))

            else:  # y_max墙
                y = y_max
                x_center = np.random.uniform(x_min + width/2, x_max - width/2)
                z_center = z_min + np.random.uniform(height/2, room_height - height/2)

                n_edge = 100
                theta = np.linspace(0, 2*np.pi, n_edge)
                x_edge = x_center + (width/2) * np.cos(theta)
                z_edge = z_center + (height/2) * np.sin(theta)
                y_edge = np.full(n_edge, y)

                points = np.column_stack([x_edge, y_edge, z_edge])
                normals = np.tile([0, -1, 0], (n_edge, 1))

            opening_points.append(points)
            opening_normals.append(normals)

        if opening_points:
            opening_points = np.vstack(opening_points)
            opening_normals = np.vstack(opening_normals)
            opening_labels = np.full(len(opening_points), 3, dtype=np.int32)

            all_points.append(opening_points)
            all_normals.append(opening_normals)
            all_labels.append(opening_labels)
            current_points += len(opening_points)

    # 5. 随机杂点 - 标签 4
    print("添加随机杂点...")
    remaining_points = max(0, num_points - current_points)
    if remaining_points > 0:
        random_points = np.random.uniform(
            low=[x_min, y_min, z_min],
            high=[x_max, y_max, z_max],
            size=(remaining_points, 3)
        )

        random_normals = np.random.randn(remaining_points, 3)
        random_normals = random_normals / np.linalg.norm(random_normals, axis=1, keepdims=True)
        random_labels = np.full(remaining_points, 4, dtype=np.int32)

        all_points.append(random_points)
        all_normals.append(random_normals)
        all_labels.append(random_labels)

    # 合并所有点
    points = np.vstack(all_points)
    normals = np.vstack(all_normals)
    labels = np.concatenate(all_labels)

    # 添加噪声
    if noise_level > 0:
        points += np.random.randn(*points.shape) * noise_level

    # 打乱顺序
    indices = np.random.permutation(len(points))
    points = points[indices]
    normals = normals[indices]
    labels = labels[indices]

    print(f"生成完成: {len(points)} 个点")
    print(f"标签分布:")
    label_names = ["结构", "家具", "管道", "开口", "杂点"]
    for i, name in enumerate(label_names):
        count = np.sum(labels == i)
        print(f"  {name}({i}): {count} 点 ({count/len(points)*100:.1f}%)")

    return points, normals, labels


def save_as_ply(filename: str, points: np.ndarray, normals: Optional[np.ndarray] = None,
               labels: Optional[np.ndarray] = None) -> bool:
    """
    保存为PLY点云文件

    Returns:
        bool: 是否成功保存
    """
    if not OPEN3D_AVAILABLE:
        print(f"[跳过] 无法保存PLY: {filename}，open3d未安装")
        return False

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        if labels is not None:
            # 颜色映射
            colors = np.zeros((len(points), 3))

            # 结构 - 灰色
            mask = labels == 0
            colors[mask] = [0.5, 0.5, 0.5]

            # 家具 - 棕色
            mask = labels == 1
            colors[mask] = [0.65, 0.5, 0.39]

            # 管道 - 蓝色
            mask = labels == 2
            colors[mask] = [0, 0, 1]

            # 开口 - 黄色
            mask = labels == 3
            colors[mask] = [1, 1, 0]

            # 杂点 - 浅灰色
            mask = labels == 4
            colors[mask] = [0.8, 0.8, 0.8]

            pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(filename, pcd)
        print(f"[OK] 保存 PLY: {filename}, {len(points)} 个点")
        return True
    except Exception as e:
        print(f"[错误] 保存PLY失败 {filename}: {e}")
        return False


def save_as_npy(filename: str, points: np.ndarray, normals: np.ndarray, labels: np.ndarray) -> None:
    """保存为训练数据格式 [X, Y, Z, Nx, Ny, Nz, Label]"""
    data = np.hstack([points, normals, labels.reshape(-1, 1)])
    np.save(filename, data)
    print(f"[OK] 保存 NPY: {filename}, {data.shape} 个点")


def main():
    parser = argparse.ArgumentParser(description='生成复杂的室内场景背景点云')
    parser.add_argument('--output_dir', type=str, default='complex_indoor_background',
                       help='输出目录 (默认: complex_indoor_background)')
    parser.add_argument('--num_points', type=int, default=50000,
                       help='总点数 (默认: 50000)')
    parser.add_argument('--noise_level', type=float, default=0.01,
                       help='噪声水平 (默认: 0.01)')
    parser.add_argument('--bounds', type=float, nargs=6,
                       default=[-5, 5, -5, 5, 0, 3],
                       help='场景范围 [x_min, x_max, y_min, y_max, z_min, z_max] (默认: -5 5 -5 5 0 3)')
    parser.add_argument('--no_furniture', action='store_true',
                       help='不添加家具')
    parser.add_argument('--no_pipes', action='store_true',
                       help='不添加管道')
    parser.add_argument('--no_openings', action='store_true',
                       help='不添加门窗开口')
    parser.add_argument('--num_scenes', type=int, default=1,
                       help='生成场景数量 (默认: 1)')
    parser.add_argument('--no_ply', action='store_true',
                       help='不生成PLY文件（即使open3d可用）')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)

    bounds = [[args.bounds[0], args.bounds[1]],
              [args.bounds[2], args.bounds[3]],
              [args.bounds[4], args.bounds[5]]]

    print("="*60)
    print("复杂室内场景背景点云生成器")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print(f"场景范围: {bounds}")
    print(f"总点数: {args.num_points}")
    print(f"噪声水平: {args.noise_level}")
    print(f"场景数量: {args.num_scenes}")
    print(f"添加家具: {not args.no_furniture}")
    print(f"添加管道: {not args.no_pipes}")
    print(f"添加门窗: {not args.no_openings}")
    print("="*60)

    for scene_idx in range(args.num_scenes):
        print(f"\n生成场景 {scene_idx + 1}/{args.num_scenes}...")

        points, normals, labels = generate_complex_indoor_background(
            bounds=bounds,
            num_points=args.num_points,
            noise_level=args.noise_level,
            add_furniture=not args.no_furniture,
            add_pipes=not args.no_pipes,
            add_openings=not args.no_openings
        )

        # 保存文件
        base_name = f"indoor_scene_{scene_idx:04d}"

        # PLY文件（如果open3d可用且未指定--no_ply）
        if not args.no_ply:
            ply_file = output_dir / f"{base_name}.ply"
            save_as_ply(str(ply_file), points, normals, labels)
        else:
            print(f"[跳过] 不生成PLY文件（--no_ply参数指定）")

        # NPY文件
        npy_file = processed_dir / f"{base_name}.npy"
        save_as_npy(str(npy_file), points, normals, labels)

    print("\n" + "="*60)
    print("[成功] 所有场景生成完成！")
    print("="*60)

    if not args.no_ply and OPEN3D_AVAILABLE:
        print(f"PLY文件: {output_dir}/*.ply")
    elif args.no_ply:
        print(f"PLY文件: 未生成（--no_ply参数指定）")
    else:
        print(f"PLY文件: 未生成（open3d未安装）")

    print(f"NPY文件: {processed_dir}/*.npy")
    print("\n标签说明:")
    print("  0: 结构 (墙壁、地板、天花板)")
    print("  1: 家具 (桌子、柜子等)")
    print("  2: 管道 (水管、通风管)")
    print("  3: 开口 (门、窗)")
    print("  4: 杂点 (随机点)")
    print("="*60)


if __name__ == "__main__":
    main()