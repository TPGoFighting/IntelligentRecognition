"""
使用真实背景生成训练数据
Real background-based training data generation
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from core.config_loader import load_config


def load_real_backgrounds(background_dir='test_data'):
    """加载真实背景点云"""
    bg_dir = Path(background_dir)
    bg_files = list(bg_dir.glob('*.ply'))

    backgrounds = []
    for f in bg_files:
        try:
            pcd = o3d.io.read_point_cloud(str(f))
            points = np.asarray(pcd.points)

            # 计算法向量
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            normals = np.asarray(pcd.normals)

            backgrounds.append({
                'name': f.name,
                'points': points,
                'normals': normals
            })
            print(f"加载背景: {f.name} ({len(points)} 点)")
        except Exception as e:
            print(f"无法加载 {f}: {e}")

    return backgrounds


def generate_cylinder(center, radius, height, direction, num_points=1000):
    """生成圆柱体点云"""
    # 生成圆柱体表面点
    theta = np.random.uniform(0, 2*np.pi, num_points)
    z = np.random.uniform(0, height, num_points)

    # 圆柱体坐标
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    points = np.column_stack([x, y, z])

    # 旋转到指定方向
    if not np.allclose(direction, [0, 0, 1]):
        # 计算旋转矩阵
        z_axis = np.array([0, 0, 1])
        direction = direction / np.linalg.norm(direction)

        v = np.cross(z_axis, direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction)

        if s > 1e-6:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
            points = points @ R.T

    # 平移到中心
    points = points + center

    # 计算法向量（径向）
    normals = points - center
    normals[:, 2] = 0  # 只保留径向分量
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)

    return points, normals


def generate_scene_with_real_background(backgrounds, config, num_cylinders=(2, 5)):
    """使用真实背景生成场景"""
    # 随机选择一个背景
    bg = np.random.choice(backgrounds)
    bg_points = bg['points'].copy()
    bg_normals = bg['normals'].copy()

    # 随机采样背景点
    num_bg = min(len(bg_points), 5000)  # 限制背景点数
    bg_indices = np.random.choice(len(bg_points), num_bg, replace=False)
    bg_points = bg_points[bg_indices]
    bg_normals = bg_normals[bg_indices]

    # 计算背景范围
    bg_min = bg_points.min(axis=0)
    bg_max = bg_points.max(axis=0)

    # 生成圆柱体
    n_cyl = np.random.randint(num_cylinders[0], num_cylinders[1] + 1)

    cyl_points_list = []
    cyl_normals_list = []

    for i in range(n_cyl):
        # 在背景范围内随机放置圆柱体
        center = np.random.uniform(bg_min + 1, bg_max - 1)

        # 随机半径和高度
        radius = np.random.uniform(0.2, 1.0)
        height = np.random.uniform(2.0, 8.0)

        # 随机方向（主要是竖直或水平）
        if np.random.rand() < 0.7:
            direction = np.array([0, 0, 1])  # 竖直
        else:
            angle = np.random.uniform(0, 2*np.pi)
            direction = np.array([np.cos(angle), np.sin(angle), 0])  # 水平

        cyl_points, cyl_normals = generate_cylinder(center, radius, height, direction, num_points=1000)
        cyl_points_list.append(cyl_points)
        cyl_normals_list.append(cyl_normals)

    # 合并所有点
    all_points = np.vstack([bg_points] + cyl_points_list)
    all_normals = np.vstack([bg_normals] + cyl_normals_list)

    # 生成标签
    labels = np.zeros(len(all_points), dtype=np.int32)
    offset = len(bg_points)
    for i, cyl_points in enumerate(cyl_points_list):
        labels[offset:offset+len(cyl_points)] = 2  # 圆柱体标签
        offset += len(cyl_points)

    return all_points, all_normals, labels


def main():
    parser = argparse.ArgumentParser(description='使用真实背景生成训练数据')
    parser.add_argument('--config', type=str, default='config/cylinder_aggressive_config.yaml')
    parser.add_argument('--background_dir', type=str, default='test_data')
    parser.add_argument('--output', type=str, default='data/cylinder_real_bg')
    parser.add_argument('--num_train', type=int, default=500)
    parser.add_argument('--num_test', type=int, default=100)

    args = parser.parse_args()

    print("="*60)
    print("使用真实背景生成训练数据")
    print("="*60)

    # 加载配置
    config = load_config(args.config)

    # 加载真实背景
    print(f"\n加载真实背景从: {args.background_dir}")
    backgrounds = load_real_backgrounds(args.background_dir)

    if len(backgrounds) == 0:
        print("\n[错误] 没有找到背景点云文件！")
        print(f"请确保 {args.background_dir} 目录中有 .ply 文件")
        sys.exit(1)

    print(f"\n加载了 {len(backgrounds)} 个背景场景")

    # 生成训练数据
    print(f"\n生成训练数据...")
    train_dir = Path(args.output) / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(args.num_train), desc='训练场景'):
        points, normals, labels = generate_scene_with_real_background(
            backgrounds, config, num_cylinders=(2, 5)
        )

        # 保存
        np.savez(
            train_dir / f'scene_{i:04d}.npz',
            points=points,
            normals=normals,
            labels=labels
        )

    # 生成测试数据
    print(f"\n生成测试数据...")
    test_dir = Path(args.output) / 'test'
    test_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(args.num_test), desc='测试场景'):
        points, normals, labels = generate_scene_with_real_background(
            backgrounds, config, num_cylinders=(2, 5)
        )

        # 保存
        np.savez(
            test_dir / f'scene_{i:04d}.npz',
            points=points,
            normals=normals,
            labels=labels
        )

    print(f"\n{'='*60}")
    print("[成功] 数据生成完成")
    print(f"{'='*60}")
    print(f"训练数据: {train_dir}")
    print(f"测试数据: {test_dir}")
    print(f"\n训练命令:")
    print(f"  python scripts/train_universal.py --config {args.config} --data {args.output}")


if __name__ == "__main__":
    main()
