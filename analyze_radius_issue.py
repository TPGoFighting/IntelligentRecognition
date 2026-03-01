import numpy as np
import open3d as o3d
import os
import sys

def analyze_test_data():
    """分析测试数据中的管道和隧道壁尺寸问题"""
    print("=== 点云数据尺寸分析 ===")

    # 加载small_tunnel.ply
    ply_path = "test_data/small_tunnel.ply"
    if not os.path.exists(ply_path):
        print(f"错误: 找不到文件 {ply_path}")
        return

    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    # 通过颜色区分管道和背景
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    if colors is None:
        print("错误: 点云没有颜色信息")
        return

    # 管道点：红色 [1, 0, 0]
    pipe_mask = (colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)
    # 隧道壁点：灰色 [0.5, 0.5, 0.5]
    wall_mask = (colors[:, 0] > 0.4) & (colors[:, 0] < 0.6) & \
                (colors[:, 1] > 0.4) & (colors[:, 1] < 0.6) & \
                (colors[:, 2] > 0.4) & (colors[:, 2] < 0.6)

    pipe_points = points[pipe_mask]
    wall_points = points[wall_mask]
    other_points = points[~(pipe_mask | wall_mask)]

    print(f"总点数: {len(points)}")
    print(f"管道点数: {len(pipe_points)} ({len(pipe_points)/len(points)*100:.1f}%)")
    print(f"隧道壁点数: {len(wall_points)} ({len(wall_points)/len(points)*100:.1f}%)")
    print(f"其他点数: {len(other_points)} ({len(other_points)/len(points)*100:.1f}%)")

    # 计算到原点的距离（近似半径）
    if len(pipe_points) > 0:
        pipe_dist = np.linalg.norm(pipe_points[:, :2], axis=1)  # XY平面距离
        print(f"\n管道点到原点距离:")
        print(f"  最小值: {pipe_dist.min():.3f}m")
        print(f"  最大值: {pipe_dist.max():.3f}m")
        print(f"  平均值: {pipe_dist.mean():.3f}m")
        print(f"  中位数: {np.median(pipe_dist):.3f}m")

        # 管道半径（根据生成脚本，应该是0.25-0.5m）
        print(f"\n预期管道半径范围: 0.25-0.5m")

    if len(wall_points) > 0:
        wall_dist = np.linalg.norm(wall_points[:, :2], axis=1)  # XY平面距离
        print(f"\n隧道壁点到原点距离:")
        print(f"  最小值: {wall_dist.min():.3f}m")
        print(f"  最大值: {wall_dist.max():.3f}m")
        print(f"  平均值: {wall_dist.mean():.3f}m")
        print(f"  中位数: {np.median(wall_dist):.3f}m")

        # 隧道壁半径（根据生成脚本，应该是3.0m ± 噪声）
        print(f"\n预期隧道壁半径: 3.0m")

    # 分析AI检测的问题
    print("\n=== AI检测问题分析 ===")
    print("问题: AI可能将隧道壁误判为管道")
    print("原因:")
    print("  1. 隧道壁（半径3m）也是圆柱体，几何特征与管道（半径0.25-0.5m）相似")
    print("  2. 隧道壁点数多，可能被模型检测为'管道'")
    print("  3. 法线特征: 隧道壁法线向外，管道法线向内/向外，可能混淆")

    # 检查法线方向
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        if len(pipe_points) > 0:
            pipe_normals = normals[pipe_mask]
            # 法线是否指向圆柱中心？
            pipe_to_center = -pipe_points / np.linalg.norm(pipe_points, axis=1, keepdims=True)
            pipe_norm_dot = np.sum(pipe_normals * pipe_to_center, axis=1)
            print(f"\n管道法线分析:")
            print(f"  法线朝向中心的比例: {(pipe_norm_dot > 0).sum()}/{len(pipe_normals)}")

        if len(wall_points) > 0:
            wall_normals = normals[wall_mask]
            # 隧道壁法线应该向外
            wall_to_center = -wall_points / np.linalg.norm(wall_points, axis=1, keepdims=True)
            wall_norm_dot = np.sum(wall_normals * wall_to_center, axis=1)
            print(f"\n隧道壁法线分析:")
            print(f"  法线朝外的比例: {(wall_norm_dot > 0).sum()}/{len(wall_normals)}")

    print("\n=== 解决方案 ===")
    print("1. 立即修复:")
    print("   - 收紧物理过滤: if 0.15 < radius < 0.8 and len(inliers) > 50")
    print("   - 降低RANSAC阈值: thresh=0.03 (更严格)")
    print("   - 添加密度检查: 管道点应该更密集")

    print("\n2. 数据生成改进:")
    print("   - 减小隧道壁半径: 1.5-2.0m (与管道更易区分)")
    print("   - 增加管道与隧道壁的距离")
    print("   - 添加更多非圆柱背景，减少圆柱混淆")

    print("\n3. 模型训练改进:")
    print("   - 增加管道与隧道壁的样本区分")
    print("   - 使用多尺度特征，学习大小圆柱的区别")

    print("\n=== 建议操作顺序 ===")
    print("1. 先收紧物理过滤条件 (最快见效)")
    print("2. 修改generate_test_data.py，优化生成数据")
    print("3. 用新数据重新训练模型")
    print("4. 测试实际效果")

def check_ransac_on_real_pipes():
    """在真实的管道点上测试RANSAC"""
    print("\n=== 在纯管道点上测试RANSAC ===")

    ply_path = "test_data/small_tunnel.ply"
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    if colors is None:
        return

    # 只取管道点
    pipe_mask = (colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)
    pure_pipe_points = points[pipe_mask]

    print(f"纯管道点数: {len(pure_pipe_points)}")

    # 尝试RANSAC拟合
    try:
        import pyransac3d as pyrsc
        cylinder = pyrsc.Cylinder()
        center, axis, radius, inliers = cylinder.fit(
            pure_pipe_points, thresh=0.05, maxIteration=2000
        )

        print(f"RANSAC拟合结果 (纯管道点):")
        print(f"  半径: {radius:.3f}m")
        print(f"  内点数: {len(inliers)}/{len(pure_pipe_points)}")
        print(f"  物理验证 (0.15-0.8m): {0.15 < radius < 0.8}")

    except Exception as e:
        print(f"RANSAC拟合失败: {e}")
        print("可能原因: 管道点来自多个不同管道，不是一个圆柱体")

if __name__ == "__main__":
    analyze_test_data()
    check_ransac_on_real_pipes()