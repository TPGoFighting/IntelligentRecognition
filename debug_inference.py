import sys
import os
import numpy as np
import open3d as o3d
import torch
from models.pointnet2_sem_seg import get_model

def test_ai_on_generated_data():
    """测试AI模型在生成数据上的表现"""
    print("=== AI推理诊断测试 ===")

    # 1. 加载生成的点云
    ply_path = "test_data/small_tunnel.ply"
    if not os.path.exists(ply_path):
        print(f"错误: 找不到文件 {ply_path}")
        return

    print(f"加载点云: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None

    print(f"点数: {len(points)}")
    print(f"法线存在: {normals is not None}")

    # 检查标签（颜色）
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        # 管道点应为红色 [1, 0, 0]
        pipe_mask = (colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)
        print(f"真实管道点数: {np.sum(pipe_mask)}")
    else:
        print("警告: 点云没有颜色信息")

    # 2. 加载AI模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model_path = "checkpoints/best_pipe_model.pth"
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return

    model = get_model(num_classes=3).to(device)  # 三分类：管道(2)、隧道壁(1)、其他背景(0)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("模型加载成功")

    # 3. 模拟GUI中的预处理
    print("\n模拟GUI预处理...")
    # 与main.py中相同的预处理
    pcd_processed = o3d.geometry.PointCloud()
    pcd_processed.points = o3d.utility.Vector3dVector(points)

    if normals is not None:
        pcd_processed.normals = o3d.utility.Vector3dVector(normals)
        print("使用文件中的法线")
    else:
        print("计算法线")
        pcd_processed.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30)
        )

    # 下采样（体素大小0.05）
    downpcd = pcd_processed.voxel_down_sample(voxel_size=0.05)
    if not downpcd.has_normals():
        downpcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30)
        )

    points_down = np.asarray(downpcd.points)
    normals_down = np.asarray(downpcd.normals)

    print(f"下采样后点数: {len(points_down)}")

    # 4. 模拟滑窗推理（简化版）
    print("\n模拟滑窗推理...")
    # 使用与main.py相同的参数
    block_size, stride = 3.0, 1.5
    all_labels = np.zeros(len(points_down))
    counts = np.zeros(len(points_down))

    xyz_min, xyz_max = points_down.min(0), points_down.max(0)
    z_range = xyz_max[2] - xyz_min[2]
    num_windows = int(np.ceil(z_range / stride))

    # 限制窗口数量
    max_windows = 200
    if num_windows > max_windows:
        stride = max(z_range / max_windows, block_size * 0.8)
        num_windows = int(np.ceil(z_range / stride))
        print(f"调整步长: {stride:.2f}, 窗口数: {num_windows}")

    z_centers = np.linspace(xyz_min[2] + block_size/2, xyz_max[2] - block_size/2, num_windows)

    processed_windows = 0
    for i, z in enumerate(z_centers):
        if i % 20 == 0:
            print(f"  窗口 {i+1}/{num_windows}")

        mask = (points_down[:, 2] >= z - block_size/2) & (points_down[:, 2] < z + block_size/2)
        idx = np.where(mask)[0]

        if len(idx) < 1024:
            continue

        # 采样固定点数
        if len(idx) >= 4096:
            sel = np.random.choice(idx, 4096, replace=False)
        else:
            sel = np.random.choice(idx, 4096, replace=True)

        # 中心化并提取特征
        block_pts = points_down[sel] - points_down[sel].mean(0)
        block_feat = np.hstack((block_pts, normals_down[sel]))

        # 推理
        input_tensor = torch.FloatTensor(block_feat).unsqueeze(0).transpose(2, 1).to(device)
        with torch.no_grad():
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type):
                pred = model(input_tensor)
                pred_label = torch.argmax(pred, dim=2).cpu().numpy()[0]

        # 统计管道类别(2)的投票数
        all_labels[sel] += (pred_label == 2).astype(np.int32)
        counts[sel] += 1
        processed_windows += 1

    print(f"处理了 {processed_windows}/{num_windows} 个窗口")

    # 5. 分析结果
    valid_mask = counts > 0
    if not np.any(valid_mask):
        print("错误: 没有有效的推理点")
        return

    pipe_mask_inference = (all_labels[valid_mask] / counts[valid_mask] > 0.3)
    pipe_points = points_down[valid_mask][pipe_mask_inference]

    print(f"\n=== 诊断结果 ===")
    print(f"AI检测到的管道点数: {len(pipe_points)}")
    print(f"占总点数的比例: {len(pipe_points)/len(points_down)*100:.1f}%")

    # 如果有真实标签，计算准确率
    if pcd.has_colors():
        # 将下采样点映射回原始点（近似）
        # 简单检查：如果下采样点靠近红色点，则认为是管道点
        from scipy.spatial import KDTree
        tree = KDTree(points)
        distances, indices = tree.query(points_down, k=1)
        true_pipe = pipe_mask[indices]

        # 计算预测管道点中真正管道点的比例
        if len(pipe_points) > 0:
            # 需要映射预测的管道点
            tree2 = KDTree(points_down[valid_mask][pipe_mask_inference])
            # 检查每个预测管道点是否靠近真实管道点
            # 简化：使用下采样点的标签
            pred_pipe_mask = np.zeros(len(points_down), dtype=bool)
            pred_pipe_mask[valid_mask] = pipe_mask_inference

            true_positives = np.sum(pred_pipe_mask & true_pipe)
            precision = true_positives / np.sum(pred_pipe_mask) if np.sum(pred_pipe_mask) > 0 else 0
            recall = true_positives / np.sum(true_pipe) if np.sum(true_pipe) > 0 else 0

            print(f"精确率 (Precision): {precision:.3f}")
            print(f"召回率 (Recall): {recall:.3f}")

    # 6. 检查RANSAC是否能提取圆柱体
    print("\n=== RANSAC测试 ===")
    if len(pipe_points) > 200:
        import pyransac3d as pyrsc
        cylinder = pyrsc.Cylinder()
        try:
            center, axis, radius, inliers = cylinder.fit(pipe_points, thresh=0.04, maxIteration=2000)
            print(f"RANSAC拟合结果:")
            print(f"  半径: {radius:.3f}m")
            print(f"  内点数: {len(inliers)}/{len(pipe_points)}")
            print(f"  物理验证: {0.15 < radius < 0.8 and len(inliers) > 50}")
        except Exception as e:
            print(f"RANSAC拟合失败: {e}")
    else:
        print("管道点数不足，跳过RANSAC测试")

    print("\n=== 诊断完成 ===")

if __name__ == "__main__":
    test_ai_on_generated_data()