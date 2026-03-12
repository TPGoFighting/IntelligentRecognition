import numpy as np
import open3d as o3d
import torch
from models.pointnet2_sem_seg import get_model

def test_full_pipeline(ply_file):
    print(f"\n{'='*60}")
    print(f"测试: {ply_file}")
    print(f"{'='*60}")

    # 1. 加载点云
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    if colors is not None:
        pipe_mask_true = (colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)
        print(f"真实管道点数: {np.sum(pipe_mask_true)}")

    # 2. 降采样
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    if not downpcd.has_normals():
        downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))

    points_down = np.asarray(downpcd.points)
    normals_down = np.asarray(downpcd.normals)
    print(f"降采样后点数: {len(points_down)}")

    # 3. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=3).to(device)
    model.load_state_dict(torch.load("checkpoints/best_pipe_model.pth", map_location=device, weights_only=True))
    model.eval()

    # 4. 滑窗推理
    block_size, stride = 3.0, 1.5
    all_labels = np.zeros(len(points_down))
    counts = np.zeros(len(points_down))

    xyz_min, xyz_max = points_down.min(0), points_down.max(0)
    z_range = xyz_max[2] - xyz_min[2]
    num_windows = int(np.ceil(z_range / stride))
    z_centers = np.linspace(xyz_min[2] + block_size/2, xyz_max[2] - block_size/2, num_windows)

    for i, z in enumerate(z_centers):
        mask = (points_down[:, 2] >= z - block_size/2) & (points_down[:, 2] < z + block_size/2)
        idx = np.where(mask)[0]
        if len(idx) < 1024:
            continue

        if len(idx) >= 4096:
            sel = np.random.choice(idx, 4096, replace=False)
        else:
            sel = np.random.choice(idx, 4096, replace=True)

        block_pts = points_down[sel] - points_down[sel].mean(0)
        block_feat = np.hstack((block_pts, normals_down[sel]))

        input_tensor = torch.FloatTensor(block_feat).unsqueeze(0).transpose(2, 1).to(device)
        with torch.no_grad():
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type):
                pred = model(input_tensor)
                pred_label = torch.argmax(pred, dim=2).cpu().numpy()[0]

        all_labels[sel] += (pred_label == 2).astype(np.int32)
        counts[sel] += 1

    # 5. 提取管道点
    valid_mask = counts > 0
    pipe_mask = (all_labels[valid_mask] / counts[valid_mask] > 0.1)
    pipe_points = points_down[valid_mask][pipe_mask]

    print(f"AI检测到管道点: {len(pipe_points)}")

    if len(pipe_points) < 200:
        print("点数不足，无法拟合")
        return

    # 6. 简单圆柱拟合
    center_xy = pipe_points[:, :2].mean(axis=0)
    radial_distances = np.linalg.norm(pipe_points[:, :2] - center_xy, axis=1)
    radius = np.median(radial_distances)
    center_z = pipe_points[:, 2].mean()
    center = np.array([center_xy[0], center_xy[1], center_z])

    threshold = 0.08
    distances_to_surface = np.abs(radial_distances - radius)
    inliers = np.where(distances_to_surface < threshold)[0]

    print(f"\n圆柱拟合结果:")
    print(f"  中心: {center}")
    print(f"  半径: {radius:.3f}m")
    print(f"  内点数: {len(inliers)}/{len(pipe_points)} ({len(inliers)/len(pipe_points)*100:.1f}%)")
    status = "OK" if 0.15 < radius < 0.8 and len(inliers) > 50 else "FAIL"
    print(f"  物理验证: {status}")

# 测试所有数据集
test_full_pipeline("test_data/small_tunnel.ply")
test_full_pipeline("test_data/medium_tunnel.ply")
