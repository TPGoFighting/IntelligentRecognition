import numpy as np
import open3d as o3d
import torch
from models.pointnet2_sem_seg import get_model

# 加载点云
pcd = o3d.io.read_point_cloud("test_data/small_tunnel.ply")
points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals) if pcd.has_normals() else None
colors = np.asarray(pcd.colors) if pcd.has_colors() else None

# 获取真实标签
if colors is not None:
    pipe_mask_true = (colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)
else:
    pipe_mask_true = None

# 降采样
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
if not downpcd.has_normals():
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))

points_down = np.asarray(downpcd.points)
normals_down = np.asarray(downpcd.normals)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=3).to(device)
model.load_state_dict(torch.load("checkpoints/best_pipe_model.pth", map_location=device, weights_only=True))
model.eval()

# 滑窗推理
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

# 分析投票比例
valid_mask = counts > 0
vote_ratios = all_labels[valid_mask] / counts[valid_mask]

print("=== 投票比例分析 ===")
print(f"有效点数: {np.sum(valid_mask)}")
print(f"投票比例统计:")
print(f"  最小值: {vote_ratios.min():.3f}")
print(f"  最大值: {vote_ratios.max():.3f}")
print(f"  平均值: {vote_ratios.mean():.3f}")
print(f"  中位数: {np.median(vote_ratios):.3f}")

# 不同阈值下的召回率
thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
print("\n不同阈值下的效果:")
for thresh in thresholds:
    pipe_mask_pred = vote_ratios > thresh
    num_detected = np.sum(pipe_mask_pred)
    print(f"  阈值 {thresh:.2f}: 检测到 {num_detected} 个点 ({num_detected/len(vote_ratios)*100:.1f}%)")
