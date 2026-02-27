import numpy as np
import torch
import open3d as o3d
import pyransac3d as pyrsc
import os
import sys
sys.path.append('.')
from models.pointnet2_sem_seg import get_model

def test_simple_cylinder_detection():
    """测试简单的圆柱体检测"""
    print("=== 圆柱体检测调试 ===")

    # 1. 创建一个简单的圆柱体点云
    print("\n1. 生成简单圆柱体点云...")
    radius = 0.3
    height = 5.0
    num_points = 5000

    # 生成圆柱体侧面点
    theta = np.random.rand(num_points) * 2 * np.pi
    z = np.random.rand(num_points) * height - height/2

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # 添加一些噪声
    points = np.column_stack([x, y, z])
    points += np.random.randn(*points.shape) * 0.02  # 2cm噪声

    print(f"  生成的圆柱体: 半径={radius}m, 高度={height}m, 点数={len(points)}")

    # 2. 计算法向量（近似圆柱法向）
    print("\n2. 计算法向量...")
    # 对于圆柱体，法向量是从中心轴指向外的方向
    center = np.array([0, 0, 0])
    axis = np.array([0, 0, 1])  # z轴
    normals = points - center
    normals[:, 2] = 0  # z方向为0，因为是圆柱侧面
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norms > 1e-6, norms, 1.0)

    # 3. 直接测试RANSAC（不经过AI）
    print("\n3. 直接RANSAC拟合测试...")
    try:
        cylinder = pyrsc.Cylinder()
        center_ransac, axis_ransac, radius_ransac, inliers = cylinder.fit(
            points, thresh=0.05, maxIteration=1000
        )
        print(f"  RANSAC结果:")
        print(f"    - 拟合半径: {radius_ransac:.3f}m (真实: {radius}m)")
        print(f"    - 拟合内点数: {len(inliers)}/{len(points)}")
        print(f"    - 拟合误差: {abs(radius_ransac - radius)/radius*100:.1f}%")

        if 0.2 < radius_ransac < 1.5 and len(inliers) > 150:
            print("  [OK] RANSAC成功检测到圆柱体")
        else:
            print("  ❌ RANSAC检测失败")
    except Exception as e:
        print(f"  RANSAC错误: {e}")

    # 4. 检查模型权重
    print("\n4. 检查模型权重...")
    model_path = "checkpoints/best_pipe_model.pth"
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"  模型权重文件: {model_path} ({file_size/1024/1024:.1f} MB)")

        # 尝试加载模型
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  使用设备: {device}")

            model = get_model(num_classes=2).to(device)
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            print("  [OK] 模型加载成功")

            # 5. 测试模型推理
            print("\n5. 测试模型推理...")
            # 准备输入数据: [B, N, 6] -> [B, 6, N]
            features = np.hstack([points, normals])

            # 中心化
            features[:, 0:3] = features[:, 0:3] - features[:, 0:3].mean(0)

            # 转换为tensor
            input_tensor = torch.FloatTensor(features).unsqueeze(0).transpose(2, 1).to(device)
            print(f"  输入形状: {input_tensor.shape}")

            with torch.no_grad():
                device_type = 'cuda' if device.type == 'cuda' else 'cpu'
                with torch.amp.autocast(device_type):
                    pred = model(input_tensor)
                    pred_label = torch.argmax(pred, dim=2).cpu().numpy()[0]

            print(f"  预测结果:")
            unique, counts = np.unique(pred_label, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"    类别 {u}: {c} 个点 ({c/len(pred_label)*100:.1f}%)")

            # 分析管道点（类别1）
            pipe_indices = np.where(pred_label == 1)[0]
            if len(pipe_indices) > 0:
                print(f"  [OK] AI检测到 {len(pipe_indices)} 个管道点 ({len(pipe_indices)/len(pred_label)*100:.1f}%)")

                # 在这些点上测试RANSAC
                pipe_points = points[pipe_indices]
                try:
                    cylinder = pyrsc.Cylinder()
                    center_ai, axis_ai, radius_ai, inliers_ai = cylinder.fit(
                        pipe_points, thresh=0.05, maxIteration=1000
                    )
                    print(f"  AI+RANSAC结果:")
                    print(f"    - 拟合半径: {radius_ai:.3f}m (真实: {radius}m)")
                    print(f"    - 拟合内点数: {len(inliers_ai)}/{len(pipe_points)}")

                    if 0.2 < radius_ai < 1.5 and len(inliers_ai) > 50:
                        print("  [OK] AI+RANSAC成功检测到圆柱体")
                    else:
                        print("  ❌ AI+RANSAC检测失败")
                except Exception as e:
                    print(f"  AI+RANSAC错误: {e}")
            else:
                print("  ❌ AI未检测到任何管道点")

        except Exception as e:
            print(f"  模型加载/推理错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ❌ 模型权重文件不存在: {model_path}")
        print("  可能需要重新训练模型")

    # 6. 测试生成的数据文件
    print("\n6. 检查生成的数据文件...")
    test_files = [
        "test_data/tiny_tunnel.ply",
        "test_data/small_tunnel.ply",
        "test_data/medium_tunnel.ply",
        "test_data/large_tunnel.ply"
    ]

    for file in test_files:
        if os.path.exists(file):
            try:
                pcd = o3d.io.read_point_cloud(file)
                points_count = len(pcd.points)
                has_normals = pcd.has_normals()
                has_colors = pcd.has_colors()
                print(f"  {file}: {points_count}点, 法向量: {has_normals}, 颜色: {has_colors}")
            except Exception as e:
                print(f"  {file}: 读取错误 - {e}")
        else:
            print(f"  {file}: 文件不存在")

    print("\n=== 调试完成 ===")

def test_voting_threshold():
    """测试不同的投票阈值"""
    print("\n=== 投票阈值测试 ===")

    # 模拟投票结果
    np.random.seed(42)
    num_points = 1000

    # 模拟一些投票：假设一些点被多次投票为管道
    all_labels = np.zeros(num_points)
    counts = np.ones(num_points) * 10  # 每个点被投票10次

    # 前300个点是管道（被投票为管道的概率高）
    for i in range(300):
        pipe_votes = np.random.randint(6, 10)  # 6-9次投票为管道
        all_labels[i] = pipe_votes

    # 后700个点是背景（被投票为管道的概率低）
    for i in range(300, num_points):
        pipe_votes = np.random.randint(0, 4)  # 0-3次投票为管道
        all_labels[i] = pipe_votes

    print(f"模拟数据: {num_points}个点")
    print(f"  前300个点: 管道点")
    print(f"  后700个点: 背景点")

    # 测试不同阈值
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    for threshold in thresholds:
        pipe_mask = (all_labels / counts > threshold)
        pipe_points_count = np.sum(pipe_mask)
        true_positives = np.sum(pipe_mask[:300])  # 前300个是真正的管道点
        false_positives = np.sum(pipe_mask[300:])  # 后700个是背景点

        precision = true_positives / pipe_points_count if pipe_points_count > 0 else 0
        recall = true_positives / 300

        print(f"  阈值 {threshold:.1f}: {pipe_points_count}个管道点, "
              f"精确率: {precision:.3f}, 召回率: {recall:.3f}")

if __name__ == "__main__":
    test_simple_cylinder_detection()
    test_voting_threshold()