import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
import time


def generate_mock_tunnel_scene():
    """生成模拟的隧道场景数据（包含管道 + 杂乱的隧道壁背景）"""
    print("[1] 正在生成模拟隧道点云数据...")
    # 1. 生成管道 (圆柱体)
    pipe_points = []
    radius = 0.8
    for _ in range(5000):
        h = np.random.uniform(0, 10)
        theta = np.random.uniform(0, 2 * np.pi)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        pipe_points.append([x, y, h])
    pipe_points = np.array(pipe_points)
    # 加入真实世界的扫描噪声
    pipe_points += np.random.normal(0, 0.02, pipe_points.shape)

    # 2. 生成隧道壁 (背景噪点)
    wall_points = np.random.uniform(low=[-3, -3, 0], high=[3, 3, 10], size=(10000, 3))

    # 合并为原始场景点云
    scene_points = np.vstack((pipe_points, wall_points))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_points)
    return pcd, len(pipe_points)


def extract_features_for_pointnet(pcd):
    """PCL/Open3D 预处理：计算法向量，这是喂给 PointNet++ 抵抗复杂背景的神器"""
    print("[2] 正在提取几何特征 (计算法向量)...")
    # 半径内搜索 30 个近邻点来估算当前点的法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    # 统一法向量方向
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # 拼接成 N x 6 的矩阵，这就是未来送入 PointNet++ 的标准输入格式
    pointnet_input = np.hstack((points, normals))
    print(f"    --> PointNet++ 输入张量形状准备完毕: {pointnet_input.shape}")
    return pointnet_input


def simulate_pointnet_inference(pcd, true_pipe_count):
    """模拟 PointNet++ 的推理过程：从 N 个点中分割出属于管道的点"""
    print("[3] 模拟 PointNet++ 语义分割推理...")
    time.sleep(1)  # 模拟 GPU 推理耗时

    # 现实中，这里是 model(tensor) 并做 argmax
    # 这里我们模拟网络成功找到了大部分管道点 (假设前 true_pipe_count 个点是管道)
    # 并不可避免地包含了一些网络的误判噪点
    points = np.asarray(pcd.points)
    predicted_pipe_indices = list(range(int(true_pipe_count * 0.95))) + list(
        range(true_pipe_count, true_pipe_count + 500))

    predicted_pipe_pcd = pcd.select_by_index(predicted_pipe_indices)
    print(f"    --> 网络提取出 {len(predicted_pipe_indices)} 个疑似管道点。")
    return predicted_pipe_pcd


def fit_cylinder_and_verify(pipe_pcd):
    """使用 RANSAC 算法进行严格的圆柱体几何校验和参数提取"""
    print("[4] 启动 RANSAC 圆柱体拟合与校验...")

    points = np.asarray(pipe_pcd.points)

    # 初始化 pyransac3d 的圆柱体拟合器
    cylinder = pyrsc.Cylinder()

    # fit 函数参数：点云，内点距离阈值(容忍的噪声厚度)
    # 返回值：圆心坐标, 轴线方向向量, 半径, 属于圆柱的内点索引
    center, axis, radius, inliers = cylinder.fit(points, thresh=0.05, maxIteration=2000)

    print("\n========== 🎯 最终检测结果 ==========")
    if radius > 0.1 and radius < 2.0:  # 物理规则校验：管道半径必须在合理范围内
        print(f"✅ 校验通过！发现有效管道。")
        print(f"   📏 半径: {radius:.3f} 米")
        print(f"   📍 轴心点: {center}")
        print(f"   🧭 轴线方向: {axis}")
        print(f"   ✨ 有效内点数: {len(inliers)} 个")
    else:
        print(f"❌ 校验失败！拟合半径 {radius:.3f} 米不符合物理常理，判定为网络误识别的背景杂点。")
    print("=====================================\n")


if __name__ == "__main__":
    # 1. 拿数据
    scene_pcd, pipe_point_count = generate_mock_tunnel_scene()

    # 2. 算特征 (给 AI 准备食物)
    network_input = extract_features_for_pointnet(scene_pcd)

    # 3. 过网络 (AI 寻找管道)
    suspected_pipe_pcd = simulate_pointnet_inference(scene_pcd, pipe_point_count)

    # 4. 几何校验 (传统算法兜底把关)
    fit_cylinder_and_verify(suspected_pipe_pcd)