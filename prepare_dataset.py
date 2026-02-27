import open3d as o3d
import numpy as np
import os


def process_and_merge(pipe_path, bg_path, output_path):
    print(f"正在处理: {pipe_path} 和 {bg_path}")

    # 1. 读取管道和背景
    pipe_pcd = o3d.io.read_point_cloud(pipe_path)
    bg_pcd = o3d.io.read_point_cloud(bg_path)

    # 2. 计算法向量 (赋予光影几何特征，极大提升 PointNet++ 精度)
    pipe_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    bg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 3. 提取坐标和法向
    pipe_points = np.asarray(pipe_pcd.points)
    pipe_normals = np.asarray(pipe_pcd.normals)
    bg_points = np.asarray(bg_pcd.points)
    bg_normals = np.asarray(bg_pcd.normals)

    # 4. 生成标签 (管道为 1，背景为 0)
    pipe_labels = np.ones((pipe_points.shape[0], 1))
    bg_labels = np.zeros((bg_points.shape[0], 1))

    # 5. 拼接特征：[X, Y, Z, Nx, Ny, Nz, Label]
    pipe_data = np.hstack((pipe_points, pipe_normals, pipe_labels))
    bg_data = np.hstack((bg_points, bg_normals, bg_labels))

    # 6. 合并管道和背景，打乱顺序
    full_data = np.vstack((pipe_data, bg_data))
    np.random.shuffle(full_data)

    # 7. 保存为 NPY 文件
    np.save(output_path, full_data)
    print(f"✅ 成功生成训练数据: {output_path}, 形状: {full_data.shape}")


if __name__ == "__main__":
    # 假设你把抠好的数据放在了 data/raw/ 目录下
    # process_and_merge("data/raw/pipe_01.ply", "data/raw/bg_01.ply", "data/tunnel_train_01.npy")
    print("请配置好路径后解除上方代码的注释运行。")