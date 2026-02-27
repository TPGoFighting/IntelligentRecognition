import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TunnelDataset(Dataset):
    def __init__(self, data_root, num_points=4096, block_size=2.0, train=True):
        """
        隧道管道语义分割数据集
        :param data_root: 存放 .npy 文件的文件夹路径
        :param num_points: 每次喂给网络的点数 (默认 4096)
        :param block_size: 每次裁剪的局部区块大小 (默认 2.0米)
        :param train: 是否为训练模式 (训练模式下加数据增强)
        """
        super().__init__()
        self.num_points = num_points
        self.block_size = block_size
        self.train = train

        # 获取所有 .npy 文件路径
        self.file_paths = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.npy')]
        if len(self.file_paths) == 0:
            print("[警告] 没有找到任何 .npy 数据文件！请检查路径。")

        # 把所有数据加载到内存中 (如果数据极大，这里要改成只存路径，在 __getitem__ 中读取)
        self.data_list = []
        for path in self.file_paths:
            data = np.load(path)  # shape: (N, 7) -> X, Y, Z, Nx, Ny, Nz, Label
            self.data_list.append(data)

        print(f"成功加载 {len(self.data_list)} 个隧道片段。")

    def __len__(self):
        # 为了增加每个 epoch 的迭代次数，我们可以让一个大文件被随机采样多次
        # 假设每个文件我们随机切 100 个 block 作为一次完整的 epoch
        return len(self.data_list) * 100

    def __getitem__(self, idx):
        # 1. 随机选择一个隧道场景
        scene_idx = idx % len(self.data_list)
        points_data = self.data_list[scene_idx]  # (N, 7)

        points = points_data[:, 0:3]  # [X, Y, Z]
        normals = points_data[:, 3:6]  # [Nx, Ny, Nz]
        labels = points_data[:, 6]  # [Label]

        # 确保标签是整数 (0或1)
        labels = labels.astype(np.int64)

        # 数据验证：检查标签范围
        unique_labels = np.unique(labels)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(f"标签数据异常：发现非0/1标签 {unique_labels}")

        # 特征归一化：确保法向量是单位向量（如果数据有问题）
        norm_norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norm_norms > 1e-6, normals / norm_norms, normals)

        # 2. 随机选取一个点作为区块的中心 (Center Point)
        center_idx = np.random.choice(points.shape[0], 1)[0]
        center_point = points[center_idx, :]

        # 3. 找出所有距离中心点在 block_size (例如2米) 范围内的点
        # 这是一个简单的长方体范围切块 (Bounding Box Crop)
        min_bound = center_point - self.block_size / 2.0
        max_bound = center_point + self.block_size / 2.0

        # 寻找在边界内的点的索引
        valid_indices = np.where((points[:, 0] >= min_bound[0]) & (points[:, 0] <= max_bound[0]) &
                                 (points[:, 1] >= min_bound[1]) & (points[:, 1] <= max_bound[1]) &
                                 (points[:, 2] >= min_bound[2]) & (points[:, 2] <= max_bound[2]))[0]

        # 如果切出来的点太少（比如边缘区域），就退而求其次，直接随机取点
        if len(valid_indices) < 100:
            if points.shape[0] >= self.num_points:
                valid_indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            else:
                valid_indices = np.random.choice(points.shape[0], self.num_points, replace=True)

        # 4. 固定点数 (Sampling)：PointNet++ 需要绝对固定的输入数量
        if len(valid_indices) >= self.num_points:
            # 点太多了，无放回随机下采样到 4096
            selected_indices = np.random.choice(valid_indices, self.num_points, replace=False)
        else:
            # 点不够 4096，有放回地重复采样填充 (Pad)
            selected_indices = np.random.choice(valid_indices, self.num_points, replace=True)

        # 提取选中的数据
        selected_points = points[selected_indices, :]
        selected_normals = normals[selected_indices, :]
        selected_labels = labels[selected_indices]

        # 5. 坐标归一化 (极其重要的一步！！！)
        # 神经网络对绝对坐标不敏感。我们需要把这 4096 个点的中心平移到 (0,0,0)
        # 这样网络学习到的是“管道的局部形状”，而不是“管道在地球上的坐标”
        selected_points = selected_points - center_point

        # 6. 拼接特征：网络输入形如 (N, 6)，即 XYZ + 法向量
        # 法向量是相对角度，不需要归一化平移
        features = np.hstack((selected_points, selected_normals))

        # 转换为 PyTorch Tensor 返回
        return torch.FloatTensor(features), torch.LongTensor(selected_labels)


# --- 本地测试代码 ---
if __name__ == "__main__":
    # 假设你在 data 目录下放了 npy 文件
    # 造一个假的 npy 文件用来测试代码是否跑通
    os.makedirs("dummy_data", exist_ok=True)
    dummy_data = np.random.rand(10000, 7)  # 1万个点，7个通道
    np.save("dummy_data/test.npy", dummy_data)

    # 实例化 Dataset
    dataset = TunnelDataset(data_root="dummy_data", num_points=4096)

    # 使用 PyTorch DataLoader 包装
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 模拟一次训练迭代
    for batch_idx, (features, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f" -> 输入特征形状 (Batch, N, C): {features.shape}")  # 预期: [8, 4096, 6]
        print(f" -> 输出标签形状 (Batch, N): {labels.shape}")  # 预期: [8, 4096]
        break  # 测试成功就退出