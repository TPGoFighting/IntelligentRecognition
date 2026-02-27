import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.pointnet2_sem_seg import get_model, get_loss
from core.dataset import TunnelDataset

# 假设你已经将 PointNet++ 的模型代码放在了 models 文件夹下
# 这里提供一个占位导入，你需要根据你克隆的仓库实际结构修改
# from models.pointnet2_sem_seg import get_model, get_loss

def train():
    print("🚀 启动隧道管道 PointNet++ 语义分割训练...")

    # 1. 基础配置
    data_root = "data/processed"
    batch_size = 16
    epochs = 50
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 当前使用计算设备: {device}")

    # 2. 加载数据集
    # 2. 加载数据集 (这行代码本身其实不用大改，主要是它吃进去的 data_root 变了)
    train_dataset = TunnelDataset(
        data_root=data_root,
        num_points=4096,  # 每次塞给显卡的点数，4096是经典配置
        block_size=3.0,  # 💡 进阶建议：我把你原来的 2.0 改成了 3.0
        train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 替换 train.py 中的这一部分


    # 3. 初始化模型与损失函数 (2个类别：背景=0，管道=1)
    print("🧠 正在初始化 PointNet++ 网络...")
    model = get_model(num_classes=2).to(device)
    criterion = get_loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. 训练循环
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (points, labels) in enumerate(train_loader):
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()

            # 【核心修改】：PyTorch 的 Conv1d 要求输入是 [Batch, Channels, N]
            # 我们 DataLoader 输出的是 [Batch, N, 6]，所以必须 transpose 一下！
            points = points.transpose(2, 1)

            # 前向传播
            predictions = model(points)

            # 计算损失
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # ... 后续保存权重的逻辑保持不变 ...

        avg_loss = total_loss / len(train_loader)
        print(f"🏁 Epoch {epoch + 1} 结束，平均 Loss: {avg_loss:.4f}")

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/best_pipe_model.pth")
            print(f"💾 发现更低 Loss，已保存最优模型至 checkpoints/best_pipe_model.pth")


if __name__ == "__main__":
    train()