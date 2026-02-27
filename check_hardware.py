import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.pointnet2_sem_seg import get_model, get_loss
from core.dataset import TunnelDataset


def train():
    # ==========================================================
    # 1. 针对 RTX 4060 & 32核 CPU 的专项配置
    # ==========================================================
    data_root = "data/processed"
    batch_size = 32  # 8GB 显存建议从 32 开始，如果报错再调回 16
    epochs = 50
    learning_rate = 0.001

    # 启用底层算法自动优化
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  检测到硬件加速: {torch.cuda.get_device_name(0)}")
    print(f"🚀 核心配置: Batch Size={batch_size}, Device={device}")

    # 2. 加载数据集
    train_dataset = TunnelDataset(data_root=data_root, num_points=4096, block_size=3.0, train=True)

    # 优化点：Windows 下 32核 CPU 建议设置 num_workers 为 4 或 8
    # pin_memory=True 能显著加快内存到显存的传输
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True  # 丢弃最后不满足 Batch 的数据，保持计算步长一致
    )

    # 3. 初始化模型与损失函数
    model = get_model(num_classes=2).to(device)
    criterion = get_loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. 初始化 AMP (自动混合精度) 缩放器
    scaler = torch.cuda.amp.GradScaler()

    # 5. 训练循环
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float('inf')

    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (points, labels) in enumerate(train_loader):
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()

            # 数据转置 [B, N, 6] -> [B, 6, N]
            points = points.transpose(2, 1)

            # --- AMP 自动混合精度核心逻辑 ---
            with torch.cuda.amp.autocast():
                predictions = model(points)
                loss = criterion(predictions, labels)

            # 缩放损失并回传
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Step [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"🏁 Epoch {epoch + 1} 结束，平均 Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/best_pipe_model.pth")
            print(f"💾 权重已更新: checkpoints/best_pipe_model.pth")


if __name__ == "__main__":
    train()