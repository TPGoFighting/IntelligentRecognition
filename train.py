import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

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
    data_root = "test_data/processed"  # 改为测试数据目录
    batch_size = 8  # 增加batch size以提高稳定性
    epochs = 50  # 增加到50个epoch以充分训练
    learning_rate = 0.001  # 提高学习率加快收敛
    grad_clip = 10.0  # 增加梯度裁剪阈值
    weight_decay = 1e-4  # 增加权重衰减，防止过拟合

    # 启用底层算法自动优化
    torch.backends.cudnn.benchmark = True

    # 使用GPU如果可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")  # 调试时使用CPU
    print(f"[GPU]  检测到硬件加速: {torch.cuda.get_device_name(0)}")
    print(f"[ROCKET] 核心配置: Batch Size={batch_size}, Device={device}")
    print(f"[CHART] 训练参数: LR={learning_rate}, 梯度裁剪={grad_clip}, 权重衰减={weight_decay}")

    # 2. 加载数据集
    train_dataset = TunnelDataset(data_root=data_root, num_points=4096, block_size=3.0, train=True)

    # 优化点：Windows 下 32核 CPU 建议设置 num_workers 为 4 或 8
    # pin_memory=True 能显著加快内存到显存的传输
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=True  # 丢弃最后不满足 Batch 的数据，保持计算步长一致
    )

    # 3. 初始化模型与损失函数
    model = get_model(num_classes=3).to(device)  # 三分类：管道(2)、隧道壁(1)、其他背景(0)

    # 设置类别权重：给管道类更高的权重以应对数据不平衡
    class_weights = torch.FloatTensor([1.0, 1.0, 3.0]).to(device)  # [其他背景, 隧道壁, 管道]
    criterion = get_loss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 学习率调度器：每10个epoch衰减到原来的0.5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 4. 禁用 AMP (自动混合精度) 缩放器，因为可能导致NaN
    # scaler = torch.amp.GradScaler(device.type)

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

            # 前向传播
            predictions = model(points)
            loss = criterion(predictions, labels)

            # 检查loss是否为NaN
            if torch.isnan(loss):
                print(f"[WARN]  Warning: NaN loss at batch {batch_idx}, skipping this batch")
                continue

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            # 更新参数
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Step [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"[FLAG] Epoch {epoch + 1} 结束，平均 Loss: {avg_loss:.4f}")

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[UP] 学习率更新: {current_lr:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/best_pipe_model.pth")
            print(f"[DISK] 权重已更新: checkpoints/best_pipe_model.pth")


if __name__ == "__main__":
    train()