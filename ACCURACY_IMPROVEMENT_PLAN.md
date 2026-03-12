# 准确率提升方案 - 激进优化

## 问题诊断

### 当前问题
- **准确率**: 33.54% (远低于目标70-85%)
- **根本原因**: 类别严重不平衡
  - 背景点: 8000 (66.7%)
  - 圆柱体点: 4000 (33.3%)
- **结果**: 模型倾向于预测背景，准确率卡在33%

## 激进优化方案

### 核心策略：平衡类别 + 增强学习

| 参数 | 旧值 | 新值 | 改进 |
|------|------|------|------|
| **background_density** | 8000 | 3000 | -62.5% 背景点 |
| **class_weights** | [1.0, 1.0, 8.0] | [0.5, 0.5, 20.0] | 圆柱体权重×2.5 |
| **learning_rate** | 0.001 | 0.003 | 学习率×3 |
| **batch_size** | 16 | 32 | 批次×2 |
| **epochs** | 100 | 150 | 训练轮数+50% |
| **noise_level** | 0.015 | 0.01 | 减少噪声 |
| **num_train** | 500 | 800 | 训练数据+60% |
| **objects_min/max** | 2-5 | 3-6 | 更多圆柱体 |
| **num_workers** | 4 | 8 | 数据加载×2 |
| **warmup_epochs** | 5 | 10 | 预热×2 |

### 预期效果

**数据平衡改善：**
- 旧配置: 背景66.7% vs 圆柱体33.3% (比例2:1)
- 新配置: 背景42.9% vs 圆柱体57.1% (比例1:1.33) ✅

**训练改进：**
- 更高的学习率 → 更快收敛
- 更大的batch size → 更稳定的梯度
- 更多的训练数据 → 更好的泛化
- 更强的类别权重 → 更关注圆柱体

**预期准确率：70-85%** 🎯

## 执行步骤

### 1. 运行激进优化训练

```bash
python scripts/train_aggressive.py
```

这将：
- 生成800个训练场景 + 150个测试场景
- 使用优化的配置训练150个epochs
- 自动使用GPU加速

### 2. 监控训练进度

在另一个终端：
```bash
# 每30秒检查一次
watch -n 30 "python -c \"
import torch
ckpt = torch.load('models/cylinder_aggressive/best_model.pth', map_location='cpu', weights_only=False)
print(f'Epoch: {ckpt.get(\\\"epoch\\\", \\\"N/A\\\")}')
print(f'Best Acc: {ckpt.get(\\\"best_acc\\\", 0):.4f}')
print(f'Val Acc: {ckpt.get(\\\"val_acc\\\", 0):.4f}')
\""
```

### 3. 测试优化模型

训练完成后：
```bash
python test_optimized_model.py
```

## 其他提升准确率的方法

### 方法1: 数据增强（推荐）

在训练时添加数据增强：

```python
# 在 UniversalPointCloudDataset.__getitem__ 中添加
def augment_points(points, normals):
    # 随机旋转
    angle = np.random.uniform(0, 2*np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[cos_a, -sin_a, 0],
                          [sin_a, cos_a, 0],
                          [0, 0, 1]])
    points = points @ rot_matrix.T
    normals = normals @ rot_matrix.T

    # 随机缩放
    scale = np.random.uniform(0.9, 1.1)
    points = points * scale

    # 随机抖动
    jitter = np.random.normal(0, 0.01, points.shape)
    points = points + jitter

    return points, normals
```

### 方法2: Focal Loss（处理类别不平衡）

替换CrossEntropyLoss：

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# 使用
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

### 方法3: 标签平滑

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
```

### 方法4: 学习率调优

使用OneCycleLR：

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=150,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos'
)
```

### 方法5: 混合精度训练（更快）

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for features, labels in dataloader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(features)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 方法6: 更简单的场景

生成更简单的训练数据：

```yaml
scene:
  background_density: 2000  # 进一步减少
  noise_level: 0.005        # 更少噪声

# 生成时
--objects_min 1  # 每个场景1-3个圆柱体
--objects_max 3
```

### 方法7: 预训练策略

课程学习（从简单到困难）：

```bash
# 阶段1: 简单场景（1-2个圆柱体，少噪声）
python scripts/generate_enhanced_data.py \
    --config config/cylinder_simple.yaml \
    --num_train 500 --objects_min 1 --objects_max 2

python scripts/train_universal.py \
    --config config/cylinder_simple.yaml \
    --epochs 50

# 阶段2: 中等场景（使用阶段1的模型作为预训练）
python scripts/train_universal.py \
    --config config/cylinder_aggressive_config.yaml \
    --resume models/cylinder_simple/best_model.pth \
    --epochs 100
```

### 方法8: 检查数据质量

```python
# 检查生成的数据
import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/cylinder_aggressive/train/scene_0000.npz')
labels = data['labels']

unique, counts = np.unique(labels, return_counts=True)
plt.bar(unique, counts)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Label Distribution')
plt.savefig('label_distribution.png')

print(f"背景比例: {counts[0]/counts.sum()*100:.1f}%")
print(f"圆柱体比例: {counts[1]/counts.sum()*100:.1f}%")
```

## 快速开始

**立即执行激进优化训练：**

```bash
python scripts/train_aggressive.py
```

预计时间：
- 数据生成：10-15分钟
- 模型训练：45-60分钟（使用GPU）
- 总计：约1小时

**预期结果：准确率从33%提升到70%+**

## 如果还是不够高

如果激进优化后准确率仍然<60%，考虑：

1. **检查模型架构** - PointNet++可能需要调整
2. **检查数据生成** - 圆柱体生成可能有问题
3. **简化任务** - 先训练识别单个圆柱体
4. **使用更强的模型** - 尝试PointNet++ with更多层
5. **增加训练时间** - 训练到200-300 epochs

## 总结

**最有效的3个方法：**
1. ✅ **平衡类别** - 减少背景点到3000
2. ✅ **增加权重** - 圆柱体权重提升到20.0
3. ✅ **更多数据** - 800个训练场景

这三个改进应该能将准确率从33%提升到70%+。
