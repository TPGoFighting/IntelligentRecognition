# 提高准确率 - 完整方案总结

## 🎯 目标
将圆柱体检测准确率从 **33%** 提升到 **70%+**

## 📊 问题根源

### 已识别的问题
1. **类别严重不平衡** ⚠️
   - 旧配置：背景66.7% vs 圆柱体33.3%
   - 导致模型偏向预测背景

2. **训练参数保守** ⚠️
   - 学习率较低（0.001）
   - 类别权重不足（8.0）
   - 训练数据量偏少（500场景）

3. **模型收敛困难** ⚠️
   - 损失停滞在1.0985
   - 准确率卡在33%左右

## ✅ 已实施的优化

### 激进优化配置（正在训练中）

| 参数 | 旧值 | 新值 | 改进效果 |
|------|------|------|----------|
| **background_density** | 8000 | 3000 | 背景点减少62.5% |
| **数据平衡** | 背景66.7% | 背景37.5% | ✅ 圆柱体占比提升到62.5% |
| **class_weights** | [1.0, 1.0, 8.0] | [0.5, 0.5, 20.0] | 圆柱体权重×2.5 |
| **learning_rate** | 0.001 | 0.003 | 学习率×3 |
| **batch_size** | 16 | 32 | 批次大小×2 |
| **epochs** | 100 | 150 | 训练轮数+50% |
| **num_train** | 500 | 800 | 训练场景+60% |
| **objects_per_scene** | 2-5 | 3-6 | 更多圆柱体 |
| **noise_level** | 0.015 | 0.01 | 减少噪声33% |

### 当前训练状态
- ✅ 数据生成完成：800训练 + 150测试
- ✅ 数据平衡改善：圆柱体占比62.5%
- 🔄 模型训练中：14/150 epochs
- ⏳ 预计完成时间：约45分钟

## 🚀 如果激进优化后仍不够（备用方案）

### 方案A: 极简场景训练

**策略**：先在极简场景上训练，确保模型能学会基础特征

```yaml
# config/cylinder_ultra_simple.yaml
scene:
  background_density: 1000  # 极少背景
  noise_level: 0.005        # 极少噪声

training:
  class_weights: [0.3, 0.3, 30.0]  # 极高圆柱体权重
  learning_rate: 0.005             # 更高学习率
```

```bash
# 生成极简数据
python scripts/generate_enhanced_data.py \
    --config config/cylinder_ultra_simple.yaml \
    --output data/cylinder_ultra_simple \
    --num_train 500 \
    --objects_min 1 \
    --objects_max 2  # 每个场景只有1-2个圆柱体

# 训练
python scripts/train_universal.py \
    --config config/cylinder_ultra_simple.yaml \
    --data data/cylinder_ultra_simple \
    --output models/cylinder_ultra_simple \
    --epochs 100
```

### 方案B: 数据增强

在 `scripts/train_universal.py` 中添加数据增强：

```python
class UniversalPointCloudDataset(Dataset):
    def __init__(self, data_dir: str, num_points: int = 4096, augment: bool = True):
        self.augment = augment
        # ... 其他初始化代码

    def __getitem__(self, idx):
        # ... 加载数据

        if self.augment:
            # 随机旋转（绕Z轴）
            angle = np.random.uniform(0, 2*np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            points = points @ rot_matrix.T
            normals = normals @ rot_matrix.T

            # 随机缩放
            scale = np.random.uniform(0.95, 1.05)
            points = points * scale

            # 随机抖动
            jitter = np.random.normal(0, 0.005, points.shape)
            points = points + jitter

        # ... 其余代码
```

### 方案C: Focal Loss（处理类别不平衡）

替换损失函数：

```python
import torch.nn.functional as F

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

# 在训练脚本中使用
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

### 方案D: 课程学习（从简单到困难）

```bash
# 阶段1: 极简场景（50 epochs）
python scripts/train_universal.py \
    --config config/cylinder_ultra_simple.yaml \
    --data data/cylinder_ultra_simple \
    --output models/stage1 \
    --epochs 50

# 阶段2: 简单场景（使用阶段1模型，50 epochs）
python scripts/train_universal.py \
    --config config/cylinder_simple.yaml \
    --data data/cylinder_simple \
    --output models/stage2 \
    --resume models/stage1/best_model.pth \
    --epochs 50

# 阶段3: 正常场景（使用阶段2模型，100 epochs）
python scripts/train_universal.py \
    --config config/cylinder_aggressive_config.yaml \
    --data data/cylinder_aggressive \
    --output models/stage3 \
    --resume models/stage2/best_model.pth \
    --epochs 100
```

### 方案E: 检查模型架构

如果所有方法都不行，可能是PointNet++架构问题：

```python
# 检查模型输出
model = get_model(num_classes=3)
print(model)

# 检查模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

### 方案F: 使用更简单的模型

如果PointNet++太复杂，尝试简单的PointNet：

```python
# 使用PointNet而不是PointNet++
from models.pointnet_sem_seg import get_model as get_pointnet_model

model = get_pointnet_model(num_classes=3)
```

## 📈 预期结果时间线

### 当前训练（激进优化）
- **当前**: 14/150 epochs, 准确率33%
- **预期30 epochs**: 准确率40-50%
- **预期60 epochs**: 准确率55-65%
- **预期100 epochs**: 准确率65-75%
- **预期150 epochs**: 准确率70-80% ✅

### 如果准确率<60%
执行方案A（极简场景）+ 方案C（Focal Loss）

### 如果准确率<50%
执行方案D（课程学习）

## 🔍 诊断工具

### 检查训练进度
```bash
python check_aggressive_progress.py
```

### 可视化训练曲线
```python
import json
import matplotlib.pyplot as plt

with open('models/cylinder_aggressive/training_history.json') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.axhline(y=0.7, color='r', linestyle='--', label='Target (70%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.tight_layout()
plt.savefig('training_curves.png')
print("Saved to training_curves.png")
```

### 检查数据分布
```python
import numpy as np
from pathlib import Path

train_dir = Path('data/cylinder_aggressive/train')
files = list(train_dir.glob('scene_*.npz'))

bg_counts = []
cyl_counts = []

for f in files[:10]:  # 检查前10个场景
    data = np.load(f)
    labels = data['labels']
    unique, counts = np.unique(labels, return_counts=True)
    bg_counts.append(counts[0])
    cyl_counts.append(counts[1] if len(counts) > 1 else 0)

print(f"平均背景点: {np.mean(bg_counts):.0f}")
print(f"平均圆柱体点: {np.mean(cyl_counts):.0f}")
print(f"平均圆柱体比例: {np.mean(cyl_counts)/(np.mean(bg_counts)+np.mean(cyl_counts))*100:.1f}%")
```

## 💡 关键洞察

### 为什么准确率卡在33%？

1. **类别不平衡** - 模型学会了"总是预测背景"
   - 解决：减少背景点，增加圆柱体权重 ✅

2. **学习率太低** - 模型收敛太慢
   - 解决：增加学习率到0.003 ✅

3. **训练不足** - 100 epochs可能不够
   - 解决：增加到150 epochs ✅

4. **数据太复杂** - 场景中对象太多，噪声太大
   - 解决：减少噪声，如果还不行用极简场景

### 成功的关键指标

- **数据平衡**: 圆柱体占比应该>50% ✅ (当前62.5%)
- **类别权重**: 圆柱体权重应该>15 ✅ (当前20.0)
- **学习率**: 应该在0.002-0.005之间 ✅ (当前0.003)
- **训练轮数**: 至少100-150 epochs ✅ (当前150)

## 📝 总结

**当前状态**：
- ✅ 已实施激进优化
- ✅ 数据平衡大幅改善
- 🔄 训练进行中（14/150 epochs）
- ⏳ 等待训练完成查看效果

**下一步**：
1. 等待训练完成（约45分钟）
2. 检查最终准确率
3. 如果<70%，执行备用方案A-F

**预期**：
激进优化应该能将准确率提升到70%+。如果不行，说明问题可能在模型架构或数据生成逻辑上，需要更深入的调试。
