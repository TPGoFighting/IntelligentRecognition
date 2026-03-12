# IntelligentRecognition 项目修改记录

**修改日期:** 2026-03-11  
**修改目标:** 提升 PointNet++ 语义分割模型准确度，减少 RANSAC fallback

---

## 📊 问题诊断

### 当前状态
- 模型训练准确度 < 30%
- 推理时自动 fallback 到纯 RANSAC 模式
- 代码位置：`inference_engine.py` 第 73-97 行

### 根本原因

1. **无数据增强** - 训练集和测试集分布差异大，模型泛化能力差
2. **训练配置不合理** - epoch 数不足、学习率调度器过于激进
3. **类别权重过高** - 管道类权重 3 倍导致模型偏向，其他类别准确度下降

---

## 🔧 修改内容

### 修改 1: `core/dataset.py` - 添加数据增强

**修改位置:** `__init__` 方法

```python
# 新增代码（第 19-22 行）
# 数据增强参数
self.rotate_range = [-0.2, 0.2]  # 随机旋转 ±0.2 弧度
self.scale_range = [0.8, 1.2]    # 随机缩放 0.8-1.2 倍
self.noise_std = 0.01            # 高斯噪声标准差
```

**修改位置:** `__getitem__` 方法（坐标归一化之前）

```python
# 新增代码（训练模式下的数据增强）
# ========== 数据增强（仅训练模式）==========
if self.train:
    # 1. 随机旋转 (绕 Z 轴)
    rotate_angle = np.random.uniform(self.rotate_range[0], self.rotate_range[1])
    rotation_matrix = np.array([
        [np.cos(rotate_angle), -np.sin(rotate_angle), 0],
        [np.sin(rotate_angle), np.cos(rotate_angle), 0],
        [0, 0, 1]
    ])
    selected_points = np.dot(selected_points, rotation_matrix.T)
    selected_normals = np.dot(selected_normals, rotation_matrix.T)
    
    # 2. 随机缩放
    scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
    selected_points = selected_points * scale
    
    # 3. 加高斯噪声
    noise = np.random.normal(0, self.noise_std, selected_points.shape)
    selected_points = selected_points + noise
```

**效果:**
- 增加训练数据多样性
- 提高模型对旋转、缩放、噪声的鲁棒性
- 减少过拟合

---

### 修改 2: `train.py` - 优化训练配置

**修改位置:** 训练参数配置（第 17-22 行）

```python
# 修改前
batch_size = 8
epochs = 50
learning_rate = 0.001
grad_clip = 10.0
weight_decay = 1e-4

# 修改后
batch_size = 8
epochs = 100                    # ↑ 从 50 增加到 100，让模型充分训练
learning_rate = 0.002           # ↑ 从 0.001 提高到 0.002，加快收敛
grad_clip = 5.0                 # ↓ 从 10.0 降到 5.0，更严格的梯度控制
weight_decay = 1e-4             # 保持不变
```

**效果:**
- 更多训练轮数 → 模型收敛更充分
- 更高学习率 → 更快收敛速度
- 更严格梯度裁剪 → 训练更稳定

---

### 修改 3: `train.py` - 更换学习率调度器

**修改位置:** 学习率调度器配置（第 57 行）

```python
# 修改前
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# 问题：每 10 个 epoch 学习率减半，50 个 epoch 后只剩 1/32，后面基本不学了

# 修改后
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
# 优势：余弦退火，平滑衰减，有助于收敛到更优解
```

**效果:**
- 学习率平滑衰减，避免突变
- 有助于跳出局部最优
- 最终收敛到更好的解

---

### 修改 4: `train.py` - 调整类别权重

**修改位置:** 类别权重配置（第 53 行）

```python
# 修改前
class_weights = torch.FloatTensor([1.0, 1.0, 3.0])  # 管道权重 3 倍
# 问题：权重过高导致模型严重偏向管道类，其他类别准确度下降

# 修改后
class_weights = torch.FloatTensor([0.5, 1.0, 2.0])  # 降低管道权重到 2 倍
# 类别顺序：[其他背景 (0), 隧道壁 (1), 管道 (2)]
```

**效果:**
- 降低管道类权重，减少模型偏向
- 提升隧道壁和背景类别的准确度
- 整体 mIoU 更均衡

---

## 📈 预期效果

### 训练指标提升
- **训练准确度:** 从 <30% → 目标 >80%
- **验证 mIoU:** 从 <0.3 → 目标 >0.7
- **推理 fallback 率:** 从 >70% → 目标 <20%

### 训练时间
- **原配置:** 50 epochs ≈ 30-40 分钟
- **新配置:** 100 epochs ≈ 60-80 分钟

---

## 🚀 使用方法

### 1. 重新训练模型

```bash
cd /home/admin/.openclaw/workspace/IntelligentRecognition
python train.py
```

### 2. 监控训练过程

观察以下指标：
- **Loss 曲线:** 应该持续下降并趋于平稳
- **学习率变化:** CosineAnnealing 会平滑衰减
- **检查点保存:** `checkpoints/best_pipe_model.pth`

### 3. 测试新模型

```bash
python inference_engine.py --checkpoint checkpoints/best_pipe_model.pth
```

---

## ⚠️ 注意事项

### 1. 数据质量
- 确保 `test_data/processed/` 目录下有足够的高质量标注数据
- 建议每个类别至少有 1000+ 样本

### 2. 硬件要求
- **GPU:** 推荐 RTX 3060 或更高（6GB+ 显存）
- **内存:** 16GB+（数据集会加载到内存）
- **CPU:** 多核 CPU 可加快数据预处理

### 3. 训练中断恢复
- 训练中断后直接重新运行 `train.py` 即可
- 检查点会自动保存在 `checkpoints/` 目录

### 4. 类别不平衡问题
- 如果训练后发现某类别准确度仍然很低，可进一步调整 `class_weights`
- 建议：先按当前配置训练，根据结果再微调

---

## 📝 后续优化建议

### 短期（1-2 周）
1. **添加验证集** - 分割 20% 数据作为验证集，监控过拟合
2. **添加 mIoU 评估** - 训练时计算每个 epoch 的 mIoU
3. **早停机制** - 验证集 mIoU 不再提升时提前停止

### 中期（1 个月）
1. **模型加深** - 增加 PointNet++ 的层数或 filter 数量
2. **数据扩充** - 合成更多训练数据（不同形状、噪声水平）
3. **多尺度训练** - 使用不同的 `block_size` 训练

### 长期（3 个月+）
1. **尝试 Point Transformer** - 比 PointNet++ 更强的 backbone
2. **自监督预训练** - 在大规模点云数据上预训练
3. **在线学习** - 部署后持续收集数据并更新模型

---

## 📞 问题排查

### Q1: 训练 Loss 不下降
- 检查学习率是否过高（尝试降到 0.001）
- 检查数据标签是否正确（0/1/2）
- 检查梯度是否爆炸（查看 grad_clip 是否生效）

### Q2: 训练准确度很高但测试很差
- 过拟合！增加数据增强强度
- 增加 weight_decay
- 添加 Dropout 层

### Q3: CUDA Out of Memory
- 减小 `batch_size`（从 8 降到 4 或 2）
- 减小 `num_points`（从 4096 降到 2048）

---

**修改完成！** 有问题随时问。🦞
