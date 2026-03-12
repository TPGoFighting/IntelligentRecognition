# 增强训练系统使用指南（中文版）

## 系统改进总结

本次优化包含以下重要改进：

### 1. GPU训练支持 ✅
- 自动检测CUDA设备
- 显示GPU型号和显存信息
- 优化数据加载（pin_memory）

### 2. 增加训练数据 ✅
- 默认：500个训练场景 + 100个测试场景
- 可自定义场景数量
- 每个场景2-5个目标对象

### 3. 扩展场景多样性 ✅
- 场景范围：[-15, 15] × [-15, 15] × [0, 25]（原来是[-10, 10]）
- 对象尺寸范围扩大
- 增加噪声水平（0.015）

### 4. 优化超参数 ✅
- **Epochs**: 50 → 100
- **Batch Size**: 8 → 16
- **优化器**: Adam → AdamW（带权重衰减）
- **学习率调度**: StepLR → CosineAnnealingLR（余弦退火）
- **权重衰减**: 0.0001（防止过拟合）
- **预热**: 5个epochs

### 5. 类别平衡 ✅
- 背景点数：50000 → 8000（减少背景比例）
- 类别权重：[1.0, 1.0, 8.0]（目标形状权重×8）

### 6. 单一形状训练 ✅
- 每个配置文件只训练一种形状
- 4个独立配置：cylinder, cuboid, sphere, plane
- 避免多类别混淆

### 7. 预训练权重支持 ✅
- 可加载预训练模型
- 支持迁移学习
- 灵活的权重匹配

## 配置文件对比

### 旧配置（shape_config.yaml）
```yaml
training:
  batch_size: 8
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "step"
  # 同时训练4种形状
```

### 新配置（例如 cylinder_only_config.yaml）
```yaml
training:
  batch_size: 16          # ↑ 增加
  epochs: 100             # ↑ 增加
  learning_rate: 0.001
  weight_decay: 0.0001    # ✨ 新增
  optimizer: "adamw"      # ✨ 改进
  scheduler: "cosine"     # ✨ 改进
  warmup_epochs: 5        # ✨ 新增
  min_lr: 0.00001        # ✨ 新增
  use_gpu: true          # ✨ 新增
  num_workers: 4         # ✨ 新增
  pretrained_weights: null  # ✨ 新增
  class_weights: [1.0, 1.0, 8.0]  # ✨ 优化
  # 只训练1种形状
```

## 快速开始

### 最简单的方式（一键训练）

```bash
# 训练圆柱体识别
python scripts/quick_train.py --shape cylinder --generate_data

# 训练长方体识别
python scripts/quick_train.py --shape cuboid --generate_data

# 训练球体识别
python scripts/quick_train.py --shape sphere --generate_data

# 训练平面识别
python scripts/quick_train.py --shape plane --generate_data
```

### 自定义数据量

```bash
# 生成1000个训练场景，200个测试场景
python scripts/quick_train.py \
    --shape cylinder \
    --generate_data \
    --num_train 1000 \
    --num_test 200
```

## 分步执行（高级用法）

### 第一步：生成数据

```bash
# 圆柱体数据（500训练+100测试）
python scripts/generate_enhanced_data.py \
    --config config/cylinder_only_config.yaml \
    --output data/cylinder_enhanced \
    --num_train 500 \
    --num_test 100 \
    --objects_min 2 \
    --objects_max 5
```

参数说明：
- `--config`: 配置文件路径
- `--output`: 数据保存目录
- `--num_train`: 训练场景数量
- `--num_test`: 测试场景数量
- `--objects_min`: 每个场景最少对象数
- `--objects_max`: 每个场景最多对象数

### 第二步：训练模型

```bash
# 训练圆柱体模型
python scripts/train_universal.py \
    --config config/cylinder_only_config.yaml \
    --data data/cylinder_enhanced \
    --output models/cylinder_model
```

训练过程中会显示：
```
Using device: cuda
GPU: NVIDIA GeForce RTX 3080
GPU Memory: 10.00 GB

Epoch 1/100
----------------------------------------------------------
Training: 100%|████████| 32/32 [00:15<00:00]
Validation: 100%|████████| 7/7 [00:02<00:00]

Epoch 1 Results:
  Train Loss: 0.3245, Train Acc: 0.8912
  Val Loss: 0.2876, Val Acc: 0.9123
  Learning Rate: 0.001000
  Per-class accuracy:
    background: 0.9456
    cylinder: 0.8790
```

### 第三步：继续训练（可选）

如果想继续训练或微调：

```bash
python scripts/train_universal.py \
    --config config/cylinder_only_config.yaml \
    --data data/cylinder_enhanced \
    --output models/cylinder_model \
    --resume models/cylinder_model/latest_model.pth
```

## 配置文件详解

### 形状参数（shapes）

```yaml
shapes:
  cylinder:
    label: 2  # 类别标签（0=背景，1=未标记，2=圆柱体）
    params:
      radius_range: [0.1, 1.2]    # 半径范围（扩大）
      height_range: [0.8, 12.0]   # 高度范围（扩大）
      direction: "auto"            # 自动检测方向
    fitting:
      algorithm: "ransac_cylinder"
      threshold: 0.08
      min_inliers: 50
      max_iterations: 1000
    min_points: 100
```

### 场景参数（scene）

```yaml
scene:
  type: "tunnel"  # 场景类型
  bounds: [[-15, 15], [-15, 15], [0, 25]]  # 场景范围（扩大）
  background_density: 8000   # 背景点数（减少）
  noise_level: 0.015         # 噪声水平（增加）
```

### 训练参数（training）

```yaml
training:
  batch_size: 16          # 批次大小
  epochs: 100             # 训练轮数
  learning_rate: 0.001    # 初始学习率
  weight_decay: 0.0001    # L2正则化
  num_classes: 3          # 类别数
  class_weights: [1.0, 1.0, 8.0]  # 类别权重
  optimizer: "adamw"      # 优化器
  scheduler: "cosine"     # 学习率调度器
  warmup_epochs: 5        # 预热轮数
  min_lr: 0.00001        # 最小学习率
  use_gpu: true          # 使用GPU
  num_workers: 4         # 数据加载线程
  pretrained_weights: null  # 预训练权重
```

## 优化器和调度器选择

### 优化器（optimizer）

1. **adamw**（推荐）
   - 带权重衰减的Adam
   - 更好的泛化性能
   - 适合大多数情况

2. **adam**
   - 标准Adam优化器
   - 快速收敛
   - 可能过拟合

3. **sgd**
   - 随机梯度下降
   - 需要更多调参
   - 训练时间更长

### 学习率调度器（scheduler）

1. **cosine**（推荐）
   - 余弦退火
   - 平滑降低学习率
   - 更好的收敛

2. **step**
   - 阶梯式降低
   - 简单直接
   - 需要设置step_size

3. **plateau**
   - 自适应调整
   - 验证准确率不提升时降低
   - 更灵活

## 性能调优指南

### 问题1：训练太慢

**解决方案：**
```yaml
training:
  batch_size: 32        # 增加批次大小（需要更多显存）
  num_workers: 8        # 增加数据加载线程
```

### 问题2：GPU内存不足

**解决方案：**
```yaml
training:
  batch_size: 8         # 减少批次大小
inference:
  num_points: 2048      # 减少点数
```

### 问题3：过拟合（训练准确率高，验证准确率低）

**解决方案：**
```yaml
training:
  weight_decay: 0.0005  # 增加正则化
scene:
  noise_level: 0.02     # 增加噪声
  background_density: 10000  # 增加背景点
```

生成更多数据：
```bash
python scripts/generate_enhanced_data.py \
    --num_train 1000 \
    --num_test 200
```

### 问题4：欠拟合（训练和验证准确率都低）

**解决方案：**
```yaml
training:
  epochs: 150           # 增加训练轮数
  learning_rate: 0.002  # 增加学习率
  weight_decay: 0.00005 # 减少正则化
```

### 问题5：类别不平衡

**解决方案：**
```yaml
training:
  class_weights: [1.0, 1.0, 10.0]  # 进一步增加目标权重
scene:
  background_density: 5000  # 减少背景点
```

增加对象数量：
```bash
python scripts/generate_enhanced_data.py \
    --objects_min 3 \
    --objects_max 7
```

## 预训练权重使用

### 方法1：配置文件指定

编辑配置文件：
```yaml
training:
  pretrained_weights: "models/cylinder_model/best_model.pth"
```

### 方法2：命令行参数

```bash
python scripts/train_universal.py \
    --config config/cylinder_only_config.yaml \
    --data data/cylinder_enhanced \
    --output models/cylinder_model_v2 \
    --resume models/cylinder_model/best_model.pth
```

### 迁移学习示例

假设你已经训练了一个圆柱体模型，想用它来加速球体模型的训练：

1. 修改 `sphere_only_config.yaml`：
```yaml
training:
  pretrained_weights: "models/cylinder_model/best_model.pth"
  learning_rate: 0.0005  # 使用较小的学习率
```

2. 训练球体模型：
```bash
python scripts/train_universal.py \
    --config config/sphere_only_config.yaml \
    --data data/sphere_enhanced \
    --output models/sphere_model
```

## 训练结果分析

### 查看训练历史

```python
import json
import matplotlib.pyplot as plt

# 加载训练历史
with open('models/cylinder_model/training_history.json', 'r') as f:
    history = json.load(f)

# 绘制损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
```

### 模型文件说明

训练完成后，`models/cylinder_model/` 目录包含：

- `best_model.pth` - 验证准确率最高的模型（用于推理）
- `latest_model.pth` - 最新的检查点（用于继续训练）
- `training_history.json` - 训练历史记录

## 完整训练示例

### 示例1：训练圆柱体识别（标准配置）

```bash
# 生成数据
python scripts/generate_enhanced_data.py \
    --config config/cylinder_only_config.yaml \
    --output data/cylinder_enhanced \
    --num_train 500 \
    --num_test 100

# 训练模型
python scripts/train_universal.py \
    --config config/cylinder_only_config.yaml \
    --data data/cylinder_enhanced \
    --output models/cylinder_model
```

### 示例2：训练长方体识别（大数据集）

```bash
# 生成更多数据
python scripts/generate_enhanced_data.py \
    --config config/cuboid_only_config.yaml \
    --output data/cuboid_large \
    --num_train 1000 \
    --num_test 200 \
    --objects_min 3 \
    --objects_max 6

# 训练模型
python scripts/train_universal.py \
    --config config/cuboid_only_config.yaml \
    --data data/cuboid_large \
    --output models/cuboid_model
```

### 示例3：使用预训练权重

```bash
# 第一阶段：训练基础模型
python scripts/quick_train.py --shape cylinder --generate_data

# 第二阶段：使用预训练权重继续训练
python scripts/train_universal.py \
    --config config/cylinder_only_config.yaml \
    --data data/cylinder_enhanced \
    --output models/cylinder_model_v2 \
    --resume models/cylinder_model/best_model.pth \
    --epochs 50 \
    --learning_rate 0.0005
```

## 系统要求

### 硬件要求

- **CPU**: 4核以上
- **内存**: 16GB以上
- **GPU**: NVIDIA GPU with CUDA（推荐）
  - 最小显存：4GB
  - 推荐显存：8GB+
  - 支持的GPU：GTX 1060及以上

### 软件要求

- Python 3.8+
- PyTorch 1.10+ with CUDA
- NumPy, Open3D等依赖

### 检查GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
```

## 常见问题（FAQ）

**Q1: 训练时显示"Using device: cpu"，没有使用GPU？**

A: 检查：
1. 是否安装了CUDA版本的PyTorch
2. CUDA驱动是否正确安装
3. 配置文件中 `use_gpu: true`

**Q2: 训练中途中断了，如何继续？**

A: 使用 `--resume` 参数：
```bash
python scripts/train_universal.py \
    --config config/cylinder_only_config.yaml \
    --data data/cylinder_enhanced \
    --output models/cylinder_model \
    --resume models/cylinder_model/latest_model.pth
```

**Q3: 验证准确率一直不提升？**

A: 尝试：
1. 检查数据质量
2. 调整学习率（减小或增大）
3. 增加训练数据
4. 调整类别权重

**Q4: 训练速度很慢？**

A: 优化：
1. 确保使用GPU
2. 增加 `batch_size`
3. 增加 `num_workers`
4. 减少 `num_points`

**Q5: 如何选择最佳的超参数？**

A: 建议：
1. 先使用默认配置训练
2. 观察训练曲线
3. 根据过拟合/欠拟合情况调整
4. 进行小规模实验

## 下一步

训练完成后，可以：

1. **评估模型性能**
   - 在测试集上评估
   - 计算各类别准确率
   - 分析混淆矩阵

2. **使用模型推理**
   - 加载训练好的模型
   - 对新点云进行预测
   - 可视化结果

3. **模型优化**
   - 模型剪枝
   - 量化加速
   - 导出ONNX格式

4. **部署应用**
   - 集成到实际系统
   - 实时推理
   - 性能监控
