# 增强训练系统使用指南

## 概述

本系统已优化，支持单一形状的高质量训练，包含以下改进：

- ✅ **GPU训练支持** - 自动检测并使用GPU加速
- ✅ **更多训练数据** - 默认500个训练场景，100个测试场景
- ✅ **更多场景变化** - 扩大场景范围，增加对象尺寸变化
- ✅ **优化的超参数** - 100 epochs，AdamW优化器，余弦退火学习率
- ✅ **类别平衡** - 减少背景点，增加目标形状权重
- ✅ **预训练权重支持** - 可加载预训练模型继续训练
- ✅ **单一形状训练** - 每次只训练一种几何形状

## 快速开始

### 方法1: 使用快速训练脚本（推荐）

```bash
# 训练圆柱体识别（生成数据+训练）
python scripts/quick_train.py --shape cylinder --generate_data --num_train 500 --num_test 100

# 训练长方体识别
python scripts/quick_train.py --shape cuboid --generate_data --num_train 500 --num_test 100

# 训练球体识别
python scripts/quick_train.py --shape sphere --generate_data --num_train 500 --num_test 100

# 训练平面识别
python scripts/quick_train.py --shape plane --generate_data --num_train 500 --num_test 100
```

### 方法2: 分步执行

#### 步骤1: 生成训练数据

```bash
# 圆柱体数据
python scripts/generate_enhanced_data.py \
    --config config/cylinder_only_config.yaml \
    --output data/cylinder_enhanced \
    --num_train 500 \
    --num_test 100 \
    --objects_min 2 \
    --objects_max 5

# 长方体数据
python scripts/generate_enhanced_data.py \
    --config config/cuboid_only_config.yaml \
    --output data/cuboid_enhanced \
    --num_train 500 \
    --num_test 100

# 球体数据
python scripts/generate_enhanced_data.py \
    --config config/sphere_only_config.yaml \
    --output data/sphere_enhanced \
    --num_train 500 \
    --num_test 100

# 平面数据
python scripts/generate_enhanced_data.py \
    --config config/plane_only_config.yaml \
    --output data/plane_enhanced \
    --num_train 500 \
    --num_test 100
```

#### 步骤2: 训练模型

```bash
# 训练圆柱体模型
python scripts/train_universal.py \
    --config config/cylinder_only_config.yaml \
    --data data/cylinder_enhanced \
    --output models/cylinder_model

# 训练长方体模型
python scripts/train_universal.py \
    --config config/cuboid_only_config.yaml \
    --data data/cuboid_enhanced \
    --output models/cuboid_model

# 训练球体模型
python scripts/train_universal.py \
    --config config/sphere_only_config.yaml \
    --data data/sphere_enhanced \
    --output models/sphere_model

# 训练平面模型
python scripts/train_universal.py \
    --config config/plane_only_config.yaml \
    --data data/plane_enhanced \
    --output models/plane_model
```

## 配置文件说明

每种形状都有独立的配置文件：

- `config/cylinder_only_config.yaml` - 圆柱体配置
- `config/cuboid_only_config.yaml` - 长方体配置
- `config/sphere_only_config.yaml` - 球体配置
- `config/plane_only_config.yaml` - 平面配置

### 主要配置参数

```yaml
training:
  batch_size: 16          # 批次大小（GPU内存允许可增大）
  epochs: 100             # 训练轮数
  learning_rate: 0.001    # 初始学习率
  weight_decay: 0.0001    # 权重衰减（正则化）
  num_classes: 3          # 类别数（背景+未标记+目标形状）
  class_weights: [1.0, 1.0, 8.0]  # 类别权重（增加目标形状权重）
  optimizer: "adamw"      # 优化器（adamw/adam/sgd）
  scheduler: "cosine"     # 学习率调度器（cosine/step/plateau）
  warmup_epochs: 5        # 预热轮数
  min_lr: 0.00001        # 最小学习率
  use_gpu: true          # 使用GPU
  num_workers: 4         # 数据加载线程数
  pretrained_weights: null  # 预训练权重路径
```

## 使用预训练权重

如果你已经训练了一个模型，可以用它作为预训练权重：

1. 修改配置文件中的 `pretrained_weights` 参数：

```yaml
training:
  pretrained_weights: "models/cylinder_model/best_model.pth"
```

2. 或者在训练时使用 `--resume` 参数继续训练：

```bash
python scripts/train_universal.py \
    --config config/cylinder_only_config.yaml \
    --data data/cylinder_enhanced \
    --output models/cylinder_model \
    --resume models/cylinder_model/latest_model.pth
```

## 训练监控

训练过程中会显示：
- 训练损失和准确率
- 验证损失和准确率
- 每个类别的准确率
- 当前学习率
- GPU使用情况（如果使用GPU）

训练完成后会保存：
- `best_model.pth` - 验证准确率最高的模型
- `latest_model.pth` - 最新的模型检查点
- `training_history.json` - 训练历史记录

## 调优建议

### 如果训练过拟合（训练准确率高，验证准确率低）：
- 增加 `weight_decay`（例如 0.0005）
- 增加训练数据量
- 增加数据噪声 `noise_level`

### 如果训练欠拟合（训练和验证准确率都低）：
- 增加 `epochs`
- 调整 `learning_rate`
- 减少 `weight_decay`
- 检查数据质量

### 如果类别不平衡：
- 调整 `class_weights`，增加少数类的权重
- 减少 `background_density`
- 增加目标对象数量

### 如果GPU内存不足：
- 减少 `batch_size`
- 减少 `num_points`（在inference配置中）

## 示例：完整训练流程

```bash
# 1. 生成圆柱体训练数据（500训练+100测试）
python scripts/generate_enhanced_data.py \
    --config config/cylinder_only_config.yaml \
    --output data/cylinder_enhanced \
    --num_train 500 \
    --num_test 100

# 2. 训练模型（使用GPU，100 epochs）
python scripts/train_universal.py \
    --config config/cylinder_only_config.yaml \
    --data data/cylinder_enhanced \
    --output models/cylinder_model

# 3. 查看训练结果
# 模型保存在: models/cylinder_model/best_model.pth
# 训练历史: models/cylinder_model/training_history.json
```

## GPU要求

- 推荐：NVIDIA GPU with CUDA support
- 最小显存：4GB
- 推荐显存：8GB+

检查GPU是否可用：
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## 常见问题

**Q: 如何知道训练是否使用了GPU？**
A: 训练开始时会显示 "Using device: cuda" 和GPU信息。

**Q: 可以同时训练多种形状吗？**
A: 不推荐。当前系统优化为单一形状训练，这样可以获得更好的准确率。

**Q: 训练需要多长时间？**
A: 取决于GPU性能和数据量。使用GPU训练500个场景约需30-60分钟。

**Q: 如何使用训练好的模型？**
A: 使用现有的推理脚本，指定对应的配置文件和模型路径。
