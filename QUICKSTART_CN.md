# 快速入门指南 - 通用形状识别系统

## 安装

```bash
# 进入项目目录
cd IntelligentRecognition

# 安装依赖
pip install torch torchvision
pip install numpy pyvista pyransac3d pyyaml scikit-learn scipy tqdm
```

## 验证安装

```bash
# 运行测试套件
python tests/test_system.py
```

预期输出：`所有测试通过！系统已准备就绪。`

## 基本工作流程

### 1. 生成训练数据

```bash
python scripts/generate_universal_data.py \
    --config config/shape_config.yaml \
    --output data/universal \
    --num_train 100 \
    --num_test 20
```

这将创建：
- `data/universal/train/` - 100个训练场景
- `data/universal/test/` - 20个测试场景

每个场景包含多种形状（圆柱体、球体、立方体）和背景点。

### 2. 训练模型

```bash
python scripts/train_universal.py \
    --config config/shape_config.yaml \
    --data data/universal \
    --output models/universal
```

训练将：
- 加载训练/测试数据
- 训练PointNet++模型50轮
- 保存最佳模型到 `models/universal/best_model.pth`
- 保存训练历史到 `models/universal/training_history.json`

### 3. 运行推理

```bash
# 单个文件
python scripts/batch_inference.py \
    --config config/shape_config.yaml \
    --model models/universal/best_model.pth \
    --input data/universal/test/scene_0000.npz \
    --output results/

# 批量处理并可视化
python scripts/batch_inference.py \
    --config config/shape_config.yaml \
    --model models/universal/best_model.pth \
    --input "data/universal/test/*.npz" \
    --output results/ \
    --visualize
```

结果保存到：
- `results/scene_0000_results.json` - 检测结果
- `results/scene_0000_viz.png` - 可视化（如果使用 --visualize）
- `results/batch_summary.json` - 所有文件的汇总

## 配置示例

### 示例1：仅圆柱体（向后兼容）

使用 `config/cylinder_config.yaml`：

```yaml
shapes:
  cylinder:
    label: 2
    params:
      radius_range: [0.15, 0.8]
      direction: [0, 0, 1]  # 固定Z轴
scene:
  type: "tunnel"
```

### 示例2：室内场景

使用 `config/indoor_config.yaml`：

```yaml
shapes:
  plane:    # 墙壁/地板
  cuboid:   # 家具
  cylinder: # 管道/柱子
scene:
  type: "indoor"
  bounds: [[-5, 5], [-5, 5], [0, 3]]
```

### 示例3：自定义配置

创建你自己的配置文件：

```yaml
shapes:
  sphere:
    label: 2
    params:
      radius_range: [0.5, 2.0]
    fitting:
      threshold: 0.05
      min_inliers: 50

scene:
  type: "generic"
  bounds: [[-10, 10], [-10, 10], [0, 10]]
  background_density: 30000

training:
  batch_size: 8
  epochs: 30
  learning_rate: 0.001
```

## Python API使用

### 数据生成

```python
from core.config_loader import load_config
from core.data_generator import UniversalDataGenerator

# 加载配置
config = load_config("config/shape_config.yaml")

# 创建生成器
generator = UniversalDataGenerator(config)

# 生成单个场景
points, normals, labels = generator.generate_scene({
    "cylinder": 2,
    "sphere": 1
})

# 保存场景
import numpy as np
np.savez("my_scene.npz", points=points, normals=normals, labels=labels)
```

### 推理

```python
from core.config_loader import load_config
from core.inference_engine import UniversalInferenceEngine
import numpy as np

# 加载配置和模型
config = load_config("config/shape_config.yaml")
engine = UniversalInferenceEngine(
    model_path="models/universal/best_model.pth",
    config=config,
    device='cuda'
)

# 加载点云
data = np.load("my_scene.npz")
points = data['points']
normals = data['normals']

# 运行推理
detected_objects = engine.infer(points, normals)

# 处理结果
for obj in detected_objects:
    print(f"检测到 {obj.shape_type}:")
    print(f"  置信度: {obj.confidence:.3f}")
    print(f"  点数: {obj.num_points}")
    print(f"  参数: {obj.params}")
```

## 添加自定义形状

### 步骤1：创建插件

创建 `core/shape_plugins/my_shape_plugin.py`：

```python
from .base_shape import BaseShape
import numpy as np

class MyShapePlugin(BaseShape):
    def generate_points(self, **params):
        # 生成合成点
        points = ...
        normals = ...
        return points, normals

    def fit(self, points, **kwargs):
        # 使用RANSAC拟合
        params = ...
        return params

    def validate(self, params):
        # 检查约束
        return True

    def compute_inliers(self, points, params, threshold):
        # 找到内点
        inliers = ...
        return inliers

    def visualize(self, plotter, params, points, color):
        # 使用PyVista可视化
        pass
```

### 步骤2：注册插件

编辑 `core/shape_plugins/__init__.py`：

```python
from .my_shape_plugin import MyShapePlugin

def get_plugin_class(shape_name):
    plugins = {
        'cylinder': CylinderPlugin,
        'sphere': SpherePlugin,
        'myshape': MyShapePlugin,  # 在此添加
    }
    return plugins[shape_name]
```

### 步骤3：添加到配置

```yaml
shapes:
  myshape:
    label: 6
    params:
      # 你的参数
    fitting:
      algorithm: "ransac_myshape"
      threshold: 0.05
      min_inliers: 30
```

## 故障排除

### 问题：导入错误
```bash
# 安装缺失的依赖
pip install pyyaml scikit-learn pyransac3d pyvista scipy
```

### 问题：CUDA内存不足
```bash
# 使用CPU
python scripts/train_universal.py --config config/shape_config.yaml --device cpu
```

或减少配置中的批次大小：
```yaml
training:
  batch_size: 4  # 从8减少
```

### 问题：检测准确率低
- 生成更多训练数据（增加 --num_train）
- 调整配置中的RANSAC阈值
- 检查参数范围是否匹配你的数据
- 增加训练轮数

### 问题：模型不收敛
- 检查配置中的类别权重
- 验证数据质量（标签、法线）
- 增加学习率或训练轮数
- 尝试不同的优化器

## 性能优化建议

1. **GPU加速**：使用CUDA可获得5倍加速
2. **批量处理**：并行处理多个文件
3. **数据缓存**：在实验中重用生成的数据
4. **模型检查点**：从最新检查点恢复训练

## 后续步骤

1. **使用真实数据训练**：用真实点云替换合成数据
2. **微调参数**：根据你的用例调整RANSAC阈值
3. **添加自定义形状**：为特定领域的形状实现插件
4. **部署**：使用batch_inference.py进行生产处理

## 支持

- 文档：`README_UNIVERSAL_CN.md`
- 测试套件：`python tests/test_system.py`
- 示例：查看 `config/` 目录的配置示例

## 快速参考

```bash
# 测试系统
python tests/test_system.py

# 生成数据
python scripts/generate_universal_data.py --config CONFIG --output DIR

# 训练模型
python scripts/train_universal.py --config CONFIG --data DIR --output DIR

# 运行推理
python scripts/batch_inference.py --config CONFIG --model PATH --input PATTERN --output DIR
```

就这样！你已准备好使用通用形状识别系统。
