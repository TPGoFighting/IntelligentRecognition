# Quick Start Guide - Universal Shape Recognition System

## Installation

```bash
# Navigate to project directory
cd IntelligentRecognition

# Install dependencies
pip install torch torchvision
pip install numpy pyvista pyransac3d pyyaml scikit-learn scipy tqdm
```

## Verify Installation

```bash
# Run test suite
python tests/test_system.py
```

Expected output: `ALL TESTS PASSED! System is ready to use.`

## Basic Workflow

### 1. Generate Training Data

```bash
python scripts/generate_universal_data.py \
    --config config/shape_config.yaml \
    --output data/universal \
    --num_train 100 \
    --num_test 20
```

This creates:
- `data/universal/train/` - 100 training scenes
- `data/universal/test/` - 20 test scenes

Each scene contains multiple shapes (cylinders, spheres, cuboids) with background points.

### 2. Train the Model

```bash
python scripts/train_universal.py \
    --config config/shape_config.yaml \
    --data data/universal \
    --output models/universal
```

Training will:
- Load training/test data
- Train PointNet++ model for 50 epochs
- Save best model to `models/universal/best_model.pth`
- Save training history to `models/universal/training_history.json`

### 3. Run Inference

```bash
# Single file
python scripts/batch_inference.py \
    --config config/shape_config.yaml \
    --model models/universal/best_model.pth \
    --input data/universal/test/scene_0000.npz \
    --output results/

# Batch processing with visualization
python scripts/batch_inference.py \
    --config config/shape_config.yaml \
    --model models/universal/best_model.pth \
    --input "data/universal/test/*.npz" \
    --output results/ \
    --visualize
```

Results saved to:
- `results/scene_0000_results.json` - Detection results
- `results/scene_0000_viz.png` - Visualization (if --visualize)
- `results/batch_summary.json` - Summary of all files

## Configuration Examples

### Example 1: Cylinder-Only (Backward Compatible)

Use `config/cylinder_config.yaml`:

```yaml
shapes:
  cylinder:
    label: 2
    params:
      radius_range: [0.15, 0.8]
      direction: [0, 0, 1]  # Fixed Z-axis
scene:
  type: "tunnel"
```

### Example 2: Indoor Scene

Use `config/indoor_config.yaml`:

```yaml
shapes:
  plane:    # Walls/floor
  cuboid:   # Furniture
  cylinder: # Pipes/columns
scene:
  type: "indoor"
  bounds: [[-5, 5], [-5, 5], [0, 3]]
```

### Example 3: Custom Configuration

Create your own config file:

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

## Python API Usage

### Data Generation

```python
from core.config_loader import load_config
from core.data_generator import UniversalDataGenerator

# Load config
config = load_config("config/shape_config.yaml")

# Create generator
generator = UniversalDataGenerator(config)

# Generate single scene
points, normals, labels = generator.generate_scene({
    "cylinder": 2,
    "sphere": 1
})

# Save scene
import numpy as np
np.savez("my_scene.npz", points=points, normals=normals, labels=labels)
```

### Inference

```python
from core.config_loader import load_config
from core.inference_engine import UniversalInferenceEngine
import numpy as np

# Load config and model
config = load_config("config/shape_config.yaml")
engine = UniversalInferenceEngine(
    model_path="models/universal/best_model.pth",
    config=config,
    device='cuda'
)

# Load point cloud
data = np.load("my_scene.npz")
points = data['points']
normals = data['normals']

# Run inference
detected_objects = engine.infer(points, normals)

# Process results
for obj in detected_objects:
    print(f"Detected {obj.shape_type}:")
    print(f"  Confidence: {obj.confidence:.3f}")
    print(f"  Points: {obj.num_points}")
    print(f"  Parameters: {obj.params}")
```

## Adding Custom Shapes

### Step 1: Create Plugin

Create `core/shape_plugins/my_shape_plugin.py`:

```python
from .base_shape import BaseShape
import numpy as np

class MyShapePlugin(BaseShape):
    def generate_points(self, **params):
        # Generate synthetic points
        points = ...
        normals = ...
        return points, normals

    def fit(self, points, **kwargs):
        # Fit using RANSAC
        params = ...
        return params

    def validate(self, params):
        # Check constraints
        return True

    def compute_inliers(self, points, params, threshold):
        # Find inlier points
        inliers = ...
        return inliers

    def visualize(self, plotter, params, points, color):
        # Visualize with PyVista
        pass
```

### Step 2: Register Plugin

Edit `core/shape_plugins/__init__.py`:

```python
from .my_shape_plugin import MyShapePlugin

def get_plugin_class(shape_name):
    plugins = {
        'cylinder': CylinderPlugin,
        'sphere': SpherePlugin,
        'myshape': MyShapePlugin,  # Add here
    }
    return plugins[shape_name]
```

### Step 3: Add to Config

```yaml
shapes:
  myshape:
    label: 6
    params:
      # Your parameters
    fitting:
      algorithm: "ransac_myshape"
      threshold: 0.05
      min_inliers: 30
```

## Troubleshooting

### Issue: Import errors
```bash
# Install missing dependencies
pip install pyyaml scikit-learn pyransac3d pyvista scipy
```

### Issue: CUDA out of memory
```bash
# Use CPU instead
python scripts/train_universal.py --config config/shape_config.yaml --device cpu
```

Or reduce batch size in config:
```yaml
training:
  batch_size: 4  # Reduce from 8
```

### Issue: Low detection accuracy
- Generate more training data (increase --num_train)
- Adjust RANSAC threshold in config
- Check parameter ranges match your data
- Increase training epochs

### Issue: Model not converging
- Check class weights in config
- Verify data quality (labels, normals)
- Increase learning rate or epochs
- Try different optimizer

## Performance Tips

1. **GPU Acceleration**: Use CUDA for 5x speedup
2. **Batch Processing**: Process multiple files in parallel
3. **Data Caching**: Reuse generated data across experiments
4. **Model Checkpointing**: Resume training from latest checkpoint

## Next Steps

1. **Train on Real Data**: Replace synthetic data with real point clouds
2. **Fine-tune Parameters**: Adjust RANSAC thresholds for your use case
3. **Add Custom Shapes**: Implement plugins for domain-specific shapes
4. **Deploy**: Use batch_inference.py for production processing

## Support

- Documentation: `README_UNIVERSAL.md`
- Test Suite: `python tests/test_system.py`
- Examples: See `config/` directory for configuration examples

## Quick Reference

```bash
# Test system
python tests/test_system.py

# Generate data
python scripts/generate_universal_data.py --config CONFIG --output DIR

# Train model
python scripts/train_universal.py --config CONFIG --data DIR --output DIR

# Run inference
python scripts/batch_inference.py --config CONFIG --model PATH --input PATTERN --output DIR
```

That's it! You're ready to use the universal shape recognition system.
