"""
Diagnostic script to check model and data generation.
"""
import numpy as np
import torch
from pathlib import Path

from core.config_loader import load_config
from core.data_generator import UniversalDataGenerator
from models.pointnet2_sem_seg import get_model

print("=" * 60)
print("MODEL AND DATA DIAGNOSTIC")
print("=" * 60)

# Load config
config = load_config('config/shape_config.yaml')
print(f"\nConfig: {config.num_classes} classes")
print(f"Shapes: {config.shape_names}")
print(f"Label mapping: {config.label_to_shape}")
print(f"Class weights: {config.get_class_weights()}")

# Check model
model_path = 'models/universal/best_model.pth'
print(f"\n=== Checking model: {model_path} ===")

checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
print(f"Checkpoint keys: {list(checkpoint.keys())}")

if 'epoch' in checkpoint:
    print(f"Trained for: {checkpoint['epoch'] + 1} epochs")
if 'best_acc' in checkpoint:
    print(f"Best accuracy: {checkpoint['best_acc']:.4f}")

state_dict = checkpoint['model_state_dict']
for key in state_dict:
    if key.endswith('conv2.weight'):
        print(f"Output layer shape: {state_dict[key].shape}")
        print(f"  (num_classes={state_dict[key].shape[0]})")

# Generate test data
print(f"\n=== Generating test data ===")
generator = UniversalDataGenerator(config)

# Generate single scene with known shapes
num_objects = {'cylinder': 1, 'sphere': 1, 'cuboid': 1, 'plane': 1}
points, normals, labels = generator.generate_scene(num_objects, noise_level=0.01)

print(f"Total points: {len(points)}")
print(f"Label distribution:")
unique, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique, counts):
    if label in config.label_to_shape:
        print(f"  {config.label_to_shape[label]} (label {label}): {count} points")
    else:
        print(f"  unmapped (label {label}): {count} points")

# Test model prediction
print(f"\n=== Testing model prediction ===")
device = 'cpu'
model = get_model(config.num_classes).to(device)
model.load_state_dict(state_dict)
model.eval()

# Sample a fixed number of points
num_test_points = 4096
if len(points) > num_test_points:
    indices = np.random.choice(len(points), num_test_points, replace=False)
else:
    indices = np.random.choice(len(points), num_test_points, replace=True)

test_points = points[indices]
test_normals = normals[indices]

# Normalize
center = test_points.mean(axis=0)
test_points = test_points - center

# Combine features
features = np.concatenate([test_points, test_normals], axis=1)
features_tensor = torch.FloatTensor(features).transpose(2, 1).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(features_tensor)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    pred_label = probs.argmax(axis=0)

print(f"\nModel prediction:")
print(f"  Predicted label: {pred_label}")
print(f"  Probabilities per class:")
for i, prob in enumerate(probs):
    if i in config.label_to_shape:
        print(f"    {config.label_to_shape[i]} (label {i}): {prob:.4f}")
    else:
        print(f"    unmapped (label {i}): {prob:.4f}")

print(f"\n  Max probability: {probs.max():.4f}")
print(f"  Expected random prob: {1.0/config.num_classes:.4f}")

# Check if model is better than random
random_prob = 1.0 / config.num_classes
if probs.max() > random_prob * 2:
    print(f"  [OK] Model seems to have learned something")
else:
    print(f"  [PROBLEM] Model is not learning (max prob {probs.max():.4f} ~ random {random_prob:.4f})")

print("\n" + "=" * 60)
