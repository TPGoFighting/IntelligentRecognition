"""
Debug the universal shape detection system.
Tests data generation, model training, and inference.
"""
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.config_loader import load_config
from core.data_generator import UniversalDataGenerator
from core.inference_engine import UniversalInferenceEngine
from core.shape_plugins import get_plugin_class

def test_data_generation():
    """Test if data generation works correctly."""
    print("=== Testing Data Generation ===")

    config = load_config("config/shape_config.yaml")
    print(f"Config loaded: {config.shape_names}")

    generator = UniversalDataGenerator(config)

    # Generate a test scene with one of each shape
    num_objects = {shape_name: 1 for shape_name in config.shape_names}
    points, normals, labels = generator.generate_scene(num_objects)

    print(f"\nGenerated {len(points)} points")

    # Check label distribution
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        shape_name = config.label_to_shape.get(label, f"unknown({label})")
        print(f"  {shape_name} (label {label}): {count} points")

    # Verify each shape has reasonable point count
    for shape_name, shape_config in config.shapes.items():
        label = shape_config.label
        shape_points = points[labels == label]
        print(f"\n{shape_name}:")
        print(f"  Points: {len(shape_points)} (min_points: {shape_config.min_points})")
        if len(shape_points) > 0:
            print(f"  Bounds: [{shape_points.min(axis=0)}] to [{shape_points.max(axis=0)}]")
            print(f"  Mean: {shape_points.mean(axis=0)}")

    return points, normals, labels, config

def test_ransac_detection(points, normals, labels, config):
    """Test if RANSAC can detect shapes in generated data."""
    print("\n=== Testing RANSAC Detection ===")

    detected_objects = []

    for shape_name, shape_config in config.shapes.items():
        label = shape_config.label
        shape_points = points[labels == label]

        if len(shape_points) < shape_config.min_points:
            print(f"\n{shape_name}: Not enough points ({len(shape_points)} < {shape_config.min_points})")
            continue

        print(f"\n{shape_name} (label {label}, {len(shape_points)} points):")

        # Get plugin
        plugin_class = get_plugin_class(shape_name)
        config_dict = {
            'params': shape_config.params,
            'fitting': shape_config.fitting,
            'min_points': shape_config.min_points,
            'scene_bounds': config.scene['bounds']
        }
        plugin = plugin_class(config_dict)

        # Try RANSAC
        params = plugin.fit(
            shape_points,
            threshold=shape_config.fitting['threshold'],
            max_iterations=shape_config.fitting['max_iterations']
        )

        if params is None:
            print(f"  RANSAC failed to fit {shape_name}")
        else:
            # Validate
            if plugin.validate(params):
                print(f"  RANSAC success!")

                # Compute inliers
                inlier_indices = plugin.compute_inliers(
                    shape_points, params, shape_config.fitting['threshold']
                )
                inlier_points = shape_points[inlier_indices]
                confidence = len(inlier_indices) / len(shape_points)

                print(f"    Inliers: {len(inlier_indices)}/{len(shape_points)}")
                print(f"    Confidence: {confidence:.3f}")
                print(f"    Params: {params}")

                # Create a simple detected object
                detected_objects.append({
                    'shape_type': shape_name,
                    'params': params,
                    'points': inlier_points,
                    'confidence': confidence
                })
            else:
                print(f"  RANSAC fit but validation failed")

    print(f"\nTotal RANSAC detections: {len(detected_objects)}")
    return detected_objects

def test_inference_engine():
    """Test the inference engine with generated data."""
    print("\n=== Testing Inference Engine ===")

    config = load_config("config/shape_config.yaml")

    # Check if model exists
    model_path = "models/universal/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Will test pure RANSAC mode")
        model_path = "models/universal/latest_model.pth"  # Try latest

    # Generate test data
    generator = UniversalDataGenerator(config)
    num_objects = {shape_name: 1 for shape_name in config.shape_names}
    points, normals, labels = generator.generate_scene(num_objects)

    print(f"Generated {len(points)} points for inference test")

    # Initialize engine
    try:
        engine = UniversalInferenceEngine(model_path, config, device='cpu')

        # Run inference
        print("\nRunning inference...")
        objects = engine.infer(points, normals)

        print(f"\nInference results: {len(objects)} objects detected")
        for i, obj in enumerate(objects):
            print(f"  #{i+1}: {obj.shape_type}, confidence={obj.confidence:.3f}, points={obj.num_points}")

    except Exception as e:
        print(f"Inference engine error: {e}")
        import traceback
        traceback.print_exc()

        # Try pure RANSAC as fallback
        print("\nTrying pure RANSAC fallback...")
        detected_objects = []
        for shape_name, shape_config in config.shapes.items():
            plugin_class = get_plugin_class(shape_name)
            config_dict = {
                'params': shape_config.params,
                'fitting': shape_config.fitting,
                'min_points': shape_config.min_points,
                'scene_bounds': config.scene['bounds']
            }
            plugin = plugin_class(config_dict)

            # Try RANSAC on all points
            params = plugin.fit(
                points,
                threshold=shape_config.fitting['threshold'],
                max_iterations=shape_config.fitting['max_iterations']
            )

            if params is not None and plugin.validate(params):
                # Compute inliers
                inlier_indices = plugin.compute_inliers(
                    points, params, shape_config.fitting['threshold']
                )
                if len(inlier_indices) >= shape_config.fitting['min_inliers']:
                    inlier_points = points[inlier_indices]
                    confidence = len(inlier_indices) / len(points)

                    detected_objects.append({
                        'shape_type': shape_name,
                        'params': params,
                        'points': inlier_points,
                        'confidence': confidence
                    })

        print(f"Pure RANSAC results: {len(detected_objects)} objects detected")
        for i, obj in enumerate(detected_objects):
            print(f"  #{i+1}: {obj['shape_type']}, confidence={obj['confidence']:.3f}, points={len(obj['points'])}")

def check_model_training():
    """Check if model is properly trained."""
    print("\n=== Checking Model Training ===")

    model_path = "models/universal/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False

    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        print(f"Checkpoint keys: {list(checkpoint.keys())}")

        if 'best_acc' in checkpoint:
            accuracy = checkpoint['best_acc']
            print(f"Best accuracy in checkpoint: {accuracy:.4f}")
            if accuracy < 0.3:
                print(f"WARNING: Model accuracy is low ({accuracy:.4f} < 0.3)")
                return False
            else:
                print(f"Model accuracy OK: {accuracy:.4f}")
                return True
        elif 'val_acc' in checkpoint:
            accuracy = checkpoint['val_acc']
            print(f"Validation accuracy in checkpoint: {accuracy:.4f}")
            if accuracy < 0.3:
                print(f"WARNING: Model validation accuracy is low ({accuracy:.4f} < 0.3)")
                return False
            else:
                print(f"Model validation accuracy OK: {accuracy:.4f}")
                return True
        else:
            print("WARNING: No accuracy information in checkpoint")
            return False

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

def main():
    """Run all diagnostic tests."""
    print("=" * 60)
    print("Universal Shape Detection System Diagnostic")
    print("=" * 60)

    # Test 1: Data generation
    points, normals, labels, config = test_data_generation()

    # Test 2: RANSAC detection
    test_ransac_detection(points, normals, labels, config)

    # Test 3: Model training status
    model_ok = check_model_training()

    # Test 4: Inference engine
    test_inference_engine()

    print("\n" + "=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)

    if not model_ok:
        print("\nRECOMMENDATIONS:")
        print("1. Model is not well-trained. Consider retraining with better parameters.")
        print("2. Use pure RANSAC mode for now (inference engine should fallback automatically).")
        print("3. Check training data balance and class weights.")

if __name__ == "__main__":
    main()