"""
Test script to verify the universal shape recognition system.
Tests configuration loading, data generation, and plugin functionality.
"""

import sys
from pathlib import Path
import numpy as np
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.append(str(Path(__file__).parent.parent))

from core.config_loader import load_config
from core.data_generator import UniversalDataGenerator
from core.shape_plugins import get_plugin_class


def test_config_loading():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("TEST 1: Configuration Loading")
    print("="*60)

    try:
        config = load_config("config/shape_config.yaml")
        print(f"[PASS] Configuration loaded successfully")
        print(f"   Number of classes: {config.num_classes}")
        print(f"   Shape names: {config.shape_names}")
        print(f"   Label mapping: {config.label_to_shape}")
        return True
    except Exception as e:
        print(f"[FAIL] Configuration loading failed: {e}")
        return False


def test_plugin_loading():
    """Test shape plugin loading."""
    print("\n" + "="*60)
    print("TEST 2: Plugin Loading")
    print("="*60)

    shapes = ['cylinder', 'sphere', 'cuboid', 'plane']
    all_passed = True

    for shape_name in shapes:
        try:
            plugin_class = get_plugin_class(shape_name)
            print(f"[PASS] {shape_name}: {plugin_class.__name__} loaded")
        except Exception as e:
            print(f"[FAIL] {shape_name}: Failed to load - {e}")
            all_passed = False

    return all_passed


def test_data_generation():
    """Test data generation for each shape."""
    print("\n" + "="*60)
    print("TEST 3: Data Generation")
    print("="*60)

    try:
        config = load_config("config/shape_config.yaml")
        generator = UniversalDataGenerator(config)

        # Test generating each shape type
        for shape_name in config.shape_names:
            print(f"\n  Testing {shape_name}...")
            plugin = generator.shape_plugins[shape_name]

            # Sample parameters
            params = plugin.sample_parameters()

            # Generate points
            points, normals = plugin.generate_points(num_points=500, **params)

            print(f"    [PASS] Generated {len(points)} points")
            print(f"       Points shape: {points.shape}")
            print(f"       Normals shape: {normals.shape}")
            print(f"       Points range: [{points.min():.2f}, {points.max():.2f}]")

        print(f"\n[PASS] All shape generation tests passed")
        return True

    except Exception as e:
        print(f"[FAIL] Data generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shape_fitting():
    """Test shape fitting for each plugin."""
    print("\n" + "="*60)
    print("TEST 4: Shape Fitting")
    print("="*60)

    try:
        config = load_config("config/shape_config.yaml")
        generator = UniversalDataGenerator(config)

        for shape_name in config.shape_names:
            print(f"\n  Testing {shape_name} fitting...")
            plugin = generator.shape_plugins[shape_name]

            # Generate test data
            params_true = plugin.sample_parameters()
            points, normals = plugin.generate_points(num_points=1000, noise=0.01, **params_true)

            # Fit shape
            params_fitted = plugin.fit(points)

            if params_fitted is None:
                print(f"    [FAIL] Fitting failed")
                continue

            # Validate
            is_valid = plugin.validate(params_fitted)
            print(f"    [PASS] Fitting succeeded, valid={is_valid}")
            print(f"       Fitted params: {list(params_fitted.keys())}")

            # Compute inliers
            inliers = plugin.compute_inliers(points, params_fitted, threshold=0.1)
            print(f"       Inliers: {len(inliers)}/{len(points)} ({len(inliers)/len(points)*100:.1f}%)")

        print(f"\n[PASS] All shape fitting tests passed")
        return True

    except Exception as e:
        print(f"[FAIL] Shape fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scene_generation():
    """Test complete scene generation."""
    print("\n" + "="*60)
    print("TEST 5: Scene Generation")
    print("="*60)

    try:
        config = load_config("config/shape_config.yaml")
        generator = UniversalDataGenerator(config)

        # Generate a scene with multiple objects
        num_objects = {
            "cylinder": 2,
            "sphere": 1,
            "cuboid": 1
        }

        points, normals, labels = generator.generate_scene(num_objects)

        print(f"[PASS] Scene generated successfully")
        print(f"   Total points: {len(points)}")
        print(f"   Points shape: {points.shape}")
        print(f"   Normals shape: {normals.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"\n   Label distribution:")

        for label_id, shape_name in config.label_to_shape.items():
            count = np.sum(labels == label_id)
            percentage = count / len(labels) * 100
            print(f"     {shape_name} (label {label_id}): {count} points ({percentage:.1f}%)")

        return True

    except Exception as e:
        print(f"[FAIL] Scene generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_configs():
    """Test loading different configuration files."""
    print("\n" + "="*60)
    print("TEST 6: Multiple Configurations")
    print("="*60)

    configs = [
        "config/shape_config.yaml",
        "config/cylinder_config.yaml",
        "config/indoor_config.yaml"
    ]

    all_passed = True
    for config_path in configs:
        try:
            if not Path(config_path).exists():
                print(f"[WARN] {config_path}: File not found, skipping")
                continue

            config = load_config(config_path)
            print(f"[PASS] {config_path}")
            print(f"   Shapes: {config.shape_names}")
            print(f"   Num classes: {config.num_classes}")

        except Exception as e:
            print(f"[FAIL] {config_path}: Failed - {e}")
            all_passed = False

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" UNIVERSAL SHAPE RECOGNITION SYSTEM - TEST SUITE")
    print("="*70)

    tests = [
        ("Configuration Loading", test_config_loading),
        ("Plugin Loading", test_plugin_loading),
        ("Data Generation", test_data_generation),
        ("Shape Fitting", test_shape_fitting),
        ("Scene Generation", test_scene_generation),
        ("Multiple Configurations", test_multiple_configs)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n[FAIL] {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")

    print("\n" + "-"*70)
    print(f"Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\nALL TESTS PASSED! System is ready to use.")
        return 0
    else:
        print(f"\n{total_count - passed_count} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
