"""
Batch inference script for processing multiple point cloud files.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))

from core.config_loader import load_config
from core.inference_engine import UniversalInferenceEngine


def load_point_cloud(file_path: str):
    """
    Load point cloud from various formats.

    Supports:
    - .npz (numpy compressed)
    - .npy (numpy)
    - .txt (space-separated x y z nx ny nz)
    """
    file_path = Path(file_path)

    if file_path.suffix == '.npz':
        data = np.load(file_path)
        points = data['points']
        normals = data.get('normals', None)

        if normals is None:
            # Compute normals if not provided (simple estimation)
            normals = np.zeros_like(points)
            normals[:, 2] = 1.0  # Default upward normals

        return points, normals

    elif file_path.suffix == '.npy':
        data = np.load(file_path)
        if data.shape[1] == 6:
            points = data[:, :3]
            normals = data[:, 3:]
        else:
            points = data
            normals = np.zeros_like(points)
            normals[:, 2] = 1.0

        return points, normals

    elif file_path.suffix == '.txt':
        data = np.loadtxt(file_path)
        if data.shape[1] >= 6:
            points = data[:, :3]
            normals = data[:, 3:6]
        else:
            points = data[:, :3]
            normals = np.zeros_like(points)
            normals[:, 2] = 1.0

        return points, normals

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def main():
    parser = argparse.ArgumentParser(description='Batch inference on point cloud files')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory or file pattern (e.g., data/*.npz)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization for each result')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Initialize inference engine
    print(f"Loading model from {args.model}")
    engine = UniversalInferenceEngine(args.model, config, device=args.device)

    # Find input files
    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob('*.npz')) + list(input_path.glob('*.npy'))
    else:
        # Glob pattern
        files = list(Path('.').glob(args.input))

    if len(files) == 0:
        print(f"No files found matching: {args.input}")
        return

    print(f"Found {len(files)} files to process")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    all_results = {}

    for file_path in tqdm(files, desc='Processing files'):
        try:
            # Load point cloud
            points, normals = load_point_cloud(file_path)

            # Run inference
            detected_objects = engine.infer(points, normals)

            # Save results
            result_file = output_dir / f"{file_path.stem}_results.json"
            engine.save_results(detected_objects, result_file)

            # Store summary
            all_results[str(file_path)] = {
                'num_objects': len(detected_objects),
                'objects': [obj.to_dict() for obj in detected_objects]
            }

            # Visualize if requested
            if args.visualize:
                visualize_results(points, detected_objects, config,
                                output_dir / f"{file_path.stem}_viz.png")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Save summary
    summary_file = output_dir / 'batch_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Batch processing complete")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_file}")


def visualize_results(points, detected_objects, config, output_path):
    """Generate visualization of detection results."""
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True)

    # Add background points
    bg_cloud = pv.PolyData(points)
    plotter.add_mesh(bg_cloud, color='gray', point_size=1, opacity=0.3)

    # Add detected objects
    from core.shape_plugins import get_plugin_class

    for obj in detected_objects:
        plugin_class = get_plugin_class(obj.shape_type)
        shape_config = config.shapes[obj.shape_type]
        plugin = plugin_class({
            'params': shape_config.params,
            'fitting': shape_config.fitting,
            'min_points': shape_config.min_points
        })

        color = config.visualization['shape_colors'].get(obj.shape_type, 'red')
        plugin.visualize(plotter, obj.params, obj.points, color=color)

    plotter.camera_position = 'iso'
    plotter.screenshot(output_path)
    plotter.close()


if __name__ == "__main__":
    main()
