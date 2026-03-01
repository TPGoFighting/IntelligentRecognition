"""
Universal inference engine for multi-shape point cloud recognition.
Handles sliding window inference and iterative shape extraction.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from core.config_loader import UniversalConfig
from core.shape_plugins import get_plugin_class, DetectedObject


class UniversalInferenceEngine:
    """
    Universal inference engine for shape detection.
    Supports multiple shape types through plugin architecture.
    """

    def __init__(self, model_path: str, config: UniversalConfig, device: str = 'cuda'):
        """
        Initialize inference engine.

        Args:
            model_path: Path to trained model checkpoint
            config: UniversalConfig object
            device: 'cuda' or 'cpu'
        """
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.shape_plugins = self._load_plugins()

        print(f"Inference engine initialized on {self.device}")
        print(f"Loaded {len(self.shape_plugins)} shape plugins: {list(self.shape_plugins.keys())}")

    def _load_model(self, model_path: str):
        """Load trained PointNet++ model."""
        from models.pointnet2_sem_seg import get_model

        # 先加载检查点以确定类别数
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 从检查点中推断类别数
        num_classes = self.config.num_classes  # 默认使用配置
        for key in state_dict:
            if key.endswith('conv2.weight'):
                checkpoint_num_classes = state_dict[key].shape[0]
                if checkpoint_num_classes != num_classes:
                    print(f"[WARNING] Checkpoint has {checkpoint_num_classes} classes, but config specifies {num_classes}. Using checkpoint value.")
                    num_classes = checkpoint_num_classes
                break

        # 使用正确的类别数创建模型
        model = get_model(num_classes).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded from {model_path} with {num_classes} classes")
        return model

    def _load_plugins(self) -> Dict:
        """Load shape plugins based on configuration."""
        plugins = {}
        for shape_name, shape_config in self.config.shapes.items():
            plugin_class = get_plugin_class(shape_name)
            config_dict = {
                'params': shape_config.params,
                'fitting': shape_config.fitting,
                'min_points': shape_config.min_points,
                'scene_bounds': self.config.scene['bounds']
            }
            plugins[shape_name] = plugin_class(config_dict)
        return plugins

    def infer(self, points: np.ndarray, normals: np.ndarray) -> List[DetectedObject]:
        """
        Run complete inference pipeline on point cloud.

        Args:
            points: (N, 3) array of 3D points
            normals: (N, 3) array of normals

        Returns:
            List of DetectedObject instances
        """
        print(f"\n=== Starting inference on {len(points)} points ===")

        # Try deep learning model first, but fallback to RANSAC if it's not trained
        use_model = True
        try:
            # Check if model is trained (accuracy > 30%)
            if hasattr(self, 'model_path') and self.model_path:
                model_path = Path(self.model_path)
                if model_path.exists():
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    # Check if checkpoint has accuracy metrics
                    if 'best_acc' in checkpoint:
                        if checkpoint['best_acc'] < 0.3:
                            print(f"  Model accuracy is low ({checkpoint['best_acc']:.2f}), using pure RANSAC mode")
                            use_model = False
                        else:
                            print(f"  Model accuracy: {checkpoint['best_acc']:.2f}, using model+RANSAC mode")
                    elif 'val_acc' in checkpoint:
                        if checkpoint['val_acc'] < 0.3:
                            print(f"  Model validation accuracy is low ({checkpoint['val_acc']:.2f}), using pure RANSAC mode")
                            use_model = False
                        else:
                            print(f"  Model validation accuracy: {checkpoint['val_acc']:.2f}, using model+RANSAC mode")
                    else:
                        # No accuracy info, assume model is not well-trained
                        print(f"  No accuracy info in checkpoint, using pure RANSAC mode")
                        use_model = False
        except Exception as e:
            print(f"  Failed to check model accuracy: {e}, using pure RANSAC mode")
            use_model = False

        if use_model:
            # Step 1: Sliding window inference
            vote_results = self._sliding_window_inference(points, normals)

            # Step 2: Extract shapes for each class
            detected_objects = []
            for shape_name, shape_config in self.config.shapes.items():
                label = shape_config.label
                plugin = self.shape_plugins[shape_name]

                # Extract points predicted as this shape
                shape_mask = vote_results == label
                shape_points = points[shape_mask]

                # If too few points predicted, use RANSAC directly on all points
                if len(shape_points) < shape_config.min_points:
                    print(f"  {shape_name}: insufficient predicted points ({len(shape_points)} < {shape_config.min_points}), skipping")
                    continue

                print(f"  {shape_name}: {len(shape_points)} candidate points")

                # Step 3: Iterative extraction of multiple instances
                objects = self._iterative_extraction(shape_points, plugin, shape_config, shape_name)
                detected_objects.extend(objects)
        else:
            # Pure RANSAC mode - try each shape on all points
            print(f"  Using pure RANSAC mode (model not well-trained)")
            detected_objects = []
            for shape_name, shape_config in self.config.shapes.items():
                plugin = self.shape_plugins[shape_name]
                print(f"  {shape_name}: trying RANSAC on all {len(points)} points")

                # Use RANSAC directly on all points
                objects = self._iterative_extraction(points, plugin, shape_config, shape_name)
                detected_objects.extend(objects)

        print(f"\n[OK] Detected {len(detected_objects)} objects total")
        return detected_objects

    def _sliding_window_inference(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """
        Perform sliding window inference with voting.

        Args:
            points: (N, 3) array
            normals: (N, 3) array

        Returns:
            vote_results: (N,) array of predicted labels
        """
        block_size = self.config.inference['block_size']
        stride = self.config.inference['stride']
        num_points = self.config.inference['num_points']
        batch_size = self.config.inference.get('batch_size', 32)
        max_windows = self.config.inference.get('max_windows', 200)

        # Initialize vote matrix
        vote_matrix = np.zeros((len(points), self.config.num_classes), dtype=np.float32)

        # Compute bounding box
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)

        # Determine main dimension for sliding (use the dimension with largest range)
        ranges = max_bound - min_bound
        main_dim = np.argmax(ranges)  # 0=x, 1=y, 2=z

        print(f"  Point cloud bounds: min={min_bound}, max={max_bound}")
        print(f"  Main dimension: {main_dim} (range={ranges[main_dim]:.2f})")

        # Generate sliding windows along the main dimension
        windows = []
        num_windows = int(np.ceil((ranges[main_dim] - block_size) / stride)) + 1

        # Ensure we have at least some windows
        if num_windows < 1:
            num_windows = 1

        # Generate window positions along main dimension
        for i in range(num_windows):
            pos = min_bound[main_dim] + i * stride
            windows.append((main_dim, pos))

        # Limit number of windows
        if len(windows) > max_windows:
            indices = np.random.choice(len(windows), max_windows, replace=False)
            windows = [windows[i] for i in indices]

        print(f"  Processing {len(windows)} windows along dimension {main_dim}...")

        # Process windows in batches
        for batch_start in range(0, len(windows), batch_size):
            batch_windows = windows[batch_start:batch_start + batch_size]
            batch_data = []
            batch_indices = []

            for main_dim, pos in batch_windows:
                # Extract points in window (along main dimension)
                mask = (points[:, main_dim] >= pos) & (points[:, main_dim] < pos + block_size)

                window_indices = np.where(mask)[0]

                if len(window_indices) < 100:
                    continue

                # Sample to num_points (with replacement if needed)
                sampled = np.random.choice(window_indices, num_points, replace=len(window_indices) < num_points)
                # Ensure sampled is 1D array
                sampled = np.atleast_1d(sampled).flatten()

                window_points = points[sampled]
                window_normals = normals[sampled]

                # Normalize to window center
                center = window_points.mean(axis=0)
                window_points = window_points - center

                # Combine features
                features = np.concatenate([window_points, window_normals], axis=1)
                batch_data.append(features)
                batch_indices.append(sampled)

            if not batch_data:
                continue

            # Run inference on batch
            batch_tensor = torch.FloatTensor(np.array(batch_data)).transpose(2, 1).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

            # Accumulate votes
            for i, indices in enumerate(batch_indices):
                # Ensure indices is 1D array
                indices = np.asarray(indices).ravel()
                n_idx = len(indices)

                # probs[i] shape is (num_classes, num_points) = (6, 4096)
                # Need to transpose to (num_points, num_classes) = (4096, 6)
                prob_transposed = probs[i].T  # Should be (4096, 6)
                # Ensure it's 2D and has correct shape
                prob_transposed = prob_transposed.reshape(-1, self.config.num_classes)

                # Only use first n_idx rows if needed
                if n_idx < prob_transposed.shape[0]:
                    prob_to_add = prob_transposed[:n_idx, :]
                else:
                    prob_to_add = prob_transposed

                # Add to vote matrix
                vote_matrix[indices] += prob_to_add

        # Get final predictions
        # Also compute max probability for each point
        max_probs = vote_matrix.max(axis=1)
        vote_results = vote_matrix.argmax(axis=1)

        # Print distribution for all labels (including unmapped ones)
        print(f"  Prediction distribution:")
        for label_id in range(self.config.num_classes):
            count = np.sum(vote_results == label_id)
            if label_id in self.config.label_to_shape:
                print(f"    {self.config.label_to_shape[label_id]} (label {label_id}): {count} points")
            else:
                print(f"    unmapped (label {label_id}): {count} points")

        # Print probability statistics
        print(f"  Probability statistics:")
        print(f"    Max probability: {max_probs.max():.4f}")
        print(f"    Mean probability: {max_probs.mean():.4f}")
        print(f"    Min probability: {max_probs.min():.4f}")

        # Debug: check if model has learned anything (if mean prob >> 1/num_classes)
        random_prob = 1.0 / self.config.num_classes
        if max_probs.mean() < random_prob * 1.5:
            print(f"  [WARNING] Model probabilities too low (mean {max_probs.mean():.4f} vs expected ~{random_prob:.4f})")
            print(f"            Model may not be trained properly or data has issues")

        return vote_results

    def _iterative_extraction(self, points: np.ndarray, plugin: Any,
                             shape_config: Any, shape_name: str) -> List[DetectedObject]:
        """
        Iteratively extract multiple instances of the same shape type.

        Args:
            points: Candidate points for this shape
            plugin: Shape plugin instance
            shape_config: Shape configuration
            shape_name: Name of shape

        Returns:
            List of DetectedObject instances
        """
        objects = []
        remaining_points = points.copy()
        max_iterations = 10  # Prevent infinite loops

        iteration = 0
        while len(remaining_points) >= shape_config.min_points and iteration < max_iterations:
            iteration += 1

            # Fit shape
            params = plugin.fit(remaining_points,
                              threshold=shape_config.fitting['threshold'],
                              max_iterations=shape_config.fitting['max_iterations'])

            if params is None:
                break

            # Validate
            if not plugin.validate(params):
                print(f"    {shape_name} #{iteration}: validation failed")
                break

            # Compute inliers
            inlier_indices = plugin.compute_inliers(
                remaining_points, params, shape_config.fitting['threshold']
            )

            if len(inlier_indices) < shape_config.fitting['min_inliers']:
                print(f"    {shape_name} #{iteration}: insufficient inliers ({len(inlier_indices)})")
                break

            # Create detected object
            inlier_points = remaining_points[inlier_indices]
            confidence = len(inlier_indices) / len(remaining_points)

            obj = DetectedObject(
                shape_type=shape_name,
                params=params,
                points=inlier_points,
                confidence=confidence
            )
            objects.append(obj)

            print(f"    ✅ {shape_name} #{iteration}: {len(inlier_indices)} points, confidence={confidence:.3f}")

            # Remove inliers from remaining points
            remaining_points = np.delete(remaining_points, inlier_indices, axis=0)

        return objects

    def infer_batch(self, point_clouds: List[Tuple[np.ndarray, np.ndarray]]) -> List[List[DetectedObject]]:
        """
        Run inference on multiple point clouds.

        Args:
            point_clouds: List of (points, normals) tuples

        Returns:
            List of detection results for each point cloud
        """
        results = []
        for i, (points, normals) in enumerate(point_clouds):
            print(f"\n=== Processing point cloud {i+1}/{len(point_clouds)} ===")
            objects = self.infer(points, normals)
            results.append(objects)
        return results

    def save_results(self, detected_objects: List[DetectedObject], output_path: str):
        """
        Save detection results to file.

        Args:
            detected_objects: List of DetectedObject instances
            output_path: Path to save results (JSON format)
        """
        import json

        results = {
            'num_objects': len(detected_objects),
            'objects': [obj.to_dict() for obj in detected_objects]
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Test inference engine
    from core.config_loader import load_config

    config = load_config("config/shape_config.yaml")

    # This would normally load a trained model
    # engine = UniversalInferenceEngine("models/best_model.pth", config)

    print("Inference engine test completed")
