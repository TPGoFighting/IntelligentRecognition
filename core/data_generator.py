"""
Universal data generator for multi-shape point cloud recognition.
Replaces the tunnel-specific data generator with a configurable system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config_loader import UniversalConfig
from core.shape_plugins import get_plugin_class


class UniversalDataGenerator:
    """
    Generate synthetic point cloud data with multiple shape types.
    Configuration-driven and extensible.
    """

    def __init__(self, config: UniversalConfig):
        """
        Initialize generator with configuration.

        Args:
            config: UniversalConfig object
        """
        self.config = config
        self.shape_plugins = self._load_plugins()

    def _load_plugins(self) -> Dict:
        """Load shape plugins based on configuration."""
        plugins = {}
        for shape_name, shape_config in self.config.shapes.items():
            plugin_class = get_plugin_class(shape_name)
            # Pass full config dict to plugin
            config_dict = {
                'params': shape_config.params,
                'fitting': shape_config.fitting,
                'min_points': shape_config.min_points,
                'scene_bounds': self.config.scene['bounds']
            }
            plugins[shape_name] = plugin_class(config_dict)
        return plugins

    def generate_scene(self, num_objects: Dict[str, int],
                      noise_level: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a complete scene with multiple shapes and background.

        Args:
            num_objects: Dictionary mapping shape names to counts
                        e.g., {"cylinder": 3, "sphere": 2}
            noise_level: Gaussian noise std dev (uses config default if None)

        Returns:
            points: (N, 3) array of all points
            normals: (N, 3) array of all normals
            labels: (N,) array of labels (0=background, 1+=shapes)
        """
        if noise_level is None:
            noise_level = self.config.scene.get('noise_level', 0.01)

        all_points = []
        all_normals = []
        all_labels = []

        # Generate each shape type
        for shape_name, count in num_objects.items():
            if shape_name not in self.shape_plugins:
                print(f"Warning: Unknown shape '{shape_name}', skipping")
                continue

            plugin = self.shape_plugins[shape_name]
            label = self.config.shapes[shape_name].label

            for i in range(count):
                # Sample parameters
                params = plugin.sample_parameters()

                # Generate points
                pts, norms = plugin.generate_points(noise=noise_level, **params)

                all_points.append(pts)
                all_normals.append(norms)
                all_labels.append(np.full(len(pts), label, dtype=np.int32))

                print(f"Generated {shape_name} {i+1}/{count}: {len(pts)} points")

        # Generate background
        bg_density = self.config.scene.get('background_density', 50000)
        if bg_density > 0:
            bg_pts, bg_norms = self._generate_background(bg_density, noise_level)
            all_points.append(bg_pts)
            all_normals.append(bg_norms)
            all_labels.append(np.zeros(len(bg_pts), dtype=np.int32))
            print(f"Generated background: {len(bg_pts)} points")

        # Combine all
        points = np.vstack(all_points)
        normals = np.vstack(all_normals)
        labels = np.hstack(all_labels)

        # Shuffle
        indices = np.random.permutation(len(points))
        points = points[indices]
        normals = normals[indices]
        labels = labels[indices]

        print(f"Total scene: {len(points)} points")
        return points, normals, labels

    def _generate_background(self, num_points: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate background points based on scene type.

        Args:
            num_points: Number of background points
            noise: Noise level

        Returns:
            points: (N, 3) array
            normals: (N, 3) array
        """
        scene_type = self.config.scene.get('type', 'generic')
        bounds = self.config.scene['bounds']

        if scene_type == 'tunnel':
            # Generate tunnel-like background
            return self._generate_tunnel_background(num_points, bounds, noise)
        elif scene_type == 'indoor':
            # Generate indoor scene (walls, floor, ceiling)
            return self._generate_indoor_background(num_points, bounds, noise)
        elif scene_type == 'outdoor':
            # Generate outdoor scene (ground plane with noise)
            return self._generate_outdoor_background(num_points, bounds, noise)
        else:
            # Generic: uniform random points
            return self._generate_generic_background(num_points, bounds, noise)

    def _generate_generic_background(self, num_points: int, bounds: List,
                                    noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate uniform random background."""
        points = np.random.uniform(
            low=[bounds[0][0], bounds[1][0], bounds[2][0]],
            high=[bounds[0][1], bounds[1][1], bounds[2][1]],
            size=(num_points, 3)
        )

        # Random normals
        normals = np.random.randn(num_points, 3)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        return points, normals

    def _generate_tunnel_background(self, num_points: int, bounds: List,
                                   noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate tunnel-shaped background."""
        # Tunnel parameters
        tunnel_radius = 5.0
        tunnel_length = bounds[2][1] - bounds[2][0]

        # Generate points on tunnel surface
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        z = np.random.uniform(bounds[2][0], bounds[2][1], num_points)

        x = tunnel_radius * np.cos(theta)
        y = tunnel_radius * np.sin(theta)

        points = np.stack([x, y, z], axis=1)
        points += np.random.randn(*points.shape) * noise

        # Normals point inward
        normals = -np.stack([np.cos(theta), np.sin(theta), np.zeros(num_points)], axis=1)

        return points, normals

    def _generate_indoor_background(self, num_points: int, bounds: List,
                                   noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate indoor scene background (walls, floor, ceiling)."""
        points_per_surface = num_points // 6

        all_points = []
        all_normals = []

        # Floor
        x = np.random.uniform(bounds[0][0], bounds[0][1], points_per_surface)
        y = np.random.uniform(bounds[1][0], bounds[1][1], points_per_surface)
        z = np.full(points_per_surface, bounds[2][0])
        all_points.append(np.stack([x, y, z], axis=1))
        all_normals.append(np.tile([0, 0, 1], (points_per_surface, 1)))

        # Ceiling
        z = np.full(points_per_surface, bounds[2][1])
        all_points.append(np.stack([x, y, z], axis=1))
        all_normals.append(np.tile([0, 0, -1], (points_per_surface, 1)))

        # 4 walls
        for wall_idx in range(4):
            if wall_idx == 0:  # x_min wall
                x = np.full(points_per_surface, bounds[0][0])
                y = np.random.uniform(bounds[1][0], bounds[1][1], points_per_surface)
                normal = [1, 0, 0]
            elif wall_idx == 1:  # x_max wall
                x = np.full(points_per_surface, bounds[0][1])
                y = np.random.uniform(bounds[1][0], bounds[1][1], points_per_surface)
                normal = [-1, 0, 0]
            elif wall_idx == 2:  # y_min wall
                x = np.random.uniform(bounds[0][0], bounds[0][1], points_per_surface)
                y = np.full(points_per_surface, bounds[1][0])
                normal = [0, 1, 0]
            else:  # y_max wall
                x = np.random.uniform(bounds[0][0], bounds[0][1], points_per_surface)
                y = np.full(points_per_surface, bounds[1][1])
                normal = [0, -1, 0]

            z = np.random.uniform(bounds[2][0], bounds[2][1], points_per_surface)
            all_points.append(np.stack([x, y, z], axis=1))
            all_normals.append(np.tile(normal, (points_per_surface, 1)))

        points = np.vstack(all_points)
        normals = np.vstack(all_normals)

        # Add noise
        points += np.random.randn(*points.shape) * noise

        return points, normals

    def _generate_outdoor_background(self, num_points: int, bounds: List,
                                    noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate outdoor scene background (ground plane)."""
        x = np.random.uniform(bounds[0][0], bounds[0][1], num_points)
        y = np.random.uniform(bounds[1][0], bounds[1][1], num_points)
        z = np.full(num_points, bounds[2][0])

        points = np.stack([x, y, z], axis=1)
        points += np.random.randn(*points.shape) * noise

        # Normals point up
        normals = np.tile([0, 0, 1], (num_points, 1))

        return points, normals

    def generate_dataset(self, num_scenes: int, objects_per_scene: Dict[str, Tuple[int, int]],
                        output_dir: str) -> None:
        """
        Generate a complete dataset with multiple scenes.

        Args:
            num_scenes: Number of scenes to generate
            objects_per_scene: Dictionary mapping shape names to (min, max) counts
                              e.g., {"cylinder": (2, 5), "sphere": (1, 3)}
            output_dir: Directory to save generated data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for scene_idx in range(num_scenes):
            print(f"\n=== Generating scene {scene_idx + 1}/{num_scenes} ===")

            # Sample number of objects for this scene
            num_objects = {}
            for shape_name, (min_count, max_count) in objects_per_scene.items():
                num_objects[shape_name] = np.random.randint(min_count, max_count + 1)

            # Generate scene
            points, normals, labels = self.generate_scene(num_objects)

            # Save
            scene_file = output_path / f"scene_{scene_idx:04d}.npz"
            np.savez_compressed(
                scene_file,
                points=points,
                normals=normals,
                labels=labels
            )
            print(f"Saved to {scene_file}")

        print(f"\n[SUCCESS] Generated {num_scenes} scenes in {output_dir}")


if __name__ == "__main__":
    # Test data generation
    from core.config_loader import load_config

    config = load_config("config/shape_config.yaml")
    generator = UniversalDataGenerator(config)

    # Generate a single test scene
    num_objects = {
        "cylinder": 2,
        "sphere": 1,
        "cuboid": 1
    }

    points, normals, labels = generator.generate_scene(num_objects)

    print(f"\nGenerated scene:")
    print(f"  Total points: {len(points)}")
    print(f"  Label distribution:")
    for label_id, shape_name in config.label_to_shape.items():
        count = np.sum(labels == label_id)
        print(f"    {shape_name} (label {label_id}): {count} points")
