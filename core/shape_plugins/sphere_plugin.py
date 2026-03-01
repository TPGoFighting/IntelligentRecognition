"""
Sphere shape plugin for universal recognition system.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import pyransac3d as pyrsc
import pyvista as pv

from .base_shape import BaseShape


class SpherePlugin(BaseShape):
    """Plugin for sphere detection and fitting."""

    def generate_points(self, center: Optional[np.ndarray] = None,
                       radius: Optional[float] = None,
                       num_points: int = 1000,
                       noise: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic sphere point cloud.

        Args:
            center: (3,) center point, random if None
            radius: sphere radius, random if None
            num_points: number of points to generate
            noise: Gaussian noise std dev

        Returns:
            points: (N, 3) array
            normals: (N, 3) array
        """
        # Sample parameters if not provided
        if center is None:
            bounds = self.config.get('scene_bounds', [[-5, 5], [-5, 5], [0, 10]])
            center = np.array([
                np.random.uniform(bounds[0][0], bounds[0][1]),
                np.random.uniform(bounds[1][0], bounds[1][1]),
                np.random.uniform(bounds[2][0], bounds[2][1])
            ])

        if radius is None:
            r_min, r_max = self.params_config['radius_range']
            radius = np.random.uniform(r_min, r_max)

        # Generate points on sphere surface using spherical coordinates
        # Uniform sampling on sphere
        phi = np.random.uniform(0, 2 * np.pi, num_points)
        cos_theta = np.random.uniform(-1, 1, num_points)
        theta = np.arccos(cos_theta)

        # Convert to Cartesian
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        points = np.stack([x, y, z], axis=1) + center

        # Add noise
        if noise > 0:
            points += np.random.randn(*points.shape) * noise

        # Compute normals (radial direction from center)
        normals = points - center
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        return points, normals

    def fit(self, points: np.ndarray, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fit sphere to point cloud using RANSAC.

        Args:
            points: (N, 3) array of points
            **kwargs: Additional parameters

        Returns:
            Dictionary with keys: center, radius
        """
        if len(points) < self.min_points:
            return None

        # Get fitting parameters
        threshold = kwargs.get('threshold', self.fitting_config.get('threshold', 0.05))
        max_iterations = kwargs.get('max_iterations',
                                   self.fitting_config.get('max_iterations', 1000))

        try:
            # Use pyransac3d for sphere fitting
            sph = pyrsc.Sphere()
            center, radius, inliers = sph.fit(
                points, thresh=threshold, maxIteration=max_iterations
            )

            print(f"    [Sphere fit] center={center}, radius={radius}, inliers={len(inliers) if inliers is not None else 0}")

            if center is None or len(inliers) < self.fitting_config.get('min_inliers', 30):
                print(f"    [Sphere fit failed] Not enough inliers or center is None")
                return None

            return {
                'center': center,
                'radius': radius,
                'inliers': inliers
            }

        except Exception as e:
            print(f"Sphere fitting failed: {e}")
            return None

    def validate(self, params: Dict[str, Any]) -> bool:
        """
        Validate sphere parameters against physical constraints.

        Args:
            params: Dictionary with center, radius

        Returns:
            True if valid
        """
        if params is None:
            print("    [Sphere validation] params is None")
            return False

        # Check radius range - much more lenient validation
        r_min, r_max = self.params_config['radius_range']
        r = params['radius']

        # Allow 200% tolerance on range (0.33x to 3x)
        if r < r_min * 0.33 or r > r_max * 3.0:
            print(f"    [Sphere validation failed] radius={r:.4f}, expected range [{r_min:.4f}, {r_max:.4f}]")
            return False

        print(f"    [Sphere validation passed] radius={r:.4f}")
        return True

    def compute_inliers(self, points: np.ndarray, params: Dict[str, Any],
                       threshold: float) -> np.ndarray:
        """
        Compute inlier indices for sphere.

        Args:
            points: (N, 3) array
            params: Dictionary with center, radius
            threshold: Distance threshold

        Returns:
            Array of inlier indices
        """
        center = params['center']
        radius = params['radius']

        # Compute distance from each point to sphere surface
        distances = np.linalg.norm(points - center, axis=1)
        errors = np.abs(distances - radius)

        # Find inliers
        inliers = np.where(errors < threshold)[0]

        return inliers

    def visualize(self, plotter: Any, params: Dict[str, Any],
                 points: Optional[np.ndarray] = None, color: str = 'green'):
        """
        Visualize sphere using PyVista.

        Args:
            plotter: PyVista plotter
            params: Dictionary with center, radius
            points: Optional inlier points
            color: Color for sphere
        """
        center = params['center']
        radius = params['radius']

        # Create sphere mesh
        sphere = pv.Sphere(center=center, radius=radius, theta_resolution=30,
                          phi_resolution=30)

        plotter.add_mesh(sphere, color=color, opacity=0.3, label='Sphere')

        # Add center point
        center_point = pv.PolyData(center.reshape(1, 3))
        plotter.add_mesh(center_point, color='yellow', point_size=10,
                        render_points_as_spheres=True, label='Center')

        # Add inlier points if provided
        if points is not None and len(points) > 0:
            point_cloud = pv.PolyData(points)
            plotter.add_mesh(point_cloud, color=color, point_size=3,
                           render_points_as_spheres=True, label='Inliers')

    def sample_parameters(self) -> Dict[str, Any]:
        """Sample random sphere parameters for data generation."""
        r_min, r_max = self.params_config['radius_range']

        return {
            'radius': np.random.uniform(r_min, r_max),
            'center': None  # Will be random in generate_points
        }
