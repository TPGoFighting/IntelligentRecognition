"""
Cuboid (box) shape plugin for universal recognition system.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import pyransac3d as pyrsc
import pyvista as pv
from scipy.spatial.transform import Rotation

from .base_shape import BaseShape


class CuboidPlugin(BaseShape):
    """Plugin for cuboid (rectangular box) detection and fitting."""

    def generate_points(self, center: Optional[np.ndarray] = None,
                       size: Optional[np.ndarray] = None,
                       rotation: Optional[np.ndarray] = None,
                       num_points: int = 1000,
                       noise: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic cuboid point cloud.

        Args:
            center: (3,) center point, random if None
            size: (3,) [width, depth, height], random if None
            rotation: (3,) Euler angles in radians, random if None
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

        if size is None:
            size_ranges = self.params_config['size_range']
            size = np.array([
                np.random.uniform(size_ranges[0][0], size_ranges[0][1]),
                np.random.uniform(size_ranges[1][0], size_ranges[1][1]),
                np.random.uniform(size_ranges[2][0], size_ranges[2][1])
            ])

        if rotation is None:
            rotation = np.random.uniform(0, 2 * np.pi, 3)

        # Generate points on 6 faces
        points_per_face = num_points // 6
        all_points = []
        all_normals = []

        w, d, h = size / 2  # Half sizes

        # Define 6 faces (local coordinates)
        faces = [
            # Face, normal
            ([[-w, -d, -h], [w, -d, -h], [w, d, -h], [-w, d, -h]], [0, 0, -1]),  # Bottom
            ([[-w, -d, h], [w, -d, h], [w, d, h], [-w, d, h]], [0, 0, 1]),       # Top
            ([[-w, -d, -h], [-w, d, -h], [-w, d, h], [-w, -d, h]], [-1, 0, 0]),  # Left
            ([[w, -d, -h], [w, d, -h], [w, d, h], [w, -d, h]], [1, 0, 0]),       # Right
            ([[-w, -d, -h], [w, -d, -h], [w, -d, h], [-w, -d, h]], [0, -1, 0]),  # Front
            ([[-w, d, -h], [w, d, -h], [w, d, h], [-w, d, h]], [0, 1, 0])        # Back
        ]

        for face_corners, normal in faces:
            # Sample points on face
            u = np.random.uniform(0, 1, points_per_face)
            v = np.random.uniform(0, 1, points_per_face)

            # Bilinear interpolation
            corners = np.array(face_corners)
            face_points = ((1 - u)[:, None] * (1 - v)[:, None] * corners[0] +
                          u[:, None] * (1 - v)[:, None] * corners[1] +
                          u[:, None] * v[:, None] * corners[2] +
                          (1 - u)[:, None] * v[:, None] * corners[3])

            all_points.append(face_points)
            all_normals.append(np.tile(normal, (points_per_face, 1)))

        points = np.vstack(all_points)
        normals = np.vstack(all_normals)

        # Apply rotation
        rot = Rotation.from_euler('xyz', rotation)
        points = rot.apply(points)
        normals = rot.apply(normals)

        # Translate to center
        points += center

        # Add noise
        if noise > 0:
            points += np.random.randn(*points.shape) * noise

        return points, normals

    def fit(self, points: np.ndarray, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fit cuboid to point cloud using RANSAC.

        Args:
            points: (N, 3) array of points
            **kwargs: Additional parameters

        Returns:
            Dictionary with keys: center, size, rotation
        """
        if len(points) < self.min_points:
            return None

        # Get fitting parameters
        threshold = kwargs.get('threshold', self.fitting_config.get('threshold', 0.05))
        max_iterations = kwargs.get('max_iterations',
                                   self.fitting_config.get('max_iterations', 1000))

        try:
            # Use pyransac3d for cuboid fitting
            cuboid = pyrsc.Cuboid()
            result = cuboid.fit(
                points, thresh=threshold, maxIteration=max_iterations
            )

            # Check result format - pyransac3d may return different numbers of values
            if result is None:
                return None

            # Handle different return formats from pyransac3d
            if len(result) >= 4:
                center, size, rotation, inliers = result
            elif len(result) >= 3:
                # May not return inliers in some versions
                center, size, rotation = result
                inliers = None
                # Compute inliers manually
                rot = Rotation.from_euler('xyz', rotation)
                local_points = rot.inv().apply(points - center)
                half_size = size / 2 + threshold
                inside = np.all(np.abs(local_points) <= half_size, axis=1)
                inliers = np.where(inside)[0]
            else:
                return None

            if center is None or (inliers is not None and len(inliers) < self.fitting_config.get('min_inliers', 50)):
                return None

            return {
                'center': center,
                'size': size,
                'rotation': rotation,
                'inliers': inliers if inliers is not None else np.arange(len(points))
            }

        except Exception as e:
            print(f"Cuboid fitting failed: {e}")
            return None

    def validate(self, params: Dict[str, Any]) -> bool:
        """
        Validate cuboid parameters against physical constraints.

        Args:
            params: Dictionary with center, size, rotation

        Returns:
            True if valid
        """
        if params is None:
            return False

        # Check size ranges - more lenient validation
        size_ranges = self.params_config['size_range']
        size = params['size']

        # Allow 50% tolerance on each dimension
        for i, (s_min, s_max) in enumerate(size_ranges):
            if size[i] < s_min * 0.5 or size[i] > s_max * 1.5:
                return False

        return True

    def compute_inliers(self, points: np.ndarray, params: Dict[str, Any],
                       threshold: float) -> np.ndarray:
        """
        Compute inlier indices for cuboid.

        Args:
            points: (N, 3) array
            params: Dictionary with center, size, rotation
            threshold: Distance threshold

        Returns:
            Array of inlier indices
        """
        center = params['center']
        size = params['size']
        rotation = params['rotation']

        # Transform points to local cuboid frame
        rot = Rotation.from_euler('xyz', rotation)
        local_points = rot.inv().apply(points - center)

        # Check if points are inside expanded cuboid
        half_size = size / 2 + threshold
        inside = np.all(np.abs(local_points) <= half_size, axis=1)

        # Compute distance to nearest face
        distances = np.min(np.abs(np.abs(local_points) - size / 2), axis=1)

        # Inliers are points close to surface
        inliers = np.where(inside & (distances < threshold))[0]

        return inliers

    def visualize(self, plotter: Any, params: Dict[str, Any],
                 points: Optional[np.ndarray] = None, color: str = 'blue'):
        """
        Visualize cuboid using PyVista.

        Args:
            plotter: PyVista plotter
            params: Dictionary with center, size, rotation
            points: Optional inlier points
            color: Color for cuboid
        """
        center = params['center']
        size = params['size']
        rotation = params['rotation']

        # Create cuboid mesh
        box = pv.Box(bounds=[
            center[0] - size[0] / 2, center[0] + size[0] / 2,
            center[1] - size[1] / 2, center[1] + size[1] / 2,
            center[2] - size[2] / 2, center[2] + size[2] / 2
        ])

        # Apply rotation
        rot = Rotation.from_euler('xyz', rotation)
        rot_matrix = rot.as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        box.transform(transform)

        plotter.add_mesh(box, color=color, opacity=0.3, label='Cuboid')

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
        """Sample random cuboid parameters for data generation."""
        size_ranges = self.params_config['size_range']

        return {
            'size': np.array([
                np.random.uniform(size_ranges[0][0], size_ranges[0][1]),
                np.random.uniform(size_ranges[1][0], size_ranges[1][1]),
                np.random.uniform(size_ranges[2][0], size_ranges[2][1])
            ]),
            'rotation': None,  # Will be random in generate_points
            'center': None  # Will be random in generate_points
        }
