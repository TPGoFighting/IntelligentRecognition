"""
Plane shape plugin for universal recognition system.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import pyransac3d as pyrsc
import pyvista as pv

from .base_shape import BaseShape


class PlanePlugin(BaseShape):
    """Plugin for plane detection and fitting."""

    def generate_points(self, center: Optional[np.ndarray] = None,
                       normal: Optional[np.ndarray] = None,
                       size: Optional[Tuple[float, float]] = None,
                       num_points: int = 1000,
                       noise: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic plane point cloud.

        Args:
            center: (3,) center point, random if None
            normal: (3,) plane normal, random if None
            size: (width, height) of plane, random if None
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

        if normal is None:
            # Random normal
            normal = np.random.randn(3)
            normal = normal / np.linalg.norm(normal)

        if size is None:
            area_min, area_max = self.params_config['area_range']
            area = np.random.uniform(area_min, area_max)
            aspect = np.random.uniform(0.5, 2.0)
            width = np.sqrt(area * aspect)
            height = area / width
            size = (width, height)

        # Create orthonormal basis for plane
        if abs(normal[2]) < 0.9:
            u = np.cross(normal, [0, 0, 1])
        else:
            u = np.cross(normal, [1, 0, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        # Sample points on plane
        width, height = size
        u_coords = np.random.uniform(-width / 2, width / 2, num_points)
        v_coords = np.random.uniform(-height / 2, height / 2, num_points)

        points = (center[np.newaxis, :] +
                 u_coords[:, np.newaxis] * u[np.newaxis, :] +
                 v_coords[:, np.newaxis] * v[np.newaxis, :])

        # Add noise
        if noise > 0:
            points += np.random.randn(*points.shape) * noise

        # All normals point in same direction
        normals = np.tile(normal, (num_points, 1))

        return points, normals

    def fit(self, points: np.ndarray, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fit plane to point cloud using RANSAC.

        Args:
            points: (N, 3) array of points
            **kwargs: Additional parameters

        Returns:
            Dictionary with keys: normal, distance, center, area
        """
        if len(points) < self.min_points:
            return None

        # Get fitting parameters
        threshold = kwargs.get('threshold', self.fitting_config.get('threshold', 0.05))
        max_iterations = kwargs.get('max_iterations',
                                   self.fitting_config.get('max_iterations', 500))

        try:
            # Use pyransac3d for plane fitting
            plane = pyrsc.Plane()
            equation, inliers = plane.fit(
                points, thresh=threshold, maxIteration=max_iterations
            )

            print(f"    [Plane fit] equation={equation}, inliers={len(inliers) if inliers is not None else 0}")

            if equation is None or len(inliers) < self.fitting_config.get('min_inliers', 100):
                print(f"    [Plane fit failed] Not enough inliers or equation is None")
                return None

            # Extract plane parameters: ax + by + cz + d = 0
            a, b, c, d = equation
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)
            distance = -d / np.linalg.norm([a, b, c])

            # Compute center and area from inliers
            inlier_points = points[inliers]
            center = inlier_points.mean(axis=0)

            # Estimate area using convex hull or bounding box
            # Project points onto plane
            if abs(normal[2]) < 0.9:
                u = np.cross(normal, [0, 0, 1])
            else:
                u = np.cross(normal, [1, 0, 0])
            u = u / np.linalg.norm(u)
            v = np.cross(normal, u)

            # Project to 2D
            proj_u = np.dot(inlier_points - center, u)
            proj_v = np.dot(inlier_points - center, v)

            # Estimate area from bounding box
            width = proj_u.max() - proj_u.min()
            height = proj_v.max() - proj_v.min()
            area = width * height

            return {
                'normal': normal,
                'distance': distance,
                'center': center,
                'area': area,
                'width': width,
                'height': height,
                'inliers': inliers
            }

        except Exception as e:
            print(f"Plane fitting failed: {e}")
            return None

    def validate(self, params: Dict[str, Any]) -> bool:
        """
        Validate plane parameters against physical constraints.

        Args:
            params: Dictionary with normal, distance, area

        Returns:
            True if valid
        """
        if params is None:
            print("    [Plane validation] params is None")
            return False

        # Check area range - much more lenient
        area_min, area_max = self.params_config['area_range']
        area = params['area']

        # Allow 300% tolerance on range (0.25x to 4x)
        if area < area_min * 0.25 or area > area_max * 4.0:
            print(f"    [Plane validation failed] area={area:.4f}, expected range [{area_min:.4f}, {area_max:.4f}]")
            return False

        # Check normal is normalized (lenient)
        normal = params['normal']
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 0.5 or normal_norm > 2.0:
            print(f"    [Plane validation failed] normal norm={normal_norm:.4f}")
            return False

        print(f"    [Plane validation passed] area={area:.4f}")
        return True

    def compute_inliers(self, points: np.ndarray, params: Dict[str, Any],
                       threshold: float) -> np.ndarray:
        """
        Compute inlier indices for plane.

        Args:
            points: (N, 3) array
            params: Dictionary with normal, distance
            threshold: Distance threshold

        Returns:
            Array of inlier indices
        """
        normal = params['normal']
        distance = params['distance']

        # Compute distance from each point to plane
        # Distance = |normal · point + distance|
        distances = np.abs(np.dot(points, normal) + distance)

        # Find inliers
        inliers = np.where(distances < threshold)[0]

        return inliers

    def visualize(self, plotter: Any, params: Dict[str, Any],
                 points: Optional[np.ndarray] = None, color: str = 'yellow'):
        """
        Visualize plane using PyVista.

        Args:
            plotter: PyVista plotter
            params: Dictionary with normal, center, width, height
            points: Optional inlier points
            color: Color for plane
        """
        center = params['center']
        normal = params['normal']
        width = params.get('width', 2.0)
        height = params.get('height', 2.0)

        # Create plane mesh
        plane = pv.Plane(center=center, direction=normal,
                        i_size=width, j_size=height)

        plotter.add_mesh(plane, color=color, opacity=0.3, label='Plane')

        # Add normal vector
        arrow_start = center
        arrow_end = center + normal * 0.5
        arrow = pv.Arrow(start=arrow_start, direction=normal, scale=0.5)
        plotter.add_mesh(arrow, color='red', label='Normal')

        # Add inlier points if provided
        if points is not None and len(points) > 0:
            point_cloud = pv.PolyData(points)
            plotter.add_mesh(point_cloud, color=color, point_size=3,
                           render_points_as_spheres=True, label='Inliers')

    def sample_parameters(self) -> Dict[str, Any]:
        """Sample random plane parameters for data generation."""
        area_min, area_max = self.params_config['area_range']
        area = np.random.uniform(area_min, area_max)
        aspect = np.random.uniform(0.5, 2.0)
        width = np.sqrt(area * aspect)
        height = area / width

        return {
            'size': (width, height),
            'normal': None,  # Will be random in generate_points
            'center': None  # Will be random in generate_points
        }
