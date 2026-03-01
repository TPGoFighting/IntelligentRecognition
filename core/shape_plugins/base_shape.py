"""
Base class for shape plugins in the universal recognition system.
All shape plugins must inherit from this class and implement the abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import numpy as np


class DetectedObject:
    """Container for a detected object."""

    def __init__(self, shape_type: str, params: Dict[str, Any],
                 points: np.ndarray, confidence: float):
        self.shape_type = shape_type
        self.params = params
        self.points = points
        self.confidence = confidence
        self.num_points = len(points)

    def __repr__(self):
        return (f"DetectedObject(type={self.shape_type}, "
                f"confidence={self.confidence:.3f}, "
                f"num_points={self.num_points})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'shape_type': self.shape_type,
            'params': self.params,
            'num_points': self.num_points,
            'confidence': float(self.confidence)
        }


class BaseShape(ABC):
    """
    Abstract base class for shape plugins.

    Each shape plugin must implement:
    1. generate_points: Generate synthetic point cloud data
    2. fit: Fit shape parameters to point cloud using RANSAC or other methods
    3. validate: Validate physical constraints of fitted parameters
    4. compute_inliers: Compute inlier points for given parameters
    5. visualize: Visualize the shape in 3D
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize shape plugin with configuration.

        Args:
            config: Shape configuration dictionary from YAML
        """
        self.config = config
        self.params_config = config.get('params', {})
        self.fitting_config = config.get('fitting', {})
        self.min_points = config.get('min_points', 50)

    @abstractmethod
    def generate_points(self, **params) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic point cloud for this shape.

        Args:
            **params: Shape-specific parameters (e.g., center, radius, etc.)

        Returns:
            points: (N, 3) array of 3D points
            normals: (N, 3) array of surface normals
        """
        pass

    @abstractmethod
    def fit(self, points: np.ndarray, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fit shape parameters to point cloud.

        Args:
            points: (N, 3) array of 3D points
            **kwargs: Additional fitting parameters

        Returns:
            Dictionary of fitted parameters, or None if fitting failed
        """
        pass

    @abstractmethod
    def validate(self, params: Dict[str, Any]) -> bool:
        """
        Validate physical constraints of fitted parameters.

        Args:
            params: Dictionary of shape parameters

        Returns:
            True if parameters are physically valid, False otherwise
        """
        pass

    @abstractmethod
    def compute_inliers(self, points: np.ndarray, params: Dict[str, Any],
                       threshold: float) -> np.ndarray:
        """
        Compute inlier indices for given shape parameters.

        Args:
            points: (N, 3) array of 3D points
            params: Dictionary of shape parameters
            threshold: Distance threshold for inliers

        Returns:
            Array of inlier indices
        """
        pass

    @abstractmethod
    def visualize(self, plotter: Any, params: Dict[str, Any],
                 points: Optional[np.ndarray] = None, color: str = 'red'):
        """
        Visualize the shape in 3D using PyVista plotter.

        Args:
            plotter: PyVista plotter object
            params: Dictionary of shape parameters
            points: Optional point cloud to visualize alongside shape
            color: Color for the shape visualization
        """
        pass

    def compute_distances(self, points: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Compute distances from points to the shape surface.
        Default implementation uses inlier computation.

        Args:
            points: (N, 3) array of 3D points
            params: Dictionary of shape parameters

        Returns:
            Array of distances
        """
        # This is a default implementation that can be overridden
        # Most shapes will have a more efficient distance computation
        raise NotImplementedError("Subclass must implement compute_distances or override this method")

    def get_bounding_box(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute axis-aligned bounding box of points.

        Args:
            points: (N, 3) array of 3D points

        Returns:
            min_bound: (3,) array of minimum coordinates
            max_bound: (3,) array of maximum coordinates
        """
        return points.min(axis=0), points.max(axis=0)

    def sample_parameters(self) -> Dict[str, Any]:
        """
        Sample random parameters for data generation.
        Uses parameter ranges from configuration.

        Returns:
            Dictionary of sampled parameters
        """
        raise NotImplementedError("Subclass should implement sample_parameters for data generation")

    def __repr__(self):
        return f"{self.__class__.__name__}(min_points={self.min_points})"
