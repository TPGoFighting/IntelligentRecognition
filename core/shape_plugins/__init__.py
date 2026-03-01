"""
Shape plugin package initialization.
Provides easy access to all shape plugins.
"""

from .base_shape import BaseShape, DetectedObject
from .cylinder_plugin import CylinderPlugin
from .sphere_plugin import SpherePlugin
from .cuboid_plugin import CuboidPlugin
from .plane_plugin import PlanePlugin

__all__ = [
    'BaseShape',
    'DetectedObject',
    'CylinderPlugin',
    'SpherePlugin',
    'CuboidPlugin',
    'PlanePlugin'
]


def get_plugin_class(shape_name: str):
    """
    Get plugin class by shape name.

    Args:
        shape_name: Name of shape (cylinder, sphere, cuboid, plane)

    Returns:
        Plugin class

    Raises:
        ValueError: If shape name is unknown
    """
    plugins = {
        'cylinder': CylinderPlugin,
        'sphere': SpherePlugin,
        'cuboid': CuboidPlugin,
        'plane': PlanePlugin
    }

    if shape_name not in plugins:
        raise ValueError(f"Unknown shape: {shape_name}. Available: {list(plugins.keys())}")

    return plugins[shape_name]
