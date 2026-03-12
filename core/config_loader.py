"""
Configuration loader for universal shape recognition system.
Loads and validates YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


class ShapeConfig:
    """Configuration for a single shape type."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.label = config['label']
        self.params = config['params']
        self.fitting = config['fitting']
        self.min_points = config.get('min_points', 50)

    def __repr__(self):
        return f"ShapeConfig(name={self.name}, label={self.label})"


class UniversalConfig:
    """Main configuration class for the universal recognition system."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._raw_config = self._load_yaml()
        self._validate()
        self._parse()

    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _validate(self):
        """Validate configuration structure and values."""
        required_sections = ['shapes', 'scene', 'inference', 'training']
        for section in required_sections:
            if section not in self._raw_config:
                raise ValueError(f"Missing required section: {section}")

        # Validate shapes
        if not self._raw_config['shapes']:
            raise ValueError("At least one shape must be defined")

        # Validate unique labels
        labels = [cfg['label'] for cfg in self._raw_config['shapes'].values()]
        if len(labels) != len(set(labels)):
            raise ValueError("Shape labels must be unique")

        # Validate label 0 and 1 are reserved
        if 0 in labels or 1 in labels:
            raise ValueError("Labels 0 and 1 are reserved (0=background, 1=unlabeled)")

    def _parse(self):
        """Parse configuration into structured objects."""
        # Parse shapes
        self.shapes = {}
        for name, config in self._raw_config['shapes'].items():
            self.shapes[name] = ShapeConfig(name, config)

        # Scene config
        self.scene = self._raw_config['scene']

        # Inference config
        self.inference = self._raw_config['inference']

        # Training config
        self.training = self._raw_config['training']

        # Visualization config
        self.visualization = self._raw_config.get('visualization', {})

        # Logging config
        self.logging = self._raw_config.get('logging', {})

    @property
    def num_classes(self) -> int:
        """Total number of classes (needs to cover all label values)."""
        # Find max label value and add 1 (since labels start at 0)
        max_label = max(shape_cfg.label for shape_cfg in self.shapes.values())
        return max_label + 1

    @property
    def shape_names(self) -> List[str]:
        """List of shape names."""
        return list(self.shapes.keys())

    @property
    def label_to_shape(self) -> Dict[int, str]:
        """Mapping from label to shape name."""
        mapping = {0: 'background'}
        for name, shape_cfg in self.shapes.items():
            mapping[shape_cfg.label] = name
        return mapping

    @property
    def shape_to_label(self) -> Dict[str, int]:
        """Mapping from shape name to label."""
        mapping = {'background': 0}
        for name, shape_cfg in self.shapes.items():
            mapping[name] = shape_cfg.label
        return mapping

    def get_shape_config(self, name: str) -> ShapeConfig:
        """Get configuration for a specific shape."""
        if name not in self.shapes:
            raise ValueError(f"Unknown shape: {name}")
        return self.shapes[name]

    def get_class_weights(self) -> np.ndarray:
        """Get class weights for training."""
        weights = self.training.get('class_weights', None)
        if weights is None:
            return np.ones(self.num_classes)
        return np.array(weights[:self.num_classes])

    def save(self, output_path: str):
        """Save configuration to a new file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._raw_config, f, default_flow_style=False, sort_keys=False)

    def __repr__(self):
        return f"UniversalConfig(shapes={list(self.shapes.keys())}, num_classes={self.num_classes})"


def load_config(config_path: str) -> UniversalConfig:
    """Convenience function to load configuration."""
    return UniversalConfig(config_path)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config("config/shape_config.yaml")
    print(config)
    print(f"Number of classes: {config.num_classes}")
    print(f"Shape names: {config.shape_names}")
    print(f"Label mapping: {config.label_to_shape}")
    print(f"Class weights: {config.get_class_weights()}")
