"""Configuration management for Co-DETR.

This module provides YAML-based configuration with dot notation access
and inheritance support.

Example:
    >>> from codetr.configs import Config, load_config
    >>> config = load_config("config.yaml")
    >>> print(config.model.num_classes)
"""

from .config import Config, ConfigDict, get_default_config, load_config

__all__ = [
    "Config",
    "ConfigDict",
    "load_config",
    "get_default_config",
]
