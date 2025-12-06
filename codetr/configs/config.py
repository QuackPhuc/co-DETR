"""YAML-based configuration system for Co-DETR.

This module provides a flexible configuration system that supports:
    - Loading from YAML files
    - Dot notation access to nested values
    - Default value handling
    - Configuration merging
    - Type-safe access methods

Example:
    >>> config = Config.from_file("config.yaml")
    >>> lr = config.get("train.optimizer.lr", default=0.0001)
    >>> config.train.batch_size = 4  # dot notation access
"""

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class ConfigDict(dict):
    """Dictionary with attribute-style access.
    
    Allows accessing dictionary keys as attributes for convenience.
    Nested dictionaries are automatically converted to ConfigDict.
    
    Example:
        >>> cfg = ConfigDict({"train": {"lr": 0.001}})
        >>> print(cfg.train.lr)  # 0.001
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert nested dicts to ConfigDict
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)
            elif isinstance(value, list):
                self[key] = self._convert_list(value)
    
    def _convert_list(self, items: List) -> List:
        """Recursively convert dicts in lists to ConfigDict."""
        result = []
        for item in items:
            if isinstance(item, dict) and not isinstance(item, ConfigDict):
                result.append(ConfigDict(item))
            elif isinstance(item, list):
                result.append(self._convert_list(item))
            else:
                result.append(item)
        return result
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to dict keys."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute-style setting of dict keys."""
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        self[name] = value
    
    def __delattr__(self, name: str) -> None:
        """Allow attribute-style deletion of dict keys."""
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")
    
    def get_nested(self, key: str, default: Any = None) -> Any:
        """Get a nested value using dot notation.
        
        Args:
            key: Dot-separated key path (e.g., "train.optimizer.lr").
            default: Default value if key not found.
            
        Returns:
            The value at the key path, or default if not found.
            
        Example:
            >>> cfg.get_nested("train.optimizer.lr", 0.001)
        """
        keys = key.split(".")
        value = self
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set_nested(self, key: str, value: Any) -> None:
        """Set a nested value using dot notation.
        
        Args:
            key: Dot-separated key path (e.g., "train.optimizer.lr").
            value: Value to set.
            
        Example:
            >>> cfg.set_nested("train.optimizer.lr", 0.0001)
        """
        keys = key.split(".")
        target = self
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = ConfigDict()
            target = target[k]
        target[keys[-1]] = value
    
    def to_dict(self) -> Dict:
        """Convert to regular nested dictionary.
        
        Returns:
            Plain dict representation.
        """
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = self._list_to_dict(value)
            else:
                result[key] = value
        return result
    
    def _list_to_dict(self, items: List) -> List:
        """Recursively convert ConfigDicts in lists to dicts."""
        result = []
        for item in items:
            if isinstance(item, ConfigDict):
                result.append(item.to_dict())
            elif isinstance(item, list):
                result.append(self._list_to_dict(item))
            else:
                result.append(item)
        return result


class Config:
    """Configuration manager for Co-DETR.
    
    Provides loading, merging, and accessing configuration values
    from YAML files with support for inheritance and defaults.
    
    Attributes:
        _cfg: Internal ConfigDict storing configuration values.
        
    Example:
        >>> config = Config.from_file("train_config.yaml")
        >>> print(config.model.num_classes)
        >>> config.save("output_config.yaml")
    """
    
    def __init__(self, cfg_dict: Optional[Dict] = None):
        """Initialize Config from dictionary.
        
        Args:
            cfg_dict: Initial configuration dictionary.
        """
        if cfg_dict is None:
            cfg_dict = {}
        self._cfg = ConfigDict(cfg_dict)
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "Config":
        """Load configuration from YAML file.
        
        Args:
            filepath: Path to YAML configuration file.
            
        Returns:
            Config object with loaded values.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML parsing fails.
            
        Example:
            >>> config = Config.from_file("configs/train.yaml")
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        
        if cfg_dict is None:
            cfg_dict = {}
        
        config = cls(cfg_dict)
        
        # Handle base config inheritance
        if "_base_" in cfg_dict:
            base_path = cfg_dict.pop("_base_")
            if isinstance(base_path, str):
                base_path = [base_path]
            
            # Load and merge base configs
            base_dir = filepath.parent
            for bp in base_path:
                base_file = base_dir / bp
                base_config = cls.from_file(base_file)
                config = base_config.merge(config)
        
        return config
    
    @classmethod
    def from_dict(cls, cfg_dict: Dict) -> "Config":
        """Create Config from dictionary.
        
        Args:
            cfg_dict: Configuration dictionary.
            
        Returns:
            Config object.
        """
        return cls(cfg_dict)
    
    def merge(self, other: "Config") -> "Config":
        """Merge another config into this one.
        
        Values from other config override values in this config.
        Nested dicts are merged recursively.
        
        Args:
            other: Config to merge in.
            
        Returns:
            New merged Config.
        """
        merged = self._deep_merge(self._cfg.to_dict(), other._cfg.to_dict())
        return Config(merged)
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        result = copy.deepcopy(base)
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation.
        
        Args:
            key: Dot-separated key path.
            default: Default value if key not found.
            
        Returns:
            Configuration value or default.
            
        Example:
            >>> lr = config.get("train.optimizer.lr", 0.001)
        """
        return self._cfg.get_nested(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation.
        
        Args:
            key: Dot-separated key path.
            value: Value to set.
            
        Example:
            >>> config.set("train.batch_size", 4)
        """
        self._cfg.set_nested(key, value)
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to top-level config keys."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        try:
            return self._cfg[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute setting for top-level config keys."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
            self._cfg[name] = value
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary.
        
        Returns:
            Plain dict representation.
        """
        return self._cfg.to_dict()
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            filepath: Output YAML file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        return f"Config({self._cfg})"
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists (supports dot notation)."""
        return self.get(key) is not None


def load_config(filepath: Union[str, Path]) -> Config:
    """Convenience function to load configuration from file.
    
    Args:
        filepath: Path to YAML configuration file.
        
    Returns:
        Loaded Config object.
        
    Example:
        >>> config = load_config("configs/train.yaml")
    """
    return Config.from_file(filepath)


def get_default_config() -> Config:
    """Get default Co-DETR configuration.
    
    Returns:
        Config with default values for Co-DETR R50 training.
    """
    default_cfg = {
        "model": {
            "type": "CoDETR",
            "num_classes": 80,
            "embed_dim": 256,
            "num_queries": 300,
            "num_feature_levels": 4,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "use_rpn": True,
            "use_roi": True,
            "use_atss": True,
            "use_dn": True,
            "aux_loss_weight": 1.0,
            "pretrained_backbone": True,
            "frozen_backbone_stages": 1,
        },
        "train": {
            "epochs": 12,
            "batch_size": 2,
            "num_workers": 4,
            "optimizer": {
                "type": "AdamW",
                "lr": 0.0002,
                "weight_decay": 0.0001,
                "backbone_lr_mult": 0.1,
            },
            "lr_scheduler": {
                "type": "step",
                "step_size": 11,
                "gamma": 0.1,
                "warmup_epochs": 0,
                "warmup_lr_ratio": 0.001,
            },
            "gradient_clip": 0.1,
            "amp": True,
            "checkpoint_interval": 1,
            "log_interval": 50,
        },
        "data": {
            "train_root": "data/train",
            "val_root": "data/val",
            "train_ann": None,
            "val_ann": None,
            "img_size": 800,
            "max_size": 1333,
        },
        "eval": {
            "interval": 1,
            "score_threshold": 0.05,
            "nms_threshold": 0.5,
            "max_detections": 100,
        },
    }
    return Config(default_cfg)
