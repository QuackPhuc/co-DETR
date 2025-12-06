"""
Tests for configuration system.

This module tests the YAML-based configuration loading, merging,
and dot notation access.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from codetr.configs.config import Config, load_config, merge_config


class TestConfigCreation:
    """Tests for Config class creation and access."""
    
    def test_config_from_dict(self):
        """Test creating Config from dictionary."""
        config_dict = {
            'model': {
                'num_classes': 80,
                'embed_dim': 256,
            },
            'train': {
                'batch_size': 4,
                'lr': 1e-4,
            }
        }
        
        config = Config(config_dict)
        
        assert config.model.num_classes == 80
        assert config.train.batch_size == 4
    
    def test_dot_notation_access(self):
        """Test attribute-style access to nested config."""
        config_dict = {
            'deep': {
                'nested': {
                    'value': 42
                }
            }
        }
        
        config = Config(config_dict)
        
        assert config.deep.nested.value == 42
    
    def test_dict_style_access(self):
        """Test dictionary-style access still works."""
        config_dict = {'key': 'value'}
        
        config = Config(config_dict)
        
        assert config['key'] == 'value'


class TestConfigYAML:
    """Tests for YAML config loading."""
    
    def test_load_yaml_config(self):
        """Test loading config from YAML file."""
        yaml_content = """
model:
  num_classes: 20
  embed_dim: 128
train:
  batch_size: 2
  epochs: 10
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            
            assert config.model.num_classes == 20
            assert config.train.epochs == 10
        finally:
            Path(config_path).unlink()
    
    def test_load_invalid_yaml_raises(self):
        """Invalid YAML should raise error."""
        invalid_yaml = """
key: value
  bad_indent: oops
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            config_path = f.name
        
        try:
            with pytest.raises(Exception):  # Could be yaml.YAMLError or ValueError
                load_config(config_path)
        finally:
            Path(config_path).unlink()


class TestConfigMerge:
    """Tests for config merging and override."""
    
    def test_merge_override(self):
        """Test merging override values."""
        base_config = {
            'model': {'num_classes': 80},
            'train': {'batch_size': 4},
        }
        
        override = {
            'train': {'batch_size': 8},
        }
        
        merged = merge_config(Config(base_config), override)
        
        # Override should take effect
        assert merged.train.batch_size == 8
        # Non-overridden values should be preserved
        assert merged.model.num_classes == 80
    
    def test_merge_new_keys(self):
        """Test adding new keys via merge."""
        base_config = {'a': 1}
        override = {'b': 2}
        
        merged = merge_config(Config(base_config), override)
        
        assert merged.a == 1
        assert merged.b == 2


class TestConfigSaveLoad:
    """Tests for config save and reload."""
    
    def test_save_and_reload(self):
        """Test saving config to file and reloading."""
        original = Config({
            'model': {'layers': 6},
            'data': {'root': '/path/to/data'},
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "config.yaml"
            
            original.save(save_path)
            
            assert save_path.exists()
            
            reloaded = load_config(save_path)
            
            assert reloaded.model.layers == 6
            assert reloaded.data.root == '/path/to/data'


class TestConfigEdgeCases:
    """Tests for edge cases in config handling."""
    
    def test_empty_config(self):
        """Test empty config creation."""
        config = Config({})
        
        assert len(config) == 0
    
    def test_list_values(self):
        """Test config with list values."""
        config = Config({
            'scales': [800, 1333],
            'iou_thresholds': [0.5, 0.75, 0.95],
        })
        
        assert config.scales == [800, 1333]
        assert len(config.iou_thresholds) == 3
    
    def test_none_values(self):
        """Test config with None values."""
        config = Config({
            'optional_param': None,
        })
        
        assert config.optional_param is None
