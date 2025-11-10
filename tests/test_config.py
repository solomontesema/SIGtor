"""Tests for configuration loading."""

import os
import pytest
import tempfile
import yaml
from sigtor.utils.config import load_config, get_config_section


def test_load_config_valid():
    """Test loading a valid YAML configuration file."""
    config_data = {
        'SIGtor': {
            'source_ann_file': './test.txt',
            'destn_dir': './output/',
            'total_new_imgs': 100
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        assert 'SIGtor' in config
        assert config['SIGtor']['total_new_imgs'] == 100
    finally:
        os.unlink(config_path)


def test_load_config_missing_file():
    """Test loading a non-existent configuration file."""
    with pytest.raises(ValueError, match="Configuration file not found"):
        load_config("./nonexistent.yaml")


def test_get_config_section():
    """Test getting a specific section from configuration."""
    config = {
        'SIGtor': {'key1': 'value1'},
        'Test': {'key2': 'value2'}
    }
    
    sigtor_section = get_config_section(config, 'SIGtor')
    assert sigtor_section['key1'] == 'value1'
    
    test_section = get_config_section(config, 'Test')
    assert test_section['key2'] == 'value2'
    
    missing_section = get_config_section(config, 'Missing')
    assert missing_section == {}

