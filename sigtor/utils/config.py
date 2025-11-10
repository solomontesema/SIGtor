import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str = "./config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    if not os.path.exists(config_path):
        raise ValueError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Configuration file is empty or invalid: {config_path}")
    
    return config


def get_config_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Get a specific section from the configuration.

    Args:
        config (dict): The full configuration dictionary.
        section (str): The section name to retrieve (e.g., 'SIGtor', 'Test').

    Returns:
        dict: The configuration section, or empty dict if not found.
    """
    return config.get(section, {})


# Backward compatibility function (deprecated)
def parse_commandline_arguments(argument_filepath: str) -> Dict[str, list]:
    """
    Parse command-line arguments from a file (deprecated - use load_config instead).

    This function is kept for backward compatibility but should not be used in new code.
    Use load_config() instead.

    Args:
        argument_filepath (str): Path to the file containing arguments.

    Returns:
        dict: A dictionary of parsed arguments.
    """
    arguments = {}
    if not os.path.exists(argument_filepath):
        raise ValueError(f"Unable to find {argument_filepath}")

    with open(argument_filepath) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#[') or line.isspace():
                continue
            key, value = line.strip().split("=")
            arguments[key] = value.strip().split(',')
    return arguments
