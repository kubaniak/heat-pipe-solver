#!/usr/bin/env python
"""
params.py - Module for handling parameters for the heat pipe solver.
Reads parameters from a configuration file and provides them to the simulation.
"""

import os
import json

# Global variable to cache parameters
_params_cache = None

def load_params(config_file=None):
    """
    Load parameters from configuration file.
    
    Args:
        config_file (str, optional): Path to configuration file. If None, uses default config.
    
    Returns:
        dict: Dictionary containing all parameters.
    """
    global _params_cache
    
    # If parameters are already cached, return them
    if _params_cache is not None:
        return _params_cache
    
    # If no config file is specified, use the default one
    if config_file is None:
        config_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'config', 'heat_pipe_config.json'
        )
    
    # Read the configuration file
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {config_file}")
    
    # Flatten the configuration into a single dictionary
    params = {}
    for section in config:
        for key, value in config[section].items():
            params[key] = value
    
    # Cache the parameters
    _params_cache = params
    
    return params

def get_param(name, default=None):
    """
    Get a specific parameter by name.
    
    Args:
        name (str): Name of the parameter.
        default: Default value if parameter is not found.
    
    Returns:
        Parameter value.
    """
    params = load_params()
    return params.get(name, default)

def get_param_group(group_name):
    """
    Get a specific group of parameters by name.
    
    Args:
        group_name (str): Name of the parameter group.
    
    Returns:
        dict: Dictionary containing the parameters in the specified group.
    """
    # Read the original config file instead of the flattened parameters
    if group_name is None:
        return {}
        
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'config', 'heat_pipe_config.json'
    )
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {config_file}")
    
    # Return the specific group if it exists, otherwise an empty dictionary
    return config.get(group_name, {})

def get_all_params():
    """
    Get all parameters.
    
    Returns:
        dict: Dictionary containing all parameters.
    """
    return load_params()
