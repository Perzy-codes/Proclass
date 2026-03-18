"""
Configuration Management
=========================

Loads and manages project configuration from YAML files and environment variables.

CONCEPT: Configuration Hierarchy
-----------------------------------
Settings are loaded in this priority (highest wins):
  1. Environment variables (for secrets and deployment overrides)
  2. Config YAML files (for default settings)
  3. Hardcoded defaults (fallback)

This pattern is standard in production systems because:
  - Secrets (API keys, passwords) live in env vars (never in code/config files)
  - Default settings live in YAML (easy to read and change)
  - Hardcoded defaults prevent crashes when config is missing
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str = 'configs/training_config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable overrides."""
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Apply environment variable overrides
    env_overrides = {
        'sagemaker.training_instance': 'TRAINING_INSTANCE_TYPE',
        'sagemaker.inference_instance': 'INFERENCE_INSTANCE_TYPE',
        's3.raw_bucket': 'RAW_IMAGES_BUCKET',
        's3.processed_bucket': 'PROCESSED_IMAGES_BUCKET',
        's3.model_artifacts_bucket': 'MODEL_ARTIFACTS_BUCKET',
    }
    
    for config_key, env_var in env_overrides.items():
        env_value = os.environ.get(env_var)
        if env_value:
            _set_nested(config, config_key, env_value)
    
    return config


def _set_nested(d: Dict, key: str, value: Any) -> None:
    """Set a nested dictionary value using dot notation."""
    keys = key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get environment variable with optional requirement check."""
    value = os.environ.get(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable not set: {key}")
    return value
