"""YAML config loading with inheritance support."""

import yaml
from pathlib import Path
from typing import Union

CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "configs"


def load_config(config_path: Union[str, Path]) -> dict:
    """Load a YAML config file with inheritance support.

    Supports 'inherit' key: list of parent configs to merge (in order).
    Child values override parent values.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        # Try relative to cwd first, then fall back to CONFIG_DIR
        if not config_path.exists():
            config_path = CONFIG_DIR / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    if "inherit" in config:
        parents = config.pop("inherit")
        if isinstance(parents, str):
            parents = [parents]

        merged = {}
        for parent_path in parents:
            parent = load_config(parent_path)
            merged = _deep_merge(merged, parent)

        config = _deep_merge(merged, config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
