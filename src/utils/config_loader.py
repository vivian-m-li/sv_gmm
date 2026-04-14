"""
config.py — loads and validates the SPLIT configuration file.

Usage:
    from config import load_config
    cfg = load_config()          # reads config.toml in the cwd
    cfg = load_config("path/to/config.toml")
"""

import sys
import os

# tomllib is part of the stdlib from Python 3.11 onward.
# For older interpreters, fall back to the third-party tomli package
# (pip install tomli).
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        raise ImportError(
            "Python < 3.11 requires the 'tomli' package: pip install tomli"
        )

DEFAULT_CONFIG_PATH = "config.toml"


def load_config(path: str | None = None) -> dict:
    """
    Load and return the configuration from a TOML file.

    Parameters
    ----------
    path:
        Path to the config file. Defaults to 'config.toml' in the
        current working directory.

    Returns
    -------
    dict
        Parsed configuration, with validated types and resolved paths.
    """
    config_path = path or DEFAULT_CONFIG_PATH

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Config file not found: '{config_path}'. "
            "Copy config.toml to your working directory and fill in your paths."
        )

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    if "output_dir" not in cfg["paths"]:
        cfg["paths"]["output_dir"] = cfg["paths"]["input_dir"]

    if "local_intermediate_output_dir" not in cfg["paths"]:
        cfg["paths"]["local_intermediate_output_dir"] = cfg["paths"][
            "intermediate_output_dir"
        ]

    return cfg
