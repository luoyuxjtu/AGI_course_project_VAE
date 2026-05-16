"""
Configuration loader for the VAE project.

All hyperparameters live in YAML files under configs/.  Scripts receive the
config path via --config and call load_config() to get a SimpleNamespace
whose fields are accessed with dot notation (cfg.batch_size, cfg.beta, …).
"""

import argparse
import yaml
from pathlib import Path
from types import SimpleNamespace


def load_config_from_path(path: str | Path) -> SimpleNamespace:
    """Load a YAML file and return its contents as a SimpleNamespace.

    Args:
        path: Path to the YAML config file.

    Returns:
        A SimpleNamespace where each top-level key becomes an attribute.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    return SimpleNamespace(**data)


def load_config(argv: list[str] | None = None) -> SimpleNamespace:
    """Parse --config <path> from the command line and return the config.

    This is the standard entry point used by train.py, evaluate.py, etc.

    Args:
        argv: Argument list (defaults to sys.argv when None).

    Returns:
        Config namespace loaded from the specified YAML file.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a YAML config file, e.g. configs/baseline.yaml",
    )
    # parse_known_args lets callers add their own flags without conflicts
    args, _ = parser.parse_known_args(argv)
    return load_config_from_path(args.config)
