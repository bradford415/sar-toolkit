import glob
from pathlib import Path

import numpy as np
import yaml
from fire import Fire
from sarpy.io.phase_history.cphd import CPHDReader


def main(base_config_path: str):
    """Entrypoint for the project

    Args:
        base_config_path: Path to the base configuration file
    """

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)


if __name__ == "__main__":
    Fire(main)
