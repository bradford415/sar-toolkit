import glob
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from fire import Fire
from sarpy.io.complex.sicd import SICDReader
from sarpy.io.phase_history.cphd import CPHDReader
from sarpy.visualization.remap import Density

from stk.chip import Chipper


def main(base_config_path: str):
    """Entrypoint for the project

    Args:
        base_config_path: Path to the base configuration file
    """

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Init file paths
    output_dir = Path(base_config["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    sicd_paths = glob.glob(str(Path(base_config["data"]["root"]) / "sicds" / "*.ntf"))

    assert sicd_paths

    sicd_chipper = Chipper(256, 256)

    sicd_chipper.chip_sicds(sicd_paths)


if __name__ == "__main__":
    Fire(main)
