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

from stk.visualize import Visualizer


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
    
    sicd_reader = SICDReader(sicd_paths[0])
    sicd_pixels = sicd_reader[:]

    sicd_dir = Path(output_dir / "remapped_sicds")
    sicd_dir.mkdir(parents=True, exist_ok=True)

    visualizer = Visualizer()
    
    for sicd_path in sicd_paths:
        save_name = sicd_dir / f"{Path(sicd_path).stem}.png"
        visualizer.plot_sicd(complex_pixels=sicd_pixels, save_path=str(save_name))


if __name__ == "__main__":
    Fire(main)
