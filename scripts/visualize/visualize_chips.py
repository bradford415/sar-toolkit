import glob
import json
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

    # Calculate the start time
    start = time.time()

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Init file paths
    output_dir = Path(base_config["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    sicd_path = Path(base_config["data"]["chips"])
    sicd_name = sicd_path.name
    chip_paths = sorted(glob.glob(str(sicd_path / "*.npy")))

    # Load sicd metadata for remap
    metadata_name = output_dir / "remapped_sicds" / f"{sicd_name}.json"
    with open(metadata_name) as json_file:
        sicd_metadata = json.load(json_file)

    assert chip_paths

    visualizer = Visualizer()

    # Initialize remapper with data_mean from full scene SICD
    remapper = Density(data_mean=sicd_metadata["data_mean"])

    chip_dir = output_dir / "remapped_chips" / sicd_name
    chip_dir.mkdir(parents=True, exist_ok=True)

    for chip_path in chip_paths:
        chip_name = Path(chip_path).stem
        chip = np.load(chip_path)

        save_name = chip_dir / f"{chip_name}.png"

        visualizer.plot_sicd(
            complex_pixels=chip, remapper=remapper, save_path=save_name
        )


if __name__ == "__main__":
    Fire(main)
