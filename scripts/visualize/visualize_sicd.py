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

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Init file paths
    output_dir = Path(base_config["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    sicd_paths = glob.glob(str(Path(base_config["data"]["root"]) / "sicds" / "*.ntf"))

    assert sicd_paths

    sicd_dir = Path(output_dir / "remapped_sicds")
    sicd_dir.mkdir(parents=True, exist_ok=True)

    visualizer = Visualizer()

    for sicd_path in sicd_paths:
        sicd_reader = SICDReader(sicd_path)
        sicd_pixels = sicd_reader[:]

        # Create remapper and calculate the sicd data_mean to be used for remapping;
        # this data_mean will be saved and used to remap the chips in another script
        remapper = Density()
        remapper.calculate_global_parameters_from_reader(sicd_reader)

        sicd_metadata = {"data_mean": remapper.data_mean}

        sicd_name = Path(sicd_path).stem
        save_name_png = sicd_dir / f"{sicd_name}.png"
        visualizer.plot_sicd(
            complex_pixels=sicd_pixels, remapper=remapper, save_path=str(save_name_png)
        )

        metadata_name = sicd_dir / f"{sicd_name}.json"
        with open(metadata_name, "w") as json_file:
            json.dump(sicd_metadata, json_file)


if __name__ == "__main__":
    Fire(main)
