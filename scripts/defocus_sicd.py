import glob
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from fire import Fire
from sarpy.io.complex.sicd import SICDReader
from sarpy.io.phase_history.cphd import CPHDReader
from sarpy.visualization.remap import Density
from scipy.stats import linregress

from stk.autofocus.defocus import azimuth_defocus
from stk.chip import Chipper
from stk.utils.sicd import load_sicd_pixels
from stk.visualize.visualizer import Visualizer


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

    sicd_dir = Path(output_dir / "defocused_sicds")
    sicd_dir.mkdir(parents=True, exist_ok=True)

    sicd_pixels, sicd_reader = load_sicd_pixels(sicd_paths[0])

    visualizer = Visualizer()

    for sicd_path in sicd_paths:
        sicd_reader = SICDReader(sicd_path)
        sicd_pixels = sicd_reader[:]

        defocused_image = azimuth_defocus(
            sicd_pixels,
            ph_err_order=base_config["defocus"]["phase_err_order"],
            coeff_multiplier=base_config["defocus"]["coeff_multiplier"],
            rand_seed=base_config["defocus"]["seed"],
        )

        # Create remapper per sicd to reset attributes
        remapper = Density()
        remapper.calculate_global_parameters_from_reader(sicd_reader)

        sicd_metadata = {"data_mean": remapper.data_mean}

        sicd_name = Path(sicd_path).stem
        save_name_png = sicd_dir / f"defocused_{sicd_name}.png"
        visualizer.plot_sicd(
            complex_pixels=defocused_image,
            remapper=remapper,
            save_path=str(save_name_png),
        )

        metadata_name = sicd_dir / f"defocused_{sicd_name}.json"
        with open(metadata_name, "w") as json_file:
            json.dump(sicd_metadata, json_file)


if __name__ == "__main__":
    Fire(main)
