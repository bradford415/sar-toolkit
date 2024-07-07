import glob
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
import json

from stk.autofocus.defocus import azimuth_defocus
from stk.chip import Chipper
from stk.utils.sicd import load_sicd_pixels, load_ordered_chips
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

    # Init file paths
    output_dir = Path(base_config["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    chips_root_path = Path(base_config["data"]["chips"])
    sicd_name = chips_root_path.name

    full_chip_paths = load_ordered_chips(chips_root_path)

    # Load sicd metadata for remap
    metadata_name = output_dir / "remapped_sicds" / f"{sicd_name}.json"
    with open(metadata_name) as json_file:
        sicd_metadata = json.load(json_file)

    # Initialize remapper with data_mean from full scene SICD
    remapper = Density(data_mean=sicd_metadata["data_mean"])

    visualizer = Visualizer()

    defocused_chip_dir = output_dir / "defocused_chips_from_chips" / sicd_name
    defocused_chip_dir.mkdir(parents=True, exist_ok=True)

    for index, chip_path in enumerate(full_chip_paths):
        ## TODO: Save chip error metadata i.e. the poly coefficients as json maybe
        chip_name = Path(chip_path).stem

        sicd_pixels = np.load(chip_path)
        
        defocused_image, test_img = azimuth_defocus(sicd_pixels, ph_err_order=base_config["defocus"]["phase_err_order"])

        save_name = defocused_chip_dir / f"defocused_{chip_name}.png"
        visualizer.plot_sicd(complex_pixels=defocused_image, remapper=remapper, save_path=save_name)
        print(f"Defocused {index+1}/{len(full_chip_paths)}")\
        
        if index == 10:
            break


    ######## START HERE, test this script then implement defocus sicd to only defocus the sicd, then chip the defocused sicd

    ################# START HERE, go through https://github.com/antsfamily/torchbox/tree/main/torchbox tb.fft and see how they do the phase correction
    ################# next conver this to azimuth phase error and see if that makes a difference
    visualizer.plot_sicd(
        complex_pixels=sicd_pixels, remapper=remapper, save_path="orignal.png"
    )
    print("Finished?")


if __name__ == "__main__":
    Fire(main)
