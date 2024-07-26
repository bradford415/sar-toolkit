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
from stk.utils.sicd import load_ordered_chips, load_sicd_pixels
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

    np.random.seed(base_config["defocus"]["seed"])

    defocus_metadata = []
    for index, chip_path in enumerate(full_chip_paths):
        ## TODO: Save chip error metadata i.e. the poly coefficients as json maybe
        chip_name = Path(chip_path).stem

        sicd_pixels = np.load(chip_path)

        defocused_image, coeffs, poly_order, coe_multiplier = azimuth_defocus(
            sicd_pixels,
            ph_err_order=base_config["defocus"]["phase_err_order"],
            coeff_multiplier=base_config["defocus"]["coeff_multiplier"],
            rand_seed=base_config["defocus"]["seed"],
        )

        # Save complex image as .npy and remapped image as .png
        save_name_npy = defocused_chip_dir / f"defocused_{chip_name}.npy"
        save_name_png = defocused_chip_dir / f"defocused_{chip_name}.png"
        np.save(save_name_npy, defocused_image)
        visualizer.plot_sicd(
            complex_pixels=defocused_image, remapper=remapper, save_path=save_name_png
        )

        metadata = {
            "chip_name": chip_name,
            "phase_error_coeffs": list(coeffs),
            "poly_order": poly_order,
            "coeffs_multiplier": coe_multiplier,
        }
        defocus_metadata.append(metadata)

        if index % 50 == 0:
            print(f"Defocused {index+1}/{len(full_chip_paths)} chips...")

    print("\nWriting metadata file")
    metadata_name = defocused_chip_dir / f"ph_err_meta.json"
    with open(metadata_name, "w") as json_file:
        json.dump(defocus_metadata, json_file)

    print("\nFinished!")


if __name__ == "__main__":
    Fire(main)
