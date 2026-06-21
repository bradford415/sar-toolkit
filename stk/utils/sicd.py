# Utility file for SICD operations
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sarpy.io.complex.sicd import SICDReader
from sarpy.visualization.remap import Density


def load_sicd_pixels(sicd_path: Path, remap: bool = False) -> Tuple[np.ndarray, SICDReader]:
    """Load SICD complex data from file path with .ntf file extension

    Args:
        sicd_path: Path to SICD file with .ntf extension
        remap: Whether to perform a density remap

    Return:
        np.ndarray of complex image data
        SICDReader to be used for calculating global params in Density() class
    """
    sicd_reader = SICDReader(sicd_path)
    sicd_pixels = sicd_reader[:]
    if remap:
        print("Applying density remap")
        sicd_pixels = Density()(sicd_pixels)
    return sicd_pixels, sicd_reader


def load_ordered_chips(chips_root_path):
    """Load the sicd chips in order from a csv file and return a list of chip paths

    Args:
        chips_root_path: Path to the scid chips
    """
    sicd_csv = pd.read_csv(chips_root_path / f"{chips_root_path.stem}.csv")

    # Gather chip paths in order; glob does not have a great way to load the file paths in order
    chip_names = sicd_csv["chip_name"].tolist()
    apply_full_path = lambda file, root: f"{root}/{file}.npy"

    full_chip_paths = [
        apply_full_path(f_name, chips_root_path) for f_name in chip_names
    ]
    return full_chip_paths


def load_sicd_json():
    pass
