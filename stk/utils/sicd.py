# Utility file for SICD operations
from pathlib import Path
from typing import Tuple

import numpy as np
from sarpy.io.complex.sicd import SICDReader


def load_sicd_pixels(sicd_path: Path) -> Tuple[np.ndarray, SICDReader]:
    """Load SICD complex data from file path with .ntf file extension

    Args:
        sicd_path: Path to SICD file with .ntf extension

    Return:
        np.ndarray of complex image data
        SICDReader to be used for calculating global params in Density() class
    """
    sicd_reader = SICDReader(sicd_path)
    sicd_pixels = sicd_reader[:]
    return sicd_pixels, sicd_reader
