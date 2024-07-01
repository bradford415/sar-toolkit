# Utility file for SICD operations
from pathlib import Path

from sarpy.io.complex.sicd import SICDReader


def load_sicd(sicd_path: Path):
    """Load SICD complex data from file path with .ntf file extension

    Args:
        sicd_path: Path to SICD file with .ntf extension

    Return:
        np.ndarray of complex image data
    """
    sicd_reader = SICDReader(sicd_path)
    sicd_pixels = sicd_reader[:]
    return sicd_pixels
