from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sarpy.io.complex.sicd import SICDReader
from sarpy.visualization.remap import Density


class Visualizer:
    def __init__(self, output_dir: Path):
        """TODO"""
        self.output_dir = output_dir

    def plot_sicd(self, complex_pixels: np.ndarray, remap=True):
        """TODO"""

        remapped_img = Density()(complex_pixels)

        pil_img = Image.fromarray(remapped_img)
        pil_img.save(self.output_dir / "sicd_remap.png")
