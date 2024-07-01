from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sarpy.io.complex.sicd import SICDReader
from sarpy.visualization.remap import Density


class Visualizer:
    def __init__(self):
        pass

    def plot_sicd(self, complex_pixels: np.ndarray, save_path: str):
        """TODO"""
        remapped_img = Density()(complex_pixels)

        pil_img = Image.fromarray(remapped_img)
        pil_img.save(save_path)
