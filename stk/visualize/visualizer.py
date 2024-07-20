from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sarpy.io.complex.sicd import SICDReader
from sarpy.visualization.remap import Density


class Visualizer:
    def __init__(self):
        pass

    def plot_sicd(self, complex_pixels: np.ndarray, remapper: Density, save_path: str):
        """TODO"""
        remapped_img = remapper(complex_pixels)

        pil_img = Image.fromarray(remapped_img)
        pil_img.save(save_path)

    def plot_autofocus_chips(
        self,
        orig_chip: np.ndarray,
        defocus_chip: np.ndarray,
        focus_chip: np.ndarray,
        remapper: Density,
        save_path: str,
    ):
        fig, axs = plt.subplots(3, 1)
        all_chips_array = np.hstack((orig_chip, defocus_chip, focus_chip))

        remapped_img = remapper(all_chips_array)

        pil_img = Image.fromarray(remapped_img)
        pil_img.save(save_path)
        # all_chips = [orig_chip, defocus_chip, focus_chip]
        # for index, ax in enumerate(axs.flat):
        #     remapped_chip = remapper(all_chips[index])
        #     ax.imshow(remapped_chip)
        # fig.savefig(save_path)
