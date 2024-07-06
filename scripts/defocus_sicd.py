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
from scipy.fftpack import fft, fft2, fftshift, ifft, ifft2
from scipy.stats import linregress

from stk.chip import Chipper
from stk.utils.sicd import load_sicd_pixels
from stk.visualize.visualizer import Visualizer


# all FT's assumed to be centered at the origin
def ft(f, ax=-1):
    F = fftshift(fft(fftshift(f), axis=ax))
    return F


def ift(F, ax=-1):
    f = fftshift(ifft(fftshift(F), axis=ax))
    return f


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

    # Init file paths
    output_dir = Path(base_config["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    chips_root_path = Path(base_config["data"]["chips"])

    
    sicd_csv = pd.read_csv(chips_root_path / f"{chips_root_path.stem}.csv")
    
    # Gather chip paths in order; glob does not have a great way to load the file paths in order
    chip_names = sicd_csv["chip_name"].tolist()
    apply_full_path = lambda file, root: f"{root}/{file}.npy"

    full_chip_path = [apply_full_path(f_name, chips_root_path) for f_name in chip_names]

    #start here@@@@@@@@@@@@@@@@@@@@@@

    assert sicd_paths

    _, sicd_reader = load_sicd_pixels(sicd_paths[0])

    sicd_pixels = np.load(chip_paths[5])

    # Degrade image with random 10th order polynomial phase
    breakpoint()
    coeff = (np.random.rand(10) - 0.5) * sicd_pixels.shape[0]
    x = np.linspace(-1, 1, sicd_pixels.shape[0])
    y = np.poly1d(coeff)(x)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + np.mean(y)
    y = y - line
    ph_err = np.tile(np.array([y]).T, (1, sicd_pixels.shape[1]))
    img_err = ft(ift(sicd_pixels, ax=0) * np.exp(1j * ph_err), ax=0)

    visualizer = Visualizer()

    # Initialize remapper with data_mean from full scene SICD
    remapper = Density()
    remapper.calculate_global_parameters_from_reader(sicd_reader)

    save_name = "defocused.png"
    visualizer.plot_sicd(complex_pixels=img_err, remapper=remapper, save_path=save_name)
    print("Finished?")


if __name__ == "__main__":
    Fire(main)
