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

    full_chip_paths = [apply_full_path(f_name, chips_root_path) for f_name in chip_names]


    assert sicd_paths

    _, sicd_reader = load_sicd_pixels(sicd_paths[0])

    sicd_pixels = np.load(full_chip_paths[5])

    # Degrade image with random 9th order polynomial phase
    breakpoint()
    # Generate 10 random coefficients between [-0.5,0.5] and multiply by azimuth length; 
    # for a 256x256 chip, this will generate coefficients between [-128, 128]
    coeff = (np.random.rand(3) - 0.5) * sicd_pixels.shape[0]
    
    # Generate sequential x data points between [-1,1] for each azimuth index
    x = np.linspace(-1, 1, sicd_pixels.shape[0])
    
    # Evaluate the nth order polynomial at each x point
    y = np.poly1d(coeff)(x)
    
    # Create a line from the data using  least-squares regression to remove the linear trend;
    # I'm not sure why we need to remove the linear trend 
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + np.mean(y) # np.mean(y) is the same as intercept in this case; this is only true when the data is mean-centered i.e. subtracting the y mean from y and x mean from x
    y = y - line
    
    # Repeat the azimuth phase error y across the range bins
    # Example: y = [[1],    ph_err = [[1, 1, 1, ..., range_len]
    #               [2],              [2, 2, 2, ..., range_len]
    #               [3]]              [3, 3, 3, ..., range_len]]
    ph_err = np.tile(y[:, np.newaxis], (1, sicd_pixels.shape[1]))
    
    # Apply phase error by taking the inverse
    img_err = ft(ift(sicd_pixels, ax=0) * np.exp(1j * ph_err), ax=0)
    img_err2 = ift(ft(sicd_pixels, ax=0) * np.exp(1j * ph_err), ax=0)

    visualizer = Visualizer()

    # Initialize remapper with data_mean from full scene SICD
    remapper = Density()
    remapper.calculate_global_parameters_from_reader(sicd_reader)
    breakpoint()
    
    ################# START HERE, go through https://github.com/antsfamily/torchbox/tree/main/torchbox tb.fft and see how they do the phase correction
    ################# next conver this to azimuth phase error and see if that makes a difference

    save_name = "defocused2.png"
    save_name_test = "defocused2_test.png"
    visualizer.plot_sicd(complex_pixels=img_err, remapper=remapper, save_path=save_name)
    visualizer.plot_sicd(complex_pixels=img_err2, remapper=remapper, save_path=save_name_test)
    visualizer.plot_sicd(complex_pixels=sicd_pixels, remapper=remapper, save_path="orignal.png")
    print("Finished?")


if __name__ == "__main__":
    Fire(main)
