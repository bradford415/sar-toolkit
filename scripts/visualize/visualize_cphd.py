import glob
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from fire import Fire
from sarpy.io.complex.sicd import SICDReader
from sarpy.io.phase_history.cphd import CPHDReader
from sarpy.visualization.remap import Density

from stk.visualize import Visualizer


def main(base_config_path: str):
    """Entrypoint for the project

    Args:
        base_config_path: Path to the base configuration file
    """

    # Calculate the start time
    start = time.time()

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Init file paths
    output_dir = Path(base_config["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cphd_paths = glob.glob(str(Path(base_config["data"]["root"]) / "cphds" / "*.cphd"))
    sicd_paths = glob.glob(str(Path(base_config["data"]["root"]) / "sicds" / "*.ntf"))

    assert cphd_paths is not None
    assert sicd_paths is not None

    cphd_reader = CPHDReader(cphd_paths[0])
    sicd_reader = SICDReader(sicd_paths[0])
    sicd_pixels = sicd_reader[:]

    # Loading in all the data is much slower so for now we can just load a subset of the data
    # signal_chip = cphd_reader[:]
    # print(signal_chip.shape)
    signal_chip = cphd_reader[:10000, :4148]

    signal_time = np.fft.fftshift(np.fft.ifft(signal_chip, axis=1), axes=1)

    # Compressing other dimension should produce an image focused in the center.
    signal_img = np.fft.fftshift(np.fft.fft(signal_time, axis=0), axes=0)

    sti = 10 * np.log10(np.abs(signal_img))
    sti = sti - np.nanmax(sti)

    visualizer = Visualizer(output_dir)
    visualizer.plot_sicd(complex_pixels=sicd_pixels)
    exit()

    # remapped_img = Density()(signal_img)
    remapped_img = Density()(sicd_pixels)

    fig, ax = plt.subplots()
    # ax = plt.imshow(remapped_img, cmap="gray", vmin=-35.0, vmax=0.0)\
    ax.axis("off")
    ax.imshow(remapped_img, cmap="gray", vmin=0.0, vmax=255.0)
    plt.savefig(output_dir / "sicd_remap.png", bbox_inches="tight", dpi=400)

    end = time.time()
    print(f"Time taken to execute {end - start}")


if __name__ == "__main__":
    Fire(main)
