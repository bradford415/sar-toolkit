import glob
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml
from fire import Fire
from sarpy.io.phase_history.cphd import CPHDReader


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

    print(cphd_paths)

    cphd_reader = CPHDReader(cphd_paths[0])

    # Loading in all the data is much slower so for now we can just load a subset of the data
    signal_chip = cphd_reader[:]
    #signal_chip = cphd_reader[:2000, :2000]
    

    signal_time = np.fft.fftshift(np.fft.ifft(signal_chip, axis=1), axes=1)

    # Compressing other dimension should produce an image focused in the center.
    signal_img = np.fft.fftshift(np.fft.fft(signal_time, axis=0), axes=0)

    sti = 10 * np.log10(np.abs(signal_img))
    sti = sti - np.nanmax(sti)
    plt.figure()
    ax = plt.imshow(sti, cmap='gray', vmin=0.0, vmax=255.0)
    plt.colorbar()
    plt.savefig(output_dir / "chip.png")
    

    end = time.time()
    print(f"Time taken to execute{end - start}")

if __name__ == "__main__":
    Fire(main)
