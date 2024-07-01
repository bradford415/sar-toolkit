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

    chip_paths = glob.glob(str(Path(base_config["data"]["chips"]) / "*.npy"))

    assert chip_paths

    visualizer = Visualizer()
    
    chip_name = Path(chip_paths[0]).stem
    chip_dir = (output_dir / "remapped_chips" / chip_name)
    chip_dir.mkdir(parents=True, exist_ok=True)
    
    for index, chip_path in enumerate(chip_paths):
        chip = np.load(chip_path)

        save_name = chip_dir / f"{chip_name}_{index}.png"
        
        visualizer.plot_sicd(complex_pixels=chip, save_path=save_name)
        
        ######## TODO: FIGURE OUT WHY CHIPS DON'T HAVE BLACK BACKGROUND LIKE SICD ###########
        

if __name__ == "__main__":
    Fire(main)
