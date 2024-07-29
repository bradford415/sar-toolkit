import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from fire import Fire
from PIL import Image
from sarpy.visualization.remap import Density

from stk.utils.sicd import load_ordered_chips, load_sicd_pixels
from stk.visualize import Visualizer
from stk.autofocus.pga import pga_az_focus

# Map of autofocus algorithms
autofocus_map = {"PGA": pga_az_focus}


def main(base_config_path: str):
    """Entrypoint for the program

    Args:
        base_config_path: Path to the base configuration file
    """
    
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Init file paths
    output_dir = Path(base_config["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    chips_root_path = Path(base_config["data"]["chips"])
    sicd_name = chips_root_path.name
    
    full_chip_paths = load_ordered_chips(chips_root_path)
    
    # Load sicd metadata for remap
    metadata_name = output_dir / "remapped_sicds" / f"{sicd_name}.json"
    with open(metadata_name) as json_file:
        sicd_metadata = json.load(json_file)
        
    # Initialize remapper with data_mean from full scene SICD
    remapper = Density(data_mean=sicd_metadata["data_mean"])

    visualizer = Visualizer()

    defocused_chip_dir = output_dir / "autofocused_chips" / sicd_name
    defocused_chip_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(base_config["defocus"]["seed"])
    
    # TODO: START HERE
    
    for index, chip_path in enumerate(full_chip_paths):
       

if __name__ == "__main__":
    orig_chip_path = "/home/bselee/programming/sar-toolkit/output/sicd_chips/CAPELLA_C02_SM_SICD_HH_20210215180158_20210215180202/CAPELLA_C02_SM_SICD_HH_20210215180158_20210215180202_5.npy"
    chip_path = "/home/bselee/programming/sar-toolkit/output/defocused_chips_from_chips/CAPELLA_C02_SM_SICD_HH_20210215180158_20210215180202/defocused_CAPELLA_C02_SM_SICD_HH_20210215180158_20210215180202_5.npy"
    orig_complex_pixels = np.load(orig_chip_path)
    defocus_complex_pixels = np.load(chip_path)

    visualizer = Visualizer()

    # Load sicd metadata for remap
    metadata_name = "/home/bselee/programming/sar-toolkit/output/remapped_sicds/CAPELLA_C02_SM_SICD_HH_20210215180158_20210215180202.json"
    with open(metadata_name) as json_file:
        sicd_metadata = json.load(json_file)
    remapper = Density(data_mean=sicd_metadata["data_mean"])
    # rg_comp_ph_hist = ft(complex_pixels, ax=1)
    # focused_imaged, _, _ = pga_az_focus(rg_comp_ph_hist, remap=remapper)
    focused_imaged, _, _ = pga_az_focus(
        defocus_complex_pixels, max_iters=10, remap=remapper
    )
    save_name_png = "pga_focused_image.png"

    ### START HERE, investigate why both of these are true
    temp_1 = Density(data_mean=sicd_metadata["data_mean"])(defocus_complex_pixels)
    temp_2 = Density(data_mean=sicd_metadata["data_mean"])(focused_imaged)
    print(np.allclose(temp_1, temp_2))

    visualizer.plot_sicd(
        complex_pixels=focused_imaged, remapper=remapper, save_path=str(save_name_png)
    )
    visualizer.plot_autofocus_chips(
        orig_chip=orig_complex_pixels,
        defocus_chip=defocus_complex_pixels,
        focus_chip=focused_imaged,
        remapper=remapper,
        save_path="3_imgs.png",
    )
