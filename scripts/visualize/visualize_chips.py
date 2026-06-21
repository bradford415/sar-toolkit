import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sarpy.visualization.remap import Density

from stk.visualize import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize chipped SICD numpy files.")
    parser.add_argument(
        "chip_dir",
        type=str,
        help="Path to the directory containing chip .npy files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output root containing remapped_sicds metadata and where PNGs will be saved",
    )

    return parser.parse_args()


def main(chip_dir: str, output_dir: str):
    """Entrypoint for the project

    Args:
        input_dir: Path to the directory containing chip .npy files
        output_dir: Output root containing metadata and visualization output
    """

    # Init file paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chips_path = Path(chip_dir)
    sicd_name = chips_path.name

    ## TODO: Load chips with csv, not glob

    # Gather all chip paths
    chip_paths = sorted(glob.glob(str(chips_path / "*.npy")))

    # Load sicd metadata for remap
    is_remapped = False if np.load(chip_paths[0]).dtype == np.complex64 else True
    
    if not is_remapped:
        metadata_name = output_dir / "remapped_sicds" / f"{sicd_name}.json"
        with open(metadata_name) as json_file:
            sicd_metadata = json.load(json_file)

        assert chip_paths
        # Initialize remapper with data_mean from full scene SICD
        remapper = Density(data_mean=sicd_metadata["data_mean"])
    else:
        remapper = None
        
    visualizer = Visualizer()

    chip_dir = output_dir / "remapped_chips" / sicd_name
    chip_dir.mkdir(parents=True, exist_ok=True)

    for index, chip_path in enumerate(chip_paths):
        print(f"Saving chip {index +1}/{len(chip_paths)}")
        chip_name = Path(chip_path).stem
        chip = np.load(chip_path)

        save_name = chip_dir / f"{chip_name}.png"

        visualizer.plot_sicd(
            pixels=chip, remapper=remapper, save_path=save_name
        )


if __name__ == "__main__":
    args = parse_args()
    main(args.chip_dir, args.output_dir)
