import argparse
import glob
from pathlib import Path

import yaml

from stk.chip import Chipper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chip SICD files into square chips.")
    parser.add_argument("sicd_dir", type=str, help="Path to a directory of SICDS (.nitf or .ntf)")
    parser.add_argument(
        "--chip-size",
        type=int,
        default=512,
        help="Desired square chip size in pixels",
    )
    parser.add_argument("--remap", default=False, action='store_true', help="Whether to perform a density remap on the chips")
    parser.add_argument("--output_path", type=str, default="output", help="Output path for where to save the chips")
    
    return parser.parse_args()


def main(sicd_dir: str, chip_size: int, remap: bool, output_path: str):
    """Entrypoint for the project

    Args:
        output_path: path to save the output files to
        chip_size: Desired square chip size in pixels
    """

    # Init file paths
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    sicd_dir = Path(sicd_dir)
    if sicd_dir.is_dir():
        sicd_paths = []
        sicd_paths += glob.glob(str(Path(sicd_dir) / "*.ntf"))
        sicd_paths += glob.glob(str(Path(sicd_dir) / "*.nitf"))
    else:
        sicd_paths = [str(sicd_dir)]

    assert sicd_paths, "sicd_paths cannot be empty"

    sicd_chipper = Chipper(chip_size, chip_size, remap)
    sicd_chipper.chip_sicds(sicd_paths)


if __name__ == "__main__":
    args = parse_args()
    main(args.sicd_dir, args.chip_size, args.remap, args.output_path)
