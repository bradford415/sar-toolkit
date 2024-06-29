import glob
from pathlib import Path

import numpy as np
import yaml
from fire import Fire
from sarpy.io.phase_history.cphd import CPHDReader


def main(base_config_path: str):
    """Entrypoint for the project

    Args:
        base_config_path: Path to the base configuration file
    """

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    cphd_paths = glob.glob(str(Path(base_config["data"]["root"]) / "cphds" / "*.cphd"))

    print(cphd_paths)

    cphd_reader = CPHDReader(cphd_paths[0])
    signal = cphd_reader.read_signal_block()["0"]
    print(np.array(cphd_reader[:]).shape)
    print(signal.shape)


if __name__ == "__main__":
    Fire(main)
