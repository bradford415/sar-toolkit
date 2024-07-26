from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from stk.utils.sicd import load_sicd_pixels


class Chipper:
    """Chip a full scene SAR image of complex data"""

    def __init__(
        self,
        chip_h: int = 256,
        chip_w: int = 256,
        output_dir: Path = Path("output/sicd_chips"),
    ):
        """Initalize parameters for chipping the SAR image

        Args:
            chip_h: Chip height
            chip_w: Chip Width
            output_dir: Output directory to save the chips
        """
        self.chip_h = chip_h
        self.chip_w = chip_w

        self.output_dir = output_dir

    def chip_sicds(self, sicd_paths: List[Path]) -> None:
        """TODO"""

        assert sicd_paths, "sicd_paths cannot be empty"

        for sicd_path in sicd_paths:
            complex_pixels, _ = load_sicd_pixels(sicd_path)

            # Crop image to make it divisble by the desired chip dimensions
            h, w = complex_pixels.shape
            num_rows, num_cols = h // self.chip_h, w // self.chip_w
            new_h, new_w = num_rows * self.chip_h, num_cols * self.chip_w

            complex_pixels = complex_pixels[:new_h, :new_w]

            # (num_chips, chip_h, chip_w)
            chipped_pixels = (
                complex_pixels.reshape(
                    new_h // self.chip_h, self.chip_h, -1, self.chip_w
                )
                .swapaxes(1, 2)
                .reshape(-1, self.chip_h, self.chip_w)
            )

            # Initialize output directory
            sicd_name = Path(sicd_path).stem
            save_dir = self.output_dir / sicd_name
            save_dir.mkdir(exist_ok=True, parents=True)

            # Save chips and csv; the csv will load chips easier for future use
            chip_ids = []
            chip_names = []
            for chip_number, chip in enumerate(chipped_pixels):
                ############### TODO, validate chips, i.e. don't save all black chips; then visualize chips

                # Save chip as numpy file
                chip_name = f"{sicd_name}_{chip_number}"
                save_name = save_dir / f"{chip_name}.npy"
                np.save(save_name, chip)

                # Save metadata for the csv chips
                chip_ids.append(chip_number)
                chip_names.append(chip_name)

            # Create csv
            chips_dict = {"chip_id": chip_ids, "chip_name": chip_names}
            chips_df = pd.DataFrame(chips_dict)
            chips_df.to_csv(Path(save_dir) / f"{sicd_name}.csv", index=False)
