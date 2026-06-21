from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

from stk.utils.sicd import load_sicd_pixels


class Chipper:
    """Chip a full scene SAR image of complex data"""

    def __init__(
        self,
        chip_h: int = 256,
        chip_w: int = 256,
        remap: bool = False,
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
        
        self.remap = remap

        self.output_dir = output_dir

    def chip_sicds(self, sicd_paths: List[Path]) -> None:
        """Chip the list of sicds"""

        assert sicd_paths, "sicd_paths cannot be empty"

        for index, sicd_path in enumerate(sicd_paths):
            print(f"Chipping SICD {index+1}/{len(sicd_paths)}")

            if Path(sicd_path).suffix == ".png":
                pil_image = Image.open(sicd_path)
                pixels = np.array(pil_image)
            else:
                pixels, _ = load_sicd_pixels(sicd_path, self.remap)

            # Crop image to make it divisble by the desired chip dimensions
            h, w = pixels.shape
            num_rows, num_cols = h // self.chip_h, w // self.chip_w
            new_h, new_w = num_rows * self.chip_h, num_cols * self.chip_w

            pixels = pixels[:new_h, :new_w]

            # (num_chips, chip_h, chip_w)
            chipped_pixels = (
                pixels.reshape(
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
