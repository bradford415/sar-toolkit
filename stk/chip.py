from pathlib import Path

import numpy as np

from stk.utils.sicd import load_sicd


class Chipper:
    """Chip a full scene SAR image of complex data"""

    def __init__(
        self,
        chip_h: int = 256,
        chip_w: int = 256,
        output_dir: Path = Path("output/sicd_chips"),
    ):
        """Initalize parameters for chipping the SAR image"""
        self.chip_h = chip_h
        self.chip_w = chip_w

        self.output_dir = output_dir

    def chip_sicds(self, sicd_paths: list[Path]) -> None:
        """TODO"""

        for sicd_path in sicd_paths:
            complex_pixels = load_sicd(sicd_path)

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

            for index, chip in enumerate(chipped_pixels):
                ############### TODO, validate chips, i.e. don't save all black chips; then visualize chips
                self._save_chip(chip, sicd_path, index)

    def _save_chip(self, complex_chip: np.ndarray, sicd_path: str, chip_number: int):
        """Save the complex chip as a .npy file

        Args:
            complex_chip: Chip with complex pixel data (chip_h, chip_w)
            chip_number: Chip number; this will be used to make a unique name to save the chip as
        """
        sicd_name = Path(sicd_path).stem
        save_dir = self.output_dir / sicd_name
        save_name = save_dir / f"{sicd_name}_{chip_number}.npy"

        save_dir.mkdir(exist_ok=True, parents=True)
        np.save(save_name, complex_chip)
