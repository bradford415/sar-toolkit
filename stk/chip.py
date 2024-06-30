from pathlib import Path


from stk.utils.sicd import load_sicd

class Chipper:
    """Chip a full scene SAR image of complex data"""

    def __init__(self, chip_h, chip_w):
        """Initalize parameters for chipping the SAR image"""
        self.chip_h = chip_h
        self.chip_w = chip_w

    def chip_sicds(self, sicd_paths: list[Path]):
        
        for sicd_path in sicd_paths:
            complex_pixels = load_sicd(sicd_path)

            # Crop image to make it divisble by the desired chip dimensions
            h, w = complex_pixels.shape
            num_rows, num_cols = h // self.chip_h, w // self.chip_w
            new_h, new_w = num_rows*self.chip_h, num_cols*self.chip_w

            complex_pixels = complex_pixels[:new_h, :new_w]

            chipped_pixels = complex_pixels.reshape(new_h//self.chip_h, self.chip_h, -1, self.chip_w).swapaxes(1,2).reshape(-1, self.chip_h, self.chip_w)
            ############## S
            breakpoint()