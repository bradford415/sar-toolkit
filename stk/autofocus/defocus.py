import glob
import time
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from fire import Fire
from sarpy.io.complex.sicd import SICDReader
from sarpy.io.phase_history.cphd import CPHDReader
from sarpy.visualization.remap import Density
from scipy.stats import linregress

from stk.chip import Chipper
from stk.utils.sicd import load_sicd_pixels
from stk.utils.signal import ft, ift
from stk.visualize.visualizer import Visualizer


def azimuth_defocus(
    complex_pixels: np.ndarray,
    ph_err_order: int = 10,
    coeff_multiplier=64,
    rand_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Defocus a complex_image by an nth order polynomial phase error in the azimuth direction.
    This implementation is largely based on:

    This function assumes the range is the y-axis and the azimuth is the x-direction.
    In the original implementation, it assumes the opposite.

    Args:
        complex_pixels: An image of complex pixels that have been range and azimuth compressed; a standard sicd
        ph_err_order: Order of the random phase error polynomial to the degrade the image by;
                      must be greater than 1 becuase removing the linear trend would cancel out the phase error
        coeff_multiplier: A constant/2 to multiply the phase error coefficients

    Return:
        np.ndarray: Complex data of the defocused image; image is range and azimuth compressed
        np.ndaaray: The coefficients of the polynomial phase error
    """
    if ph_err_order <= 1:
        raise ValueError("Polynomial order of the phase error must be greater than 1.")

    # Since np.poly1d includes the constant, we need to increment by for it to be true nth order polynomial
    ph_err_order += 1

    # Degrade image with random 9th order polynomial phase;
    # Generate 10 random coefficients between [-0.5,0.5] and multiply by azimuth length;
    # for a 256x256 chip, this will generate coefficients between [-128, 128]
    coeffs = (np.random.rand(ph_err_order) - 0.5) * (
        coeff_multiplier / 2
    )  # (complex_pixels.shape[1]/2)

    # Generate sequential x data points between [-1,1] for each azimuth index
    x = np.linspace(-1, 1, complex_pixels.shape[1])

    # Evaluate the nth order polynomial at each x point
    poly = np.poly1d(coeffs)
    y = poly(x)

    # Create a line from the data using  least-squares regression to remove the linear trend;
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + np.mean(
        y
    )  # np.mean(y) is the same as intercept in this case; this is only true when the data is mean-centered i.e. subtracting the y mean from y and x mean from x

    y = y - line

    # Repeat the azimuth phase error across the range bins;
    # Example: y = [[1, 2, 3]]    ph_err = [[1,     2,     3    ]
    #                                       [1,     2,     3    ]
    #                                       [1,     2,     3    ]
    #                                       [...,   ...,   ...  ]
    #                                       [range, range, range]]
    ph_err = np.tile(y[np.newaxis, :], (complex_pixels.shape[0], 1))

    # Apply phase error by taking the FT along the azimuth direction, then multiply the phase error element-wise
    # by the range-compressed phase history domain data; after, take the IFT in azimuth to visualize the image
    # In the the original implementation, the ift is taken then the phase error is applied.
    # I cannot understand why this is but both methods return similar images.
    defocused_img = ift(ft(complex_pixels, ax=1) * np.exp(1j * ph_err), ax=1)

    return defocused_img, coeffs, poly.order, coeff_multiplier
