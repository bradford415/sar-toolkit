from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import yaml
from fire import Fire
from PIL import Image
from sarpy.visualization.remap import Density
from scipy.fftpack import fft, fft2, fftshift, ifft, ifft2

from stk.utils.signal import ft, ift
from stk.visualize import Visualizer


# --------------------------------------------------------------------------------------
#   Adapted from the autoFocus2 function originally written by Douglas Macdonald
# as part of the RITSAR Python package located at https://github.com/dm6718/RITSAR and
# https://github.com/dm6718/RITSAR/blob/master/ritsar/imgTools.py.
#
#   We modified the phase gradient estimate to use the ML estimate from:
# Jakowatz, Charles V., and Daniel E. Wahl. "Eigenvector method for maximum-likelihood
#   estimation of phase errors in synthetic-aperture-radar imagery."
#   JOSA A 10.12 (1993): 2539-2546.
#
#  Flag for shadow PGA from:
# Prater, et al. J Prater, D Bryner, and S Synnes. "SHADOW BASED PHASE GRADIENT
#    AUTOFOCUS FOR SYNTHETIC APERTURE SONAR." 5th annual Institute of Acoustics
#    SAS/SAR Conference. Lerici, Italy. 2023.
#
#   Assumes SLC (single look complex) azimuth is vertical dimension and range increases left to right
# along the horizontal dimension.
#
# "np" is the numpy package.
# "sig" is the signal package from RITSAR.
# --------------------------------------------------------------------------------------
def pga_range_focus(img, win="auto", win_params=[100, 0.5], shadow_pga=False):
    """TODO need to modify this like pga_az_focus except keep the axis as the original version

    page 269 in Carrara

    Args:

    """
    return NotImplementedError

    ## NOTE: Might have to start with only range compressed data... i.e take the FFT in the azimuth direction
    ## Will have to test this out on a defocused chip

    # Derive parameters
    npulses = int(img.shape[0])
    nsamples = int(img.shape[1])

    # Initialize loop variables
    img_af = 1.0 * img
    max_iter = 30
    af_ph = 0
    rms = []

    # Compute phase error and apply correction
    for iteration in range(max_iter):
        # Not sure why the range bins are the rows; I think this may be wrong depending on the image
        # Find brightest azimuth sample in each range bin; [0] = darkest [-1] = brightest
        if shadow_pga:
            index = np.argsort(np.abs(img_af), axis=0)[0]
        else:
            index = np.argsort(np.abs(img_af), axis=0)[-1]

        # Circularly shift image so max values line up along azimuth center
        shifted_image = np.zeros(img.shape) + 0j
        for sample in range(nsamples):
            shifted_image[:, sample] = np.roll(
                img_af[:, sample], int(npulses / 2 - index[sample])
            )

        # Create a 1D window centered around the azimuth center; this window limits the azimuth only, not the range bins
        if win == "auto":
            # Compute window width by summing along rand bins
            range_sum = np.sum(shifted_image * np.conj(shifted_image), axis=-1)

            # Caluclate power ratio in decibels (dBs); log10(x) = # of bels and 10log10(x) = # of decibels
            range_dbs = 10 * np.log10(range_sum / range_sum.max())

            # For first iteration, use all azimuth data
            if iteration == 0:
                win_width = npulses
            # For second iteration, use half azimuth data
            elif iteration == 1:
                win_width = npulses // 2
            # For all other iterations, use twice the 10 dB threshold
            else:
                win_width = np.sum(range_dbs > -10)

            # Create window centered around azimuth (win_width,)
            window = np.arange(
                npulses / 2 - win_width / 2, npulses / 2 + win_width / 2
            ).astype(np.int32)
        else:
            # Compute window width using win_params if win not set to 'auto'
            win_width = int(win_params[0] * win_params[1] ** iteration)
            window = np.arange(
                npulses / 2 - win_width / 2, npulses / 2 + win_width / 2
            ).astype(np.int32)
            if win_width < 5:
                break

        # Extract the window of values in the center shifted image
        windowed_image = np.zeros(img.shape) + 0j
        windowed_image[window] = shifted_image[window]  # (window_length, npulses)

        # Fourier Transform along azimuth axis

        # Might need to change this name convention if it's not signals
        windowed_signals = ift(windowed_image, ax=0)

        ### TODO: start here and finish, understanding algorithm; figure out why they use the ift and not ft; also look through autofocus demo in ritsar

        # Maximum Likelihood method:
        #   1. Multiply the windowed conjugate by the window shifted down by 1 in azimuth
        #   2. Sum the result across the range bins
        #   3. Extract the angle of each complex sum
        phi_dot = np.angle(
            np.sum(np.conj(windowed_signals[:-1, :]) * windowed_signals[1:, :], axis=1)
        )

        # Integrate to obtain estimate of the phase error;
        # np.cumsum integrates by calculating a Riemann sum on a certain inverval.
        # For example, if you take a cumsum on 5-element-array the output array will represent
        # the integration on the interval of the star of the array and the index
        # i.e. cumsum_output[3] will be the value of integration from [0,3).
        # A [0] is prepended because in phi_dot we multiplied by (window_azimuth-1, window_range)
        phi = np.concatenate([[0], np.cumsum(phi_dot)])

        # TODO: Understand what this does and why its needed
        phi = np.unwrap(phi)

        # Remove linear trend; I think this straightens out a signal
        ## START HERE
        t = np.arange(0, nsamples)  # nsmaples = range
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t, phi)
        line = slope * t + intercept
        phi = phi - line
        if shadow_pga:
            phi = -phi
        rms.append(np.sqrt(np.mean(phi**2)))

        if win == "auto":
            if rms[iteration] < 0.01:
                break

        # Apply correction
        phi2 = np.tile(np.array([phi]).T, (1, nsamples))
        IMG_af = ift(img_af, ax=0)
        IMG_af = IMG_af * np.exp(-1j * phi2)
        img_af = ft(
            IMG_af, ax=0
        )  # FIXME: gets back in az signal domain because thats what the algo requires maybe?

        # Store phase
        af_ph += phi

    print("number of iterations: {}".format(iteration + 1))

    return (img_af, np.flip(af_ph), rms)


def pga_az_focus(
    complex_image: np.ndarray,
    max_iters: int = 30,
    win: str = "auto",
    win_params: List[int, int] = [100, 0.5],
    shadow_pga: bool = False,
) -> np.ndarray:
    """PGA algorithm to autofocus SAR images in the azimuth direction. This function assumes
    the image is in the complex domain, range is in the vertical direction (rows)
    while azimuth is in the horizontal direction (columns), and the image is compressed in
    both range and azimuth.

    Algorithm from Jakowatz 1994
    "Phase Gradient Autofocus — A Robust Tool for High Resolution SAR Phase Correction"

    This implementation was based on: https://github.com/dm6718/RITSAR/blob/master/ritsar/imgTools.py#L1029

    This method was then slightly modified to use the maximum-likelihood estimator described by Jakowatz in
    "Eigenvector method for maximum-likelihood estimation of phase errors in synthetic-aperture-radar imagery"

    The code modification to use the ML method is here: https://github.com/isaacgerg/synthetic_aperture_sonar_autofocus

    Args:
        complex_image: numpy array of the complex image where rows is the range and cols is the
                       azimuth/cross-range; this image should be range and azimuth compressed
        max_iters: The max iterations to run the algorithm for
        win: The window mode to use; if 'auto' automatically determine the window size in the algorithm,
             else manually specify the window size
        win_params: If win is not set to auto, these win_params will be the window size for the algorithm
                    if win is auto, this win_params is ignored
        shadow_pga: Whether to use the shadow version of PGA; I'm not sure when this is used but
                    I think it looks for the darkest scatters rather than the brightest ones

    Returns:
        The focused complex image which is range (y) and azimuth (x) compressed

        Note: The image still needs to be remapped to visualize
    """
    # Extract the number of pulses (azimuth index) and samples (range index)
    npulses = int(complex_image.shape[1])
    nsamples = int(complex_image.shape[0])

    # Initialize loop variables
    img_af = 1.0 * complex_image
    max_iter = max_iters
    af_ph = 0
    rms = []

    # Compute phase error and apply correction
    for iteration in range(max_iter):
        # Find brightest azimuth sample in each range bin; [0] = darkest [-1] = brightest
        if shadow_pga:
            index = np.argsort(np.abs(img_af), axis=1)[:, 0]
        else:
            index = np.argsort(np.abs(img_af), axis=1)[:, -1]

        # Circularly shift image so max values line up along azimuth center
        shifted_image = np.zeros(complex_image.shape) + 0j
        for sample in range(nsamples):
            shifted_image[sample, :] = np.roll(
                img_af[sample, :], int(npulses / 2 - index[sample])
            )

        # Create a 1D window centered around the azimuth center; this window limits the azimuth only, not the range bins
        if win == "auto":
            # Compute window width by summing along range direction
            range_sum = np.sum(shifted_image * np.conj(shifted_image), axis=0)

            # Caluclate power ratio in decibels (dBs); log10(x) = # of bels and 10log10(x) = # of decibels
            range_dbs = 10 * np.log10(range_sum / range_sum.max())

            # For first iteration, use all azimuth data
            if iteration == 0:
                win_width = npulses
            # For second iteration, use half azimuth data
            elif iteration == 1:
                win_width = npulses // 2
            # For all other iterations, use twice the 10 dB threshold
            else:
                win_width = np.sum(range_dbs > -10)

            # Create window centered around azimuth center (win_width,)
            window = np.arange(
                npulses / 2 - win_width / 2, npulses / 2 + win_width / 2
            ).astype(np.int32)
        else:
            # Compute window width using win_params if win not set to 'auto'
            win_width = int(win_params[0] * win_params[1] ** iteration)
            window = np.arange(
                npulses / 2 - win_width / 2, npulses / 2 + win_width / 2
            ).astype(np.int32)
            if win_width < 5:
                break

        # Extract the window of values in the center shifted image
        windowed_image = np.zeros(complex_image.shape) + 0j
        windowed_image[window] = shifted_image[window]  # (window_length, npulses)

        # Fourier Transform along azimuth axis
        # breakpoint()
        windowed_image_ift = ft(windowed_image, ax=1)

        ### TODO: start here and finish, understanding algorithm; figure out why they use the ift and not ft; also look through autofocus demo in ritsar

        # Maximum Likelihood method:
        #   1. Multiply the windowed conjugate by the window shifted down by 1 in azimuth
        #   2. Sum the result across the range direction
        #   3. Extract the angle of each complex sum
        phi_dot = np.angle(
            np.sum(
                np.conj(windowed_image_ift[:, :-1]) * windowed_image_ift[:, 1:], axis=0
            )
        )

        # Integrate to obtain estimate of the phase error;
        # np.cumsum integrates by calculating a Riemann sum on a certain inverval.
        # For example, if you take a cumsum on 5-element-array the output array will represent
        # the integration on the interval of the star of the array and the index
        # i.e. cumsum_output[3] will be the value of integration from [0,3).
        # A [0] is prepended because in phi_dot we multiplied by (window_azimuth-1, window_range)
        phi = np.concatenate([[0], np.cumsum(phi_dot)])

        # TODO: Understand what this does and why its needed
        phi = np.unwrap(phi)

        # Remove linear trend
        # From the paper "Phase Gradient Autofocus - A Robust Tool for HIgh Resolution SAR Phase Correction" - Jakowatz
        # Removing any linear trend in the phase-error estimate prevents image shifting
        t = np.arange(0, nsamples)  # nsamples = range
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t, phi)
        line = slope * t + intercept
        phi = phi - line
        if shadow_pga:
            phi = -phi
        rms.append(np.sqrt(np.mean(phi**2)))

        if win == "auto":
            if rms[iteration] < 0.01:
                break

        # Repeat the azimuth phase error along the range direction
        phi2 = np.tile(np.array([phi]), (nsamples, 1))

        # Still don't understand why the ift is used
        IMG_af = ft(img_af, ax=1)

        # Apply correction, element-wise
        IMG_af = IMG_af * np.exp(-1j * phi2)

        # Return paritally correct phase error back to complex image domain
        # for the next iteration of PGA
        img_af = ift(IMG_af, ax=1)

        # Store phase
        af_ph += phi

    print("number of iterations: {}".format(iteration + 1))

    return (img_af, np.flip(af_ph), rms)
