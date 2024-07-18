import numpy as np
import scipy.stats
from PIL import Image
from scipy.fftpack import fft, fft2, fftshift, ifft, ifft2


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
def pga(img, win="auto", win_params=[100, 0.5], shadow_pga=False):
    """TODO

    page 269 in Carrara

    Args:

    """

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
        #breakpoint()
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


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# C Schlick Rational Tone Mapping Operator
# targetBrightness is 0,1.
def schlick(L, targetBrightness=0.3, medianFlag=True):
    L = np.squeeze(L)
    if np.iscomplex(L).sum() > 0:
        L = np.abs(L)

    L = normalize(L.astype("float32"))

    # determine b
    if medianFlag:
        m = np.median(L[np.where(L > 0)])
    else:
        m = np.sqrt(np.sum(L**2) / (2 * np.prod(L.shape)))
    #

    if np.isnan(m):
        return np.zeros_like(L)

    b = (targetBrightness - targetBrightness * m) / (m - targetBrightness * m)
    b = np.clip(b, 1, 99999999)

    # apply b
    L = (b * L) / ((b - 1) * L + 1 + 1e-9)

    return L


# -----------------------------------------------------------------------------------------------------------------------------------------------------
def imwrite(mat, filename, normalize_data=True):
    if mat.ndim == 3:
        mat = mat[:, :, 0:3]
        if normalize_data:
            mat = (normalize(mat) * 255).astype("uint8")

    else:  # grayscale
        if normalize_data:
            mat = (normalize(mat) * 255).astype("uint8")
        else:
            mat = (mat * 255).astype("uint8")
        #
    img = Image.fromarray(mat)
    img.save(filename)
    return


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normalizes array to [0,1]
def normalize(arr):
    arr -= arr.min()
    arr /= arr.max() + 1e-9
    return arr


# -----------------------------------------------------------------------------------------------------------------------------------------------------
def get_fig_as_numpy(fig):
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)

    # Open the image with PIL and convert to NumPy array
    image = Image.open(buf)
    image_array = np.array(image)

    # Close the BytesIO object
    buf.close()

    # Resize the image to 256x256 using PIL
    resized_image = Image.fromarray(image_array).resize((512, 512), Image.ANTIALIAS)
    resized_image_array = np.array(resized_image)

    return resized_image_array


# all FT's assumed to be centered at the origin
def ft(f, ax=-1):
    F = fftshift(fft(fftshift(f), axis=ax))
    return F


def ift(F, ax=-1):
    f = fftshift(ifft(fftshift(F), axis=ax))
    return f


from stk.utils.sicd import load_sicd_pixels

if __name__ == "__main__":
    chip_path = "/home/bselee/programming/sar-toolkit/output/sicd_chips/CAPELLA_C02_SM_SICD_HH_20210215180158_20210215180202/CAPELLA_C02_SM_SICD_HH_20210215180158_20210215180202_5.npy"
    complex_pixels = np.load(chip_path)
    breakpoint()
    pga(complex_pixels)
    breakpoint()
