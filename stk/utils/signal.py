from scipy.fftpack import fft, fft2, fftshift, ifft, ifft2


# All FT's assumed to be centered at the origin
def ft(img_domain_data, ax=-1):
    """1D Fourier transform complex image domain into signal domain along a specific axis

    Args:
        img_domain_data: Data in the time domain; typically, this will be complex image data
        ax: Axis to perform the 1D fourier transform along

    """
    F = fftshift(fft(fftshift(img_domain_data), axis=ax))
    return F


def ift(sig_domain_data, ax=-1):
    """1D Inverse fourier transform. Transform frequency domain into the image domain
    along a specific axis.

    Args:
        sig_domain_data: Data in the signal/frquency domain
        ax: Axis to perform the 1D inverse fourier transform along
    """
    f = fftshift(ifft(fftshift(sig_domain_data), axis=ax))
    return f
