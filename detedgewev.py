from typing import Tuple, Callable, Union
import pywt
import numpy as np
from detedge import FFilter, Range, Spectrum, Mask, lpf, hpf, bpf


def detect_edges_wavelet(img: np.ndarray, ffilter: FFilter, r: Union[float, Range], wavelet='db6') -> Tuple[np.ndarray, Tuple[Spectrum, Mask]]:
    """
    Compute wdt in the given image and filter the coefficients using the mask
    returned by ffilter function in the given radio or radio_range

    :param img: image as 2-D numpy.ndarray

    :param ffilter: one of the functions lpf, hpf, bpf

    :param r: if ffilter is lpf or hpf is a float representing the radio
            if ffilter is bpf is 2-items tuple with (inner_radio, outer_radio)

    :return: image as numpy.array with the applied filter
    """

    wdt: np.ndarray = pywt.wavedec2(np.float32(img), wavelet=wavelet, level=6)
    arr, slices = pywt.coeffs_to_array(wdt)
    mask: np.ndarray = ffilter(arr, r)
    wdt_shift: np.ndarray = np.fft.fftshift(arr)

    magnitude_spectrum: np.ndarray = 20 * np.log(np.abs(wdt_shift))

    # apply a mask and inverse WDT
    wdt_shift *= mask

    fshift_mask_mag = 2000 * np.log(np.abs(wdt_shift))

    iwdt_shift = np.fft.ifftshift(wdt_shift)
    coeff_from_arr = pywt.array_to_coeffs(iwdt_shift, slices, output_format='wavedecn')
    wdt_rec = pywt.waverecn(coeff_from_arr, wavelet=wavelet)
    img_back = np.abs(wdt_rec)

    return img_back, (magnitude_spectrum, fshift_mask_mag)
