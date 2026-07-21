"""
filters.py
----------
Classical / "traditional DSP" denoising filters used as the baseline
against the convolutional autoencoder. Each filter represents a
different denoising strategy:

- Gaussian filter : linear, weights nearby pixels by a Gaussian kernel.
                     Fast, but blurs edges along with noise.
- Median filter    : nonlinear, replaces each pixel with the median of
                     its neighborhood. Very effective on salt-and-pepper
                     noise, less so on Gaussian noise.
- Bilateral filter : edge-preserving smoothing -- like Gaussian blur, but
                     the weight also depends on intensity similarity, so
                     it smooths flat regions while preserving edges.
- Wiener filter    : a statistically optimal linear filter that adapts
                     its smoothing strength to local image variance
                     (smooths more in flat/noisy regions, less in
                     high-detail regions).
"""

import numpy as np
from scipy.signal import wiener
from skimage.filters import gaussian, median
from skimage.restoration import denoise_bilateral
from skimage.morphology import disk


def apply_gaussian(noisy_image, sigma=1.0):
    return gaussian(noisy_image, sigma=sigma)


def apply_median(noisy_image, radius=2):
    # median() expects a footprint; a disk gives an isotropic neighborhood
    return median(noisy_image, footprint=disk(radius))


def apply_bilateral(noisy_image, sigma_color=0.15, sigma_spatial=5):
    return denoise_bilateral(
        noisy_image, sigma_color=sigma_color, sigma_spatial=sigma_spatial
    )


def apply_wiener(noisy_image, mysize=5):
    return np.clip(wiener(noisy_image, mysize=mysize), 0.0, 1.0)


def apply_all_filters(noisy_image):
    """Run every traditional filter and return a name -> result dict."""
    return {
        "Gaussian Filter": apply_gaussian(noisy_image),
        "Median Filter": apply_median(noisy_image),
        "Bilateral Filter": apply_bilateral(noisy_image),
        "Wiener Filter": apply_wiener(noisy_image),
    }
