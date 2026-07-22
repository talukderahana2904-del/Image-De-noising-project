import numpy as np
from scipy.signal import wiener
from skimage.filters import gaussian, median
from skimage.restoration import denoise_bilateral
from skimage.morphology import disk
def apply_gaussian(noisy_image, sigma=1.0):
    return gaussian(noisy_image, sigma=sigma)
def apply_median(noisy_image, radius=2):
    return median(noisy_image, footprint=disk(radius))
def apply_bilateral(noisy_image, sigma_color=0.15, sigma_spatial=5):
    return denoise_bilateral(noisy_image, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
def apply_wiener(noisy_image, mysize=5):
    return np.clip(wiener(noisy_image, mysize=mysize), 0.0, 1.0)
def apply_all_filters(noisy_image):
    return {
        "Gaussian Filter": apply_gaussian(noisy_image),
        "Median Filter": apply_median(noisy_image),
        "Bilateral Filter": apply_bilateral(noisy_image),
        "Wiener Filter": apply_wiener(noisy_image),
    }
