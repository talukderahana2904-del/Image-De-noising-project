"""
metrics.py
----------
Image-quality metrics used to score every denoising method against
the known-clean ground truth.

- PSNR (Peak Signal-to-Noise Ratio): measures pixel-level reconstruction
  error in decibels; higher is better. Sensitive to any deviation from
  the original pixel values, regardless of perceptual relevance.

- SSIM (Structural Similarity Index): measures similarity in local
  luminance, contrast, and structure between two images; ranges from
  -1 to 1, higher is better. Correlates more closely with how a human
  actually perceives image quality than PSNR does, since it accounts
  for structural/edge information rather than only raw pixel error.
"""

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_metrics(clean_image, test_image):
    psnr = peak_signal_noise_ratio(clean_image, test_image, data_range=1.0)
    ssim = structural_similarity(clean_image, test_image, data_range=1.0)
    return {"PSNR": psnr, "SSIM": ssim}


def compute_all_metrics(clean_image, results_dict):
    """results_dict: {method_name: denoised_image}. Returns a
    {method_name: {"PSNR": ..., "SSIM": ...}} dict."""
    return {name: compute_metrics(clean_image, img) for name, img in results_dict.items()}
