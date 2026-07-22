

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics(clean_image, test_image):
    psnr = peak_signal_noise_ratio(clean_image, test_image, data_range=1.0)
    ssim = structural_similarity(clean_image, test_image, data_range=1.0)
    return {"PSNR": psnr, "SSIM": ssim}
def compute_all_metrics(clean_image, results_dict):
    return {name: compute_metrics(clean_image, img) for name, img in results_dict.items()}
