import numpy as np
from skimage import data, color, img_as_float
from skimage.transform import resize
from skimage.util import view_as_windows


def load_grayscale_images():
   
    sources = [
        data.astronaut(),
        data.chelsea(),
        data.coffee(),
        data.rocket(),
    ]
    grays = []
    for img in sources:
        gray = color.rgb2gray(img)
        gray = resize(gray, (256, 256), anti_aliasing=True)
        grays.append(img_as_float(gray))
    return grays


def add_gaussian_noise(image, sigma=0.1, seed=None):
   
    rng = np.random.default_rng(seed)
    noisy = image + rng.normal(0, sigma, image.shape)
    return np.clip(noisy, 0.0, 1.0)


def extract_patches(image, patch_size=32, stride=16):
   
    windows = view_as_windows(image, (patch_size, patch_size), step=stride)
    patches = windows.reshape(-1, patch_size, patch_size)
    return patches


def build_training_set(patch_size=32, stride=16, sigma=0.1, seed=42):
   
    images = load_grayscale_images()
    train_images, test_image = images[:-1], images[-1]

    clean_patches = []
    for img in train_images:
        clean_patches.append(extract_patches(img, patch_size, stride))
    clean_patches = np.concatenate(clean_patches, axis=0)

    rng = np.random.default_rng(seed)
    noisy_patches = np.clip(
        clean_patches + rng.normal(0, sigma, clean_patches.shape), 0.0, 1.0
    )

    # Add channel dimension for Conv2D layers: (N, H, W, 1)
    clean_patches = clean_patches[..., np.newaxis].astype(np.float32)
    noisy_patches = noisy_patches[..., np.newaxis].astype(np.float32)

    return noisy_patches, clean_patches, test_image
