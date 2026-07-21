"""
autoencoder.py
--------------
A small convolutional autoencoder for image denoising.

Architecture idea: the ENCODER progressively downsamples the noisy
patch through convolution + max-pooling, forcing the network to
compress the image into a compact feature representation that
captures structure but discards noise (noise, being random, doesn't
compress well and gets dropped). The DECODER then upsamples that
representation back to full resolution, reconstructing a clean
version of the patch.

This is intentionally small (a handful of conv layers) since it is
trained on 32x32 patches for a portfolio-scale project -- production
denoising networks (e.g. DnCNN) are deeper and trained on much larger
datasets.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_shape=(32, 32, 1)):
    inputs = keras.Input(shape=input_shape)

    # ---- Encoder ----
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D(2, padding="same")(x)

    # ---- Decoder ----
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    autoencoder = keras.Model(inputs, decoded, name="denoising_autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_autoencoder(noisy_patches, clean_patches, epochs=15, batch_size=64):
    model = build_autoencoder(input_shape=noisy_patches.shape[1:])
    history = model.fit(
        noisy_patches,
        clean_patches,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2,
    )
    return model, history


def denoise_full_image(model, image, patch_size=32, stride=16):
    """Apply a patch-trained autoencoder to a full-resolution image.

    Non-overlapping tiling causes visible grid artifacts at patch
    boundaries (each patch is denoised independently, so neighboring
    patches don't agree at the seam). To avoid this we instead use
    OVERLAPPING patches (stride < patch_size) and average every pixel
    across all the patch predictions that covered it -- this blends
    the seams smoothly, the same idea as overlap-add reconstruction
    in classical signal processing.
    """
    import numpy as np

    h, w = image.shape
    pad_h = (-((h - patch_size)) % stride) if h > patch_size else patch_size - h
    pad_w = (-((w - patch_size)) % stride) if w > patch_size else patch_size - w
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape

    accum = np.zeros_like(padded, dtype=np.float64)
    weight = np.zeros_like(padded, dtype=np.float64)

    # Collect every patch position (including a final flush pass against
    # the far edge so the last few pixels are always covered too).
    ys = list(range(0, ph - patch_size + 1, stride))
    if ys[-1] != ph - patch_size:
        ys.append(ph - patch_size)
    xs = list(range(0, pw - patch_size + 1, stride))
    if xs[-1] != pw - patch_size:
        xs.append(pw - patch_size)

    batch, coords = [], []
    for i in ys:
        for j in xs:
            batch.append(padded[i : i + patch_size, j : j + patch_size])
            coords.append((i, j))
    batch = np.array(batch)[..., np.newaxis].astype("float32")

    predictions = model.predict(batch, verbose=0)[..., 0]

    for (i, j), pred in zip(coords, predictions):
        accum[i : i + patch_size, j : j + patch_size] += pred
        weight[i : i + patch_size, j : j + patch_size] += 1.0

    output = accum / weight
    return output[:h, :w]
