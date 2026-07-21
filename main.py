"""
main.py
-------
End-to-end pipeline for the "Image Denoising using Traditional DSP
Filters and Autoencoder" project:

  1. Build a training set of (noisy, clean) patches from sample images.
  2. Train a small convolutional autoencoder to denoise patches.
  3. Take a held-out test image, add Gaussian noise to it.
  4. Denoise it with four traditional DSP filters AND the autoencoder.
  5. Score every method against the clean ground truth using PSNR/SSIM.
  6. Save a comparison table (CSV) and a side-by-side visual figure.

Run with:  python3 main.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import build_training_set, add_gaussian_noise
from filters import apply_all_filters
from autoencoder import train_autoencoder, denoise_full_image
from metrics import compute_all_metrics

OUTPUT_DIR = "outputs"
NOISE_SIGMA = 0.1
PATCH_SIZE = 32


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/5] Building training patch dataset...")
    noisy_patches, clean_patches, test_image = build_training_set(
        patch_size=PATCH_SIZE, stride=8, sigma=NOISE_SIGMA
    )
    print(f"      {noisy_patches.shape[0]} training patches of size "
          f"{PATCH_SIZE}x{PATCH_SIZE}")

    print("[2/5] Training convolutional autoencoder...")
    model, history = train_autoencoder(noisy_patches, clean_patches, epochs=25)

    print("[3/5] Preparing held-out test image and adding noise...")
    noisy_test = add_gaussian_noise(test_image, sigma=NOISE_SIGMA, seed=7)

    print("[4/5] Running traditional DSP filters + autoencoder on test image...")
    results = apply_all_filters(noisy_test)
    results["Convolutional Autoencoder"] = denoise_full_image(
        model, noisy_test, patch_size=PATCH_SIZE
    )

    print("[5/5] Scoring every method with PSNR / SSIM...")
    # Include the untouched noisy image as a baseline row
    baseline = compute_all_metrics(test_image, {"No Denoising (Noisy Input)": noisy_test})
    scored = compute_all_metrics(test_image, results)
    all_scores = {**baseline, **scored}

    df = pd.DataFrame(all_scores).T.sort_values("PSNR", ascending=False)
    df.index.name = "Method"
    csv_path = os.path.join(OUTPUT_DIR, "comparison_results.csv")
    df.to_csv(csv_path, float_format="%.4f")
    print("\n=== PSNR / SSIM Comparison (higher is better for both) ===")
    print(df.round(4).to_string())
    print(f"\nSaved metrics table to {csv_path}")

    save_visual_comparison(test_image, noisy_test, results, df)
    save_metric_bar_chart(df)


def save_visual_comparison(clean, noisy, results, df):
    """Side-by-side figure: clean, noisy, and every denoised result,
    each labeled with its PSNR/SSIM score."""
    methods = ["No Denoising (Noisy Input)"] + list(results.keys())
    images = [noisy] + list(results.values())

    n = len(images) + 1  # +1 for the clean reference
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    axes[0].imshow(clean, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth (Clean)")
    axes[0].axis("off")

    for ax, method, img in zip(axes[1:], methods, images):
        psnr = df.loc[method, "PSNR"]
        ssim = df.loc[method, "SSIM"]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{method}\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.3f}", fontsize=10)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "visual_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visual comparison to {out_path}")


def save_metric_bar_chart(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    df["PSNR"].plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("PSNR by Method (higher = better)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].tick_params(axis="x", rotation=40)

    df["SSIM"].plot(kind="bar", ax=axes[1], color="seagreen")
    axes[1].set_title("SSIM by Method (higher = better)")
    axes[1].set_ylabel("SSIM")
    axes[1].tick_params(axis="x", rotation=40)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "metric_bar_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metric bar chart to {out_path}")


if __name__ == "__main__":
    main()
