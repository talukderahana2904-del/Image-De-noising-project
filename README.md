# Image Denoising: Traditional DSP Filters vs. Convolutional Autoencoder

Matches the CV project bullet: *"Applied computer vision pipeline for
automated image quality analysis; PSNR/SSIM metrics used to compare
traditional DSP filters with modern convolutional autoencoder."*

## What it does

1. Loads grayscale sample images (from scikit-image's built-in gallery
   — no dataset download needed) and adds synthetic Gaussian noise.
2. Trains a small convolutional autoencoder on noisy/clean image
   patches to learn how to remove that noise.
3. Denoises a held-out test image using four traditional DSP filters
   (Gaussian, Median, Bilateral, Wiener) AND the trained autoencoder.
4. Scores every method against the known-clean ground truth using
   PSNR and SSIM.
5. Saves a results table (CSV), a side-by-side visual comparison, and
   a metric bar chart to `outputs/`.

## Project structure

```
image_denoising/
├── data.py          # loads images, adds noise, builds training patches
├── filters.py        # Gaussian / Median / Bilateral / Wiener filters
├── autoencoder.py     # CNN autoencoder model + training + inference
├── metrics.py         # PSNR / SSIM scoring
├── main.py            # runs the full pipeline end to end
├── requirements.txt
└── outputs/            # generated after running main.py
    ├── comparison_results.csv
    ├── visual_comparison.png
    └── metric_bar_chart.png
```

## How to run

```bash
pip install -r requirements.txt
python3 main.py
```

Takes roughly 1-2 minutes on CPU (no GPU required). Results (metrics
table, comparison image, bar chart) are written to `outputs/`.

## Notes on the results

On a single test image with moderate Gaussian noise (sigma = 0.1),
the Wiener filter and Gaussian filter score highest on PSNR/SSIM here
— this is expected and honest: classical filters are hard to beat on
*pure, well-understood* Gaussian noise with a small autoencoder
trained on limited data. The autoencoder's real advantage shows up on
noise types that don't fit a simple statistical model (e.g. structured
or spatially varying noise), or once it's trained on a much larger,
more diverse dataset. Feel free to:

- Increase `epochs` or training patch count in `main.py` for a
  stronger autoencoder,
- Swap `add_gaussian_noise` for a different noise model (e.g.
  salt-and-pepper, speckle) to see the autoencoder pull ahead,
- Swap in your own images by editing `load_grayscale_images()` in
  `data.py`.
