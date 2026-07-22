# Image Denoising: Traditional DSP Filters vs. Convolutional Autoencoder

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/talukderahana2904-del/Image-De-noising-project/blob/main/Image_Denoising_DSP.ipynb)

I built this to answer a question I kept running into while studying signal processing and machine learning side by side: for something as basic as removing noise from an image, when does a "classical" DSP filter actually hold up against a neural network, and when does it fall apart?
So I built a small pipeline that puts four traditional filters and a convolutional autoencoder through the same test, on the same noisy image, scored the same way — and let the numbers answer the question instead of assuming deep learning automatically wins.

## Methodology

1. Take a clean grayscale image, add synthetic Gaussian noise to it (simulating something like sensor noise in low light).
2. Try to recover the clean image five different ways:
   - **Gaussian filter** — blur-based, fast, but blurs edges along with the noise
   - **Median filter** — swaps each pixel for the median of its neighborhood, great against salt-and-pepper noise
   - **Bilateral filter** — smooths flat areas but tries to preserve edges by factoring in intensity similarity, not just spatial distance
   - **Wiener filter** — a statistically "optimal" linear filter that adapts to local image variance
   - **A small convolutional autoencoder** — trained from scratch on noisy/clean image patches to learn how to denoise directly from data
3. Score every result against the real clean image using **PSNR** (pixel-level error, in dB) and **SSIM** (structural/perceptual similarity).
4. Save a comparison table and side-by-side visuals so the differences are actually visible, not just numbers in a table.

## Results and analysis

The classical filters won on this particular test:

| Method | PSNR (dB) | SSIM |
|---|---|---|
| Wiener Filter | 29.44 | 0.708 |
| Gaussian Filter | 28.89 | 0.634 |
| Convolutional Autoencoder | 28.21 | 0.668 |
| Median Filter | 27.44 | 0.532 |
| Bilateral Filter | 25.81 | 0.507 |
| No Denoising (baseline) | 20.28 | 0.202 |

I could've tuned this until the autoencoder "won," but I'd rather report what actually happened. Plain Gaussian noise is exactly the kind of well-behaved, statistically predictable noise that a Wiener filter is mathematically built to handle — so it's not surprising it holds its own against a small network trained on limited data. The autoencoder's advantage tends to show up on messier, less textbook noise (structured artifacts, compression noise, anything that doesn't fit a clean statistical model), or once it's trained on a lot more data than a handful of sample images. That's the more interesting follow-up experiment, and the code is set up so it's easy to try.

## Project layout

 data.py :loads sample images, adds noise, builds training patches
 
 filters.py : the four classical DSP filters
 
 autoencoder.py : the CNN autoencoder: architecture, training, inference
 
 metrics.py : PSNR / SSIM scoring
 
 main.py : runs everything end to end
 requirements.txt
 Image_Denoising_Colab.ipynb :same pipeline, executed on GoogleColab
 outputs :results land here after running main.py


## Running it

**Locally:**
\`\`\`bash
pip install -r requirements.txt
python3 main.py
\`\`\`
Takes about a minute or two on a regular CPU — no GPU needed for a model this small.

**On Colab:** click the badge at the top of this README, or open `Image_Denoising_Colab.ipynb` directly. It clones this repo, installs what's needed, and runs the whole thing with the results displayed inline.

Everything uses scikit-image's built-in sample images, so there's no dataset to download — it just runs.

## Future scope for betterment:
- Train the autoencoder on a much larger, more varied image set instead of a handful of samples
- Test against noise types that don't fit a clean statistical model — structured noise, compression artifacts, real sensor noise instead of synthetic Gaussian
- Try a deeper network (this one's intentionally small — two conv layers each way — to keep training fast and the comparison honest about what a *lightweight* model can do)

## Conclusion
I'm a final-year ECE student interested in hardware and signal processing, and this was as much about understanding *why* a filter works as it was about writing the code — the kind of question that comes up constantly in DSP and embedded systems, whether the "processing" is happening on a chip or in a neural net.
- Swap in your own images by editing `load_grayscale_images()` in
  `data.py`.
