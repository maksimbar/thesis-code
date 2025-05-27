This package is structured as follows:

- `audio/noisy`: Contains the audio recording `F_BG014_02-a0125` with added noise at various Signal-to-Noise Ratios (SNRs) from 0 dB to 33 dB (in 3 dB increments).
- `audio/denoised/original`: Contains audio denoised using the original SDnCNN model. Model weights (`best_model.pt`) are available [here](../../packages/results/original/training/2025-05-11_08-57-06).
- `audio/denoised/proposed`: Contains audio denoised using our modified (proposed) SDnCNN model. Model weights (`best_model.pt`) are available [here](../../packages/results/proposed/training/2025-05-14_14-35-06/).

You can access the demo by opening the `index.html` file in a browser, or using link - https://maksimbar.github.io/denoising-example/