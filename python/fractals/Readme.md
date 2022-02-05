# Readme

Some examples of how to generate fractals using Python.

Emphasis on fractional Brownian motion in 1 and 2-d.

There is a separate document containing relevant [links and resources](Notes.md).

## Programs

### Deterministic

* [Sierpinski triangle](sierp.py) - every baby's first fractal
* [Koch curve](koch.py) - showing steps in the generation
* [Mandelbrot set](mandel.py) - a very basic construction

### Random

These use numpy (and some use scipy):

#### 1d - 1 Euclidean dimension, so fractal dimension between 1 and 2

* [Brownian motion](bm.py) - a simple random fractal (fBm for H=0.5)
* [Exact fBm](fbm.py) - simple fractional Brownian motion - exact sampling using Cholesky decomposition
* [fBm RMD](fbmrmd.py) - approximate fBm using random midpoint displacement
* [fBm FS](fbmfs.py) - approximate fBm using an elementary Fourier synthesis approach
* [fBm FFT](fbmfft.py) - appromimate fBm using the Fast Fourier Transform (FFT) - the classic spectral synthesis approach
* [fBm DCT](fbmdct.py) - using the Discrete Cosine Transform (DCT) instead of the FFT - an arguably simpler and better spectral synthesis approach, but requires (very basic) knowledge of the DCT

#### 2d - 2d versions of most of the above algorithms, so fractal dimension between 2 and 3

* [fBm 2D diamond square](fbm2ds.py) - the diamond square algorithm - the natural 2d adaptation of the random midpoint displacement approach
* [fBm 2D FS](fbm2fs.py) - an elementary Fourier synthesis approach - very slow and inefficient
* [fBm 2D FFT](fbm2fft.py) - FFT is much faster and more efficient - but need to understand the FFT quite well to see why the coefficients get set the way they do
* [fBm 2D DCT](fbm2dct.py) - again, the DCT is arguably a simpler and better approach

