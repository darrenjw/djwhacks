#!/usr/bin/env python3
# filter.py
# test out some filters on a test dataset to understand how well they smooth and especially how well they do at smoothing near the ends of the series

import numpy as np
import scipy as sp
from scipy.signal import savgol_filter, butter, sosfiltfilt
import matplotlib.pyplot as plt

# average atlantic sea surface temperatures from the CANARI LE
v1 = np.load("a_mean_sst.npy")
print(v1.shape)
v2 = np.load("future_a_mean_sst.npy")
print(v2.shape)

sst = np.hstack([v1, v2])
print(sst.shape)
sst_ts = sst.reshape((40, 150*12))
print(sst_ts.shape)

def moving_average(x, window=120):
    """Centred moving average over `window` points (default 120 months = 1 decade).
    Returns an array the same length as x, with NaN where the window is incomplete
    (i.e. the first and last ~window/2 points), so the end effects are visible."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ma = np.convolve(x, np.ones(window) / window, mode="valid")
    start = (window - 1) // 2
    out[start:start + len(ma)] = ma
    return out

def savgol(x, window=121, polyorder=3):
    """Savitzky-Golay smoother: local least-squares polynomial fit of degree
    `polyorder` over `window` points (must be odd; default ~1 decade of months).
    Uses mode="interp", so the ends are handled by fitting a polynomial to the
    last `window` points rather than by padding."""
    return savgol_filter(np.asarray(x, dtype=float), window, polyorder, mode="interp")

def butterworth(x, period=120, order=4):
    """Zero-phase Butterworth low-pass filter with cutoff at 1/`period` cycles
    per sample (default 1/120 per month, i.e. decadal). Applied forwards and
    backwards (sosfiltfilt) so there is no phase lag; the effective order is
    doubled. Ends are handled by odd-reflection padding (scipy default)."""
    sos = butter(order, 1.0 / period, btype="low", fs=1.0, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=float))

def fft_lowpass(x, period=120, width=0.5):
    """FFT-based low-pass filter. Frequencies below 1/`period` cycles per sample
    are kept, those above (1+`width`)/`period` are removed, with a raised-cosine
    roll-off in between (a hard brick-wall cut causes Gibbs ringing). The FFT
    assumes the signal is periodic, so to avoid wrap-around artefacts the series
    is first extended at both ends by odd reflection (as for the Butterworth
    filter), filtered, and then the padding is discarded."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    pad = min(n - 1, 2 * period)
    left = 2 * x[0] - x[pad:0:-1]
    right = 2 * x[-1] - x[-2:-pad - 2:-1]
    xp = np.concatenate([left, x, right])
    f = np.fft.rfftfreq(len(xp))
    f0, f1 = 1.0 / period, (1.0 + width) / period
    gain = np.clip((f1 - f) / (f1 - f0), 0.0, 1.0)
    gain = 0.5 - 0.5 * np.cos(np.pi * gain)
    yp = np.fft.irfft(np.fft.rfft(xp) * gain, n=len(xp))
    return yp[pad:pad + n]

def hanning(x, window=241):
    """Hanning (Hann, raised-cosine) weighted moving average over `window`
    points (should be odd so that it is centred). The weights taper smoothly to
    zero at the edges, which gives much smaller sidelobes than the boxcar moving
    average. A Hann window has roughly twice the width of a boxcar with the same
    main-lobe cutoff, so the default of 241 months gives decadal-ish smoothing.
    Near the ends the window is truncated and the remaining weights are
    renormalised to sum to one, so the output is the same length as x with no
    NaNs (but correspondingly noisier and more one-sided at the ends)."""
    x = np.asarray(x, dtype=float)
    w = np.hanning(window + 2)[1:-1]  # drop the zero endpoints
    num = np.convolve(x, w, mode="same")
    den = np.convolve(np.ones_like(x), w, mode="same")
    return num / den

run = sst_ts[0, :]
smoothed = moving_average(run, 120)
sg = savgol(run, 121, 3)
bw = butterworth(run, 120, 4)
ff = fft_lowpass(run, 120)
hn = hanning(run, 241)

plt.plot(run, color="lightgrey", linewidth=0.8, label="raw (monthly)")
plt.plot(smoothed, color="C0", linewidth=2, label="decadal moving average")
plt.plot(sg, color="C1", linewidth=2, label="Savitzky-Golay (121, cubic)")
plt.plot(bw, color="C2", linewidth=2, label="Butterworth low-pass (decadal, order 4)")
plt.plot(ff, color="C3", linewidth=2, label="FFT low-pass (decadal, cosine taper)")
plt.plot(hn, color="C4", linewidth=2, label="Hanning (241)")
plt.xlabel("month")
plt.ylabel("SST")
plt.legend()
plt.savefig("run1.pdf")







# eof

