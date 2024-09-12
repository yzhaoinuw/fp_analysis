# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 20:54:41 2024

@author: yzhao
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fft import rfft, rfftfreq


def plot_fft(eeg_seg=[], eeg_frequency=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=(6.0, 4.0))
    if len(eeg_seg) == 0:
        return fig
    n = len(eeg_seg)
    T = 1 / eeg_frequency
    f = rfftfreq(n, T)  # frequency bins range from 0 Hz to Nyquist freq.
    XX = rfft(eeg_seg)  # one-sided magnitude spectrum
    ax.set(
        title="FFT",
        xlabel="Frequency (Hz)",
        ylabel=r"Magnitude $|X(f)|/\tau$",
        xlim=(0, 20),
    )
    ax.plot(
        f,
        abs(XX),
        "-",
        linewidth=1,
    )
    xticks = np.arange(0, 21, 1)
    ax.set_xticks(xticks)
    ax.grid(True)
    return fig


if __name__ == "__main__":
    mat = loadmat("C:/Users/yzhao/python_projects/time_series/data/arch_392.mat")
    eeg = mat["eeg"].flatten()
    eeg_frequency = mat["eeg_frequency"].item()

    start = 30000
    duration = 100
    eeg_seg = eeg[start : round(start + duration * eeg_frequency)]
    fig = plot_fft(eeg_seg, eeg_frequency)
