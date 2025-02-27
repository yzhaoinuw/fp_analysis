# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 21:39:24 2024

@author: yzhao
"""

import numpy as np

import plotly.io as io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.io import loadmat
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming
from scipy.ndimage import gaussian_filter, gaussian_filter1d


def plot_spectrogram(eeg, eeg_frequency, window_duration=5, start_time=0, mfft=None):
    nperseg = round(eeg_frequency * window_duration)
    noverlap = round(nperseg / 2)
    window = hamming(nperseg)
    SFT = ShortTimeFFT(
        window,
        hop=noverlap,
        fs=eeg_frequency,
        fft_mode="onesided",
        mfft=mfft,  # potentially can be set to power of 2 for speed up
        scale_to="psd",
    )
    Sx = SFT.spectrogram(eeg)
    time = SFT.t(len(eeg)) + start_time
    frequencies = SFT.f

    freq_mask = frequencies <= 30
    frequencies = frequencies[freq_mask]
    Sx = Sx[freq_mask, :]
    Sx_db = 10 * np.log10(Sx)

    delta_mask = np.where((frequencies > 1) & (frequencies <= 4))[0]
    theta_mask = np.where((frequencies > 4) & (frequencies <= 8))[0]
    delta_power = np.mean(Sx_db[delta_mask, :], axis=0)
    theta_power = np.mean(Sx_db[theta_mask, :], axis=0)
    # theta_delta_ratio = theta_power / delta_power
    theta_delta_ratio = (
        delta_power / theta_power
    )  # flip delta and theta because their "magnitude" is negative
    theta_delta_ratio = gaussian_filter1d(theta_delta_ratio, 4)

    Sx_db = gaussian_filter(Sx_db, sigma=4)

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[
            [{"secondary_y": True, "r": -0.05}]
        ],  # Allow dual y-axes and reduce the padding on the right side
    )

    spectrogram_matrix = go.Heatmap(
        x=time,
        y=frequencies,
        z=Sx_db,
        colorscale="viridis",
        showscale=True,
        colorbar=dict(
            title="Power (dB)",
            orientation="h",
            thicknessmode="fraction",  # set the mode of thickness to fraction
            thickness=0.05,  # the thickness of the colorbar
            lenmode="fraction",  # set the mode of length to fraction
            len=0.15,  # the length of the colorbar
            y=0.95,  # the y position of the colorbar
            xanchor="right",  # anchor the colorbar at the left
            x=0.8,  # the x position of the colorbar
            tickfont=dict(size=8),
        ),
    )

    fig.add_trace(spectrogram_matrix, secondary_y=False, row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=time,
            y=theta_delta_ratio,
            mode="lines",
            name="Theta/Delta",
            line=dict(color="white", width=1),
            opacity=0.4,
        ),
        secondary_y=True,  # This is for the right y-axis
        row=1,
        col=1,
    )

    fig.update_layout(
        # autosize=True,
        margin=dict(t=20, l=10, r=0, b=0),
        height=200,
        title=dict(
            text="Spectrogram and Theta/Delta Ratio",
            font=dict(size=16),
            y=0.98,
            x=0.5,
            yref="container",
            xanchor="center",
            yanchor="top",
        ),
        xaxis=dict(title="Time (s)", tickformat="digits", title_standoff=5),
        yaxis=dict(title="Frequency (Hz)", title_standoff=5),  # Left y-axis
        yaxis2=dict(
            title="Theta/Delta",  # Right y-axis
            overlaying="y",
            side="right",
            title_standoff=5,
        ),
        font=dict(size=10),
        modebar_remove=["lasso2d", "zoom", "autoScale"],
    )
    fig.update_xaxes(range=[time[0], time[-1]], row=1, col=1)

    return fig


# %%
if __name__ == "__main__":
    io.renderers.default = "browser"

    mat = loadmat("C:/Users/yzhao/python_projects/time_series/data/arch_392.mat")
    eeg = mat["eeg"].flatten()
    eeg_frequency = mat["eeg_frequency"].item()
    start_time = mat.get("start_time")
    if start_time is None:
        start_time = 0
    else:
        start_time = start_time.item()
    fig = plot_spectrogram(eeg, eeg_frequency, start_time=start_time)
    fig.show(config={"scrollZoom": True})
