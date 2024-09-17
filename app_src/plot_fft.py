# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:18:15 2024

@author: yzhao
"""

import numpy as np
import plotly.io as io
from scipy.io import loadmat
import plotly.graph_objects as go
from scipy.fft import rfft, rfftfreq


def plot_fft(eeg_seg=[], eeg_frequency=None):
    fig = go.Figure()
    fig.update_layout(
        height=150,
        width=300,
        title=dict(
            text="Spectral Estimation of EEG",
            font=dict(size=10),
            xanchor="center",
            x=0.5,
            yanchor="top",
            yref="container",
            automargin=True,
        ),
        xaxis=dict(
            title_text="Frequency (Hz)",
            titlefont=dict(size=8),
            range=[0, 20],
            dtick=1,
            tickfont=dict(size=6),
        ),
        yaxis=dict(
            title_text=r"$|X(f)|/\tau$",
            titlefont=dict(size=8),
            rangemode="tozero",
            tickfont=dict(size=6),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    if len(eeg_seg) == 0:
        return fig
    n = len(eeg_seg)
    T = 1 / eeg_frequency
    f = rfftfreq(n, T)  # frequency bins range from 0 Hz to Nyquist freq.
    XX = rfft(eeg_seg)  # one-sided magnitude spectrum
    fig.add_trace(
        go.Scatter(
            x=f,
            y=abs(XX),
            line=dict(width=1),
        )
    )

    return fig


if __name__ == "__main__":
    io.renderers.default = "browser"

    mat = loadmat("C:/Users/yzhao/python_projects/time_series/data/arch_392.mat")
    eeg = mat["eeg"].flatten()
    eeg_frequency = mat["eeg_frequency"].item()
    start = 30000
    duration = 100
    eeg_seg = eeg[start : round(start + duration * eeg_frequency)]
    fig = plot_fft(eeg_seg, eeg_frequency)
    fig.show(config=dict({"staticPlot": True}))
