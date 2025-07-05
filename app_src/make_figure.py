# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:36:14 2023

@author: Yue

Notes
1. A common reason that sleep scores, which are a heatmap,
   don't show up is that they have shape of (N,), instead of (1, N). The heatmap
   only works with 2d arrays.
"""

import math
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB


# set up color config
SLEEP_SCORE_OPACITY = 1
STAGE_COLORS = [
    "rgb(124, 124, 251)",  # Wake,
    "rgb(251, 124, 124)",  # NREM,
    "rgb(123, 251, 123)",  # REM,
    "rgb(255, 255, 0)",  # MA yellow
]
STAGE_NAMES = ["Wake: 1", "NREM: 2", "REM: 3", "MA: 4"]
COLORSCALE = {
    3: [[0, STAGE_COLORS[0]], [0.5, STAGE_COLORS[1]], [1, STAGE_COLORS[2]]],
    4: [
        [0, STAGE_COLORS[0]],
        [1 / 3, STAGE_COLORS[1]],
        [2 / 3, STAGE_COLORS[2]],
        [1, STAGE_COLORS[3]],
    ],
}
RANGE_QUANTILE = 0.99
HEATMAP_WIDTH = 40
RANGE_PADDING_PERCENT = 0.2


def get_padded_period_labels(period_labels: np.ndarray, duration: int) -> np.ndarray:
    """Make a period laebl array the same size as the duration."""

    if period_labels.size == 0:
        # if unscored, initialize with nan
        period_labels = np.zeros(duration)
        period_labels[:] = np.nan
    else:
        # manually scored, but may contain missing scores
        period_labels = period_labels.astype(float)

        # period_labels need to have the length of duration. pad if necessary
        pad_len = duration - period_labels.size
        if pad_len > 0:
            period_labels = np.pad(
                period_labels, (0, pad_len), "constant", constant_values=np.nan
            )
    return period_labels


def make_figure(
    mat,
    plot_name="",
    period_labels=np.array([]),
    default_n_shown_samples=2048,
    num_class=3,
):
    # Time span and frequencies
    fp_signal_names = mat["fp_signal_names"]
    num_signals = len(fp_signal_names)
    subplot_titles = fp_signal_names + [""] * (4 - num_signals)
    fp_signals = [mat[signal_name] for signal_name in fp_signal_names]
    signal_lengths = [len(fp_signals[k]) for k in range(num_signals)]
    assert all(length == signal_lengths[0] for length in signal_lengths)

    signal_length = signal_lengths[0]
    fp_signals = np.vstack(fp_signals)
    fp_freq = mat.get("fp_frequency")
    start_time = mat.get("start_time", 0)
    if mat.get("num_class") is not None:
        num_class = mat["num_class"]

    duration = math.ceil(
        (signal_length - 1) / fp_freq
    )  # need to round duration to an int for later

    # scored fully or partially or unscored
    period_labels = get_padded_period_labels(period_labels, duration)
    np.place(
        period_labels, period_labels == -1, [np.nan]
    )  # convert -1 to None for heatmap visualization

    # convert flat array to 2D array for visualization to work
    if len(period_labels.shape) == 1:
        period_labels = np.expand_dims(period_labels, axis=0)

    signal_end_time = duration + start_time

    # Create the time sequences
    time = np.linspace(start_time, signal_end_time, signal_length)
    end_time = math.ceil(time[-1])
    signal_ranges = [
        max(
            abs(np.nanquantile(fp_signals[k], RANGE_QUANTILE)),
            abs(np.nanquantile(fp_signals[k], 1 - RANGE_QUANTILE)),
        )
        for k in range(num_signals)
    ]

    fig = FigureResampler(
        make_subplots(
            rows=num_signals,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            row_heights=[1 / num_signals] * num_signals,
        ),
        default_n_shown_samples=default_n_shown_samples,
        default_downsampler=MinMaxLTTB(parallel=True),
    )

    # Create a heatmap for stages
    period_labels = go.Heatmap(
        x0=start_time + 0.5,
        dx=1,
        y0=0,
        dy=HEATMAP_WIDTH,  # assuming that the max abs value of eeg, emg, or ne is no more than 10
        z=period_labels,
        name="Period Labels",
        hoverinfo="none",
        colorscale=COLORSCALE[num_class],
        showscale=False,
        opacity=SLEEP_SCORE_OPACITY,
        zmax=num_class - 1,
        zmin=0,
        showlegend=False,
        xgap=0.05,  # add small gaps to serve as boundaries / ticks
    )

    for k in range(num_signals):
        # Add the time series to the figure
        fig.add_trace(
            go.Scattergl(
                name=fp_signal_names[k],
                line=dict(width=1),
                marker=dict(size=2, color="black"),
                showlegend=False,
                mode="lines+markers",
                hovertemplate="<b>time</b>: %{x:.2f}"
                + "<br><b>y</b>: %{y}<extra></extra>",
            ),
            hf_x=time,
            hf_y=fp_signals[k],
            row=k + 1,
            col=1,
        )

    for i, color in enumerate(STAGE_COLORS[:num_class]):
        fig.add_trace(
            go.Scatter(
                x=[-100],
                y=[0.2],
                name=STAGE_NAMES[i],
                mode="markers",
                marker=dict(
                    size=8, color=color, symbol="square", opacity=SLEEP_SCORE_OPACITY
                ),
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    # add the heatmap last so that their indices can be accessed using last indices
    for k in range(num_signals):
        fig.add_trace(period_labels, row=k + 1, col=1)
        fig.update_xaxes(
            range=[start_time, end_time], tickformat="digits", row=k + 1, col=1
        )
        fig.update_yaxes(
            range=[
                signal_ranges[k] * -(1 + RANGE_PADDING_PERCENT),
                signal_ranges[k] * (1 + RANGE_PADDING_PERCENT),
            ],
            fixedrange=(k + 1 == num_signals),  # fix y range on the last subplot
            row=k + 1,
            col=1,
        )

    fig.update_layout(
        autosize=True,
        margin=dict(t=50, l=10, r=5, b=20),
        height=800,
        hovermode="x unified",  # gives crosshair in one subplot
        hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.6)"),
        title=dict(
            text=plot_name,
            font=dict(size=16),
            xanchor="left",
            x=0.03,
            # yanchor="bottom",
            automargin=True,
            yref="paper",
        ),
        # yaxis4=dict(tickvals=[]),  # suppress y ticks on the heatmap
        # xaxis4=dict(tickformat="digits"),
        legend=dict(
            x=0.6,  # adjust these values to position the sleep score legend STAGE_NAMES
            y=1.03,
            yref="paper",
            orientation="h",  # makes legend items horizontal
            bgcolor="rgba(0,0,0,0)",  # transparent legend background
            font=dict(size=10),  # adjust legend text size
        ),
        modebar_remove=["lasso2d", "zoom", "autoScale"],
        dragmode="pan",
        clickmode="event",
    )

    fig.update_traces(xaxis=f"x{num_signals}")  # gives crosshair across all subplots
    fig.update_xaxes(
        range=[start_time, end_time],
        row=num_signals,
        col=1,
        title_text="<b>Time (s)</b>",
        title_standoff=10,
        ticklabelstandoff=5,  # keep some distance between tick label and the minor ticks
        minor=dict(
            tick0=0,
            dtick=3600,
            tickcolor="black",
            ticks="outside",
            ticklen=5,
            tickwidth=2,
        ),
    )

    fig.update_annotations(font_size=14)  # subplot title size
    fig["layout"]["annotations"][-1]["font"]["size"] = 14

    return fig


if __name__ == "__main__":
    import os
    import plotly.io as io
    from scipy.io import loadmat

    io.renderers.default = "browser"
    data_path = "../data/"
    mat_file = "F268.mat"
    mat = loadmat(os.path.join(data_path, mat_file), squeeze_me=True)
    mat_name = os.path.basename(mat_file)
    fig = make_figure(mat, plot_name=mat_name)
    fig.show_dash(config={"scrollZoom": True})
