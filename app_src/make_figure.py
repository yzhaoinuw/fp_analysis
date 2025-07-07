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
PERIOD_LABEL_OPACITY = 1

LABEL_COLORS = [
    "rgb(31, 119, 180)",  # blue
    "rgb(255, 127, 14)",  # orange
    "rgb(44, 160, 44)",  # green
    "rgb(214, 39, 40)",  # red
    "rgb(148, 103, 189)",  # purple
    "rgb(140, 86, 75)",  # brown
    "rgb(227, 119, 194)",  # pink
    "rgb(127, 127, 127)",  # gray
    "rgb(188, 189, 34)",  # olive
    "rgb(23, 190, 207)",  # cyan
]


def get_colorscale(num_class):
    colorscale = [[i * 1 / (num_class - 1), LABEL_COLORS[i]] for i in range(num_class)]
    return colorscale


RANGE_QUANTILE = 0.99
HEATMAP_WIDTH = 40
RANGE_PADDING_PERCENT = 0.2


def get_padded_labels(labels: np.ndarray, duration: int) -> np.ndarray:
    """Make a period laebl array the same size as the duration."""

    if labels.size == 0:
        # if unscored, initialize with nan
        labels = np.zeros(duration)
        labels[:] = np.nan
    else:
        # manually scored, but may contain missing scores
        labels = labels.astype(float)

        # labels need to have the length of duration. pad if necessary
        pad_len = duration - labels.size
        if pad_len > 0:
            labels = np.pad(labels, (0, pad_len), "constant", constant_values=np.nan)
    return labels


def make_figure(
    mat,
    plot_name="",
    label_dict={},
    default_n_shown_samples=2048,
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

    duration = math.ceil(
        (signal_length - 1) / fp_freq
    )  # need to round duration to an int for later

    if label_dict:
        label_names = label_dict["label_names"]
        labels = label_dict["labels"]
    else:
        label_names = []
        labels = np.array([])

    # scored fully or partially or unscored
    labels = get_padded_labels(labels, duration)
    np.place(
        labels, labels == -1, [np.nan]
    )  # convert -1 to None for heatmap visualization

    num_class = max(2, len(label_names))
    # convert flat array to 2D array for visualization to work
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, axis=0)

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
    colorscale = get_colorscale(num_class)
    # Create a heatmap for stages
    labels = go.Heatmap(
        x0=start_time + 0.5,
        dx=1,
        y0=0,
        dy=HEATMAP_WIDTH,  # assuming that the max abs value of eeg, emg, or ne is no more than 10
        z=labels,
        name="Period Labels",
        hoverinfo="none",
        colorscale=colorscale,
        showscale=False,
        opacity=PERIOD_LABEL_OPACITY,
        zmax=max(num_class - 1, 0),
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

    for i, color in enumerate(LABEL_COLORS[: len(label_names)]):
        fig.add_trace(
            go.Scatter(
                x=[-100],
                y=[0.2],
                name=label_names[i],
                mode="markers",
                marker=dict(
                    size=8, color=color, symbol="square", opacity=PERIOD_LABEL_OPACITY
                ),
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    # add the heatmap last so that their indices can be accessed using last indices
    for k in range(num_signals):
        fig.add_trace(labels, row=k + 1, col=1)
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
            x=0.6,  # adjust these values to position the sleep score legend biosignal_names
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

    from event_analysis import make_perievent_labels

    io.renderers.default = "browser"
    DATA_PATH = "../data/"
    fp_name = "F268"
    fp_file = os.path.join(DATA_PATH, f"{fp_name}.mat")
    fp_data = loadmat(fp_file, squeeze_me=True)
    # biosignal_names = fp_data["fp_signal_names"]
    biosignal_name = "NE2m"
    biosignal = fp_data[biosignal_name]

    event_file = os.path.join(DATA_PATH, "Transitions_F268.xlsx")
    fp_freq = fp_data["fp_frequency"]
    nsec_before = 60
    nsec_after = 60
    duration = int(np.ceil(len(biosignal) / fp_freq))
    min_time = nsec_before
    max_time = duration - nsec_after
    perievent_label_dict = make_perievent_labels(
        event_file, duration, nsec_before=nsec_before, nsec_after=nsec_after
    )
    fig = make_figure(fp_data, plot_name=fp_name, label_dict=perievent_label_dict)
    fig.show_dash(config={"scrollZoom": True})
