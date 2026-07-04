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
from html import escape

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB

from fp_analysis_app.mat_utils import get_visualization_signal_data


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
EVENT_TIMESTAMP_TRACE_PREFIX = "Event timestamp: "
EVENT_TIMESTAMP_LINE_OPACITY = 0.55
EVENT_TIMESTAMP_LINE_WIDTH = 1.25
EVENT_TIMESTAMP_LEGEND_X = 0.6
EVENT_TIMESTAMP_LEGEND_MIN_X = 0.03
EVENT_TIMESTAMP_LEGEND_Y = 1.035
EVENT_TIMESTAMP_LEGEND_CHAR_WIDTH = 0.006
EVENT_TIMESTAMP_LEGEND_ITEM_GAP = 0.025
EVENT_TIMESTAMP_LEGEND_ITEM_INDENT = -0.05
EVENT_TIMESTAMP_LEGEND_MIN_ITEM_WIDTH = 0.045
EVENT_TIMESTAMP_LEGEND_LINE_HEIGHT = 0.035
EVENT_TIMESTAMP_LEGEND_MAX_X = 0.98


def get_padded_labels(labels: np.ndarray, duration: int) -> np.ndarray:
    """Make label array the same size as the duration."""

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


def _build_event_timestamp_trace(event_name, event_times, y_range, color):
    event_times = np.asarray(event_times, dtype=float).ravel()
    if event_times.size == 0:
        return None

    x_values = []
    y_values = []
    for event_time in event_times:
        x_values.extend([event_time, event_time, None])
        y_values.extend([y_range[0], y_range[1], None])

    return go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        name=f"{EVENT_TIMESTAMP_TRACE_PREFIX}{event_name}",
        line=dict(color=color, width=EVENT_TIMESTAMP_LINE_WIDTH),
        opacity=EVENT_TIMESTAMP_LINE_OPACITY,
        hovertemplate=(
            f"<b>{event_name}</b><br>"
            "event time: %{x:.2f} s<extra></extra>"
        ),
        showlegend=False,
    )


def add_event_timestamp_traces(fig, event_time_dict, signal_ranges, num_signals):
    if not event_time_dict:
        return []

    legend_items = []
    for event_index, event_name in enumerate(sorted(event_time_dict.keys())):
        color = LABEL_COLORS[event_index % len(LABEL_COLORS)]
        legend_items.append((event_name, color))
        for signal_index in range(num_signals):
            signal_range = signal_ranges[signal_index] * (1 + RANGE_PADDING_PERCENT)
            if signal_range == 0:
                signal_range = 1
            trace = _build_event_timestamp_trace(
                event_name=event_name,
                event_times=event_time_dict[event_name],
                y_range=(-signal_range, signal_range),
                color=color,
            )
            if trace is None:
                continue
            fig.add_trace(trace, row=signal_index + 1, col=1)
    return legend_items


def _get_event_timestamp_legend_item_width(event_name):
    return max(
        EVENT_TIMESTAMP_LEGEND_MIN_ITEM_WIDTH,
        (
            (len(str(event_name)) + 2) * EVENT_TIMESTAMP_LEGEND_CHAR_WIDTH
            + EVENT_TIMESTAMP_LEGEND_ITEM_GAP
            + EVENT_TIMESTAMP_LEGEND_ITEM_INDENT
        ),
    )


def _get_event_timestamp_legend_start_x(legend_items):
    total_width = sum(
        _get_event_timestamp_legend_item_width(event_name)
        for event_name, _ in legend_items
    )
    return max(
        EVENT_TIMESTAMP_LEGEND_MIN_X,
        min(EVENT_TIMESTAMP_LEGEND_X, EVENT_TIMESTAMP_LEGEND_MAX_X - total_width),
    )


def add_event_timestamp_legend_annotations(fig, legend_items):
    row_start_x = _get_event_timestamp_legend_start_x(legend_items)
    x = row_start_x
    y = EVENT_TIMESTAMP_LEGEND_Y
    for event_name, color in legend_items:
        item_width = _get_event_timestamp_legend_item_width(event_name)
        if (
            x > row_start_x
            and x + item_width > EVENT_TIMESTAMP_LEGEND_MAX_X
        ):
            x = row_start_x
            y -= EVENT_TIMESTAMP_LEGEND_LINE_HEIGHT

        fig.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=(
                f'<span style="color:{color}">&#9632;</span>'
                f'&nbsp;{escape(str(event_name))}'
            ),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=10, color="black"),
        )
        x += item_width


def make_figure(
    mat,
    plot_name="",
    label_dict=None,
    event_time_dict=None,
    show_period_labels=True,
    default_n_shown_samples=2048,
):
    # Time span and frequencies
    fp_signal_names, fp_signals, fp_freq = get_visualization_signal_data(mat)
    num_signals = len(fp_signal_names)
    subplot_titles = fp_signal_names + [""] * (4 - num_signals)
    signal_lengths = [len(fp_signals[k]) for k in range(num_signals)]
    assert all(length == signal_lengths[0] for length in signal_lengths)

    signal_length = signal_lengths[0]
    fp_signals = np.vstack(fp_signals)
    start_time = mat.get("start_time", 0)

    duration = math.ceil(
        (signal_length - 1) / fp_freq
    )  # need to round duration to an int for later

    if show_period_labels and label_dict:
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
            row=1,
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

    event_timestamp_legend_items = add_event_timestamp_traces(
        fig,
        event_time_dict,
        signal_ranges,
        num_signals,
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
    add_event_timestamp_legend_annotations(fig, event_timestamp_legend_items)

    return fig


if __name__ == "__main__":
    import os
    import plotly.io as io
    from scipy.io import loadmat

    from event_analysis import Event_Utils

    io.renderers.default = "browser"
    DATA_PATH = "../data/"
    fp_name = "M67_RS-new"
    fp_file = os.path.join(DATA_PATH, f"{fp_name}.mat")
    fp_data = loadmat(fp_file, squeeze_me=True)
    # biosignal_names = fp_data["fp_signal_names"]
    fp_signal_names, fp_signals, fp_freq = get_visualization_signal_data(fp_data)
    biosignal = fp_signals[0]

    event_file = os.path.join(DATA_PATH, "M67_RS_Transitions_from_SleepBouts.xlsx")
    fp_freq = fp_data["fp_frequency"]
    duration = int(np.ceil(len(biosignal) / fp_freq))

    event_utils = Event_Utils(fp_freq, duration)
    perievent_label_dict = event_utils.make_perievent_labels(event_file)
    fig = make_figure(fp_data, plot_name=fp_name, label_dict=perievent_label_dict)
    fig.show_dash(config={"scrollZoom": True}, mode="external")
