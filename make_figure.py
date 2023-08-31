# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:36:14 2023

@author: Yue
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler

from config import annotation_config, annotation_color_map, sleep_score_opacity

# load custom colorscale
stage_colors = list(annotation_color_map.keys())


def make_figure(pred):
    # Time span and frequencies
    start_time, end_time = 0, pred["eeg"].shape[0]
    freq_x1, freq_x2, freq_x3 = (
        pred["eeg"].shape[1] * pred["eeg"].shape[0],
        pred["emg"].shape[1] * pred["emg"].shape[0],
        pred["ne"].shape[1] * pred["ne"].shape[0],  # example frequencies
    )

    # Create the time sequences
    time_x1 = np.linspace(start_time, end_time, freq_x1)
    time_x2 = np.linspace(start_time, end_time, freq_x2)
    time_x3 = np.linspace(start_time, end_time, freq_x3)
    time = np.arange(start_time, end_time)

    # Create some example y-values
    y_x1 = pred["eeg"].flatten()
    y_x2 = pred["emg"].flatten()
    y_x3 = pred["ne"].flatten()
    eeg_min, eeg_max = min(y_x1), max(y_x2)
    emg_min, emg_max = min(y_x2), max(y_x2)
    ne_min, ne_max = min(y_x3), max(y_x3)
    predictions = pred["pred_labels"].flatten()
    confidence = pred["scores"].flatten()
    user_annotation_dummy = np.zeros(end_time)

    fig = FigureResampler(
        make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "EEG",
                "EMG",
                "NE",
                "Prediction Confidence",
                "User Annotation",
            ),
            row_heights=[0.25, 0.25, 0.25, 0.1, 0.15],
        )
    )

    colorscale = [[0, stage_colors[0]], [0.5, stage_colors[1]], [1, stage_colors[2]]]

    # Create a heatmap for stages
    hovertext = [
        f"time: {time[i]}\nconfidence: {confidence[i]:.2f}"
        for i in range(len(confidence))
    ]
    sleep_scores = go.Heatmap(
        x=time,
        y0=0,
        dy=20,  # assuming that the max abs value of eeg, emg, or ne is no more than 10
        z=[predictions],
        text=[hovertext],
        hoverinfo="text",
        colorscale=colorscale,
        showscale=False,
        opacity=sleep_score_opacity,
    )

    conf = go.Heatmap(
        x=time,
        z=[confidence],
        text=[hovertext],
        hoverinfo="text",
        colorscale="speed",
        colorbar=dict(
            thicknessmode="fraction",  # set the mode of thickness to fraction
            thickness=0.005,  # the thickness of the colorbar
            lenmode="fraction",  # set the mode of length to fraction
            len=0.15,  # the length of the colorbar
            yanchor="bottom",  # anchor the colorbar at the top
            y=0.24,  # the y position of the colorbar
            xanchor="right",  # anchor the colorbar at the left
            x=0.75,  # the x position of the colorbar
            tickfont=dict(size=8),
        ),
        showscale=True,
    )

    user_annotation = go.Heatmap(
        x=time,
        z=[user_annotation_dummy],
        hoverinfo="x",
        colorscale="gray",
        showscale=False,
        opacity=0.1,
    )

    # Add the time series to the figure
    fig.add_trace(
        go.Scattergl(
            line=dict(width=1),
            marker=dict(size=2),
            showlegend=False,
            mode="lines+markers",
            hoverinfo="x+y",
        ),
        hf_x=time_x1,
        hf_y=y_x1,
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            line=dict(width=1),
            marker=dict(size=2),
            showlegend=False,
            mode="lines+markers",
            hoverinfo="x+y",
        ),
        hf_x=time_x2,
        hf_y=y_x2,
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            line=dict(width=1),
            marker=dict(size=2),
            showlegend=False,
            mode="lines+markers",
            hoverinfo="x+y",
        ),
        hf_x=time_x3,
        hf_y=y_x3,
        row=3,
        col=1,
    )
    fig.add_trace(sleep_scores, row=1, col=1)
    fig.add_trace(sleep_scores, row=2, col=1)
    fig.add_trace(sleep_scores, row=3, col=1)
    fig.add_trace(conf, row=4, col=1)
    fig.add_trace(user_annotation, row=5, col=1)

    stage_names = ["Wake", "SWS", "REM"]  # Adjust this to match your stages
    for i, color in enumerate(stage_colors):
        fig.add_trace(
            go.Scatter(
                x=[-100],
                y=[0.2],
                mode="markers",
                marker=dict(
                    size=5, color=color, symbol="square", opacity=sleep_score_opacity
                ),
                name=stage_names[i],
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        autosize=True,
        margin=dict(t=50, l=20, r=20, b=40),
        height=800,
        hovermode="x unified",  # gives crosshair in one subplot
        title_text="Predicted Sleep Scores on EEG, EMG, and NE.",
        yaxis4=dict(tickvals=[]),  # suppress y ticks on the heatmap
        yaxis5=dict(tickvals=[]),
        legend=dict(
            x=0.6,  # adjust these values to position the sleep score legend
            y=1.05,  # stage_names
            orientation="h",  # makes legend items horizontal
            bgcolor="rgba(0,0,0,0)",  # transparent legend background
            font=dict(size=10),  # adjust legend text size
        ),
        font=dict(
            size=12,  # title font size
        ),
        modebar_remove=["lasso2d"],
        modebar_add=["eraseshape"],
    )

    fig.update_traces(xaxis="x5")  # gives crosshair across all subplots
    fig.update_traces(colorbar_orientation="h", selector=dict(type="heatmap"))
    fig.update_xaxes(range=[start_time, end_time], row=1, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=2, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=3, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=4, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=5, col=1)
    fig.update_xaxes(
        range=[start_time, end_time], row=5, col=1, title_text="<b>Time (s)</b>"
    )

    fig.update_yaxes(
        range=[
            eeg_min - 0.1 * (eeg_max - eeg_min),
            eeg_max + 0.1 * (eeg_max - eeg_min),
        ],
        fixedrange=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[
            emg_min - 0.1 * (emg_max - emg_min),
            emg_max + 0.1 * (emg_max - emg_min),
        ],
        fixedrange=True,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        range=[ne_min - 0.1 * (ne_max - ne_min), ne_max + 0.1 * (ne_max - ne_min)],
        fixedrange=True,
        row=3,
        col=1,
    )

    fig.update_yaxes(fixedrange=True, row=4, col=1)
    fig.update_yaxes(range=[0, 0.5], fixedrange=True, row=5, col=1)
    fig.update_annotations(font_size=14)  # subplot title size

    # load annotations if exist and non-empty
    if pred.get("annotated", False):
        x0, x1, fillcolor = (
            pred["annotation_x0"].squeeze(axis=0),
            pred["annotation_x1"].squeeze(axis=0),
            pred["annotation_fillcolor"],
        )
        for i in range(len(x0)):
            fig.add_shape(
                x0=x0[i],
                x1=x1[i],
                fillcolor=fillcolor[i],
                **annotation_config,
            )

    fig["layout"]["annotations"][-1]["font"]["size"] = 14

    return fig


if __name__ == "__main__":
    import plotly.io as io
    from scipy.io import loadmat

    io.renderers.default = "browser"
    path = ".\\"
    pred = loadmat(path + "data_predictions.mat")
    fig = make_figure(pred)
    fig.show_dash()
