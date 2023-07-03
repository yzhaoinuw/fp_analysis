# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:36:14 2023

@author: Yue
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler


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
    predictions = pred["pred_labels"].flatten()
    confidence = pred["scores"].flatten()

    fig = FigureResampler(
        make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=("EEG", "EMG", "NE", "Predicted Sleep Scores"),
        )
    )

    # Define custom colorscale
    stage_colors = [
        "rgb(255, 51, 153)",
        "rgb(51, 255, 51)",
        "rgb(153, 51, 255)",
    ]  # colors for the legend
    colorscale = [[0, stage_colors[0]], [0.5, stage_colors[1]], [1, stage_colors[2]]]
    # Create a heatmap for stages
    hovertext = [
        f"time: {time[i]}\nconfidence: {confidence[i]:.2f}"
        for i in range(len(confidence))
    ]
    heatmap = go.Heatmap(
        x=time,
        z=[predictions],
        text=[hovertext],
        hoverinfo="text",
        colorscale=colorscale,
        colorbar=dict(y=0.8, len=0.5),
        showscale=False,
        opacity=0.6,
    )

    # Add the time series to the figure
    fig.add_trace(
        go.Scattergl(
            line=dict(width=1),
            marker=dict(size=3),
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
            marker=dict(size=3),
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
            marker=dict(size=3),
            showlegend=False,
            mode="lines+markers",
            hoverinfo="x+y",
        ),
        hf_x=time_x3,
        hf_y=y_x3,
        row=3,
        col=1,
    )
    fig.add_trace(heatmap, row=4, col=1)

    stage_names = ["Wake", "SWS", "REM"]  # Adjust this to match your stages
    for i, color in enumerate(stage_colors):
        fig.add_trace(
            go.Scatter(
                x=[-100],
                y=[0.4],
                mode="markers",
                marker=dict(size=10, color=color, symbol="square"),
                name=stage_names[i],
                showlegend=True,
            ),
            row=4,
            col=1,
        )

    # Update layout to include yaxis2 with its own range
    fig.update_layout(
        autosize=True,
        margin=dict(t=50, l=20, r=20, b=20),
        height=800,
        hovermode="x unified",  # gives crosshair in one subplot
        # title_text="EMG, EEG and NE with Predicted Sleep Scores",
        yaxis4=dict(tickvals=[]),  # suppress y ticks on the heatmap
        xaxis4_title="Time (s)",
        legend=dict(
            x=0.6,  # adjust these values to position the legend
            y=0.26,  # adjust these values to position the legend
            orientation="h",  # makes legend items horizontal
            bgcolor="rgba(0,0,0,0)",  # transparent legend background
            font=dict(size=10),  # adjust legend text size
        ),
    )

    fig.update_traces(xaxis="x4")  # gives crosshair across all subplots
    fig.update_xaxes(range=[start_time, end_time], row=1, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=2, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=3, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=4, col=1)
    return fig
