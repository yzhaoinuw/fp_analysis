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
            rows=6,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                "EEG",
                "EMG",
                "NE",
                "Predicted Sleep Scores",
                "Prediction Confidence",
                "",
            ),
            row_heights=[0.25, 0.25, 0.25, 0.05, 0.05, 0.15],
        )
    )

    # Define custom colorscale
    stage_colors = [
        "rgb(102, 178, 255)",
        "rgb(255, 102, 255)",
        "rgb(102, 255, 102)",
    ]  # colors for the legend
    colorscale = [[0, stage_colors[0]], [0.5, stage_colors[1]], [1, stage_colors[2]]]

    # Create a heatmap for stages
    hovertext = [
        f"time: {time[i]}\nconfidence: {confidence[i]:.2f}"
        for i in range(len(confidence))
    ]
    sleep_scores = go.Heatmap(
        x=time,
        z=[predictions],
        text=[hovertext],
        hoverinfo="text",
        colorscale=colorscale,
        showscale=False,
        opacity=0.6,
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
            y=0.19,  # the y position of the colorbar
            xanchor="right",  # anchor the colorbar at the left
            x=0.75,  # the x position of the colorbar
            tickfont=dict(size=8),
        ),
        showscale=True,
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
    fig.add_trace(sleep_scores, row=4, col=1)
    fig.add_trace(conf, row=5, col=1)

    stage_names = ["Wake", "SWS", "REM"]  # Adjust this to match your stages
    for i, color in enumerate(stage_colors):
        fig.add_trace(
            go.Scatter(
                x=[-100],
                y=[0.2],
                mode="markers",
                marker=dict(size=6, color=color, symbol="square"),
                name=stage_names[i],
                showlegend=True,
            ),
            row=4,
            col=1,
        )

    fig.add_annotation(
        dict(
            x=0.5,
            y=0.12,  # you may need to adjust this
            showarrow=False,
            text="<b>Time (s)</b>",
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
        )
    )

    fig.update_layout(
        autosize=True,
        margin=dict(t=50, l=20, r=20, b=20),
        height=800,
        hovermode="x unified",  # gives crosshair in one subplot
        title_text="EEG, EMG, and NE with Predicted Sleep Scores",
        yaxis4=dict(tickvals=[]),  # suppress y ticks on the heatmap
        yaxis5=dict(tickvals=[]),
        # xaxis5_title="Time (s)",
        legend=dict(
            x=0.6,  # adjust these values to position the legend
            y=0.3,  # stage_names
            orientation="h",  # makes legend items horizontal
            bgcolor="rgba(0,0,0,0)",  # transparent legend background
            font=dict(size=9),  # adjust legend text size
        ),
        font=dict(
            size=10,  # title font size
        ),
    )

    fig.update_traces(xaxis="x5")  # gives crosshair across all subplots
    fig.update_traces(colorbar_orientation="h", selector=dict(type="heatmap"))
    fig.update_xaxes(range=[start_time, end_time], row=1, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=2, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=3, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=4, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=5, col=1, showticklabels=True)
    fig.update_annotations(font_size=12)
    fig["layout"]["annotations"][-1]["font"]["size"] = 14
    return fig


if __name__ == "__main__":
    import plotly.io as io
    from scipy.io import loadmat

    io.renderers.default = "browser"
    # register_plotly_resampler(mode="auto")

    path = "C:\\Users\\Yue\\python_projects\\sleep_scoring\\"
    pred = loadmat(path + "final_results.mat")
    fig = make_figure(pred)
    fig.show_dash()
