# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:36:14 2023

@author: Yue
"""

import numpy as np
from scipy import signal

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB


# set up color config
sleep_score_opacity = 0.5
stage_colors = [
    "rgb(102, 178, 255)",  # Wake,
    "rgb(255, 102, 255)",  # SWS,
    "rgb(102, 255, 102)",  # REM,
    "rgb(255, 255, 0)",  # MA yellow
]
stage_names = ["Wake: 1", "SWS: 2", "REM: 3", "MA: 4"]
colorscale = {
    3: [[0, stage_colors[0]], [0.5, stage_colors[1]], [1, stage_colors[2]]],
    4: [
        [0, stage_colors[0]],
        [1 / 3, stage_colors[1]],
        [2 / 3, stage_colors[2]],
        [1, stage_colors[3]],
    ],
}
range_quantile = 0.9999
range_padding_percent = 0.2


def make_figure(pred, default_n_shown_samples=4000, ne_fs=10):
    # Time span and frequencies
    start_time, end_time = 0, pred["trial_eeg"].shape[0]

    eeg, emg, ne = pred.get("trial_eeg"), pred.get("trial_emg"), pred.get("trial_ne")
    freq_x1, freq_x2 = (eeg.shape[1] * eeg.shape[0], emg.shape[1] * emg.shape[0])

    # Create the time sequences
    time_x1 = np.linspace(start_time, end_time, freq_x1)
    time_x2 = np.linspace(start_time, end_time, freq_x2)
    time = np.arange(start_time, end_time)
    y_x1 = eeg.flatten()
    y_x2 = emg.flatten()
    eeg_lower_range, eeg_upper_range = np.quantile(
        y_x1, 1 - range_quantile
    ), np.quantile(y_x1, range_quantile)
    emg_lower_range, emg_upper_range = np.quantile(
        y_x2, 1 - range_quantile
    ), np.quantile(y_x2, range_quantile)
    eeg_range = max(abs(eeg_lower_range), abs(eeg_upper_range))
    emg_range = max(abs(emg_lower_range), abs(emg_upper_range))

    predictions = pred["pred_labels"].flatten()
    confidence = pred["confidence"].flatten()
    num_class = pred["num_class"].item()
    end_time_clipped = len(confidence)

    time = time[
        :end_time_clipped
    ]  # clip time if last last couple of seconds don't preds

    # align heatmap and xticks and hoverinfo over x axis
    time = time + 0.5
    hovertext = []
    hovertext.extend(
        [
            # f"time: {round(time[i]+0.5)}\nconfidence: {confidence[i]:.2f}"
            f"time: {round(time[i]+0.5)}"
            for i in range(len(confidence))
        ]
    )

    fig = FigureResampler(
        make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "EEG",
                "EMG",
                "NE",
                "Prediction Confidence",
            ),
            row_heights=[0.3, 0.3, 0.3, 0.1],
        ),
        default_n_shown_samples=default_n_shown_samples,
        default_downsampler=MinMaxLTTB(parallel=True),
    )

    ne_lower_range, ne_upper_range = 0, 0
    if ne.size > 1:
        # downsample ne because the user doesn't need its high frequency
        ne_resampled = signal.resample(ne, ne_fs, axis=1)
        freq_x3 = ne_resampled.shape[1] * ne_resampled.shape[0]
        time_x3 = np.linspace(start_time, end_time, freq_x3)
        y_x3 = ne_resampled.flatten()
        ne_lower_range, ne_upper_range = np.quantile(
            y_x3, 1 - range_quantile
        ), np.quantile(y_x3, range_quantile)
        fig.add_trace(
            go.Scattergl(
                line=dict(width=1),
                marker=dict(size=2, color="black"),
                showlegend=False,
                mode="lines+markers",
                hoverinfo="y",
            ),
            hf_x=time_x3,
            hf_y=y_x3,
            row=3,
            col=1,
        )

    ne_range = max(abs(ne_lower_range), abs(ne_upper_range))
    heatmap_width = max(
        20, 2 * (1 + range_padding_percent) * max([eeg_range, emg_range, ne_range])
    )

    # Create a heatmap for stages
    sleep_scores = go.Heatmap(
        x=time,
        y0=0,
        dy=heatmap_width,  # assuming that the max abs value of eeg, emg, or ne is no more than 10
        z=[predictions],
        text=[hovertext],
        hoverinfo="text",
        colorscale=colorscale[num_class],
        showscale=False,
        opacity=sleep_score_opacity,
        zmax=num_class - 1,
        zmin=0,
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
            y=0.08,  # the y position of the colorbar
            xanchor="right",  # anchor the colorbar at the left
            x=0.75,  # the x position of the colorbar
            tickfont=dict(size=8),
        ),
        showscale=True,
        xgap=0.05,  # add small gaps to serve as boundaries / ticks
    )

    # Add the time series to the figure
    fig.add_trace(
        go.Scattergl(
            line=dict(width=1),
            marker=dict(size=2, color="black"),
            showlegend=False,
            mode="lines+markers",
            hoverinfo="y",
        ),
        hf_x=time_x1,
        hf_y=y_x1,
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            line=dict(width=1),
            marker=dict(size=2, color="black"),
            showlegend=False,
            mode="lines+markers",
            hoverinfo="y",
        ),
        hf_x=time_x2,
        hf_y=y_x2,
        row=2,
        col=1,
    )

    for i, color in enumerate(stage_colors[:num_class]):
        fig.add_trace(
            go.Scatter(
                x=[-100],
                y=[0.2],
                mode="markers",
                marker=dict(
                    size=8, color=color, symbol="square", opacity=sleep_score_opacity
                ),
                name=stage_names[i],
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # add the heatmap last so that their indices can be accessed using last indices
    fig.add_trace(sleep_scores, row=1, col=1)
    fig.add_trace(sleep_scores, row=2, col=1)
    fig.add_trace(sleep_scores, row=3, col=1)
    fig.add_trace(conf, row=4, col=1)

    fig.update_layout(
        autosize=True,
        margin=dict(t=50, l=20, r=20, b=40),
        height=800,
        hovermode="x unified",  # gives crosshair in one subplot
        hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.6)"),
        title_text="Predicted Sleep Scores",
        yaxis4=dict(tickvals=[]),  # suppress y ticks on the heatmap
        xaxis4=dict(tickformat="digits"),
        legend=dict(
            x=0.6,  # adjust these values to position the sleep score legend stage_names
            y=1.05,
            orientation="h",  # makes legend items horizontal
            bgcolor="rgba(0,0,0,0)",  # transparent legend background
            font=dict(size=10),  # adjust legend text size
        ),
        font=dict(
            size=12,  # title font size
        ),
        modebar_remove=["lasso2d", "zoom", "autoScale"],
        dragmode="pan",
        clickmode="event",
    )

    fig.update_traces(xaxis="x4")  # gives crosshair across all subplots
    fig.update_traces(colorbar_orientation="h", selector=dict(type="heatmap"))
    fig.update_xaxes(range=[start_time, end_time], row=1, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=2, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=3, col=1)
    fig.update_xaxes(
        range=[start_time, end_time],
        row=4,
        col=1,
        title_text="<b>Time (s)</b>",
    )
    fig.update_yaxes(
        range=[
            eeg_range * -(1 + range_padding_percent),
            eeg_range * (1 + range_padding_percent),
        ],
        # fixedrange=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[
            emg_range * -(1 + range_padding_percent),
            emg_range * (1 + range_padding_percent),
        ],
        # fixedrange=True,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        range=[
            ne_range * -(1 + range_padding_percent),
            ne_range * (1 + range_padding_percent),
        ],
        # fixedrange=True,
        row=3,
        col=1,
    )
    fig.update_yaxes(range=[0, 0.5], fixedrange=True, row=4, col=1)
    fig.update_annotations(font_size=14)  # subplot title size
    fig["layout"]["annotations"][-1]["font"]["size"] = 14

    return fig


if __name__ == "__main__":
    import plotly.io as io
    from scipy.io import loadmat

    io.renderers.default = "browser"
    path = ".\\"
    # mat_file = "115_35_data_prediction_msda_3class.mat"
    # mat_file = "Klaudia_datatest_prediction_msda_3class.mat"
    # mat_file = "data_prediction_msda_3class.mat"
    mat_file = "data_no_ne_prediction_msda_3class.mat"
    pred = loadmat(path + mat_file)
    fig = make_figure(pred)
    fig.show_dash(config={"scrollZoom": True})
