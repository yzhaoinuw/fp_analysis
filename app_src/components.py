# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:27:03 2023

@author: yzhao
"""

from dash import dcc, html
from dash_extensions import EventListener


home_div = html.Div(
    [
        html.Div(
            style={"display": "flex"},
            children=[
                html.Div(
                    dcc.RadioItems(
                        id="task-selection",
                        options=[
                            {"label": "Generate prediction", "value": "pred"},
                            {"label": "Visualize existing prediction", "value": "vis"},
                        ],
                        style={"marginRight": "50px"},
                    )
                ),
                html.Div(
                    id="model-choice-container",
                    children=[
                        dcc.RadioItems(
                            id="model-choice",
                            options=[
                                {"label": "sDREAMER", "value": "sdreamer"},
                            ],
                            value="sdreamer",
                        )
                    ],
                    style={"display": "none"},
                ),
                html.Div([" "], id="invisible-gap", style={"marginRight": "50px"}),
            ],
        ),
        html.Div(id="upload-container"),
        html.Div(id="data-upload-message"),
        dcc.Store(id="prediction-ready-store"),
        dcc.Store(id="visualization-ready-store"),
        dcc.Download(id="prediction-download-store"),
    ]
)

graph = dcc.Graph(
    id="graph",
    config={
        "scrollZoom": True,
        "editable": True,
        "edits": {
            "axisTitleText": False,
            "titleText": False,
            "colorbarTitleText": False,
            "annotationText": False,
        },
    },
)

fft_graph = dcc.Graph(
    id="fft-graph",
    # style={'width': '200', 'height': '200'},
    config={
        "scrollZoom": False,
        "editable": False,
        "staticPlot": True,
    },
)

visualization_div = html.Div(
    children=[
        html.Div(
            style={"display": "flex"},
            children=[
                html.Div(
                    ["Sampling Level"],
                    style={
                        "display": "inline-block",
                        "marginRight": "10px",
                        "lineHeight": "40px",
                    },
                ),
                html.Div(
                    children=[
                        dcc.Dropdown(
                            ["x1", "x2", "x4"],
                            "x1",
                            id="n-sample-dropdown",
                        )
                    ],
                    style={
                        "width": "50px",
                        "height": "auto",
                        "textAlign": "left",
                        "margin": "2px",
                        "marginRight": "20px",
                        "display": "inline-block",
                    },
                ),
                dcc.Store(id="update-fft-store"),
                html.Div(
                    id="fft-div",
                    children=[fft_graph],
                    style={"display": "none"},
                ),
            ],
        ),
        graph,
        html.Div(
            style={"display": "flex"},
            children=[
                dcc.Store(id="box-select-store"),
                dcc.Store(id="annotation-store"),
                EventListener(
                    id="keyboard",
                    events=[{"event": "keydown", "props": ["key"]}],
                ),
                html.Div(
                    style={"display": "flex", "margin-right": "5px"},
                    children=[
                        html.Button("Save Annotations", id="save-button"),
                        dcc.Download(id="download-annotations"),
                        dcc.Download(id="download-spreadsheet"),
                        html.Button(
                            "Undo Annotation",
                            id="undo-button",
                            style={"display": "none"},
                        ),
                    ],
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=1 * 1000,  # in milliseconds
                    max_intervals=0,  # stop after the first interval
                ),
                html.Div(id="annotation-message"),
                html.Div(id="debug-message"),
            ],
        ),
    ],
)


# %%
class Components:
    def __init__(self):
        self.home_div = home_div
        self.graph = graph
        self.visualization_div = visualization_div
