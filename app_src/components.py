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
                            {"label": "Visualize a recording", "value": "vis"},
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
    config={"scrollZoom": True, "editable": False, "displayModeBar": True},
)

visualization_div = html.Div(
    children=[
        html.Div(
            style={"display": "flex", "marginBottom": "0px"},
            children=[
                html.Div(
                    ["Sampling Level"],
                    style={
                        "display": "inline-block",
                        "marginRight": "5px",
                        "marginLeft": "10px",
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
                        "marginLeft": "5px",
                        # "marginRight": "20px",
                        "display": "inline-block",
                    },
                ),
            ],
        ),
        html.Div(
            children=[fft_graph], style={"marginTop": "0px", "marginLeft": "48px"}
        ),
        html.Div(
            children=[graph],
            style={"marginTop": "1px", "marginLeft": "30px", "marginRight": "28px"},
        ),
        html.Div(
            style={"display": "flex"},
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "marginRight": "5px",
                        "marginLeft": "10px",
                    },
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
                html.Div(id="annotation-message"),
                html.Div(id="debug-message"),
                dcc.Store(id="box-select-store"),
                dcc.Store(id="annotation-store"),
                dcc.Store(id="update-fft-store"),
                EventListener(
                    id="keyboard",
                    events=[{"event": "keydown", "props": ["key"]}],
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=1 * 1000,  # in milliseconds
                    max_intervals=0,  # stop after the first interval
                ),
            ],
        ),
    ],
)

upload_box_style = {
    "fontSize": "18px",
    "width": "15%",
    "height": "auto",
    "minHeight": "auto",
    "lineHeight": "auto",
    "borderWidth": "1px",
    "borderStyle": "none",
    "textAlign": "center",
    "margin": "5px",  # spacing between the upload box and the div it's in
    "borderRadius": "10px",  # rounded corner
    "backgroundColor": "lightgrey",
}


# %%
class Components:
    def __init__(self):
        self.home_div = home_div
        self.graph = graph
        self.fft_graph = fft_graph
        self.visualization_div = visualization_div
        self.upload_box_style = upload_box_style
