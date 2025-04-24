# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:27:03 2023

@author: yzhao
"""

import dash_uploader as du
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash_extensions import EventListener


home_div = html.Div(
    [
        html.Div(
            style={"display": "flex", "marginLeft": "10px", "marginTop": "10px"},
            children=[
                html.Div(
                    dcc.RadioItems(
                        id="task-selection",
                        options=[
                            {"label": "Generate prediction", "value": "pred"},
                            {"label": "Visualize a recording", "value": "vis"},
                        ],
                        style={"marginRight": "50px"},
                    ),
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
                # html.Div([" "], id="invisible-gap", style={"marginRight": "50px"}),
            ],
        ),
        html.Div(id="upload-container", style={"marginLeft": "10px"}),
        html.Div(id="data-upload-message", style={"marginLeft": "10px"}),
        dcc.Store(id="prediction-ready-store"),
        dcc.Store(id="visualization-ready-store"),
        dcc.Download(id="prediction-download-store"),
    ]
)


graph = dcc.Graph(
    id="graph",
    config={
        "scrollZoom": True,
    },
)

visualization_div = html.Div(
    children=[
        html.Div(
            style={
                "display": "flex",
                "marginLeft": "10px",
                "marginRight": "10px",
                "marginBottom": "0px",
            },
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
                        "display": "inline-block",
                    },
                ),
            ],
        ),
        html.Div(
            html.Button(
                "Check Video",
                id="video-button",
                style={"display": "none"},
            ),
            style={"marginLeft": "50px"},
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Video")),
                dbc.ModalBody(html.Div(id="video-container")),
                dbc.ModalFooter(html.Div(id="video-message")),
            ],
            id="video-modal",
            size="lg",
            is_open=False,
            backdrop="static",  # the user must clicks the "x" to exit
            centered=True,
        ),
        html.Div(
            children=[graph],
            style={"marginTop": "1px", "marginLeft": "20px", "marginRight": "28px"},
        ),
        html.Div(
            style={"display": "flex"},
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "marginRight": "10px",
                        "marginLeft": "10px",
                        "marginBottom": "10px",
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
                dcc.Store(id="video-path-store"),
                dcc.Store(id="clip-name-store"),
                dcc.Store(id="clip-range-store"),
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

# set up dash uploader
pred_upload_box = du.Upload(
    id="pred-data-upload",
    text="Click here to select File",
    text_completed="Completed loading",
    cancel_button=True,
    filetypes=["mat"],
    upload_id="",
    default_style=upload_box_style,
)

vis_upload_box = du.Upload(
    id="vis-data-upload",
    text="Click here to select File",
    text_completed="Completed loading",
    cancel_button=True,
    filetypes=["mat"],
    upload_id="",
    default_style=upload_box_style,
)

video_upload_box_style = {
    "fontSize": "18px",
    "width": "100%",
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

video_upload_box = du.Upload(
    id="video-upload",
    text="Select avi File",
    text_completed="Completed loading",
    cancel_button=True,
    filetypes=["avi"],
    max_file_size=2048,
    upload_id="",
    default_style=video_upload_box_style,
)


# %%
class Components:
    def __init__(self):
        self.home_div = home_div
        self.graph = graph
        self.visualization_div = visualization_div
        self.pred_upload_box = pred_upload_box
        self.vis_upload_box = vis_upload_box
        self.video_upload_box = video_upload_box

    def configure_du(self, app, folder):
        du.configure_upload(app, folder, use_upload_id=True)
        return du
