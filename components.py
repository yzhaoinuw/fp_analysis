# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:27:03 2023

@author: yzhao
"""

from dash import dcc, html
from trace_updater import TraceUpdater
from dash_extensions import EventListener
from plotly_resampler import FigureResampler

mat_upload_box = dcc.Upload(
    # id="data-upload",
    children=html.Div(["Select File"], className="upload-button"),
    style={
        "width": "100%",
        "height": "60px",
        "lineHeight": "60px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
        "margin": "10px",
    },
    multiple=False,
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


visualization_div = html.Div(
    children=[
        dcc.Dropdown(
            ["x1", "x2", "x4"],
            "x1",
            id="n-sample-dropdown",
            placeholder="Select a sampling level",
        ),
        graph,
        TraceUpdater(id="trace-updater", gdID="graph"),
        html.Div(
            style={"display": "flex"},
            children=[
                dcc.Store(id="box-select-store"),
                EventListener(
                    id="keyboard",
                    events=[{"event": "keydown", "props": ["key"]}],
                ),
                html.Div(
                    style={"display": "flex", "margin-right": "5px"},
                    children=[
                        html.Button("Save Annotations", id="save-button"),
                        dcc.Download(id="download-annotations"),
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
