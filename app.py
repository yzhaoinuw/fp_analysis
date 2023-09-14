# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:53:49 2023

@author: Yue
"""

import os
import json
import base64
import tempfile
import webbrowser
from io import BytesIO
from threading import Timer
from collections import defaultdict

from trace_updater import TraceUpdater
import dash
from dash import Dash, dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash_extensions import EventListener

from plotly_resampler import FigureResampler

import numpy as np
from scipy.io import loadmat, savemat

from inference import run_inference
from config import annotation_config
from make_figure import make_figure, stage_colors


app = Dash(__name__)
port = 8050
fig = FigureResampler(default_n_shown_samples=2000)
fig.register_update_graph_callback(
    app=app, graph_id="graph-1", trace_updater_id="trace-updater-1"
)
TEMP_PATH = os.path.join(tempfile.gettempdir(), "sleep_scoring_app_data")


def run_app():
    app.layout = html.Div(
        [
            dcc.RadioItems(
                id="task-selection",
                options=[
                    {"label": "Generate prediction", "value": "gen"},
                    {"label": "Visualize existing prediction", "value": "vis"},
                ],
            ),
            html.Div(id="upload-container"),
            html.Div(id="output-data-upload"),
            dcc.Store(id="validate-file-extension"),
            dcc.Store(id="mat-filename"),
            dcc.Download(id="download-prediction"),
        ]
    )


@app.callback(Output("upload-container", "children"), Input("task-selection", "value"))
def show_upload(task):
    if task is None:
        raise PreventUpdate
    else:
        return dcc.Upload(
            id="upload-data",
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


@app.callback(
    Output("output-data-upload", "children", allow_duplicate=True),
    Output("validate-file-extension", "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("task-selection", "value"),
    prevent_initial_call=True,
)
def show_mat_read_status(contents, filename, task):
    if contents is None:
        return html.Div(["Please select a .mat file."]), dash.no_update

    if not filename.endswith(".mat"):
        return html.Div(["Please select a .mat file only."]), dash.no_update

    if task == "gen":
        return (
            html.Div(
                ["Generating sleep score predictions. This may take up to 60 seconds."]
            ),
            True,
        )
    return (
        html.Div(
            ["Preparing visualizations of the results. This may take up to 20 seconds."]
        ),
        True,
    )


@app.callback(
    [
        Output("output-data-upload", "children"),
        Output("mat-filename", "data"),
        Output("download-prediction", "data"),
    ],
    Input("validate-file-extension", "data"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("task-selection", "value"),
    prevent_initial_call=True,
)
def update_output(file_validated, contents, filename, task):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    mat = loadmat(BytesIO(decoded))
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)

    # clear TEMP_PATH regularly
    for temp_file in os.listdir(TEMP_PATH):
        os.remove(os.path.join(TEMP_PATH, temp_file))
    temp_mat_path = os.path.join(TEMP_PATH, filename)

    if task == "gen":
        output_path = os.path.splitext(temp_mat_path)[0] + "_predictions.mat"
        run_inference(mat, model_path=None, output_path=output_path)
        return (
            html.Div(["The predictions have been generated successfully."]),
            None,
            dcc.send_file(output_path),
        )
    else:  # task == 'vis'
        try:
            savemat(temp_mat_path, mat)
            fig.replace(make_figure(mat))
            div = html.Div(
                children=[
                    dcc.Graph(
                        id="graph-1",
                        config={
                            "editable": True,
                            "edits": {
                                "axisTitleText": False,
                                "titleText": False,
                                "colorbarTitleText": False,
                                "annotationText": False,
                            },
                        },
                        figure=fig,
                    ),
                    TraceUpdater(id="trace-updater-1", gdID="graph-1"),
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
                                ],
                            ),
                            dcc.Interval(
                                id="interval-component",
                                interval=1 * 1000,  # in milliseconds
                                max_intervals=0,  # stop after the first interval
                            ),
                            html.Div(id="annotation-message"),
                        ],
                    ),
                ],
            )

            return div, filename, None
        except Exception as e:
            print(e)
            return html.Div(["There was an error processing this file."]), None, None


@app.callback(
    Output("box-select-store", "data"),
    Output("annotation-message", "children", allow_duplicate=True),
    Input("graph-1", "selectedData"),
    prevent_initial_call=True,
)
def read_box_select(box_select):
    try:
        start, end = box_select["range"]["x4"]
    except KeyError:
        return None, ""
    start = int(start)
    end = int(end)
    return json.dumps([start, end]), "Press 0 for Wake, 1 for SWS, and 2 for REM."


@app.callback(
    Output("graph-1", "figure"),
    Input("box-select-store", "data"),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("graph-1", "figure"),
    prevent_initial_call=True,
)
def update_sleep_scores(box_select_range, keyboard_nevents, keyboard_event, figure):
    ctx = dash.callback_context
    if ctx.triggered:
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if input_id == "keyboard":
            label = keyboard_event.get("key")
            if label in ["0", "1", "2"] and box_select_range:
                label = int(label)
                start, end = json.loads(box_select_range)
                figure["data"][3]["z"][0][start : end + 1] = [label] * (end - start + 1)
                figure["data"][4]["z"][0][start : end + 1] = [label] * (end - start + 1)
                figure["data"][5]["z"][0][start : end + 1] = [label] * (end - start + 1)
            return figure
    return dash.no_update


@app.callback(
    Output("interval-component", "max_intervals"),
    Output("annotation-message", "children", allow_duplicate=True),
    Input("save-button", "n_clicks"),
    State("graph-1", "figure"),
    prevent_initial_call=True,
)
def show_save_annotation_status(n_clicks, figure):
    return 5, "Saving annotations. This may take up to 10 seconds."


@app.callback(
    Output("annotation-message", "children"),
    Input("interval-component", "n_intervals"),
)
def clear_display(n):
    if n == 5:
        return ""
    return dash.no_update


@app.callback(
    Output("download-annotations", "data"),
    Input("save-button", "n_clicks"),
    State("graph-1", "figure"),
    State("mat-filename", "data"),
    prevent_initial_call=True,
)
def save_annotations(n_clicks, figure, mat_filename):
    temp_mat_path = os.path.join(TEMP_PATH, mat_filename)
    mat = loadmat(temp_mat_path)
    savemat(temp_mat_path, mat)
    return dcc.send_file(temp_mat_path)


def open_browser():
    webbrowser.open_new(f"http://127.0.0.1:{port}/")


if __name__ == "__main__":
    run_app()
    Timer(1, open_browser).start()
    app.run_server(debug=True, port=8050)
