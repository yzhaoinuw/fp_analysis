# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:45:29 2023

@author: yzhao
"""

import os
import base64
import tempfile
import webbrowser
from io import BytesIO
from threading import Timer
from collections import deque

import dash
from dash import Dash, dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

import numpy as np
from flask_caching import Cache
from scipy.io import loadmat, savemat


from inference import run_inference
from make_figure import make_figure
from components import Components


app = Dash(__name__, suppress_callback_exceptions=True)
port = 8050
components = Components()
app.layout = components.home_div

TEMP_PATH = os.path.join(tempfile.gettempdir(), "sleep_scoring_app_data")
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": TEMP_PATH,
        "CACHE_THRESHOLD": 10,
        "CACHE_DEFAULT_TIMEOUT": 86400,  # to save cache for 1 day, otherwise it is default to 300 seconds
    },
)


def create_fig(mat, default_n_shown_samples=1000):
    fig = make_figure(mat, default_n_shown_samples)
    return fig


def initiate_cache(cache, filename, mat):
    cache.set("filename", filename)
    cache.set("mat", mat)
    cache.set("annotation_history", deque(maxlen=3))
    cache.set("fig_resampler", None)


@app.callback(
    Output("upload-container", "children"),
    Output("model-choice-container", "style"),
    Input("task-selection", "value"),
    Input("model-choice", "value"),
    Input("num-class", "value"),
)
def show_upload_box(task, model_choice, num_class):
    # if any of task, model choice, or num_class changes, gives a new upload box so that
    # the upload of the same file (but running with different model) is allowed
    if task is None:
        raise PreventUpdate
    if task == "gen":
        return components.mat_upload_box, {"display": "block"}
    else:
        return components.mat_upload_box, {"display": "none"}


@app.callback(
    Output("num-class-container", "style"),
    Output("model-choice-store", "data"),
    Input("model-choice", "value"),
    Input("model-choice-container", "style"),
)
def choose_model(model_choice, model_choice_style):
    if model_choice_style["display"] == "none":
        return {"display": "none"}, model_choice

    return {"display": "block"}, model_choice


@app.callback(
    Output("num-class-store", "data"),
    Input("num-class", "value"),
    State("model-choice-store", "data"),
)
def choose_num_class(num_class, model_choice):
    if model_choice == "sdreamer":
        num_class = 4
    return num_class


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("extension-validation-store", "data"),
    Input(components.mat_upload_box, "contents"),
    State(components.mat_upload_box, "filename"),
    prevent_initial_call=True,
)
def validate_extension(contents, filename):
    if contents is None:
        return html.Div(["Please select a .mat file."]), dash.no_update

    if not filename.endswith(".mat"):
        return html.Div(["Please select a .mat file only."]), dash.no_update

    return (
        html.Div(["Loading and validating file... This may take up to 10 seconds."]),
        True,
    )


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("generation-ready-store", "data"),
    Output("visualization-ready-store", "data"),
    Input("extension-validation-store", "data"),
    State(components.mat_upload_box, "contents"),
    State(components.mat_upload_box, "filename"),
    State("task-selection", "value"),
    prevent_initial_call=True,
)
def read_mat(extension_validated, contents, filename, task):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    mat = loadmat(BytesIO(decoded))

    # clear TEMP_PATH regularly
    for temp_file in os.listdir(TEMP_PATH):
        if temp_file.endswith(".mat"):
            os.remove(os.path.join(TEMP_PATH, temp_file))

    if mat.get("trial_eeg") is None:
        return (
            html.Div(["EEG data is missing. Please double check the file selected."]),
            dash.no_update,
            dash.no_update,
        )

    if mat.get("trial_emg") is None:
        return (
            html.Div(["EMG data is missing. Please double check the file selected."]),
            dash.no_update,
            dash.no_update,
        )

    if mat.get("trial_ne") is None:
        return (
            html.Div(["NE data is missing. Please double check the file selected."]),
            dash.no_update,
            dash.no_update,
        )

    initiate_cache(cache, filename, mat)
    if task == "gen":
        return (
            html.Div(
                [
                    "File validated. Generating predictions... This may take up to 2 minutes."
                ]
            ),
            True,
            dash.no_update,
        )

    return (
        html.Div(
            [
                "File validated. Creating visualizations... This may take up to 20 seconds."
            ]
        ),
        dash.no_update,
        True,
    )


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("prediction-download-store", "data"),
    Input("generation-ready-store", "data"),
    State("model-choice-store", "data"),
    State("num-class-store", "data"),
    prevent_initial_call=True,
)
def generate_prediction(ready, model_choice, num_class):
    filename = cache.get("filename")
    mat = cache.get("mat")
    temp_mat_path = os.path.join(TEMP_PATH, filename)
    output_path = os.path.splitext(temp_mat_path)[0] + "_prediction"
    _, _, output_path = run_inference(
        mat, model_choice, num_class, output_path=output_path
    )
    return (
        html.Div(["The prediction has been generated successfully."]),
        dcc.send_file(output_path),
    )


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Input("visualization-ready-store", "data"),
    prevent_initial_call=True,
)
def create_visualization(ready):
    mat = cache.get("mat")
    fig = create_fig(mat)
    cache.set("fig_resampler", fig)
    components.graph.figure = fig
    return components.visualization_div


@app.callback(
    Output("graph", "figure"),
    Input("n-sample-dropdown", "value"),
    prevent_initial_call=True,
)
def change_sampling_level(sampling_level):
    if sampling_level is None:
        return dash.no_update
    sampling_level_map = {"x1": 1000, "x2": 2000, "x4": 4000}
    n_samples = sampling_level_map[sampling_level]
    mat = cache.get("mat")
    fig = create_fig(mat, default_n_shown_samples=n_samples)
    return fig


@app.callback(
    Output("box-select-store", "data"),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Input("graph", "selectedData"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def read_box_select(box_select, figure):
    selections = figure["layout"].get("selections")
    if not selections:
        return [], dash.no_update, dash.no_update
    if len(selections) > 1:
        selections.pop(0)
    start, end = selections[0]["x0"], selections[0]["x1"]
    start, end = round(start), round(end)
    return [start, end], figure, "Press 0 for Wake, 1 for SWS, and 2 for REM."


@app.callback(
    Output("trace-updater", "updateData"),
    Input("graph", "relayoutData"),
    prevent_initial_call=True,
)
def update_fig(relayoutdata):
    fig = cache.get("fig_resampler")
    return fig.construct_update_data(relayoutdata)


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("undo-button", "style"),
    Input("box-select-store", "data"),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("graph", "figure"),
    State("num-class-store", "data"),
    prevent_initial_call=True,
)
def update_sleep_scores(
    box_select_range, keyboard_nevents, keyboard_event, figure, num_class
):
    ctx = dash.callback_context
    if ctx.triggered:
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if input_id == "keyboard":
            label = keyboard_event.get("key")
            if (
                label in ["0", "1", "2", "3"][:num_class] and box_select_range
            ):  # TODO: what about "3"?
                label = int(label)
                start, end = box_select_range
                start = max(start, 0)  # TODO: what about end capping?
                if start == end:
                    raise PreventUpdate

                annotation_history = cache.get("annotation_history")
                annotation_history.append(
                    (
                        start,
                        end,
                        figure["data"][3]["z"][0][start:end],  # previous prediction
                        figure["data"][6]["z"][0][start:end],  # previous confidence
                    )
                )
                cache.set("annotation_history", annotation_history)
                figure["data"][3]["z"][0][start:end] = [label] * (end - start)
                figure["data"][4]["z"][0][start:end] = [label] * (end - start)
                figure["data"][5]["z"][0][start:end] = [label] * (end - start)
                figure["data"][6]["z"][0][start:end] = [1] * (
                    end - start
                )  # change conf to 1

                mat = cache.get("mat")
                mat["pred_labels"] = np.array(figure["data"][3]["z"][0])
                mat["confidence"] = np.array(figure["data"][6]["z"][0])
                cache.set("mat", mat)
                return figure, {"display": "block"}
    raise PreventUpdate


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("undo-button", "style", allow_duplicate=True),
    Input("undo-button", "n_clicks"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def undo_annotation(n_clicks, figure):
    annotation_history = cache.get("annotation_history")
    prev_annotation = annotation_history.pop()
    (start, end, prev_pred, prev_conf) = prev_annotation

    # undo figure
    figure["data"][3]["z"][0][start:end] = prev_pred
    figure["data"][4]["z"][0][start:end] = prev_pred
    figure["data"][5]["z"][0][start:end] = prev_pred
    figure["data"][6]["z"][0][start:end] = prev_conf

    # undo cache
    mat = cache.get("mat")
    mat["pred_labels"] = np.array(figure["data"][3]["z"][0])
    mat["confidence"] = np.array(figure["data"][6]["z"][0])
    cache.set("mat", mat)

    # update annotation_history
    cache.set("annotation_history", annotation_history)
    if not annotation_history:
        return figure, {"display": "none"}
    return figure, {"display": "block"}


@app.callback(
    Output("download-annotations", "data"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_annotations(n_clicks):
    mat_filename = cache.get("filename")
    temp_mat_path = os.path.join(TEMP_PATH, mat_filename)
    mat = cache.get("mat")
    savemat(temp_mat_path, mat)
    return dcc.send_file(temp_mat_path)


@app.callback(
    Output("interval-component", "max_intervals"),
    Output("annotation-message", "children", allow_duplicate=True),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)
def show_save_annotation_status(n_clicks):
    return 5, "Saving annotations. This may take up to 10 seconds."


@app.callback(
    Output("annotation-message", "children"),
    Input("interval-component", "n_intervals"),
)
def clear_display(n):
    if n == 5:
        return ""
    return dash.no_update


def open_browser():
    webbrowser.open_new(f"http://127.0.0.1:{port}/")


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run_server(debug=True, port=8050)
