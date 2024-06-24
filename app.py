# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:45:29 2023

@author: yzhao
"""

import os
import math
import base64
import tempfile
import webbrowser
from io import BytesIO
from threading import Timer
from collections import deque

import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash import Dash, dcc, html, ctx, clientside_callback, Patch

import numpy as np
from flask_caching import Cache
from scipy.io import loadmat, savemat

from components import Components
from inference import run_inference
from make_figure import make_figure


app = Dash(__name__, title="Sleep Scoring App", suppress_callback_exceptions=True)
components = Components()
app.layout = components.home_div

PORT = 8050
TEMP_PATH = os.path.join(tempfile.gettempdir(), "sleep_scoring_app_data")
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

# Notes
# np.nan is converted to None when reading from chache
cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": TEMP_PATH,
        "CACHE_THRESHOLD": 10,
        "CACHE_DEFAULT_TIMEOUT": 86400,  # to save cache for 1 day, otherwise it is default to 300 seconds
    },
)


def open_browser():
    webbrowser.open_new(f"http://127.0.0.1:{PORT}/")


def create_fig(mat, mat_name, default_n_shown_samples=4000):
    fig = make_figure(mat, mat_name, default_n_shown_samples)
    return fig


def initiate_cache(cache, filename, mat):
    cache.set("filename", filename)
    cache.set("mat", mat)
    cache.set("annotation_history", deque(maxlen=3))
    cache.set("fig_resampler", None)


# %% client side callbacks below


@app.callback(
    Output("upload-container", "children"),
    Output("model-choice-container", "style"),
    Input("task-selection", "value"),
    Input("model-choice", "value"),
    Input("num-class-choice", "value"),
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


# choose_model
clientside_callback(
    """
    function(model_choice, model_choice_style) {
        if (model_choice_style.display === "none") {
            return [{"display": "none"}, model_choice];
        }
        return [{"display": "block"}, model_choice];
    }
    """,
    Output("num-class-container", "style"),
    Output("model-choice-store", "data"),
    Input("model-choice", "value"),
    Input("model-choice-container", "style"),
)


# choose_num_class
clientside_callback(
    """
    function(num_class, model_choice) {
        if (model_choice === "sdreamer") {
            num_class = 3;
        }
        return num_class;
    }
    """,
    Output("num-class-store", "data", allow_duplicate=True),
    Input("num-class-choice", "value"),
    State("model-choice-store", "data"),
    prevent_initial_call=True,
)

# validate_extension
clientside_callback(
    """
    function(contents, filename) {
        if (!contents) {
            return ["Please select a .mat file.", dash_clientside.no_update];
        }

        if (!filename.endsWith(".mat")) {
            return ["Please select a .mat file only.", dash_clientside.no_update];
        }

        return ["Loading and validating file... This may take up to 10 seconds.", true];
    }
    """,
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("extension-validation-store", "data"),
    Input(components.mat_upload_box, "contents"),
    State(components.mat_upload_box, "filename"),
    prevent_initial_call=True,
)


# pan_figures
app.clientside_callback(
    """
    function(keyboard_nevents, keyboard_event, relayoutdata, figure) {
        if (!keyboard_event || !figure) {
            return [dash_clientside.no_update, dash_clientside.no_update];
        }

        var key = keyboard_event.key;
        var xaxisRange = figure.layout.xaxis4.range;
        var x0 = xaxisRange[0];
        var x1 = xaxisRange[1];
        var newRange;

        if (key === "ArrowRight") {
            newRange = [x0 + (x1 - x0) * 0.3, x1 + (x1 - x0) * 0.3];
        } else if (key === "ArrowLeft") {
            newRange = [x0 - (x1 - x0) * 0.3, x1 - (x1 - x0) * 0.3];
        }

        if (newRange) {
            relayoutdata['xaxis4.range[0]'] = newRange[0];
            relayoutdata['xaxis4.range[1]'] = newRange[1];
            figure.layout.xaxis4.range = newRange;
            return [figure, relayoutdata];
        }

        return [dash_clientside.no_update, dash_clientside.no_update];
    }
    """,
    Output("graph", "figure", allow_duplicate=True),
    Output("graph", "relayoutData"),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("graph", "relayoutData"),
    State("graph", "figure"),
    prevent_initial_call=True,
)

# show_save_annotation_status
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks > 0) {
            return [5, "Saving annotations. This may take up to 10 seconds."];
        }
        return [dash_clientside.no_update, dash_clientside.no_update];
    }
    """,
    Output("interval-component", "max_intervals"),
    Output("annotation-message", "children", allow_duplicate=True),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)


# clear_display
clientside_callback(
    """
    function(n_intervals) {
        return n_intervals === 5 ? "" : dash_clientside.no_update;
    }
    """,
    Output("annotation-message", "children", allow_duplicate=True),
    Input("interval-component", "n_intervals"),
    prevent_initial_call=True,
)

# %% server side callbacks below


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("generation-ready-store", "data"),
    Output("visualization-ready-store", "data"),
    Output("num-class-store", "data", allow_duplicate=True),
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
    message = "File validated."
    # clear TEMP_PATH regularly
    for temp_file in os.listdir(TEMP_PATH):
        if temp_file.endswith(".mat"):
            os.remove(os.path.join(TEMP_PATH, temp_file))

    eeg = mat.get("eeg")
    if eeg is None:
        return (
            html.Div(["EEG data is missing. Please double check the file selected."]),
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    emg = mat.get("emg")
    if emg is None:
        return (
            html.Div(["EMG data is missing. Please double check the file selected."]),
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    initiate_cache(cache, filename, mat)
    if task == "gen":
        eeg_freq = mat["eeg_frequency"].item()
        if round(eeg_freq) != 512:
            message += " " + (
                f"EEG/EMG data has a sampling frequency of {eeg_freq} Hz. "
                "Will resample to 512 Hz."
            )

        ne = mat.get("ne")
        if ne is None:
            message += " " + "NE data not detected."

        message += " " + "Generating predictions... This may take up to 2 minutes."
        return (
            html.Div([message]),
            True,
            dash.no_update,
            dash.no_update,
        )

    # else visualizing predictions
    num_class = mat.get("num_class")
    if mat.get("num_class") is None:
        return (
            html.Div(
                [
                    "Missing the number of unique sleep scores. Please double check the file selected."
                ]
            ),
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )
    num_class = num_class.item()
    return (
        html.Div(
            [
                "File validated. Creating visualizations... This may take up to 20 seconds."
            ]
        ),
        dash.no_update,
        True,
        num_class,
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
    _, _, output_path = run_inference(
        mat, model_choice, num_class, output_path=temp_mat_path
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
    mat_name = cache.get("filename")
    fig = create_fig(mat, mat_name)
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
    sampling_level_map = {"x1": 4000, "x2": 8000, "x4": 16000}
    n_samples = sampling_level_map[sampling_level]
    mat = cache.get("mat")
    mat_name = cache.get("filename")
    fig = create_fig(mat, mat_name, default_n_shown_samples=n_samples)
    return fig


"""
@app.callback(
    Output("debug-message", "children", allow_duplicate=True),
    Input("graph", "selectedData"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def debug_selected_data(box_select, figure):
    if box_select is None:
        return dash.no_update
    #return str(box_select)
    return str(figure["layout"].get("selections"))

@app.callback(
    Output("debug-message", "children", allow_duplicate=True),
    #Input("keyboard", "n_events"),
    Input("keyboard", "event"),
    #State("keyboard", "event"),
    prevent_initial_call=True,
)
def debug_keypress(keyboard_event):
    return str(keyboard_event.get("key"))



@app.callback(
    Output("debug-message", "children", allow_duplicate=True),
    Input("box-select-store", "data"),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("num-class-store", "data"),
    prevent_initial_call=True,
)
def debug_annotate(box_select_range, keyboard_press, keyboard_event, num_class):
    if not (ctx.triggered_id == "keyboard" and box_select_range):
        raise PreventUpdate

    label = keyboard_event.get("key")
    if label not in ["1", "2", "3", "4"][:num_class]:
        raise PreventUpdate

    return (
        " callback triggered from: "
        + str(ctx.triggered_id)
        + ", box select range: "
        + str(box_select_range)
        + ", key pressed: "
        + str(label)
    )
"""


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("graph", "relayoutData"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata):
    fig = cache.get("fig_resampler")
    # return fig.construct_update_data(relayoutdata)
    return fig.construct_update_data_patch(relayoutdata)


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
        return [], dash.no_update, ""

    patched_figure = Patch()
    if len(selections) > 1:
        selections.pop(0)
    start, end = min(selections[0]["x0"], selections[0]["x1"]), max(
        selections[0]["x0"], selections[0]["x1"]
    )
    patched_figure["layout"][
        "selections"
    ] = selections  # patial property update: https://dash.plotly.com/partial-properties#update
    return (
        [start, end],
        patched_figure,
        "Press 1 for Wake, 2 for SWS, 3 for REM, and 4 for MA, if applicable.",
    )


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-store", "data"),
    Input("box-select-store", "data"),
    Input("keyboard", "n_events"),  # a keyboard press
    State("keyboard", "event"),
    State("num-class-store", "data"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def update_sleep_scores(
    box_select_range, keyboard_press, keyboard_event, num_class, figure
):
    if not (ctx.triggered_id == "keyboard" and box_select_range):
        raise PreventUpdate

    label = keyboard_event.get("key")
    if label not in ["1", "2", "3", "4"][:num_class]:
        raise PreventUpdate

    label = int(label) - 1
    start, end = box_select_range
    if end < 0:
        raise PreventUpdate
    start_round, end_round = round(start), round(end)

    start_round = max(start_round, 0)  # TODO: what about end capping?
    if start_round == end_round:
        if (
            start_round - start > end - end_round
        ):  # spanning over two consecutive seconds
            end_round = math.ceil(start)
            start_round = math.floor(start)
        else:
            end_round = math.ceil(end)
            start_round = math.floor(end)

    start, end = start_round, end_round
    # If the annotation does not change anything, don't add to history
    if (
        figure["data"][-2]["z"][0][start:end] == np.array([label] * (end - start))
    ).all():
        raise PreventUpdate

    patched_figure = Patch()
    prev_labels = figure["data"][-4]["z"][0][start:end]
    prev_conf = figure["data"][-1]["z"][0][start:end]
    figure["data"][-4]["z"][0][start:end] = [label] * (end - start)
    figure["data"][-1]["z"][0][start:end] = [1] * (end - start)  # change conf to 1

    patched_figure["data"][-4]["z"][0] = figure["data"][-4]["z"][0]
    patched_figure["data"][-3]["z"][0] = figure["data"][-4]["z"][0]
    patched_figure["data"][-2]["z"][0] = figure["data"][-4]["z"][0]
    patched_figure["data"][-1]["z"][0] = figure["data"][-1]["z"][0]
    # remove box select after an update is made
    # selections = figure["layout"].get("selections")
    # selections.pop()
    patched_figure["layout"]["selections"].clear()

    return patched_figure, (start, end, prev_labels, prev_conf)


@app.callback(
    Output("undo-button", "style"),
    Input("annotation-store", "data"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def make_annotation(annotation, figure):
    """write to annotation history, update mat in cache, and make undo button availabe"""
    start, end, prev_labels, prev_conf = annotation
    mat = cache.get("mat")
    annotation_history = cache.get("annotation_history")
    annotation_history.append(
        (
            start,
            end,
            prev_labels,  # previous prediction
            prev_conf,  # previous confidence
        )
    )
    cache.set("annotation_history", annotation_history)

    pred_labels = mat.get("pred_labels")
    if pred_labels is not None and pred_labels.size != 0:
        mat["pred_labels"] = np.array(figure["data"][-2]["z"][0])
    else:
        mat["sleep_scores"] = np.array(figure["data"][-2]["z"][0])
    mat["confidence"] = np.array(figure["data"][-1]["z"][0])
    cache.set("mat", mat)
    return {"display": "block"}


@app.callback(
    # Output("debug-message", "children"),
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

    patched_figure = Patch()
    # undo figure
    figure["data"][-4]["z"][0][start:end] = prev_pred
    figure["data"][-1]["z"][0][start:end] = prev_conf

    patched_figure["data"][-4]["z"][0] = figure["data"][-4]["z"][0]
    patched_figure["data"][-3]["z"][0] = figure["data"][-4]["z"][0]
    patched_figure["data"][-2]["z"][0] = figure["data"][-4]["z"][0]
    patched_figure["data"][-1]["z"][0] = figure["data"][-1]["z"][0]

    # undo cache
    mat = cache.get("mat")
    pred_labels = mat.get("pred_labels")
    if pred_labels is not None and pred_labels.size != 0:
        mat["pred_labels"][start:end] = np.array(prev_pred)
    else:
        mat["sleep_scores"][start:end] = np.array(prev_pred)

    mat["confidence"][start:end] = np.array(prev_conf)
    cache.set("mat", mat)

    # update annotation_history
    cache.set("annotation_history", annotation_history)
    if not annotation_history:
        return patched_figure, {"display": "none"}
    return patched_figure, {"display": "block"}


@app.callback(
    Output("download-annotations", "data"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_annotations(n_clicks):
    mat_filename = cache.get("filename")
    temp_mat_path = os.path.join(TEMP_PATH, mat_filename)
    mat = cache.get("mat")
    sleep_scores = mat.get("sleep_scores")
    if sleep_scores is not None and sleep_scores.size != 0:
        np.place(
            sleep_scores, sleep_scores == None, [-1.0]
        )  # convert None to -1.0 for scipy's savemat

    savemat(temp_mat_path, mat)
    return dcc.send_file(temp_mat_path)


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run_server(debug=True, port=PORT)
