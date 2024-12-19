# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:45:29 2023

@author: yzhao
"""

import os
import math
import tempfile
import webbrowser
from collections import deque

import dash
import dash_uploader as du
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash import Dash, dcc, html, ctx, clientside_callback, Patch

import numpy as np
import pandas as pd
from flask_caching import Cache
from scipy.io import loadmat, savemat

from app_src import VERSION, config
from app_src.plot_spectrogram import plot_spectrogram
from app_src.inference import run_inference
from app_src.components import Components
from app_src.make_figure import make_figure
from app_src.postprocessing import get_sleep_segments, get_pred_label_stats


app = Dash(
    __name__, title=f"Sleep Scoring App {VERSION}", suppress_callback_exceptions=True
)

TEMP_PATH = os.path.join(tempfile.gettempdir(), "sleep_scoring_app_data")
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

du.configure_upload(app, folder=TEMP_PATH, use_upload_id=True)
components = Components()
app.layout = components.home_div


# set up dash uploader
pred_upload_box = du.Upload(
    id="pred-data-upload",
    text="Click here to select File",
    text_completed="Completed loading",
    cancel_button=True,
    filetypes=["mat"],
    upload_id="",
    default_style=components.upload_box_style,
)

vis_upload_box = du.Upload(
    id="vis-data-upload",
    text="Click here to select File",
    text_completed="Completed loading",
    cancel_button=True,
    filetypes=["mat"],
    upload_id="",
    default_style=components.upload_box_style,
)

# Notes
# np.nan is converted to None when reading from cache
cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": TEMP_PATH,
        "CACHE_THRESHOLD": 20,
        "CACHE_DEFAULT_TIMEOUT": 86400,  # to save cache for 1 day, otherwise it is default to 300 seconds
    },
)


# %%
def open_browser(port):
    webbrowser.open_new(f"http://127.0.0.1:{port}/")


def create_fig(mat, mat_name, default_n_shown_samples=4000):
    fig = make_figure(mat, mat_name, default_n_shown_samples)
    return fig


def set_cache(cache, filename):
    cache.set("filename", filename)
    cache.set("start_time", 0)
    cache.set("modified_sleep_scores", None)
    cache.set("modified_confidence", None)
    cache.set("annotation_history", deque(maxlen=3))
    cache.set("fig_resampler", None)


# %% client side callbacks below


@app.callback(
    Output("upload-container", "children", allow_duplicate=True),
    Output("model-choice-container", "style"),
    Input("task-selection", "value"),
    Input("model-choice", "value"),
    prevent_initial_call=True,
)
def show_upload_box(task, model_choice):
    # if task or model choice changes, give a new upload box so that
    # the upload of the same file (but running with different model) is allowed
    if task is None:
        raise PreventUpdate
    if task == "pred":
        return pred_upload_box, {"display": "block"}
    else:
        return vis_upload_box, {"display": "none"}


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

# synch_fft_figure
app.clientside_callback(
    """
    function(relayoutData, figure, fftRelayoutData, fftFigure) {
        // Ensure the main figure's xaxis4 range exists
        if (!figure.layout.xaxis4 || !figure.layout.xaxis4.range) {
            return [fftRelayoutData, fftFigure];
        }

        // Extract the range
        var range = figure.layout.xaxis4.range;
        var fftFigStart = range[0];
        var fftFigEnd = range[1];

        // Ensure fftRelayoutData is initialized
        fftRelayoutData = fftRelayoutData || {};
        fftRelayoutData['xaxis.range[0]'] = fftFigStart;
        fftRelayoutData['xaxis.range[1]'] = fftFigEnd;

        // Create a new copy of fftFigure to ensure re-render
        let updatedFftFigure = JSON.parse(JSON.stringify(fftFigure));
        updatedFftFigure.layout = updatedFftFigure.layout || {};
        updatedFftFigure.layout.xaxis = updatedFftFigure.layout.xaxis || {};
        updatedFftFigure.layout.xaxis.range = [fftFigStart, fftFigEnd];

        return [fftRelayoutData, updatedFftFigure];
    }
    """,
    [Output("fft-graph", "relayoutData"), Output("fft-graph", "figure")],
    [Input("graph", "relayoutData")],
    [
        State("graph", "figure"),
        State("fft-graph", "relayoutData"),
        State("fft-graph", "figure"),
    ],
)


"""
@app.callback(
    Output("fft-graph", "relayoutData"),
    Output("fft-graph", "figure"),
    Input("graph", "relayoutData"),
    State("graph", "figure"),
    State("fft-graph", "relayoutData"),
    State("fft-graph", "figure"),
    prevent_initial_call=True,
)
def synch_fft_figure(relayoutdata, figure, fft_relayoutdata, fft_fig):
    fft_fig_start,  fft_fig_end = figure["layout"]["xaxis4"]["range"]
    fft_relayoutdata['xaxis.range[0]'], fft_relayoutdata['xaxis.range.[1]'] = fft_fig_start, fft_fig_end
    fft_fig["layout"]["xaxis"]["range"] = [fft_fig_start, fft_fig_end]
    return fft_relayoutdata, fft_fig
"""

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


@du.callback(
    output=[
        Output("data-upload-message", "children", allow_duplicate=True),
        Output("prediction-ready-store", "data"),
        Output("upload-container", "children", allow_duplicate=True),
    ],
    id="pred-data-upload",
)
def read_mat_pred(status):
    message = "File validated."
    mat_file = status.latest_file
    filename = os.path.basename(mat_file)
    # clean TEMP_PATH regularly by deleting temp files written there

    for temp_file in os.listdir(TEMP_PATH):
        if temp_file.endswith(".mat") or temp_file.endswith(".xlsx"):
            if temp_file == filename:
                continue
            os.remove(os.path.join(TEMP_PATH, temp_file))

    mat = loadmat(mat_file)
    eeg = mat.get("eeg")
    if eeg is None:
        return (
            html.Div(["EEG data is missing. Please double check the file selected."]),
            dash.no_update,
            pred_upload_box,
        )

    emg = mat.get("emg")
    if emg is None:
        return (
            html.Div(["EMG data is missing. Please double check the file selected."]),
            dash.no_update,
            pred_upload_box,
        )

    set_cache(cache, filename)
    eeg_freq = mat["eeg_frequency"].item()
    if round(eeg_freq) != 512:
        message += " " + (
            f"EEG/EMG data has a sampling frequency of {eeg_freq} Hz. "
            "Will resample to 512 Hz."
        )

    ne = mat.get("ne")
    if ne is None:
        message += " " + "NE data not detected."

    message += (
        " "
        + "Generating predictions... This may take up to 3 minutes. Check Terminal for the progress."
    )
    return (html.Div([message]), True, pred_upload_box)


@du.callback(
    output=[
        Output("data-upload-message", "children", allow_duplicate=True),
        Output("visualization-ready-store", "data", allow_duplicate=True),
        Output("upload-container", "children", allow_duplicate=True),
    ],
    id="vis-data-upload",
)
def read_mat_vis(status):
    # clean TEMP_PATH regularly by deleting temp files written there
    mat_file = status.latest_file
    filename = os.path.basename(mat_file)
    for temp_file in os.listdir(TEMP_PATH):
        if temp_file.endswith(".mat") or temp_file.endswith(".xlsx"):
            if temp_file == filename:
                continue
            os.remove(os.path.join(TEMP_PATH, temp_file))

    set_cache(cache, filename)

    return (
        html.Div(
            [
                "File validated. Creating visualizations... This may take up to 30 seconds."
            ]
        ),
        True,
        vis_upload_box,
    )


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("visualization-ready-store", "data", allow_duplicate=True),
    # Output("prediction-download-store", "data"),
    Input("prediction-ready-store", "data"),
    prevent_initial_call=True,
)
def generate_prediction(ready):
    filename = cache.get("filename")
    temp_mat_path = os.path.join(TEMP_PATH, filename)
    mat = loadmat(temp_mat_path)
    mat, output_path = run_inference(
        mat,
        postprocess=config["postprocess"],
        output_path=temp_mat_path,
        save_inference=True,
    )
    # it is necessart to set cache again here because the output file
    # which includes prediction and confidence has a new name (old_name + "_sdreamer"),
    # it is this file that should be used for the subsequent visualization.
    set_cache(cache, os.path.basename(output_path))
    return (
        html.Div(["The prediction has been generated successfully."]),
        True,
        # dcc.send_file(output_path),
    )


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Input("visualization-ready-store", "data"),
    prevent_initial_call=True,
)
def create_visualization(ready):
    mat_name = cache.get("filename")
    mat = loadmat(os.path.join(TEMP_PATH, mat_name))
    fig = create_fig(mat, mat_name)
    start_time = mat.get("start_time")
    if start_time is None:
        start_time = 0
    else:
        start_time = start_time.item()
    eeg_frequency = mat.get("eeg_frequency").item()
    eeg = mat.get("eeg").flatten()
    fft_fig = plot_spectrogram(eeg, eeg_frequency, start_time=start_time)
    cache.set("start_time", start_time)
    cache.set("fig_resampler", fig)

    components.graph.figure = fig
    components.fft_graph.figure = fft_fig
    return (components.visualization_div,)


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
    mat_name = cache.get("filename")
    mat = loadmat(os.path.join(TEMP_PATH, mat_name))

    # copy modified (through annotation) sleep scores and confidence over
    modified_sleep_scores = cache.get("modified_sleep_scores")
    if modified_sleep_scores is not None:
        modified_confidence = cache.get("modified_confidence")
        if mat.get("pred_labels") is not None and mat["pred_labels"].size != 0:
            mat["pred_labels"] = modified_sleep_scores.copy()
        else:
            mat["sleep_scores"] = modified_sleep_scores.copy()
        mat["confidence"] = modified_confidence.copy()

    fig = create_fig(mat, mat_name, default_n_shown_samples=n_samples)
    return fig


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("graph", "relayoutData"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata):
    fig = cache.get("fig_resampler")
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
    # allow only at most one select box in all subplots
    if len(selections) > 1:
        selections.pop(0)

    patched_figure["layout"][
        "selections"
    ] = selections  # patial property update: https://dash.plotly.com/partial-properties#update

    # take the min as start and max as end so that how the box is drawn doesn't matter
    start, end = min(selections[0]["x0"], selections[0]["x1"]), max(
        selections[0]["x0"], selections[0]["x1"]
    )

    eeg_start_time = cache.get("start_time")
    eeg_duration = len(figure["data"][-1]["z"][0])
    eeg_end_time = eeg_start_time + eeg_duration

    if end < eeg_start_time or start > eeg_end_time:
        return [], patched_figure, ""

    start_round, end_round = round(start), round(end)
    start_round = max(start_round, eeg_start_time)
    end_round = min(end_round, eeg_end_time)
    if start_round == end_round:
        if (
            start_round - start > end - end_round
        ):  # spanning over two consecutive seconds
            end_round = math.ceil(start)
            start_round = math.floor(start)
        else:
            end_round = math.ceil(end)
            start_round = math.floor(end)

    start, end = start_round - eeg_start_time, end_round - eeg_start_time

    return (
        [start, end],
        patched_figure,
        "Press 1 for Wake, 2 for SWS, 3 for REM, and 4 for MA, if applicable.",
    )


"""
@app.callback(
    Output("debug-message", "children"),
    Input("graph", "relayoutData"),
    State("graph", "figure"),
    State("fft-graph", "relayoutData"),
    State("fft-graph", "figure"),
    prevent_initial_call=True,
)
def debug_fft_fig(relayoutdata, figure, fft_relayoutdata, fft_fig):
    return str(fft_relayoutdata)
"""


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-store", "data"),
    Input("box-select-store", "data"),
    Input("keyboard", "n_events"),  # a keyboard press
    State("keyboard", "event"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def update_sleep_scores(box_select_range, keyboard_press, keyboard_event, figure):
    if not (ctx.triggered_id == "keyboard" and box_select_range):
        raise PreventUpdate

    label = keyboard_event.get("key")
    if label not in ["1", "2", "3"]:
        raise PreventUpdate

    label = int(label) - 1
    start, end = box_select_range
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
    patched_figure["layout"]["selections"].clear()

    return patched_figure, (start, end, prev_labels, prev_conf)


@app.callback(
    Output("undo-button", "style"),
    Input("annotation-store", "data"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def write_annotation(annotation, figure):
    """write to annotation history, update mat in cache, and make undo button availabe"""
    start, end, prev_labels, prev_conf = annotation
    annotation_history = cache.get("annotation_history")
    annotation_history.append(
        (
            start,
            end,
            prev_labels,  # previous prediction
            prev_conf,  # previous confidence
        )
    )
    labels = np.array(figure["data"][-2]["z"]).astype(float)
    confidence = np.array(figure["data"][-1]["z"]).astype(float)
    cache.set("annotation_history", annotation_history)
    cache.set("modified_sleep_scores", labels)
    cache.set("modified_confidence", confidence)
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
    (start, end, prev_labels, prev_conf) = prev_annotation
    prev_labels, prev_conf = np.array(prev_labels), np.array(prev_conf)

    patched_figure = Patch()
    # undo figure
    figure["data"][-4]["z"][0][start:end] = prev_labels
    figure["data"][-1]["z"][0][start:end] = prev_conf

    patched_figure["data"][-4]["z"][0] = figure["data"][-4]["z"][0]
    patched_figure["data"][-3]["z"][0] = figure["data"][-4]["z"][0]
    patched_figure["data"][-2]["z"][0] = figure["data"][-4]["z"][0]
    patched_figure["data"][-1]["z"][0] = figure["data"][-1]["z"][0]

    # undo cache
    modified_sleep_scores, modified_confidence = cache.get(
        "modified_sleep_scores"
    ), cache.get("modified_confidence")
    modified_sleep_scores[0, start:end] = prev_labels
    modified_confidence[0, start:end] = prev_conf
    cache.set("modified_sleep_scores", modified_sleep_scores)
    cache.set("modified_confidence", modified_confidence)

    # update annotation_history
    cache.set("annotation_history", annotation_history)
    if not annotation_history:
        return patched_figure, {"display": "none"}
    return patched_figure, {"display": "block"}


@app.callback(
    Output("download-annotations", "data"),
    Output("download-spreadsheet", "data"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_annotations(n_clicks):
    mat_filename = cache.get("filename")
    temp_mat_path = os.path.join(TEMP_PATH, mat_filename)
    mat = loadmat(temp_mat_path)

    # only need to replace None in sleep_scores assuming pred_labels will never have nan or None
    modified_sleep_scores = cache.get("modified_sleep_scores")
    modified_confidence = cache.get("modified_confidence")
    labels = None
    if modified_sleep_scores is not None:
        # replace any None or nan in sleep scores to -1 before saving, otherwise results in save error
        # make a copy first because we don't want to convert any nan in the cache
        modified_sleep_scores = modified_sleep_scores.copy()
        np.place(
            modified_sleep_scores, modified_sleep_scores == None, [-1]
        )  # convert None to -1 for scipy's savemat
        modified_sleep_scores = np.nan_to_num(
            modified_sleep_scores, nan=-1
        )  # convert np.nan to -1 for scipy's savemat

        if mat.get("pred_labels") is not None and mat["pred_labels"].size != 0:
            mat["pred_labels"] = modified_sleep_scores
        else:
            mat["sleep_scores"] = modified_sleep_scores
        mat["confidence"] = modified_confidence.copy()
    savemat(temp_mat_path, mat)

    # export sleep bout spreadsheet only if the manual scoring is complete
    if mat.get("sleep_scores") is not None and -1 not in mat["sleep_scores"]:
        labels = mat["sleep_scores"].flatten()

    if mat.get("pred_labels") is not None and mat["pred_labels"].size != 0:
        labels = mat["pred_labels"].flatten()

    if labels is not None:
        labels = labels.astype(int)
        df = get_sleep_segments(labels)
        df_stats = get_pred_label_stats(df)
        temp_excel_path = os.path.splitext(temp_mat_path)[0] + "_table.xlsx"
        with pd.ExcelWriter(temp_excel_path) as writer:
            df.to_excel(writer, sheet_name="Sleep_bouts")
            df_stats.to_excel(writer, sheet_name="Sleep_stats")
            worksheet = writer.sheets["Sleep_stats"]
            worksheet.set_column(0, 0, 20)

        return dcc.send_file(temp_mat_path), dcc.send_file(temp_excel_path)

    return dcc.send_file(temp_mat_path), dash.no_update


if __name__ == "__main__":
    from threading import Timer
    from functools import partial

    PORT = 8050
    Timer(1, partial(open_browser, PORT)).start()
    app.run_server(debug=False, port=PORT)
