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
import dash_player
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash import Dash, dcc, html, ctx, clientside_callback, Patch

import numpy as np
import pandas as pd
from flask_caching import Cache
from scipy.io import loadmat, savemat

from app_src import VERSION, config
from app_src.make_mp4 import avi_to_mp4
from app_src.components_dev import Components
from app_src.inference import run_inference
from app_src.make_figure_dev import make_figure

# from app_src.plot_spectrogram import plot_spectrogram
from app_src.postprocessing import get_sleep_segments, get_pred_label_stats


app = Dash(
    __name__,
    title=f"Sleep Scoring App {VERSION}",
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP
    ],  # need this for the modal to work properly
)

TEMP_PATH = os.path.join(tempfile.gettempdir(), "sleep_scoring_app_data")
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

VIDEO_DIR = "./assets/videos/"
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

components = Components()
app.layout = components.home_div
du = components.configure_du(app, TEMP_PATH)

# Notes
# np.nan is converted to None when reading from cache
cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": TEMP_PATH,
        "CACHE_THRESHOLD": 30,
        "CACHE_DEFAULT_TIMEOUT": 20
        * 24
        * 3600,  # to save cache for 20 days, otherwise it is default to 300 seconds
    },
)


# %%
def open_browser(port):
    webbrowser.open_new(f"http://127.0.0.1:{port}/")


def create_fig(mat, mat_name, default_n_shown_samples=4000):
    fig = make_figure(mat, mat_name, default_n_shown_samples)
    return fig


def reset_cache(cache, filename):
    cache.set("filename", filename)
    recent_files_with_video = cache.get("recent_files_with_video")
    if recent_files_with_video is None:
        recent_files_with_video = []
    file_video_record = cache.get("file_video_record")
    if file_video_record is None:
        file_video_record = {}
    cache.set("recent_files_with_video", recent_files_with_video)
    cache.set("file_video_record", file_video_record)
    cache.set("start_time", 0)
    cache.set("video_start_time", 0)
    cache.set("video_name", "")
    cache.set("video_path", "")
    cache.set("modified_sleep_scores", None)
    cache.set("annotation_history", deque(maxlen=3))
    cache.set("fig_resampler", None)


# %% client side callbacks below

# switch_mode by pressing "m"
app.clientside_callback(
    """
    function(keyboard_nevents, keyboard_event, figure) {
        if (!keyboard_event || !figure) {
            return dash_clientside.no_update;
        }

        var key = keyboard_event.key;

        if (key === "m" || key === "M") {
            let updatedFigure = JSON.parse(JSON.stringify(figure));
            if (figure.layout.dragmode === "pan") {
                updatedFigure.layout.dragmode = "select"
            } else if (figure.layout.dragmode === "select") {
                var selections = figure.layout.selections;
                if (selections) {
                    if (selections.length > 0) {
                        updatedFigure.layout.selections = [];  // Remove the first selection (equivalent to pop(0) in Python)
                    }
                }
                updatedFigure.layout.dragmode = "pan"
            }
            return updatedFigure;
        }

        return dash_clientside.no_update;
    }
    """,
    Output("graph", "figure"),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("graph", "figure"),
)

# pan_figures
clientside_callback(
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
        return components.pred_upload_box, {"display": "block"}
    else:
        return components.vis_upload_box, {"display": "none"}


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
            components.pred_upload_box,
        )

    emg = mat.get("emg")
    if emg is None:
        return (
            html.Div(["EMG data is missing. Please double check the file selected."]),
            dash.no_update,
            components.pred_upload_box,
        )

    reset_cache(cache, filename)
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
    return (html.Div([message]), True, components.pred_upload_box)


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

    reset_cache(cache, filename)

    return (
        html.Div(
            [
                "File validated. Creating visualizations... This may take up to 30 seconds."
            ]
        ),
        True,
        components.vis_upload_box,
    )


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("visualization-ready-store", "data", allow_duplicate=True),
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
    # it is necessary to set cache again here because the output file
    # which includes prediction has a new name (old_name + "_sdreamer"),
    # it is this file that should be used for the subsequent visualization.
    reset_cache(cache, os.path.basename(output_path))
    return (
        html.Div(["The prediction has been generated successfully."]),
        True,
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
    video_start_time = mat.get("video_start_time")
    video_path = mat.get("video_path", [])
    video_name = mat.get("video_name", [])
    if start_time is not None:
        start_time = start_time.item()
        cache.set("start_time", start_time)
    else:
        start_time = 0

    if video_start_time is not None:
        video_start_time = video_start_time.item()
        cache.set("video_start_time", video_start_time)

    if video_path:
        video_path = video_path.item()
        cache.set("video_path", video_path)
    if video_name:
        video_name = video_name.item()
        cache.set("video_name", video_name)
    # eeg_frequency = mat.get("eeg_frequency").item()
    # eeg = mat.get("eeg").flatten()
    cache.set("fig_resampler", fig)
    components.graph.figure = fig
    return components.visualization_div


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("n-sample-dropdown", "value"),
    prevent_initial_call=True,
)
def change_sampling_level(sampling_level):
    if sampling_level is None:
        return dash.no_update
    sampling_level_map = {"x1": 2000, "x2": 4000, "x4": 8000}
    n_samples = sampling_level_map[sampling_level]
    mat_name = cache.get("filename")
    mat = loadmat(os.path.join(TEMP_PATH, mat_name))

    # copy modified (through annotation) sleep scores over
    modified_sleep_scores = cache.get("modified_sleep_scores")
    if modified_sleep_scores is not None:
        if mat.get("pred_labels") is not None and mat["pred_labels"].size != 0:
            mat["pred_labels"] = modified_sleep_scores.copy()
        else:
            mat["sleep_scores"] = modified_sleep_scores.copy()

    fig = create_fig(mat, mat_name, default_n_shown_samples=n_samples)
    return fig


@app.callback(
    Output("video-modal", "is_open"),
    Output("video-path-store", "data", allow_duplicate=True),
    Output("video-container", "children", allow_duplicate=True),
    Output("video-message", "children", allow_duplicate=True),
    Input("video-button", "n_clicks"),
    State("video-modal", "is_open"),
    prevent_initial_call=True,
)
def prepare_video(n_clicks, is_open):
    file_unseen = True
    filename = cache.get("filename")
    recent_files_with_video = cache.get("recent_files_with_video")
    file_video_record = cache.get("file_video_record")
    if filename in recent_files_with_video:
        recent_files_with_video.remove(filename)
        video_info = file_video_record.get(filename)
        if video_info is not None and os.path.isfile(video_info["video_path"]):
            file_unseen = False

    recent_files_with_video.append(filename)
    cache.set("recent_files_with_video", recent_files_with_video)
    if not file_unseen:
        video_path = video_info["video_path"]
        message = "Preparing clip..."
        return (not is_open), video_path, "", message

    # if original avi has not been uploaded, ask for it
    video_path = cache.get("video_path")
    message = "Please upload the original video (an avi file) above."
    if video_path:
        message += f" You may find it at {video_path}."
    return (not is_open), dash.no_update, components.video_upload_box, message


@du.callback(
    output=[
        Output("video-path-store", "data"),
        Output("video-message", "children", allow_duplicate=True),
    ],
    id="video-upload",
)
def upload_video(status):
    avi_path = status.latest_file  # a WindowsPath
    avi_path = str(avi_path)  # need to turn WindowsPath to str for the output
    filename = cache.get("filename")
    recent_files_with_video = cache.get("recent_files_with_video")
    file_video_record = cache.get("file_video_record")
    file_video_record[filename] = {
        "video_path": avi_path,
        "video_name": os.path.basename(avi_path),
    }
    if len(recent_files_with_video) > 3:
        filename_to_remove = recent_files_with_video.pop(0)
        if filename_to_remove in file_video_record:
            avi_file_to_remove = file_video_record[filename_to_remove]["video_path"]
            file_video_record.pop(filename_to_remove)
            if os.path.isfile(avi_file_to_remove):
                os.remove(avi_file_to_remove)

    cache.set("recent_files_with_video", recent_files_with_video)
    cache.set("file_video_record", file_video_record)

    return avi_path, "Preparing clip..."


@app.callback(
    # Output("debug-message", "children"),
    Output("clip-name-store", "data"),
    Output("video-message", "children", allow_duplicate=True),
    Input("video-path-store", "data"),
    State("box-select-store", "data"),
    prevent_initial_call=True,
)
def make_clip(video_path, box_select_range):
    if not box_select_range:
        return dash.no_update, ""

    start, end = box_select_range
    video_start_time = cache.get("video_start_time")
    # start_time = cache.get("start_time")
    start = start + video_start_time
    end = end + video_start_time
    video_name = os.path.basename(video_path).split(".")[0]
    clip_name = video_name + f"_time_range_{start}-{end}" + ".mp4"
    if os.path.isfile(os.path.join(VIDEO_DIR, clip_name)):
        return clip_name, ""

    for file in os.listdir(VIDEO_DIR):
        if file.endswith(".mp4"):
            os.remove(os.path.join(VIDEO_DIR, file))

    save_path = os.path.join(VIDEO_DIR, clip_name)
    try:
        avi_to_mp4(
            video_path,
            start_time=start,
            end_time=end,
            save_path=save_path,
            # save_dir=VIDEO_DIR,
        )
    except ValueError as error_message:
        return dash.no_update, repr(error_message)

    return clip_name, ""


@app.callback(
    Output("video-container", "children"),
    Output("video-message", "children"),
    Input("clip-name-store", "data"),
    prevent_initial_call=True,
)
def show_clip(clip_name):
    clip_path = os.path.join("/assets/videos/", clip_name)
    if not os.path.isfile(os.path.join(VIDEO_DIR, clip_name)):
        return "", "Video not ready yet. Please check again in a second."
    player = dash_player.DashPlayer(
        id="player",
        url=clip_path,
        controls=True,
        width="100%",
        height="100%",
    )

    return player, ""


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("graph", "relayoutData"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata):
    fig = cache.get("fig_resampler")
    if fig is None:
        return dash.no_update

    # manually supply xaxis4.range[0] and xaxis4.range[1] after clicking
    # reset axes button because it only gives xaxis4.range. It seems
    # updating fig_resampler requires xaxis4.range[0] and xaxis4.range[1]
    if (
        relayoutdata.get("xaxis4.range") is not None
        and relayoutdata.get("xaxis4.range[0]") is None
    ):
        relayoutdata["xaxis4.range[0]"], relayoutdata["xaxis4.range[1]"] = relayoutdata[
            "xaxis4.range"
        ]
    return fig.construct_update_data_patch(relayoutdata)


@app.callback(
    Output("box-select-store", "data"),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("video-button", "style"),
    Input("graph", "selectedData"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def read_box_select(box_select, figure):
    video_button_style = {"display": "none"}
    selections = figure["layout"].get("selections")
    if not selections:
        return [], dash.no_update, "", video_button_style

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
    eeg_duration = len(figure["data"][-1]["z"][0])
    eeg_start_time = cache.get("start_time")
    eeg_end_time = eeg_start_time + eeg_duration

    if end < eeg_start_time or start > eeg_end_time:
        return [], patched_figure, "", video_button_style

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
    if 1 <= end - start <= 300:
        video_button_style = {"display": "block"}

    return (
        [start, end],
        patched_figure,
        f"You selected [{start}, {end}]. Press 1 for Wake, 2 for NREM, or 3 for REM.",
        video_button_style,
    )


"""
@app.callback(
    Output("debug-message", "children"),
    Input("box-select-store", "data"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def debug_box_select(box_select, figure):
    #time_end = figure["data"][-1]["z"][0][-1]
    return json.dumps(box_select, indent=2)
"""


@app.callback(
    # Output("debug-message", "children"),
    Output("box-select-store", "data", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("video-button", "style", allow_duplicate=True),
    Input("graph", "clickData"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def read_click(clickData, figure):
    video_button_style = {"display": "none"}
    if clickData is None:
        "", [], "", video_button_style

    dragmode = figure["layout"]["dragmode"]
    if dragmode == "pan":
        raise dash.exceptions.PreventUpdate

    # Grab clicked x value
    x_click = clickData["points"][0]["x"]

    # Determine current x-axis visible range
    x_min, x_max = figure["layout"]["xaxis4"]["range"]
    total_range = x_max - x_min

    # Decide neighborhood size: e.g., 1% of current view range
    fraction = 0.005  # 1% (adjustable)
    delta = total_range * fraction
    eeg_duration = len(figure["data"][-1]["z"][0])
    eeg_start_time = cache.get("start_time")
    eeg_end_time = eeg_start_time + eeg_duration
    x0 = max(math.floor(x_click - delta / 2), eeg_start_time)
    x1 = min(math.ceil(x_click + delta / 2), eeg_end_time)
    if x0 > x1:
        return "", [], "", video_button_style

    return (
        # f"Precise range: [{x0}, {x1}]",
        [x0, x1],
        f"You selected [{x0}, {x1}]. Press 1 for Wake, 2 for NREM, or 3 for REM.",
        video_button_style,
    )


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
    prev_labels = figure["data"][-1]["z"][0][start:end]
    figure["data"][-1]["z"][0][start:end] = [label] * (end - start)

    patched_figure["data"][-3]["z"][0] = figure["data"][-1]["z"][0]
    patched_figure["data"][-2]["z"][0] = figure["data"][-1]["z"][0]
    patched_figure["data"][-1]["z"][0] = figure["data"][-1]["z"][0]
    # remove box select after an update is made
    patched_figure["layout"]["selections"].clear()

    return patched_figure, (start, end, prev_labels)


@app.callback(
    Output("undo-button", "style"),
    Input("annotation-store", "data"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def write_annotation(annotation, figure):
    """write to annotation history, update mat in cache, and make undo button availabe"""
    start, end, prev_labels = annotation
    annotation_history = cache.get("annotation_history")
    annotation_history.append(
        (
            start,
            end,
            prev_labels,  # previous prediction
        )
    )
    labels = np.array(figure["data"][-1]["z"]).astype(float)
    cache.set("annotation_history", annotation_history)
    cache.set("modified_sleep_scores", labels)
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
    (start, end, prev_labels) = prev_annotation
    prev_labels = np.array(prev_labels)

    patched_figure = Patch()
    # undo figure
    figure["data"][-1]["z"][0][start:end] = prev_labels

    patched_figure["data"][-3]["z"][0] = figure["data"][-1]["z"][0]
    patched_figure["data"][-2]["z"][0] = figure["data"][-1]["z"][0]
    patched_figure["data"][-1]["z"][0] = figure["data"][-1]["z"][0]

    # undo cache
    modified_sleep_scores = cache.get("modified_sleep_scores")
    modified_sleep_scores[0, start:end] = prev_labels
    cache.set("modified_sleep_scores", modified_sleep_scores)

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
    app.run_server(debug=False, port=PORT, dev_tools_hot_reload=False)
