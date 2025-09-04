# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:45:29 2023

@author: yzhao
"""

import os
import math

import tempfile
import webbrowser
from pathlib import Path
from collections import deque

import dash
import dash_player
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash_extensions.pages import setup_page_components
from dash import Dash, dcc, html, ctx, clientside_callback, Patch, page_container, ALL

import numpy as np
import pandas as pd
from flask_caching import Cache
from scipy.io import loadmat, savemat


from app_src import VERSION
from app_src.make_mp4 import make_mp4_clip
from app_src.components import Components
from app_src.make_figure import get_padded_labels, make_figure
from app_src.event_analysis import Event_Utils, Perievent_Plots, Analyses
from app_src.postprocessing import get_sleep_segments, get_pred_label_stats


app = Dash(
    __name__,
    title=f"FP Visualization App {VERSION}",
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP
    ],  # need this for the modal to work properly
    use_pages=True,
)

TEMP_PATH = os.path.join(tempfile.gettempdir(), "fp_visualization_app_data")
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)


VIDEO_DIR = Path(__file__).parent / "assets" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR = Path(__file__).parent / "assets" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

components = Components()
app.layout = html.Div(
    [
        page_container,  # page layout is rendered here
        setup_page_components(),  # page components are rendered here
    ]
)

du = components.configure_du(app, TEMP_PATH)

# Note: np.nan is converted to None when reading from cache
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


def create_fig(mat, mat_name, label_dict={}, default_n_shown_samples=2048):
    fig = make_figure(
        mat,
        mat_name,
        label_dict=label_dict,
        default_n_shown_samples=default_n_shown_samples,
    )
    return fig


def make_analysis_plots(
    mat_file, annotation_file, biosignal_name, nsec_before=60, nsec_after=60
):
    event_subplots = Perievent_Plots.make_perievent_plots(
        fp_file=mat_file,
        biosignal_name=biosignal_name,
        event_file=annotation_file,
        nsec_before=nsec_before,
        nsec_after=nsec_after,
        as_base64=True,
    )
    return event_subplots


def reset_cache(cache, filename):
    prev_filename = cache.get("filename")

    # attempt for salvaging unsaved annotations
    # if prev_filename is None or prev_filename != filename:
    cache.set("labels_history", deque(maxlen=4))

    cache.set("filename", filename)
    recent_files_with_video = cache.get("recent_files_with_video")
    if recent_files_with_video is None:
        recent_files_with_video = []
    file_video_record = cache.get("file_video_record")
    if file_video_record is None:
        file_video_record = {}
    cache.set("annotation_filename", "")
    cache.set("analysis_fig", None)
    cache.set("recent_files_with_video", recent_files_with_video)
    cache.set("file_video_record", file_video_record)
    cache.set("start_time", 0)
    cache.set("end_time", 0)
    cache.set("duration", 0)
    cache.set("video_start_time", 0)
    cache.set("video_name", "")
    cache.set("video_path", "")
    cache.set("fig_resampler", None)


# %% client side callbacks below

# switch_mode by pressing "m"
app.clientside_callback(
    """
    function(keyboard_nevents, keyboard_event, figure) {
        if (!keyboard_event || !figure) {
            return [dash_clientside.no_update, dash_clientside.no_update, dash_clientside.no_update];
        }

        var key = keyboard_event.key;

        if (key === "m" || key === "M") {
            let updatedFigure = JSON.parse(JSON.stringify(figure));
            if (figure.layout.dragmode === "pan") {
                updatedFigure.layout.dragmode = "select"
            } else if (figure.layout.dragmode === "select") {
                updatedFigure.layout.selections = null;
                updatedFigure.layout.shapes = null;
                updatedFigure.layout.dragmode = "pan"
            }
            return [updatedFigure, "", {"visibility": "hidden"}];
        }

        return [dash_clientside.no_update, dash_clientside.no_update, dash_clientside.no_update];
    }
    """,
    Output("graph", "figure"),
    Output("annotation-message", "children"),
    Output("video-button", "style"),
    # Output("pred-button", "style"),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("graph", "figure"),
)

# pan_figures
clientside_callback(
    """
    function(keyboard_nevents, keyboard_event, relayoutdata, figure, num_signals) {
        if (!keyboard_event || !figure) {
            return [dash_clientside.no_update, dash_clientside.no_update];
        }

        var key = keyboard_event.key;
        var axisK = 'xaxis' + num_signals;
        var xaxisRange = figure.layout[axisK].range;
        var x0 = xaxisRange[0];
        var x1 = xaxisRange[1];
        var newRange;

        if (key === "ArrowRight") {
            newRange = [x0 + (x1 - x0) * 0.3, x1 + (x1 - x0) * 0.3];
        } else if (key === "ArrowLeft") {
            newRange = [x0 - (x1 - x0) * 0.3, x1 - (x1 - x0) * 0.3];
        }

        if (newRange) {
            relayoutdata[axisK + '.range[0]'] = newRange[0];
            relayoutdata[axisK + '.range[1]'] = newRange[1];
            figure.layout[axisK].range = newRange;
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
    State("num-signals-store", "data"),
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
    Output({"type": "analysis-image", "event": ALL}, "src"),
    Output({"type": "save-plots-button", "event": ALL}, "style"),
    Input("show-results-button", "n_clicks"),
    State("signal-select-dropdown", "value"),
    State("perievent-window-dropdown", "value"),
    State({"type": "analysis-image", "event": ALL}, "id"),
    State({"type": "save-plots-button", "event": ALL}, "id"),
    prevent_initial_call=False,
)
def show_analysis_results(
    n_clicks, selected_signals, perievent_window, img_ids, btn_ids
):
    if selected_signals is None:
        return PreventUpdate()

    mat_name = cache.get("filename")
    baseline_window = 30
    fp_name = Path(mat_name).stem
    biosignal_name = "NE2m"
    annotation_filename = cache.get("annotation_filename")
    annotation_file = os.path.join(TEMP_PATH, annotation_filename)
    mat_file = os.path.join(TEMP_PATH, mat_name)
    fp_data = loadmat(mat_file, squeeze_me=True)
    biosignal_names = fp_data["fp_signal_names"]
    biosignal = fp_data[biosignal_name]
    fp_freq = fp_data["fp_frequency"]
    duration = cache.get("duration")
    event_utils = Event_Utils(fp_freq, duration, window_len=perievent_window)
    event_time_dict = event_utils.read_events(annotation_file)
    analyses = Analyses(fp_freq=fp_freq, baseline_window=baseline_window)
    figs = {}
    fig_paths = {}
    for i, event in enumerate(sorted(event_time_dict.keys())):

        event_time = event_time_dict[event]
        perievent_windows = event_utils.make_perievent_windows(event_time)
        perievent_indices = event_utils.get_perievent_indices(perievent_windows)
        perievent_analysis_result = {}
        for biosignal_name in biosignal_names:
            biosignal = fp_data[biosignal_name]
            perievent_signals = biosignal[perievent_indices]
            perievent_analysis_result[biosignal_name] = analyses.get_perievent_analyses(
                perievent_signals
            )

        figure_name = f"{mat_name}_{event}.png"
        figure_save_path = FIGURE_DIR / figure_name
        plots = Perievent_Plots(
            fp_freq, event, fp_name, biosignal_names, perievent_window
        )
        fig = plots.make_perievent_analysis_plots(
            perievent_analysis_result, figure_save_path=figure_save_path
        )
        figs[event] = fig
        fig_paths[event] = os.path.join("/assets/figures/", figure_name)

    cache.set("analysis_fig", figs)
    # Build outputs aligned to each pattern’s IDs
    figure_urls = [fig_paths.get(img_id["event"]) for img_id in img_ids]
    styles = [{"visibility": "visible"} for _ in btn_ids]

    return figure_urls, styles


"""
@app.callback(
    Output({"type": "analysis-image", "event": ALL}, "src"),
    Output({"type": "save-plots-button", "event": ALL}, "style"),
    Input("page-url", "pathname"),
    State({"type": "analysis-image", "event": ALL}, "id"),
    State("perievent-window-dropdown", "value"),
    State({"type": "save-plots-button", "event": ALL}, "id"),
)
def navigate_pages(pathname, img_ids, perievent_window, btn_ids):
    if pathname != "/analysis":
        raise PreventUpdate
    
    mat_name = cache.get("filename")
    window_len = 120
    baseline_window = 30
    fp_name = Path(mat_name).stem
    biosignal_name = "NE2m"
    annotation_filename = cache.get("annotation_filename")
    annotation_file = os.path.join(TEMP_PATH, annotation_filename)
    mat_file = os.path.join(TEMP_PATH, mat_name)
    fp_data = loadmat(mat_file, squeeze_me=True)
    # biosignal_names = fp_data["fp_signal_names"]
    biosignal = fp_data[biosignal_name]
    fp_freq = fp_data["fp_frequency"]
    duration = cache.get("duration")
    event_utils = Event_Utils(fp_freq, duration, window_len=window_len)
    event_time_dict = event_utils.read_events(annotation_file)
    analyses = Analyses(fp_freq=fp_freq, baseline_window=baseline_window)
    figs = {}
    fig_paths = {}
    for i, event in enumerate(sorted(event_time_dict.keys())):

        event_time = event_time_dict[event]
        perievent_windows = event_utils.make_perievent_windows(event_time)
        perievent_indices = event_utils.get_perievent_indices(perievent_windows)
        perievent_signals = biosignal[perievent_indices]
        figure_name = f"{mat_name}_{biosignal_name}_{event}.png"
        figure_save_path = FIGURE_DIR / figure_name
        perievent_analysis_result = analyses.get_perievent_analyses(perievent_signals)
        plots = Perievent_Plots(perievent_signals, fp_freq, event, fp_name, biosignal_name, window_len)
        
        fig = plots.make_perievent_analysis_plots(perievent_analysis_result, figure_save_path=figure_save_path)
        figs[event] = fig
        fig_paths[event] = os.path.join("/assets/figures/", figure_name)

    cache.set("analysis_fig", figs)
    # Build outputs aligned to each pattern’s IDs
    figure_urls = [fig_paths.get(img_id["event"]) for img_id in img_ids]
    styles = [{"visibility": "visible"} for _ in btn_ids]

    return figure_urls, styles
"""


@du.callback(
    output=[
        Output("data-upload-message", "children", allow_duplicate=True),
        Output("visualization-ready-store", "data", allow_duplicate=True),
        Output("upload-container", "children", allow_duplicate=True),
        Output("net-annotation-count-store", "data", allow_duplicate=True),
        Output("annotation-message", "children", allow_duplicate=True),
    ],
    id="vis-data-upload",
)
def upload_mat(status):
    # clean TEMP_PATH regularly by deleting temp files written there
    mat_file = status.latest_file
    filename = os.path.basename(mat_file)
    for temp_file in os.listdir(TEMP_PATH):
        if temp_file.endswith(".mat") or temp_file.endswith(".xlsx"):
            if temp_file == filename:
                continue
            os.remove(os.path.join(TEMP_PATH, temp_file))

    reset_cache(cache, filename)
    message = (
        "File uploaded. Creating visualizations... This may take up to 30 seconds."
    )
    return message, True, components.vis_upload_box, 0, ""


@app.callback(
    Output("video-modal", "is_open", allow_duplicate=True),
    Output("video-container", "children", allow_duplicate=True),
    Output("video-message", "children", allow_duplicate=True),
    Input("load-annotations-button", "n_clicks"),
    State("video-modal", "is_open"),
    prevent_initial_call=True,
)
def prepare_annotation(n_clicks, is_open):
    message = "Please upload the annotation spreadsheet (an xlsx file) above."
    return (not is_open), components.annotation_upload_box, message


@du.callback(
    output=[
        Output("video-message", "children", allow_duplicate=True),
        Output("annotation-uploaded-store", "data"),
    ],
    id="annotation-upload",
)
def upload_annotation(status):
    annotation_file = status.latest_file
    annotation_filename = os.path.basename(annotation_file)
    cache.set("annotation_filename", annotation_filename)
    message = "File uploaded. It may take up to 30 seconds to update the visualizations. You can close this window now."
    return message, "uploaded"


@app.callback(
    Output("analysis-page", "children"),
    Output("graph", "figure", allow_duplicate=True),
    Output("analysis-link", "style"),
    Input("annotation-uploaded-store", "data"),
    prevent_initial_call=True,
)
def import_annotation_file(uploaded):
    mat_name = cache.get("filename")
    mat = loadmat(os.path.join(TEMP_PATH, mat_name), squeeze_me=True)
    annotation_filename = cache.get("annotation_filename")
    annotation_file = os.path.join(TEMP_PATH, annotation_filename)
    signal_names = mat.get("fp_signal_names")
    fp_freq = mat.get("fp_frequency")
    window_len = 120
    duration = cache.get("duration")
    event_utils = Event_Utils(fp_freq, duration, window_len=window_len)
    event_time_dict = event_utils.read_events(annotation_file)
    event_count_records = event_utils.count_events(event_time_dict)
    event_names = list(event_time_dict.keys())
    analysis_page_content = components.fill_analysis_page(
        event_names, event_count_records, signal_names
    )
    perievent_label_dict = event_utils.make_perievent_labels(annotation_file)
    fig = create_fig(mat, mat_name, label_dict=perievent_label_dict)
    return analysis_page_content, fig, {"visibility": "visible"}


@app.callback(
    Output("visualization-container", "children"),
    Output("num-signals-store", "data"),
    Output("data-upload-message", "children", allow_duplicate=True),
    Input("visualization-ready-store", "data"),
    prevent_initial_call=True,
)
def create_visualization(ready):
    if not ready:
        raise PreventUpdate()

    mat_name = cache.get("filename")
    mat = loadmat(os.path.join(TEMP_PATH, mat_name), squeeze_me=True)
    fp_signal_names = mat["fp_signal_names"]
    num_signals = len(fp_signal_names)

    message = "Please double check the file selected."
    if num_signals == 0:
        message = " ".join(["No FP signal found.", message])
        return message, dash.no_update, ""

    fp_signals = [mat[signal_name] for signal_name in fp_signal_names]
    signal_lengths = [len(fp_signals[k]) for k in range(num_signals)]
    if not all(length == signal_lengths[0] for length in signal_lengths):
        message = " ".join(["Not all FP signals are of the same length.", message])
        return message, dash.no_update, ""

    # salvage unsaved annotations
    labels_history = cache.get("labels_history")
    if labels_history:
        mat["labels"] = labels_history[-1]
    else:
        signal_length = signal_lengths[0]
        fp_freq = mat.get("fp_frequency")
        duration = math.ceil(
            (signal_length - 1) / fp_freq
        )  # need to round duration to an int for later
        labels = mat.get("labels", np.array([]))
        labels = get_padded_labels(labels, duration)
        np.place(labels, labels == -1, [np.nan])
        labels_history.append(labels)

    fig = create_fig(mat, mat_name)
    video_start_time = mat.get("video_start_time")
    video_path = mat.get("video_path", np.array([]))
    video_name = mat.get("video_name", np.array([]))
    time_ax = fig["data"][0]["x"]
    start_time, end_time = time_ax[0], time_ax[-1]
    cache.set("start_time", start_time)
    cache.set("end_time", end_time)
    cache.set("duration", duration)
    if video_start_time is not None:
        cache.set("video_start_time", video_start_time)
    if video_path.size != 0:
        video_path = video_path.item()
        cache.set("video_path", video_path)
    if video_name.size != 0:
        video_name = video_name.item()
        cache.set("video_name", video_name)

    cache.set("fig_resampler", fig)
    cache.set("labels_history", labels_history)
    graph = dcc.Graph(id="graph", figure=fig, config={"scrollZoom": True})
    visualization_div = components.make_visualization_div(graph)

    return visualization_div, num_signals, ""


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("n-sample-dropdown", "value"),
    prevent_initial_call=True,
)
def change_sampling_level(sampling_level):
    if sampling_level is None:
        return dash.no_update
    sampling_level_map = {"x1": 2048, "x2": 4096, "x4": 8192}
    n_samples = sampling_level_map[sampling_level]
    mat_name = cache.get("filename")
    mat = loadmat(os.path.join(TEMP_PATH, mat_name), squeeze_me=True)

    # copy modified (through annotation) sleep scores over
    labels_history = cache.get("labels_history")
    if labels_history:
        mat["labels"] = labels_history[-1]

    fig = create_fig(mat, mat_name, default_n_shown_samples=n_samples)
    return fig


@app.callback(
    Output("video-modal", "is_open", allow_duplicate=True),
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
    save_path = VIDEO_DIR / clip_name
    if save_path.is_file():
        return clip_name, ""

    for file in VIDEO_DIR.iterdir():
        if file.is_file() and file.suffix == ".mp4":
            file.unlink()

    try:
        make_mp4_clip(
            video_path,
            start_time=start,
            end_time=end,
            save_path=save_path,
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
    if not (VIDEO_DIR / clip_name).is_file():
        return "", "Video not ready yet. Please check again in a second."
    clip_path = os.path.join("/assets/videos/", clip_name)
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
    State("num-signals-store", "data"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata, num_signals):
    fig = cache.get("fig_resampler")
    if fig is None:
        return dash.no_update

    # manually supply xaxis4.range[0] and xaxis4.range[1] after clicking
    # reset axes button because it only gives xaxis4.range. It seems
    # updating fig_resampler requires xaxis4.range[0] and xaxis4.range[1]
    if (
        relayoutdata.get(f"xaxis{num_signals}.range") is not None
        and relayoutdata.get(f"xaxis{num_signals}.range[0]") is None
    ):
        (
            relayoutdata[f"xaxis{num_signals}.range[0]"],
            relayoutdata[f"xaxis{num_signals}.range[1]"],
        ) = relayoutdata[f"xaxis{num_signals}.range"]
    return fig.construct_update_data_patch(relayoutdata)


@app.callback(
    # Output("debug-message", "children"),
    Output("box-select-store", "data"),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("video-button", "style", allow_duplicate=True),
    Input("graph", "selectedData"),
    State("graph", "figure"),
    State("graph", "clickData"),
    prevent_initial_call=True,
)
def read_box_select(box_select, figure, clickData):
    video_button_style = {"visibility": "hidden"}
    selections = figure["layout"].get("selections")

    # when selections is None, it means there's not box select in the graph
    if selections is None:
        raise PreventUpdate()

    # allow only at most one select box in all subplots
    if len(selections) > 1:
        selections.pop(0)

    patched_figure = Patch()
    patched_figure["layout"][
        "selections"
    ] = selections  # patial property update: https://dash.plotly.com/partial-properties#update
    patched_figure["layout"]["shapes"] = None  # remove click select box

    # take the min as start and max as end so that how the box is drawn doesn't matter
    start, end = min(selections[0]["x0"], selections[0]["x1"]), max(
        selections[0]["x0"], selections[0]["x1"]
    )
    start_time = cache.get("start_time")
    end_time = cache.get("end_time")

    if end < start_time or start > end_time:
        return (
            [],
            patched_figure,
            f"Out of range. Please select from {start_time} to {end_time}.",
            video_button_style,
        )

    start_round, end_round = round(start), round(end)
    start_round = max(start_round, start_time)
    end_round = min(end_round, end_time)
    if start_round == end_round:
        if (
            start_round - start > end - end_round
        ):  # spanning over two consecutive seconds
            end_round = math.ceil(start)
            start_round = math.floor(start)
        else:
            end_round = math.ceil(end)
            start_round = math.floor(end)

    start, end = start_round - start_time, end_round - start_time
    if 1 <= end - start <= 300:
        video_button_style = {"visibility": "visible"}

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
    # Output("debug-message", "children", allow_duplicate=True),
    Output("box-select-store", "data", allow_duplicate=True),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("video-button", "style", allow_duplicate=True),
    Input("graph", "clickData"),
    State("graph", "figure"),
    State("num-signals-store", "data"),
    prevent_initial_call=True,
)
def read_click_select(
    clickData, figure, num_signals
):  # triggered only  if clicked within x-range
    patched_figure = Patch()
    patched_figure["layout"]["shapes"] = None
    video_button_style = {"visibility": "hidden"}
    dragmode = figure["layout"]["dragmode"]
    if clickData is None or dragmode == "pan":
        return [], patched_figure, "", video_button_style

    # remove the select box if present
    patched_figure["layout"]["selections"] = None

    # Grab clicked x value
    x_click = clickData["points"][0]["x"]
    x_click = round(x_click)
    # Determine current x-axis visible range
    x_min, x_max = figure["layout"][f"xaxis{num_signals}"]["range"]
    if x_click < x_min or x_click > x_max:
        return [], patched_figure, "", video_button_style

    total_range = x_max - x_min

    # Decide neighborhood size: e.g., 1% of current view range
    fraction = 0.005  # 0.5% (adjustable)
    delta = total_range * fraction
    start_time = cache.get("start_time")
    end_time = cache.get("end_time")
    x0, x1 = math.floor(x_click - delta / 2), math.ceil(x_click + delta / 2)
    curve_index = clickData["points"][0]["curveNumber"]
    trace = figure["data"][curve_index]
    xref = trace.get("xaxis", f"x{num_signals}")  # x4 is the shared x-axis
    yref = trace.get("yaxis", f"y{num_signals}")  # spectrogram has dual y-axis

    select_box = {
        "type": "rect",
        "xref": xref,
        "yref": yref,
        "x0": x0,
        "x1": x1,
        "y0": -20,
        "y1": 20,
        "line": {"width": 1, "dash": "dot"},
    }

    patched_figure["layout"]["shapes"] = [select_box]
    start = max(x0, start_time)
    end = min(x1, end_time)

    if 1 <= end - start <= 300:
        video_button_style = {"visibility": "visible"}
    return (
        [start, end],
        patched_figure,
        f"You selected [{start}, {end}]. Press 1 for Wake, 2 for NREM, or 3 for REM.",
        video_button_style,
    )


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("video-button", "style", allow_duplicate=True),
    Output("net-annotation-count-store", "data", allow_duplicate=True),
    Input("box-select-store", "data"),
    Input("keyboard", "n_events"),  # a keyboard press
    State("keyboard", "event"),
    State("graph", "figure"),
    State("net-annotation-count-store", "data"),
    State("num-signals-store", "data"),
    prevent_initial_call=True,
)
def add_annotation(
    box_select_range,
    keyboard_press,
    keyboard_event,
    figure,
    net_annotation_count,
    num_signals,
):
    """update sleep scores in fig and annotation history"""
    if not (
        ctx.triggered_id == "keyboard"
        and box_select_range
        and figure["layout"]["dragmode"] == "select"
    ):
        raise PreventUpdate

    label = keyboard_event.get("key")
    if label not in ["1", "2", "3"]:
        raise PreventUpdate

    label = int(label) - 1
    start, end = box_select_range
    labels_history = cache.get("labels_history")
    current_labels = labels_history[-1]  # np array
    new_labels = current_labels.copy()
    new_labels[start:end] = np.array([label] * (end - start))
    # If the annotation does not change anything, don't add to history
    if (new_labels == current_labels).all():
        raise PreventUpdate

    labels_history.append(new_labels.astype(float))
    cache.set("labels_history", labels_history)
    net_annotation_count += 1

    patched_figure = Patch()
    for k in range(1, num_signals + 1):
        patched_figure["data"][-k]["z"][0] = new_labels

    # remove box or click select after an update is made
    patched_figure["layout"]["selections"] = None
    patched_figure["layout"]["shapes"] = None
    return patched_figure, "", {"visibility": "hidden"}, net_annotation_count


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("net-annotation-count-store", "data", allow_duplicate=True),
    Input("undo-button", "n_clicks"),
    State("graph", "figure"),
    State("net-annotation-count-store", "data"),
    prevent_initial_call=True,
)
def undo_annotation(n_clicks, figure, net_annotation_count):
    labels_history = cache.get("labels_history")
    if len(labels_history) <= 1:
        raise PreventUpdate()

    net_annotation_count -= 1
    labels_history.pop()  # pop current one, then get the last one

    # undo cache
    cache.set("labels_history", labels_history)
    prev_labels = labels_history[-1]

    # undo figure
    patched_figure = Patch()
    patched_figure["data"][-3]["z"][0] = prev_labels
    patched_figure["data"][-2]["z"][0] = prev_labels
    patched_figure["data"][-1]["z"][0] = prev_labels
    return patched_figure, net_annotation_count


@app.callback(
    Output("save-button", "style"),
    Output("undo-button", "style"),
    # Output("debug-message", "children"),
    Input("net-annotation-count-store", "data"),
    # State("net-annotation-count-store", "data"),
    prevent_initial_call=True,
)
def show_hide_save_undo_button(net_annotation_count):
    # time.sleep(10)
    labels_history = cache.get("labels_history")
    save_button_style = {"visibility": "hidden"}
    undo_button_style = {"visibility": "hidden"}
    if net_annotation_count > 0:
        save_button_style = {"visibility": "visible"}
        if len(labels_history) > 1:
            undo_button_style = {"visibility": "visible"}
    return save_button_style, undo_button_style


@app.callback(
    Output("download-annotations", "data"),
    Output("download-spreadsheet", "data"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_annotations(n_clicks):
    mat_filename = cache.get("filename")
    temp_mat_path = os.path.join(TEMP_PATH, mat_filename)
    mat = loadmat(temp_mat_path, squeeze_me=True)

    # replace None in labels
    labels_history = cache.get("labels_history")
    labels = None
    if labels_history:
        # replace any None or nan in sleep scores to -1 before saving, otherwise results in save error
        # make a copy first because we don't want to convert any nan in the cache
        updated_labels = labels_history[-1]
        np.place(
            updated_labels, updated_labels == None, [-1]
        )  # convert None to -1 for scipy's savemat
        updated_labels = np.nan_to_num(
            updated_labels, nan=-1
        )  # convert np.nan to -1 for scipy's savemat

        mat["labels"] = updated_labels
    savemat(temp_mat_path, mat)

    # export sleep bout spreadsheet only if the manual scoring is complete
    if mat.get("labels") is not None and -1 not in mat["labels"]:
        labels = mat["labels"]

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


"""
@app.callback(
    Output("download-plots", "data"),
    Input("save-plots-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_plots(n_clicks):
    mat_filename = cache.get("filename")
    mat_filename = mat_filename.rstrip(".mat")
    save_zip_path = os.path.join(TEMP_PATH, f"{mat_filename}_plots.zip")
    fig = cache.get("analysis_fig")
    Perievent_Plots.zip_plots(fig, save_zip_path)
    return dcc.send_file(save_zip_path)
"""

if __name__ == "__main__":
    from threading import Timer
    from functools import partial

    PORT = 8050
    Timer(1, partial(open_browser, PORT)).start()
    app.run(debug=False, port=PORT, dev_tools_hot_reload=False)
