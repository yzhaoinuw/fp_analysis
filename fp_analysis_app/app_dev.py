# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:45:29 2023

@author: yzhao
"""

import os
import math
import tempfile
from pathlib import Path
from collections import deque

import dash
import webview
import diskcache
from flask_caching import Cache

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash_extensions.pages import setup_page_components
from dash import (
    Dash,
    dcc,
    html,
    clientside_callback,
    page_container,
    ALL,
    DiskcacheManager,
)

import numpy as np
from scipy.io import loadmat

from fp_analysis_app import VERSION
from fp_analysis_app.components_dev import Components
from fp_analysis_app.make_figure import get_padded_labels, make_figure
from fp_analysis_app.event_analysis import Event_Utils, Perievent_Plots, Analyses


app = Dash(
    __name__,
    title=f"FP Visualization App {VERSION}",
    suppress_callback_exceptions=True,
    use_pages=True,
)

TEMP_PATH = os.path.join(tempfile.gettempdir(), "fp_visualization_app_data")
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)


VIDEO_DIR = Path(__file__).parent / "assets" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR = Path(__file__).parent / "assets" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SPREADSHEET_DIR = Path(__file__).parent / "assets" / "spreadsheets"
SPREADSHEET_DIR.mkdir(parents=True, exist_ok=True)

components = Components()
app.layout = html.Div(
    [
        page_container,  # page layout is rendered here
        setup_page_components(),  # page components are rendered here
    ]
)


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

background_callback_cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(background_callback_cache)

# %%


def create_fig(mat, mat_name, label_dict={}, default_n_shown_samples=2048):
    fig = make_figure(
        mat,
        mat_name,
        label_dict=label_dict,
        default_n_shown_samples=default_n_shown_samples,
    )
    return fig


def open_mat_dialog():
    """
    Open a native file dialog (pywebview) that ONLY shows .mat files.
    Returns a single file path as a string, or None if canceled.
    """
    if not webview.windows:
        return None

    window = webview.windows[0]

    # A single filter for .mat files
    file_types = ("MAT files (*.mat)",)

    result = window.create_file_dialog(
        webview.FileDialog.OPEN,
        allow_multiple=False,
        file_types=file_types,
    )

    if not result:
        return None

    return result[0]  # return the selected file as a single string


def open_annotation_file_dialog():
    if not webview.windows:
        return None

    window = webview.windows[0]

    # A single filter for .mat files
    file_types = ("Spreadsheets (*.xlsx;*.csv)",)

    result = window.create_file_dialog(
        webview.FileDialog.OPEN,
        allow_multiple=False,
        file_types=file_types,
    )

    if not result:
        return None

    return result[0]  # return the selected file as a single string


def make_analysis_plots(
    event_time_dict: dict,
    selected_signals: tuple[str, ...],
    baseline_window: int,
    analysis_window: int,
    duration: float | None = None,
):
    filepath = cache.get("filepath")
    mat_name = os.path.splitext(os.path.basename(filepath))[0]
    fp_data = loadmat(filepath, squeeze_me=True)
    fp_freq = float(fp_data["fp_frequency"])

    # Build helpers
    event_utils = Event_Utils(
        fp_freq, duration, nsec_before=baseline_window, nsec_after=analysis_window
    )
    analyses = Analyses(fp_freq=fp_freq, baseline_window=baseline_window)

    # Indices for this event
    # event_time_dict = event_utils.read_events(event_file=annotation_file)
    perievent_signals_fig_paths = {}
    analyses_fig_paths = {}
    corr_fig_paths = {}

    for i, event in enumerate(sorted(event_time_dict.keys())):

        event_time = event_time_dict[event]
        perievent_windows = event_utils.make_perievent_windows(event_time)
        perievent_indices = event_utils.get_perievent_indices(perievent_windows)

        perievent_signals_dict = {}
        perievent_analysis_dict = {}
        perievent_signals_normalized_array = []

        for sig in selected_signals:
            biosignal = fp_data[sig]
            perievent_signals = biosignal[perievent_indices]
            perievent_signals_dict[sig] = perievent_signals
            result = analyses.get_perievent_analyses(perievent_signals)
            perievent_analysis_dict[sig] = result
            perievent_signals_normalized_array.append(
                result["perievent_signals_normalized"]
            )

        # Plot/save
        plots = Perievent_Plots(
            fp_freq, event, nsec_before=baseline_window, nsec_after=analysis_window
        )

        # NOTE: use distinct paths for the two figures (you were reusing the same var)
        perievent_signals_fig_save_path = (
            FIGURE_DIR
            / f"{mat_name}_{event}_bw{baseline_window}_aw{analysis_window}.png"
        )
        analyses_fig_save_path = (
            FIGURE_DIR
            / f"{mat_name}_{event}_analyses_bw{baseline_window}_aw{analysis_window}.png"
        )
        spreadsheet_save_path = (
            SPREADSHEET_DIR
            / f"{mat_name}_{event}_bw{baseline_window}_aw{analysis_window}.xlsx"
        )
        perievent_signals_fig_paths[event] = os.path.join(
            "/assets/figures/",
            f"{mat_name}_{event}_bw{baseline_window}_aw{analysis_window}.png",
        )
        analyses_fig_paths[event] = os.path.join(
            "/assets/figures/",
            f"{mat_name}_{event}_analyses_bw{baseline_window}_aw{analysis_window}.png",
        )
        plots.make_perievent_plots(
            perievent_signals_dict, figure_save_path=perievent_signals_fig_save_path
        )
        plots.make_perievent_analysis_plots(
            perievent_analysis_dict, figure_save_path=analyses_fig_save_path
        )

        corr_path = None
        if len(perievent_signals_normalized_array) == 2:
            corr_path = (
                FIGURE_DIR
                / f"{mat_name}_{event}_correlation_bw{baseline_window}_aw{analysis_window}.png"
            )
            plots.plot_correlation(
                perievent_signals_normalized_array[0],
                perievent_signals_normalized_array[1],
                figure_save_path=corr_path,
            )
        corr_fig_paths[event] = os.path.join(
            "/assets/figures/",
            f"{mat_name}_{event}_correlation_bw{baseline_window}_aw{analysis_window}.png",
        )
        plots.write_spreadsheet(perievent_analysis_dict, spreadsheet_save_path)
    return (
        perievent_signals_fig_paths,
        analyses_fig_paths,
        corr_fig_paths,
    )


def reset_cache(cache, filepath):
    # prev_filepath = cache.get("filepath")

    # attempt for salvaging unsaved annotations
    # if prev_filepath is None or prev_filepath != filepath:
    cache.set("labels_history", deque(maxlen=4))
    cache.set("filepath", filepath)
    # cache.set("annotation_filepath", "")
    cache.set("event_time_dict", {})
    # cache.set("analysis_fig", None)
    cache.set("start_time", 0)
    cache.set("end_time", 0)
    cache.set("duration", 0)
    cache.set("fig_resampler", None)


# %% client side callbacks below

# switch_mode by pressing "m"
app.clientside_callback(
    """
    function(keyboard_nevents, keyboard_event, figure) {
        if (!keyboard_event || !figure) {
            return [dash_clientside.no_update, dash_clientside.no_update];
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
            return [updatedFigure, {"visibility": "hidden"}];
        }

        return [dash_clientside.no_update, dash_clientside.no_update];
    }
    """,
    Output("graph", "figure"),
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
            return [5];
        }
        return [dash_clientside.no_update];
    }
    """,
    Output("interval-component", "max_intervals"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)


# %% server side callbacks below
@app.callback(
    Output({"type": "tab", "event": ALL}, "children"),
    Input("show-results-button", "n_clicks"),
    State("signal-select-dropdown", "value"),
    State("baseline-window-dropdown", "value"),
    State("analysis-window-dropdown", "value"),
    State({"type": "tab", "event": ALL}, "id"),
    background=True,
    manager=background_callback_manager,
    running=[
        (Output("show-results-button", "disabled"), True, False),
    ],
    prevent_initial_call=False,
)
def show_analysis_results(
    n_clicks,
    selected_signals,
    baseline_window,
    analysis_window,
    tabs,
):
    if not n_clicks:  # None or 0 → do nothing
        raise PreventUpdate
    if not selected_signals:
        raise PreventUpdate

    for file in FIGURE_DIR.iterdir():
        if file.is_file() and file.suffix == ".png":
            file.unlink()

    for file in SPREADSHEET_DIR.iterdir():
        if file.is_file() and file.suffix == ".xlsx":
            file.unlink()

    # annotation_filepath = cache.get("annotation_filepath")
    event_time_dict = cache.get("event_time_dict")
    duration = cache.get("duration")
    perievent_signals_fig_paths, analyses_fig_paths, corr_fig_paths = (
        make_analysis_plots(
            event_time_dict=event_time_dict,
            selected_signals=selected_signals,
            baseline_window=baseline_window,
            analysis_window=analysis_window,
            duration=duration,
        )
    )
    # Build outputs aligned to each pattern’s IDs
    tab_children = []
    for tab in tabs:
        event = tab["event"]
        children = components._fill_tab(
            event, perievent_signals_fig_paths, analyses_fig_paths, corr_fig_paths
        )
        tab_children.append(children)

    return tab_children


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("visualization-ready-store", "data", allow_duplicate=True),
    Output("upload-container", "children", allow_duplicate=True),
    Output("analysis-link", "style", allow_duplicate=True),
    Input("vis-data-upload-button", "n_clicks"),
    prevent_initial_call=True,
)
def choose_mat(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    selected_file_path = open_mat_dialog()
    if selected_file_path is None:
        raise PreventUpdate  # user canceled dialog

    reset_cache(cache, selected_file_path)
    message = (
        "File uploaded. Creating visualizations... This may take up to 30 seconds."
    )
    return message, True, components.vis_upload_button, {"visibility": "hidden"}


@app.callback(
    Output("annotation-uploaded-store", "data"),
    Input("load-annotations-button", "n_clicks"),
    prevent_initial_call=True,
)
def choose_annotation(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    selected_file_path = open_annotation_file_dialog()
    if selected_file_path is None:
        raise PreventUpdate  # user canceled dialog

    return selected_file_path


@app.callback(
    Output("analysis-page", "children"),
    Output("graph", "figure", allow_duplicate=True),
    Output("analysis-link", "style"),
    Input("annotation-uploaded-store", "data"),
    prevent_initial_call=True,
)
def import_annotation_file(annotation_filepath):
    mat_path = cache.get("filepath")
    mat = loadmat(mat_path, squeeze_me=True)
    signal_names = mat.get("fp_signal_names")
    fp_freq = mat.get("fp_frequency")
    duration = cache.get("duration")
    event_utils = Event_Utils(fp_freq, duration)
    event_time_dict = event_utils.read_events(event_file=annotation_filepath)
    cache.set("event_time_dict", event_time_dict)
    event_count_records = event_utils.count_events(event_time_dict)
    event_names = list(event_time_dict.keys())
    analysis_page_content = components.fill_analysis_page(
        event_names, event_count_records, signal_names
    )
    perievent_label_dict = event_utils.make_perievent_labels(
        event_file=annotation_filepath
    )
    fig = create_fig(mat, os.path.basename(mat_path), label_dict=perievent_label_dict)
    return analysis_page_content, fig, {"visibility": "visible"}


@app.callback(
    Output("visualization-container", "children"),
    Output("num-signals-store", "data"),
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("analysis-page", "children", allow_duplicate=True),
    Output("analysis-link", "style", allow_duplicate=True),
    Input("visualization-ready-store", "data"),
    prevent_initial_call=True,
)
def create_visualization(ready):
    if not ready:
        raise PreventUpdate

    filepath = cache.get("filepath")
    mat_name = os.path.splitext(os.path.basename(filepath))[0]
    mat = loadmat(filepath, squeeze_me=True)
    fp_signal_names = mat["fp_signal_names"]
    num_signals = len(fp_signal_names)
    fp_freq = mat.get("fp_frequency")
    # duration = cache.get("duration")
    event_data = mat.get("event")

    label_dict = {}
    analysis_page_content = dash.no_update
    analysis_link_style = {"visibility": "hidden"}

    message = "Please double check the file selected."
    if num_signals == 0:
        message = " ".join(["No FP signal found.", message])
        return message, dash.no_update, "", analysis_page_content, analysis_link_style

    fp_signals = [mat[signal_name] for signal_name in fp_signal_names]
    signal_lengths = [len(fp_signals[k]) for k in range(num_signals)]
    if not all(length == signal_lengths[0] for length in signal_lengths):
        message = " ".join(["Not all FP signals are of the same length.", message])
        return message, dash.no_update, "", analysis_page_content, analysis_link_style

    signal_length = signal_lengths[0]
    duration = math.ceil(
        (signal_length - 1) / fp_freq
    )  # need to round duration to an int for later

    if event_data is not None:
        signal_names = mat.get("fp_signal_names")
        event_utils = Event_Utils(fp_freq, duration)
        df_events = event_utils.eventdata_to_df(event_data)
        event_time_dict = event_utils.read_events(df_events=df_events)
        cache.set("event_time_dict", event_time_dict)
        event_count_records = event_utils.count_events(event_time_dict)
        event_names = list(event_time_dict.keys())
        analysis_page_content = components.fill_analysis_page(
            event_names, event_count_records, signal_names
        )
        label_dict = event_utils.make_perievent_labels(df_events=df_events)
        analysis_link_style = {"visibility": "visible"}

    # salvage unsaved annotations
    labels_history = cache.get("labels_history")
    if labels_history:
        mat["labels"] = labels_history[-1]
    else:
        labels = mat.get("labels", np.array([]))
        labels = get_padded_labels(labels, duration)
        np.place(labels, labels == -1, [np.nan])
        labels_history.append(labels)

    fig = create_fig(mat, mat_name, label_dict=label_dict)
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

    return (
        visualization_div,
        num_signals,
        "",
        analysis_page_content,
        analysis_link_style,
    )


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
    filepath = cache.get("filepath")
    mat_name = os.path.splitext(os.path.basename(filepath))[0]
    mat = loadmat(filepath, squeeze_me=True)

    # copy modified (through annotation) sleep scores over
    labels_history = cache.get("labels_history")
    if labels_history:
        mat["labels"] = labels_history[-1]

    fig = create_fig(mat, mat_name, default_n_shown_samples=n_samples)
    return fig


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
