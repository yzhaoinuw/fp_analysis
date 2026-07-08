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
from fp_analysis_app.components_dev import (
    Components,
    get_analysis_signal_select_style,
    is_valid_analysis_signal_selection,
)
from fp_analysis_app.make_figure import get_padded_labels, make_figure
from fp_analysis_app.event_analysis import Event_Utils, Perievent_Plots, Analyses
from fp_analysis_app.analysis_export import (
    build_analysis_type_checklist_options,
    get_analysis_type_checklist_values,
    write_analysis_workbooks,
)
from fp_analysis_app.event_editing import (
    event_time_dict_to_store_data,
    store_data_to_event_time_dict,
)
from fp_analysis_app.mat_utils import (
    get_fp_signal_names,
    get_visualization_signal_data,
    get_visualization_signal_names_and_frequency,
    has_embedded_event_data,
)


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
EXPORT_DOWNSAMPLE_FACTOR = 100
ANALYSIS_EXPORT_PAYLOAD_CACHE_KEY = "analysis_export_payload"
REMEMBERED_ANALYSIS_EXPORT_TYPES_CACHE_KEY = "remembered_analysis_export_types"
SAMPLING_LEVEL_TO_N_SAMPLES = {"x1": 2048, "x2": 4096, "x4": 8192}

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


def create_fig(
    mat,
    mat_name,
    label_dict=None,
    event_time_dict=None,
    show_period_labels=True,
    default_n_shown_samples=2048,
):
    fig = make_figure(
        mat,
        mat_name,
        label_dict=label_dict,
        event_time_dict=event_time_dict,
        show_period_labels=show_period_labels,
        default_n_shown_samples=default_n_shown_samples,
    )
    return fig


def get_cached_runtime_fp_metadata(mat=None):
    signal_names = cache.get("fp_signal_names")
    fp_freq = cache.get("fp_frequency")
    if signal_names and fp_freq is not None:
        return signal_names, float(fp_freq)

    if mat is None:
        filepath = cache.get("filepath")
        mat = loadmat(filepath, squeeze_me=True)

    signal_names, fp_freq = get_visualization_signal_names_and_frequency(mat)
    cache.set("fp_signal_names", signal_names)
    cache.set("fp_frequency", float(fp_freq))
    return signal_names, float(fp_freq)


def get_n_samples_for_sampling_level(sampling_level):
    return SAMPLING_LEVEL_TO_N_SAMPLES.get(sampling_level, 2048)


def build_analysis_page_content(event_time_dict):
    event_time_dict = event_time_dict or {}
    event_count_records = [
        {"Event": event, "Count": len(event_times)}
        for event, event_times in event_time_dict.items()
    ]
    signal_names = cache.get("fp_signal_names") or []
    return components.fill_analysis_page(
        list(event_time_dict.keys()),
        event_count_records,
        signal_names,
    )


def build_current_full_recording_figure(default_n_shown_samples=2048):
    filepath = cache.get("filepath")
    if not filepath:
        return None

    mat_name = os.path.splitext(os.path.basename(filepath))[0]
    mat = loadmat(filepath, squeeze_me=True)
    labels_history = cache.get("labels_history")
    if labels_history:
        mat["labels"] = labels_history[-1]

    fig = create_fig(
        mat,
        mat_name,
        event_time_dict=cache.get("event_time_dict"),
        show_period_labels=False,
        default_n_shown_samples=default_n_shown_samples,
    )
    cache.set("fig_resampler", fig)
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


def get_preferred_spreadsheet_dir(filepath):
    return Path(filepath).resolve().parent


def clear_analysis_export_payload():
    cache.set(ANALYSIS_EXPORT_PAYLOAD_CACHE_KEY, None)


def is_selection_relayout(relayoutdata):
    if not relayoutdata:
        return False
    return any(str(key).startswith("selections") for key in relayoutdata)


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
    _, fp_freq = get_cached_runtime_fp_metadata(mat=fp_data)
    signal_length = len(fp_data[selected_signals[0]]) if selected_signals else None

    # Build helpers
    event_utils = Event_Utils(
        fp_freq,
        duration,
        nsec_before=baseline_window,
        nsec_after=analysis_window,
        signal_length=signal_length,
    )
    analyses = Analyses(fp_freq=fp_freq, baseline_window=baseline_window)

    # Indices for this event
    # event_time_dict = event_utils.read_events(event_file=annotation_file)
    perievent_signals_fig_paths = {}
    analyses_fig_paths = {}
    corr_fig_paths = {}
    subject_id = mat_name
    cross_correlation_event_exports = {}
    strongest_cross_correlation_event_exports = {}
    processed_events = []

    def build_mean_trace_export(plots, perievent_signals, result):
        return plots.build_mean_trace_export_df(
            perievent_signals=perievent_signals,
            subject_id=subject_id,
            downsample_factor=EXPORT_DOWNSAMPLE_FACTOR,
        )

    def build_auc_export(plots, perievent_signals, result):
        return plots.build_auc_export_df(
            auc_values=result["reaction_signal_auc"],
            subject_id=subject_id,
        )

    def build_max_peak_magnitude_export(plots, perievent_signals, result):
        return plots.build_occurrence_value_export_df(
            values=result["max_peak_magnitude"],
            subject_id=subject_id,
        )

    def build_first_peak_time_export(plots, perievent_signals, result):
        return plots.build_occurrence_value_export_df(
            values=result["first_peak_time"],
            subject_id=subject_id,
        )

    def build_decay_time_export(plots, perievent_signals, result):
        return plots.build_occurrence_value_export_df(
            values=result["decay_time"],
            subject_id=subject_id,
        )

    export_specs = {
        "mean_trace": {
            "build_df": build_mean_trace_export,
        },
        "auc": {
            "build_df": build_auc_export,
        },
        "max_peak_magnitude": {
            "build_df": build_max_peak_magnitude_export,
        },
        "first_peak_time": {
            "build_df": build_first_peak_time_export,
        },
        "decay_time": {
            "build_df": build_decay_time_export,
        },
    }
    signal_event_exports = {
        export_name: {sig: {} for sig in selected_signals}
        for export_name in export_specs
    }

    for i, event in enumerate(sorted(event_time_dict.keys())):

        event_time = event_utils.filter_event_times(event_time_dict[event])
        if event_time.size == 0:
            continue
        processed_events.append(event)
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

        plots = Perievent_Plots(
            fp_freq, event, nsec_before=baseline_window, nsec_after=analysis_window
        )

        for sig in selected_signals:
            perievent_signals = perievent_signals_dict[sig]
            result = perievent_analysis_dict[sig]
            for export_name, export_spec in export_specs.items():
                signal_event_exports[export_name][sig][event] = export_spec[
                    "build_df"
                ](plots, perievent_signals, result)

        perievent_signals_fig_save_path = (
            FIGURE_DIR
            / f"{mat_name}_{event}_bw{baseline_window}_aw{analysis_window}.png"
        )
        analyses_fig_save_path = (
            FIGURE_DIR
            / f"{mat_name}_{event}_analyses_bw{baseline_window}_aw{analysis_window}.png"
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
            sig_a, sig_b = selected_signals
            lags_time, cross_corr = plots._compute_cross_correlation(
                perievent_signals_normalized_array[0],
                perievent_signals_normalized_array[1],
            )
            lags_time_export, mean_corr_export, se_corr_export = (
                plots.summarize_cross_correlation(
                    lags_time,
                    cross_corr,
                    downsample_factor=EXPORT_DOWNSAMPLE_FACTOR,
                )
            )
            std_corr_export = plots._bin_average_1d(
                np.nanstd(cross_corr, axis=0),
                EXPORT_DOWNSAMPLE_FACTOR,
            )
            cross_correlation_event_exports[event] = (
                plots.build_cross_correlation_export_df(
                    lags_time=lags_time_export,
                    mean_corr=mean_corr_export,
                    std_corr=std_corr_export,
                    n_occurrences=cross_corr.shape[0],
                    subject_id=subject_id,
                )
            )
            strongest_cross_corr_lag_s = (
                plots.get_lag_at_strongest_cross_correlation(
                    lags_time=lags_time,
                    cross_correlations=cross_corr,
                )
            )
            strongest_cross_correlation_event_exports[event] = (
                plots.build_strongest_cross_correlation_export_df(
                    strongest_lag_s=strongest_cross_corr_lag_s,
                    subject_id=subject_id,
                )
            )
            corr_path = (
                FIGURE_DIR
                / f"{mat_name}_{event}_correlation_bw{baseline_window}_aw{analysis_window}.png"
            )
            plots.plot_correlation(
                lags_time=lags_time_export,
                mean_corr=mean_corr_export,
                se_corr=se_corr_export,
                signal_names=(sig_a, sig_b),
                figure_save_path=corr_path,
            )

        corr_fig_paths[event] = os.path.join(
            "/assets/figures/",
            f"{mat_name}_{event}_correlation_bw{baseline_window}_aw{analysis_window}.png",
        )

    export_payload = {
        "mat_filepath": filepath,
        "subject_id": subject_id,
        "selected_signals": tuple(selected_signals),
        "baseline_window": baseline_window,
        "analysis_window": analysis_window,
        "event_names": sorted(processed_events),
        "signal_event_exports": signal_event_exports,
        "cross_correlation_event_exports": cross_correlation_event_exports,
        "strongest_cross_correlation_event_exports": (
            strongest_cross_correlation_event_exports
        ),
    }
    
    return (
        perievent_signals_fig_paths,
        analyses_fig_paths,
        corr_fig_paths,
        export_payload,
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
    cache.set("fp_signal_names", [])
    cache.set("fp_frequency", None)
    cache.set("fig_resampler", None)
    cache.set("analysis_export_status_message", "")
    clear_analysis_export_payload()


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


# store_annotation_selection_span
app.clientside_callback(
    """
    function(relayoutData, keyboardEvents, figure) {
        const no_update = dash_clientside.no_update;
        const triggered = (
            dash_clientside.callback_context &&
            dash_clientside.callback_context.triggered
        ) || [];
        const triggeredByRelayout = triggered.some(
            item => item.prop_id === "graph.relayoutData"
        );
        if (!triggeredByRelayout) {
            return [no_update, no_update, no_update];
        }
        const hasSelectionRelayout = relayoutData && Object.keys(relayoutData).some(
            key => key.startsWith("selections")
        );
        if (!hasSelectionRelayout) {
            return [no_update, no_update, no_update];
        }
        if (!figure || !figure.layout) {
            return [null, no_update, no_update];
        }
        let selections = figure.layout.selections;
        if ((!selections || selections.length === 0) && Array.isArray(relayoutData.selections)) {
            selections = relayoutData.selections;
        }
        let latest = selections && selections.length ? selections[selections.length - 1] : null;
        if (!latest) {
            const keyedSelections = {};
            for (const [key, value] of Object.entries(relayoutData || {})) {
                const match = key.match(/^selections\\[(\\d+)\\]\\.(.+)$/);
                if (!match) continue;
                const index = match[1];
                const property = match[2];
                keyedSelections[index] = keyedSelections[index] || {};
                keyedSelections[index][property] = value;
            }
            const indices = Object.keys(keyedSelections).map(Number);
            if (indices.length > 0) {
                latest = keyedSelections[String(Math.max(...indices))];
            }
        }
        if (!latest) {
            return [null, no_update, no_update];
        }
        if (latest.x0 === undefined || latest.x1 === undefined) {
            return [no_update, no_update, no_update];
        }
        const start = Math.min(Number(latest.x0), Number(latest.x1));
        const end = Math.max(Number(latest.x0), Number(latest.x1));
        if (!Number.isFinite(start) || !Number.isFinite(end)) {
            return [no_update, no_update, no_update];
        }

        const updatedFigure = JSON.parse(JSON.stringify(figure));
        updatedFigure.layout.selections = [latest];
        updatedFigure.layout.shapes = null;
        return [
            {start: start, end: end},
            updatedFigure,
            `Selected event deletion span [${start.toFixed(1)}, ${end.toFixed(1)}] s.`
        ];
    }
    """,
    Output("box-select-store", "data"),
    Output("graph", "figure", allow_duplicate=True),
    Output("debug-message", "children"),
    Input("graph", "relayoutData"),
    Input("keyboard", "n_events"),
    State("graph", "figure"),
    prevent_initial_call=True,
)


# update_event_annotations
app.clientside_callback(
    """
    function(keyboardEvents, selectedSpan, keyboardEvent, eventTimes, figure) {
        const no_update = dash_clientside.no_update;
        const triggered = (
            dash_clientside.callback_context &&
            dash_clientside.callback_context.triggered
        ) || [];
        const triggeredByKeyboard = triggered.some(
            item => item.prop_id === "keyboard.n_events"
        );
        if (!triggeredByKeyboard) {
            return [no_update, no_update, no_update, no_update];
        }
        if (!keyboardEvents || !keyboardEvent) {
            return [no_update, no_update, no_update, no_update];
        }
        const key = keyboardEvent.key;
        if (key !== "Delete" && key !== "Backspace") {
            return [no_update, no_update, no_update, no_update];
        }
        if (!selectedSpan) {
            return [
                no_update,
                no_update,
                no_update,
                "Draw an annotation rectangle before deleting event timestamps."
            ];
        }
        if (!figure || !figure.layout || !eventTimes) {
            return [no_update, no_update, no_update, no_update];
        }
        if (
            figure.layout.dragmode !== "select" ||
            !figure.layout.selections ||
            figure.layout.selections.length === 0
        ) {
            return [no_update, no_update, no_update, no_update];
        }

        const start = Math.min(Number(selectedSpan.start), Number(selectedSpan.end));
        const end = Math.max(Number(selectedSpan.start), Number(selectedSpan.end));
        if (!Number.isFinite(start) || !Number.isFinite(end)) {
            return [no_update, no_update, no_update, no_update];
        }

        const nextEvents = {};
        let removedCount = 0;
        for (const [eventName, times] of Object.entries(eventTimes || {})) {
            const kept = [];
            for (const rawTime of times || []) {
                const time = Number(rawTime);
                if (time >= start && time <= end) {
                    removedCount += 1;
                } else {
                    kept.push(rawTime);
                }
            }
            if (kept.length > 0) nextEvents[eventName] = kept;
        }

        if (removedCount === 0) {
            return [
                no_update,
                selectedSpan,
                no_update,
                `No event timestamps found in [${start.toFixed(1)}, ${end.toFixed(1)}] s.`
            ];
        }

        const prefix = "Event timestamp: ";
        const updatedFigure = JSON.parse(JSON.stringify(figure));
        for (const trace of updatedFigure.data || []) {
            if (!trace.name || !String(trace.name).startsWith(prefix)) continue;
            const eventName = String(trace.name).slice(prefix.length);
            const keptTimes = nextEvents[eventName] || [];
            const originalY = trace.y || [];
            let y0 = originalY.length > 0 ? originalY[0] : -1;
            let y1 = originalY.length > 1 ? originalY[1] : 1;
            const x = [];
            const y = [];
            for (const time of keptTimes) {
                x.push(time, time, null);
                y.push(y0, y1, null);
            }
            trace.x = x;
            trace.y = y;
        }
        updatedFigure.layout.selections = null;
        updatedFigure.layout.shapes = null;

        return [
            nextEvents,
            null,
            updatedFigure,
            `Removed ${removedCount} event timestamp(s) in [${start.toFixed(1)}, ${end.toFixed(1)}] s.`
        ];
    }
    """,
    Output("event-time-store", "data", allow_duplicate=True),
    Output("box-select-store", "data", allow_duplicate=True),
    Output("graph", "figure", allow_duplicate=True),
    Output("debug-message", "children", allow_duplicate=True),
    Input("keyboard", "n_events"),
    Input("box-select-store", "data"),
    State("keyboard", "event"),
    State("event-time-store", "data"),
    State("graph", "figure"),
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
    Output("save-spreadsheets-button", "disabled", allow_duplicate=True),
    Output("analysis-save-status", "children", allow_duplicate=True),
    Input("show-results-button", "n_clicks"),
    State("signal-select-dropdown", "value"),
    State("baseline-window-dropdown", "value"),
    State("analysis-window-dropdown", "value"),
    State("event-time-store", "data"),
    State({"type": "tab", "event": ALL}, "id"),
    background=True,
    manager=background_callback_manager,
    running=[
        (Output("show-results-button", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def show_analysis_results(
    n_clicks,
    selected_signals,
    baseline_window,
    analysis_window,
    event_time_store,
    tabs,
):
    if not n_clicks:  # None or 0 → do nothing
        raise PreventUpdate
    if not is_valid_analysis_signal_selection(selected_signals):
        raise PreventUpdate

    for file in FIGURE_DIR.iterdir():
        if file.is_file() and file.suffix == ".png":
            file.unlink()

    event_time_dict = store_data_to_event_time_dict(event_time_store)
    duration = cache.get("duration")
    (
        perievent_signals_fig_paths,
        analyses_fig_paths,
        corr_fig_paths,
        export_payload,
    ) = (
        make_analysis_plots(
            event_time_dict=event_time_dict,
            selected_signals=selected_signals,
            baseline_window=baseline_window,
            analysis_window=analysis_window,
            duration=duration,
        )
    )
    cache.set(ANALYSIS_EXPORT_PAYLOAD_CACHE_KEY, export_payload)
    # Build outputs aligned to each pattern’s IDs
    tab_children = []
    for tab in tabs:
        event = tab["event"]
        if event in perievent_signals_fig_paths:
            children = components._fill_tab(
                event, perievent_signals_fig_paths, analyses_fig_paths, corr_fig_paths
            )
        elif event in event_time_dict:
            children = html.Div(
                "No event timestamps for this event remain after applying the "
                "selected baseline and analysis windows."
            )
        else:
            children = html.Div("This event type no longer has event timestamps.")
        tab_children.append(children)

    status_message = (
        "Analysis results are ready. Click Save Spreadsheets to choose which "
        "workbooks to save."
    )
    return tab_children, False, status_message


@app.callback(
    Output("save-spreadsheets-button", "disabled", allow_duplicate=True),
    Output("show-results-button", "disabled", allow_duplicate=True),
    Output("signal-select-wrapper", "style"),
    Output("analysis-save-status", "children", allow_duplicate=True),
    Input("signal-select-dropdown", "value"),
    Input("baseline-window-dropdown", "value"),
    Input("analysis-window-dropdown", "value"),
    Input("event-time-store", "data"),
    prevent_initial_call=True,
)
def clear_export_payload_after_analysis_setting_change(
    selected_signals,
    baseline_window,
    analysis_window,
    event_time_store,
):
    clear_analysis_export_payload()
    signal_selection_is_valid = is_valid_analysis_signal_selection(selected_signals)
    show_results_disabled = not signal_selection_is_valid
    signal_select_style = get_analysis_signal_select_style(selected_signals)
    status_message = (
        "Run analysis to prepare spreadsheet exports for these settings."
        if signal_selection_is_valid
        else ""
    )
    return (
        True,
        show_results_disabled,
        signal_select_style,
        status_message,
    )


@app.callback(
    Output("save-spreadsheets-modal", "style"),
    Output("save-analysis-checklist", "options"),
    Output("save-analysis-checklist", "value"),
    Output("analysis-save-status", "children", allow_duplicate=True),
    Input("save-spreadsheets-button", "n_clicks"),
    Input("cancel-save-spreadsheets-button", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_save_spreadsheets_modal(open_clicks, cancel_clicks):
    triggered_id = dash.callback_context.triggered_id
    hidden_style = {"display": "none"}
    visible_style = {"display": "block"}

    if triggered_id == "cancel-save-spreadsheets-button":
        return hidden_style, dash.no_update, dash.no_update, dash.no_update

    export_payload = cache.get(ANALYSIS_EXPORT_PAYLOAD_CACHE_KEY)
    if not export_payload:
        return (
            hidden_style,
            dash.no_update,
            dash.no_update,
            "Run analysis before saving spreadsheets.",
        )

    options = build_analysis_type_checklist_options(export_payload)
    selected_values = get_analysis_type_checklist_values(
        options=options,
        remembered_analysis_types=cache.get(
            REMEMBERED_ANALYSIS_EXPORT_TYPES_CACHE_KEY
        ),
    )
    return visible_style, options, selected_values, dash.no_update


@app.callback(
    Output("save-spreadsheets-modal", "style", allow_duplicate=True),
    Output("analysis-save-status", "children", allow_duplicate=True),
    Input("confirm-save-spreadsheets-button", "n_clicks"),
    State("save-analysis-checklist", "value"),
    background=True,
    manager=background_callback_manager,
    running=[
        (Output("confirm-save-spreadsheets-button", "disabled"), True, False),
        (Output("cancel-save-spreadsheets-button", "disabled"), True, False),
        (
            Output("save-spreadsheets-button", "disabled", allow_duplicate=True),
            True,
            False,
        ),
    ],
    prevent_initial_call=True,
)
def save_selected_analysis_spreadsheets(n_clicks, selected_analysis_types):
    if not n_clicks:
        raise PreventUpdate
    if not selected_analysis_types:
        return {"display": "block"}, "Select at least one analysis type to save."

    export_payload = cache.get(ANALYSIS_EXPORT_PAYLOAD_CACHE_KEY)
    if not export_payload:
        return {"display": "none"}, "Run analysis before saving spreadsheets."

    primary_dir = get_preferred_spreadsheet_dir(export_payload["mat_filepath"])
    _, export_status_message = write_analysis_workbooks(
        primary_dir=primary_dir,
        fallback_dir=SPREADSHEET_DIR,
        export_payload=export_payload,
        selected_analysis_types=selected_analysis_types,
    )
    cache.set(REMEMBERED_ANALYSIS_EXPORT_TYPES_CACHE_KEY, selected_analysis_types)
    return {"display": "none"}, export_status_message


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("visualization-ready-store", "data", allow_duplicate=True),
    Output("upload-container", "children", allow_duplicate=True),
    Output("analysis-link", "style", allow_duplicate=True),
    Output("analysis-page", "children", allow_duplicate=True),
    Output("event-time-store", "data", allow_duplicate=True),
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
    return message, True, components.vis_upload_button, {"visibility": "hidden"}, [], {}


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
    Output("event-time-store", "data", allow_duplicate=True),
    Input("annotation-uploaded-store", "data"),
    prevent_initial_call=True,
)
def import_annotation_file(annotation_filepath):
    clear_analysis_export_payload()
    mat_path = cache.get("filepath")
    mat = loadmat(mat_path, squeeze_me=True)
    signal_names, fp_freq = get_cached_runtime_fp_metadata(mat=mat)
    duration = cache.get("duration")
    signal_length = len(mat[signal_names[0]]) if signal_names else None
    event_utils = Event_Utils(fp_freq, duration, signal_length=signal_length)
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
    fig = create_fig(
        mat,
        os.path.basename(mat_path),
        label_dict=perievent_label_dict,
        event_time_dict=event_time_dict,
        show_period_labels=False,
    )
    cache.set("fig_resampler", fig)
    return (
        analysis_page_content,
        fig,
        {"visibility": "visible"},
        event_time_dict_to_store_data(event_time_dict),
    )


@app.callback(
    Output("visualization-container", "children"),
    Output("num-signals-store", "data"),
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("analysis-page", "children", allow_duplicate=True),
    Output("analysis-link", "style", allow_duplicate=True),
    Output("event-time-store", "data", allow_duplicate=True),
    Input("visualization-ready-store", "data"),
    prevent_initial_call=True,
)
def create_visualization(ready):
    if not ready:
        raise PreventUpdate

    filepath = cache.get("filepath")
    mat_name = os.path.splitext(os.path.basename(filepath))[0]
    mat = loadmat(filepath, squeeze_me=True)
    label_dict = {}
    analysis_page_content = []
    analysis_link_style = {"visibility": "hidden"}
    message = "Please double check the file selected."
    try:
        fp_signal_names, fp_signals, fp_freq = get_visualization_signal_data(mat)
    except KeyError:
        message = " ".join(["No FP signal found.", message])
        return (
            message,
            dash.no_update,
            "",
            analysis_page_content,
            analysis_link_style,
            {},
        )
    cache.set("fp_signal_names", fp_signal_names)
    cache.set("fp_frequency", float(fp_freq))

    num_signals = len(fp_signal_names)
    # duration = cache.get("duration")
    event_data = mat.get("event")
    event_time_dict = {}

    signal_lengths = [len(fp_signals[k]) for k in range(num_signals)]
    if not all(length == signal_lengths[0] for length in signal_lengths):
        message = " ".join(["Not all FP signals are of the same length.", message])
        return (
            message,
            dash.no_update,
            "",
            analysis_page_content,
            analysis_link_style,
            {},
        )

    signal_length = signal_lengths[0]
    duration = math.ceil(
        (signal_length - 1) / fp_freq
    )  # need to round duration to an int for later

    if has_embedded_event_data(event_data):
        signal_names = fp_signal_names
        event_utils = Event_Utils(fp_freq, duration, signal_length=signal_length)
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

    fig = create_fig(
        mat,
        mat_name,
        label_dict=label_dict,
        event_time_dict=event_time_dict,
        show_period_labels=not event_time_dict,
    )
    video_path = mat.get("video_path", "")
    video_name = mat.get("video_name", "")
    time_ax = fig["data"][0]["x"]
    start_time, end_time = time_ax[0], time_ax[-1]
    cache.set("start_time", start_time)
    cache.set("end_time", end_time)
    cache.set("duration", duration)
        
    if not isinstance(mat.get("video_start_time"), (int, float)):
        video_start_time = 0
        cache.set("video_start_time", video_start_time)
    if not isinstance(video_path, str):
        video_path = ""  
        cache.set("video_path", video_path)
    if not isinstance(video_name, str):
        video_name = ""
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
        event_time_dict_to_store_data(event_time_dict),
    )


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("n-sample-dropdown", "value"),
    prevent_initial_call=True,
)
def change_sampling_level(sampling_level):
    if sampling_level is None:
        return dash.no_update
    n_samples = get_n_samples_for_sampling_level(sampling_level)
    filepath = cache.get("filepath")
    mat_name = os.path.splitext(os.path.basename(filepath))[0]
    mat = loadmat(filepath, squeeze_me=True)

    # copy modified (through annotation) sleep scores over
    labels_history = cache.get("labels_history")
    if labels_history:
        mat["labels"] = labels_history[-1]

    fig = create_fig(
        mat,
        mat_name,
        event_time_dict=cache.get("event_time_dict"),
        default_n_shown_samples=n_samples,
    )
    cache.set("fig_resampler", fig)
    return fig


@app.callback(
    Output("event-time-sync-store", "data"),
    Output("analysis-page", "children", allow_duplicate=True),
    Input("event-time-store", "data"),
    prevent_initial_call=True,
)
def sync_event_time_store(event_time_store):
    event_time_dict = store_data_to_event_time_dict(event_time_store)
    cache.set("event_time_dict", event_time_dict)
    clear_analysis_export_payload()
    sync_data = {
        "event_count": int(
            sum(len(event_times) for event_times in event_time_dict.values())
        )
    }
    if not cache.get("fp_signal_names"):
        return sync_data, dash.no_update
    return sync_data, build_analysis_page_content(event_time_dict)


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
    if not relayoutdata:
        return dash.no_update
    if is_selection_relayout(relayoutdata):
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
