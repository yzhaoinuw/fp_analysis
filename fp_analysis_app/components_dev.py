# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:27:03 2023

@author: yzhao
"""

import math

from dash import dcc, html, page_container, dash_table
from dash_extensions import EventListener
from dash_extensions.pages import setup_page_components

MAX_ANALYSIS_SIGNALS = 2
DEFAULT_BASELINE_WINDOW = 30
DEFAULT_ANALYSIS_WINDOW = 60
BASE_SIGNAL_SELECT_WRAPPER_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "gap": "8px",
    "padding": "4px",
    "borderRadius": "6px",
    "transition": "border-color 120ms ease, box-shadow 120ms ease, background-color 120ms ease",
}


def normalize_analysis_signal_selection(selected_signals):
    if selected_signals is None:
        return []
    if isinstance(selected_signals, str):
        return [selected_signals]
    return list(selected_signals)


def is_valid_analysis_signal_selection(selected_signals):
    selected_signals = normalize_analysis_signal_selection(selected_signals)
    return 1 <= len(selected_signals) <= MAX_ANALYSIS_SIGNALS


def get_analysis_signal_select_style(selected_signals):
    style = dict(BASE_SIGNAL_SELECT_WRAPPER_STYLE)
    if is_valid_analysis_signal_selection(selected_signals):
        style.update(
            {
                "border": "2px solid transparent",
                "boxShadow": "none",
                "backgroundColor": "transparent",
            }
        )
    else:
        style.update(
            {
                "border": "2px solid #c62828",
                "boxShadow": "0 0 0 3px rgba(198, 40, 40, 0.14)",
                "backgroundColor": "#fff7f7",
            }
        )
    return style


def get_max_analysis_window(recording_duration):
    """Return the largest whole-second window strictly below 1/4 duration."""
    if isinstance(recording_duration, bool):
        return None
    try:
        duration = float(recording_duration)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return max(math.ceil(duration / 4) - 1, 0)


def _normalize_positive_integer(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(numeric_value)
        or numeric_value <= 0
        or not numeric_value.is_integer()
    ):
        return None
    return int(numeric_value)


def validate_and_normalize_analysis_windows(
    baseline_window,
    analysis_window,
    recording_duration,
):
    """Normalize valid window inputs and return a user-facing validation error."""
    normalized_baseline = _normalize_positive_integer(baseline_window)
    normalized_analysis = _normalize_positive_integer(analysis_window)
    if normalized_baseline is None or normalized_analysis is None:
        return (
            normalized_baseline,
            normalized_analysis,
            "Baseline and analysis windows must be positive whole numbers of seconds.",
        )

    max_window = get_max_analysis_window(recording_duration)
    if max_window is None:
        return (
            normalized_baseline,
            normalized_analysis,
            "Recording duration is unavailable. Reload the MAT file before running analysis.",
        )
    if max_window < 1:
        return (
            normalized_baseline,
            normalized_analysis,
            "The recording is too short for a positive whole-second analysis window.",
        )

    oversized_windows = []
    if normalized_baseline > max_window:
        oversized_windows.append("Baseline window")
    if normalized_analysis > max_window:
        oversized_windows.append("Analysis window")
    if oversized_windows:
        names = " and ".join(oversized_windows)
        return (
            normalized_baseline,
            normalized_analysis,
            f"{names} must be less than one quarter of the recording duration "
            f"(maximum {max_window} seconds).",
        )

    return normalized_baseline, normalized_analysis, ""


def get_default_analysis_window(preferred_window, recording_duration):
    max_window = get_max_analysis_window(recording_duration)
    if max_window is None:
        return preferred_window
    if max_window < 1:
        return None
    return min(preferred_window, max_window)


def get_analysis_settings_prompt(settings_are_valid):
    if not settings_are_valid:
        return ""
    return "Click Show Results to see the analysis results"


# %% home div

upload_box_style = {
    "fontSize": "18px",
    "width": "20%",
    "height": "auto",
    "minHeight": "auto",
    "lineHeight": "auto",
    "borderWidth": "1px",
    "borderStyle": "none",
    "textAlign": "center",
    # "margin": "5px",  # spacing between the upload box and the div it's in
    "borderRadius": "10px",  # rounded corner
    "backgroundColor": "lightgrey",
    "padding": "0px",
}


vis_upload_button = html.Button(
    "Click here to select a mat File",
    id="vis-data-upload-button",
    style=upload_box_style,
)


save_div = html.Div(
    style={
        "alignItems": "center",
        "columnGap": "12px",
        "display": "grid",
        "gridTemplateColumns": "auto minmax(0, 1fr) auto",
        "marginLeft": "15px",
        "marginRight": "15px",
        "marginBottom": "10px",
        "width": "calc(100% - 30px)",
    },
    children=[
        html.Div(
            style={"alignItems": "center", "display": "flex", "gap": "10px"},
            children=[
                html.Button(
                    "Save Annotations",
                    id="save-button",
                    style={"visibility": "hidden"},
                ),
                html.Button(
                    "Undo Annotation",
                    id="undo-button",
                    style={"visibility": "hidden"},
                ),
                dcc.Download(id="download-annotations"),
                dcc.Download(id="download-spreadsheet"),
            ],
        ),
        html.Div(id="debug-message"),
        html.Div(
            id="analysis-link",
            children=[dcc.Link(children="Analysis ->", href="/analysis")],
            style={"justifySelf": "end", "visibility": "hidden"},
        ),
    ],
)
home_page = html.Div(
    id="home-page",
    children=[
        html.Div(
            id="upload-container",
            style={"marginLeft": "15px", "marginTop": "15px"},
            children=[vis_upload_button],
        ),
        html.Div(id="data-upload-message", style={"marginLeft": "10px"}),
        html.Div(id="visualization-container", style={"marginLeft": "10px"}),
        dcc.Location(id="page-url"),
        html.Div(
            style={"display": "flex", "width": "100%"},
            children=[
                save_div,
            ],
        ),
        dcc.Store(id="visualization-ready-store"),
        dcc.Store(id="annotation-uploaded-store"),
        dcc.Store(id="event-time-store", data={}),
        dcc.Store(id="event-time-sync-store"),
        # dcc.Store(id="net-annotation-count-store"),
        dcc.Store(id="num-signals-store"),
    ],
)

analysis_page = html.Div(
    id="analysis-page",
)


# ###########################################################################
# Added component: Save Spreadsheets modal for separated analysis export flow.
# ###########################################################################
def build_save_spreadsheets_modal():
    return html.Div(
        id="save-spreadsheets-modal",
        style={"display": "none"},
        children=[
            html.Div(
                style={
                    "position": "fixed",
                    "top": 0,
                    "left": 0,
                    "right": 0,
                    "bottom": 0,
                    "backgroundColor": "rgba(0, 0, 0, 0.35)",
                    "zIndex": 1000,
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
                children=[
                    html.Div(
                        style={
                            "backgroundColor": "white",
                            "border": "1px solid #ccc",
                            "borderRadius": "6px",
                            "boxShadow": "0 4px 18px rgba(0, 0, 0, 0.2)",
                            "padding": "18px",
                            "minWidth": "320px",
                            "maxWidth": "480px",
                        },
                        children=[
                            html.H4("Save Spreadsheets"),
                            dcc.Checklist(
                                id="save-analysis-checklist",
                                options=[],
                                value=[],
                                labelStyle={
                                    "display": "block",
                                    "marginBottom": "8px",
                                },
                            ),
                            html.Div(
                                style={
                                    "display": "flex",
                                    "justifyContent": "flex-end",
                                    "gap": "8px",
                                    "marginTop": "16px",
                                },
                                children=[
                                    html.Button(
                                        "Cancel",
                                        id="cancel-save-spreadsheets-button",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Confirm",
                                        id="confirm-save-spreadsheets-button",
                                        n_clicks=0,
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
            )
        ],
    )

main_div = html.Div(
    [
        page_container,  # page layout is rendered here
        setup_page_components(),  # page components are rendered here
        home_page,
    ]
)

# %% visualization div

utility_div = html.Div(
    style={
        "display": "flex",
        "marginLeft": "10px",
        "marginTop": "5px",
        "marginBottom": "0px",
        "justifyContent": "flex-start",
        "width": "100%",
        "alignItems": "center",
        "flexWrap": "nowrap",  # prevent wrap during transition
        "whiteSpace": "nowrap",
        "paddingRight": "30px",
        "boxSizing": "border-box",
    },
    children=[
        html.Div(
            style={"display": "flex", "marginLeft": "10px", "gap": "10px"},
            children=[
                html.Div(["Sampling Level"]),
                dcc.Dropdown(
                    options=["x1", "x2", "x4"],
                    value="x1",
                    id="n-sample-dropdown",
                    searchable=False,
                    clearable=False,
                ),
                html.Div(
                    [
                        html.Button(
                            "Check Video",
                            id="video-button",
                            style={"visibility": "hidden"},
                        )
                    ]
                ),
            ],
        ),
        html.Div(
            [
                html.Button(
                    "Load Annotations",
                    id="load-annotations-button",
                )
            ],
            style={"marginLeft": "auto"},  # keep the button to the right edge
        ),
    ],
)

graph = dcc.Graph(
    id="graph",
    config={
        "scrollZoom": True,
    },
)


backend_div = html.Div(
    children=[
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
    ]
)


def make_visualization_div(graph):
    visualization_div = html.Div(
        children=[
            utility_div,
            html.Div(
                children=[graph],
                style={"marginTop": "1px", "marginLeft": "20px", "marginRight": "20px"},
            ),
            backend_div,
        ],
    )
    return visualization_div


# %%
class Components:
    def __init__(self):
        self.home_page = home_page
        self.graph = graph
        self.make_visualization_div = make_visualization_div
        self.vis_upload_button = vis_upload_button
        # self.annotation_upload_box = annotation_upload_box

    def _build_event_tab(self, event_name: str):
        """A fixed template of stats/plots for one event."""
        return dcc.Tab(
            label=event_name,
            value=event_name,
            id={"type": "tab", "event": event_name},
        )

    def _fill_tab(
        self,
        event_name: str,
        perievent_signals_fig_paths: dict,
        analyses_fig_paths: dict,
        corr_fig_paths: dict,
    ):
        children = [
            html.Img(
                # id={"type": "perievent-signal-image", "event": event_name},
                style={"width": "auto", "border": "1px solid #ccc"},
                src=perievent_signals_fig_paths[event_name],
            ),
            html.H4("Analysis Plots"),
            html.Img(
                # id={"type": "analysis-image", "event": event_name},
                style={"width": "auto", "border": "1px solid #ccc"},
                src=analyses_fig_paths[event_name],
            ),
            html.Img(
                # id={"type": "correlation-image", "event": event_name},
                style={"width": "40%", "maxWidth": "400px"},
                src=corr_fig_paths[event_name],
            ),
        ]
        return children

    def _build_event_tabs(self, event_names):
        if not event_names:
            return [
                dcc.Tab(
                    label="No events",
                    value="none",
                    children=html.Div("No events found."),
                )
            ], "none"
        tabs = [self._build_event_tab(event_name) for event_name in event_names]
        return tabs, event_names[0]

    def fill_analysis_page(
        self,
        event_names,
        event_count_records,
        signal_names,
        recording_duration=None,
    ):
        event_tabs, active_tab = self._build_event_tabs(event_names)
        max_window = get_max_analysis_window(recording_duration)
        baseline_window = get_default_analysis_window(
            DEFAULT_BASELINE_WINDOW, recording_duration
        )
        analysis_window = get_default_analysis_window(
            DEFAULT_ANALYSIS_WINDOW, recording_duration
        )
        _, _, initial_window_error = validate_and_normalize_analysis_windows(
            baseline_window,
            analysis_window,
            recording_duration,
        )
        window_input_props = {
            "type": "number",
            "min": 1,
            "step": 1,
            "style": {"width": "90px"},
        }
        if max_window is not None and max_window >= 1:
            window_input_props["max"] = max_window
        children = [
            html.H3("Analysis Page"),
            html.Div(dcc.Link(children="← Back", href="/")),
            dcc.Store(
                id="analysis-recording-duration",
                data=recording_duration,
            ),
            # ################################################################
            # Edited component: analysis controls now separate running analysis
            # from saving spreadsheets.
            # ################################################################
            html.Div(
                id="analysis-controls-row",
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginLeft": "10px",
                    "gap": "10px",
                },
                children=[
                    html.Div(
                        children=[
                            html.Label(["Baseline Window (seconds)"]),
                            dcc.Input(
                                id="baseline-window-input",
                                value=baseline_window,
                                **window_input_props,
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "4px",
                        },
                    ),
                    html.Div(
                        children=[
                            html.Label(["Analysis Window (seconds)"]),
                            dcc.Input(
                                id="analysis-window-input",
                                value=analysis_window,
                                **window_input_props,
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "4px",
                        },
                    ),
                    html.Div(
                        id="signal-select-wrapper",
                        style=get_analysis_signal_select_style([]),
                        children=[
                            html.Label(["Select 1 - 2 Signals"]),
                            dcc.Dropdown(
                                id="signal-select-dropdown",
                                options=[
                                    {"label": s, "value": s} for s in signal_names
                                ],
                                multi=True,
                                placeholder="Choose up to two…",
                                value=[],
                                style={"width": "300px"},
                                clearable=True,
                            ),
                        ],
                    ),
                    html.Button(
                        "Show Results",
                        id="show-results-button",
                        n_clicks=0,
                        disabled=True,
                    ),
                    html.Button(
                        "Save Spreadsheets",
                        id="save-spreadsheets-button",
                        n_clicks=0,
                        disabled=True,
                    ),
                ],
            ),
            html.Div(
                initial_window_error,
                id="analysis-window-validation-message",
                style={
                    "color": "#c62828",
                    "marginLeft": "10px",
                    "marginTop": "2px",
                },
            ),
            html.Div(
                id="analysis-save-status",
                style={"marginLeft": "10px", "marginTop": "2px"},
            ),
            build_save_spreadsheets_modal(),
            html.Br(),
            html.Div(
                dash_table.DataTable(
                    id="event-count-table",
                    data=event_count_records,
                    style_cell={"textAlign": "center", "width": "100px"},
                ),
                style={
                    "maxWidth": "300px",
                    "marginLeft": "20px",
                    "marginRight": "auto",
                },
            ),
            html.Br(),
            dcc.Tabs(id="event-tabs", children=event_tabs, value=active_tab),
        ]
        return children
