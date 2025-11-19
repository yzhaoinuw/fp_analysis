# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:27:03 2023

@author: yzhao
"""

from dash import dcc, html, page_container, dash_table
from dash_extensions import EventListener
from dash_extensions.pages import setup_page_components


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
    "Click here to select File",
    id="vis-data-upload-button",
    style=upload_box_style,
)


save_div = html.Div(
    style={
        "display": "flex",
        "marginRight": "10px",
        "marginLeft": "10px",
        "marginBottom": "10px",
    },
    children=[
        dcc.Location(id="page-url"),
        html.Div(
            id="analysis-link",
            children=[dcc.Link(children="Analysis ->", href="/analysis")],
            style={"visibility": "hidden"},
        ),
        # html.A(html.Button("Run Analysis"), href="/analysis"),
        html.Button(
            "Save Annotations",
            id="save-button",
            style={"visibility": "hidden"},
        ),
        dcc.Download(id="download-annotations"),
        dcc.Download(id="download-spreadsheet"),
        html.Button(
            "Undo Annotation",
            id="undo-button",
            style={"visibility": "hidden"},
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
        html.Div(
            style={"display": "flex", "marginLeft": "15px"},
            children=[
                save_div,
                # html.Div(id="annotation-message"),
                html.Div(id="debug-message"),
            ],
        ),
        dcc.Store(id="visualization-ready-store"),
        dcc.Store(id="annotation-uploaded-store"),
        # dcc.Store(id="net-annotation-count-store"),
        dcc.Store(id="num-signals-store"),
    ],
)

analysis_page = html.Div(
    id="analysis-page",
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
        return tabs

    def fill_analysis_page(self, event_names, event_count_records, signal_names):
        event_tabs = self._build_event_tabs(event_names)
        children = [
            html.H3("Analysis Page"),
            html.Div(dcc.Link(children="← Back", href="/")),
            html.Div(
                style={"display": "flex", "marginLeft": "10px", "gap": "10px"},
                children=[
                    html.Label(["Baseline Window Size"]),
                    dcc.Dropdown(
                        options=[30, 60],
                        value=30,
                        style={"width": "60px"},
                        id="baseline-window-dropdown",
                        searchable=False,
                        clearable=False,
                    ),
                    html.Label(["Analysis Window Size"]),
                    dcc.Dropdown(
                        options=list(range(10, 70, 10)),
                        value=60,
                        style={"width": "60px"},
                        id="analysis-window-dropdown",
                        searchable=False,
                        clearable=False,
                    ),
                    html.Label(["Select 1 - 2 Signals"]),
                    dcc.Dropdown(
                        id="signal-select-dropdown",
                        options=[{"label": s, "value": s} for s in signal_names],
                        multi=True,
                        placeholder="Choose up to two…",
                        value=[],
                        style={"width": "300px"},
                        clearable=True,
                    ),
                    html.Button("Show Results", id="show-results-button", n_clicks=0),
                ],
            ),
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
            dcc.Tabs(id="event-tabs", children=event_tabs, value=event_names[0]),
        ]
        return children
