# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:27:03 2023

@author: yzhao
"""

import dash_uploader as du
import dash_bootstrap_components as dbc
from dash import dcc, html, page_container, dash_table
from dash_extensions import EventListener
from dash_extensions.pages import setup_page_components


# %% home div

upload_box_style = {
    "fontSize": "18px",
    "width": "15%",
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

# set up dash uploader
vis_upload_box = du.Upload(
    id="vis-data-upload",
    text="Click here to select File",
    text_completed="Completed loading ",
    cancel_button=True,
    filetypes=["mat"],
    upload_id="",
    default_style=upload_box_style,
)

video_upload_box_style = {
    "fontSize": "18px",
    "width": "100%",
    "height": "auto",
    "minHeight": "auto",
    "lineHeight": "auto",
    "borderWidth": "1px",
    "borderStyle": "none",
    "textAlign": "center",
    "margin": "5px",  # spacing between the upload box and the div it's in
    "borderRadius": "10px",  # rounded corner
    "backgroundColor": "lightgrey",
}

video_upload_box = du.Upload(
    id="video-upload",
    text="Select avi File",
    text_completed="Completed loading ",
    cancel_button=True,
    filetypes=["avi"],
    max_file_size=2048,
    upload_id="",
    default_style=video_upload_box_style,
)

annotation_upload_box_style = {
    "fontSize": "18px",
    "width": "100%",
    "height": "auto",
    "minHeight": "auto",
    "lineHeight": "auto",
    "borderWidth": "1px",
    "borderStyle": "none",
    "textAlign": "center",
    "margin": "5px",  # spacing between the upload box and the div it's in
    "borderRadius": "10px",  # rounded corner
    "backgroundColor": "lightgrey",
}

# set up dash uploader
annotation_upload_box = du.Upload(
    id="annotation-upload",
    text="Select Annotation File",
    text_completed="Completed loading ",
    cancel_button=True,
    filetypes=["xlsx"],
    upload_id="",
    default_style=annotation_upload_box_style,
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
            children=[vis_upload_box],
        ),
        html.Div(id="data-upload-message", style={"marginLeft": "10px"}),
        html.Div(id="visualization-container", style={"marginLeft": "10px"}),
        html.Div(
            style={"display": "flex", "marginLeft": "15px"},
            children=[
                save_div,
                html.Div(id="annotation-message"),
                html.Div(id="debug-message"),
            ],
        ),
        dcc.Store(id="visualization-ready-store"),
        dcc.Store(id="annotation-uploaded-store"),
        dcc.Store(id="net-annotation-count-store"),
        dcc.Store(id="num-signals-store"),
    ],
)

analysis_page = html.Div(
    id="analysis-page",
    # hidden=True,
    children=[
        html.H1("Analysis Page"),
        # html.A(html.Button("← Back"), href="/"),
        html.Div(dcc.Link(children="← Back", href="/")),
        dcc.Dropdown(
            options=[60, 120],
            value=120,
            id="perievent-window-dropdown",
            searchable=False,
            clearable=False,
        ),
        html.Br(),
        html.Div(
            dash_table.DataTable(
                id="event-count-table",
                style_cell={"textAlign": "center", "width": "100px"},
            ),
            style={"maxWidth": "300px", "marginLeft": "20px", "marginRight": "auto"},
        ),
        html.Br(),
        dcc.Tabs(id="event-tabs"),
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

video_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Video")),
        dbc.ModalBody(html.Div(id="video-container")),
        dbc.ModalFooter(html.Div(id="video-message")),
    ],
    id="video-modal",
    size="lg",
    is_open=False,
    backdrop="static",  # the user must clicks the "x" to exit
    centered=True,
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
            video_modal,
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
        self.vis_upload_box = vis_upload_box
        self.video_upload_box = video_upload_box
        self.annotation_upload_box = annotation_upload_box

    def configure_du(self, app, folder):
        du.configure_upload(app, folder, use_upload_id=True)
        return du
