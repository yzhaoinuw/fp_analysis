# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:53:49 2023

@author: Yue
"""

import base64
import webbrowser
from io import BytesIO
from threading import Timer

from trace_updater import TraceUpdater
from dash import Dash, dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from plotly_resampler import FigureResampler

from scipy.io import loadmat

from inference import run_inference
from make_figure import make_figure, stage_colors


app = Dash(__name__)
port = 8050
fig = FigureResampler()
fig.register_update_graph_callback(
    app=app, graph_id="graph-1", trace_updater_id="trace-updater-1"
)


def run_app():
    app.layout = html.Div(
        [
            dcc.RadioItems(
                id="task-selection",
                options=[
                    {"label": "Generate prediction", "value": "gen"},
                    {"label": "Visualize existing prediction", "value": "vis"},
                ],
            ),
            html.Div(id="upload-container"),
            html.Div(id="output-data-upload"),
        ]
    )


@app.callback(Output("upload-container", "children"), Input("task-selection", "value"))
def show_upload(task):
    if task is None:
        raise PreventUpdate
    else:
        return dcc.Upload(
            id="upload-data",
            children=html.Div(["Select File"], className="upload-button"),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        )


@app.callback(
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("task-selection", "value"),
)
def update_output(contents, filename, task):
    if contents is None:
        return html.Div(["Please upload a .mat file"])

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    mat = loadmat(BytesIO(decoded))

    if task == "gen":
        run_inference(mat, model_path=None, output_path=None)
        return html.Div(["The results.mat file has been generated successfully."])
    else:  # task == 'vis'
        try:
            fig.replace(make_figure(mat))
            div = html.Div(
                children=[
                    dcc.Graph(
                        id="graph-1",
                        config={
                            "editable": True,
                            "edits": {
                                "axisTitleText": False,
                                "titleText": False,
                                "colorbarTitleText": False,
                                "annotationText": False,
                            },
                        },
                        figure=fig,
                    ),
                    TraceUpdater(id="trace-updater-1", gdID="graph-1"),
                    html.Div(
                        children=[
                            dcc.Input(id="start", type="number", placeholder="start"),
                            dcc.Input(id="end", type="number", placeholder="end"),
                            dcc.Dropdown(
                                id="label",
                                options=[
                                    {"label": "0: Wake", "value": 0},
                                    {"label": "1: SWS", "value": 1},
                                    {"label": "2: REM", "value": 2},
                                ],
                                placeholder="Select a Sleep Score",
                                style={"width": "200px"},
                            ),
                            html.Button("Add Annotation", id="add-button"),
                        ],
                        style={"display": "flex"},
                    ),
                ],
            )

            return div
        except Exception as e:
            print(e)
            return html.Div(["There was an error processing this file."])


@app.callback(
    Output("graph-1", "figure"),
    Input("add-button", "n_clicks"),
    State("start", "value"),
    State("end", "value"),
    State("label", "value"),
    State("graph-1", "figure"),
)
def add_annotation(n_clicks, start, end, label, figure):
    if start is None or end is None or label is None:
        return figure

    shape = dict(
        type="rect",
        # coordinates in data reference
        xref="x6",
        yref="y6",
        x0=start,
        y0=-1,
        x1=end,
        y1=2,
        fillcolor=stage_colors[label],
        opacity=0.5,
        layer="above",
        line_width=0,
    )

    # figure['layout'].update(shapes=shapes)
    if "shapes" in figure["layout"]:
        figure["layout"]["shapes"].append(shape)
    else:
        figure["layout"]["shapes"] = [shape]
    return figure


def open_browser():
    webbrowser.open_new(f"http://127.0.0.1:{port}/")


if __name__ == "__main__":
    run_app()
    Timer(1, open_browser).start()
    app.run_server(debug=False, port=8050)
