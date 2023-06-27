# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:53:49 2023

@author: Yue
"""

import base64
import webbrowser
from threading import Timer
from io import BytesIO

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from scipy.io import loadmat

from inference import run_inference
from make_figure import make_figure


app = dash.Dash(__name__)

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
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
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
    else:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        mat = loadmat(BytesIO(decoded))

        if task == "gen":
            run_inference(mat)
            return html.Div(["The results.mat file has been generated successfully."])
        else:  # task == 'vis'
            try:
                fig = make_figure(mat)
                return dcc.Graph(figure=fig)
            except Exception as e:
                print(e)
                return html.Div(["There was an error processing this file."])


port = 8050


def open_browser():
    webbrowser.open_new(f"http://127.0.0.1:{port}/")


if __name__ == "__main__":
    Timer(3, open_browser).start()
    app.run_server(debug=False)
