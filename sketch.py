# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:20:58 2023

@author: Yue
"""

import dash
from dash.dependencies import Input, Output, State
from dash import dash_table, dcc, html

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Create a simple time series DataFrame
df = pd.DataFrame(
    {
        "Time": pd.date_range(start="1/1/2021", periods=100),
        "Value": np.random.randn(100).cumsum(),
    }
)

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        dash_table.DataTable(
            id="table-editing-simple",
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict("records"),
            editable=True,
        ),
        html.Button("Submit", id="submit-button", n_clicks=0),
        dcc.Graph(id="time-series-plot"),
    ]
)


@app.callback(
    Output("time-series-plot", "figure"),
    Input("submit-button", "n_clicks"),
    State("table-editing-simple", "data"),
)
def update_graph(n_clicks, rows):
    if n_clicks > 0:
        dff = pd.DataFrame(rows)
        dff["Time"] = pd.to_datetime(
            dff["Time"]
        )  # Convert Time back to datetime format
        fig = go.Figure(data=go.Scatter(x=dff["Time"], y=dff["Value"]))
        return fig
    else:
        raise dash.exceptions.PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True)
