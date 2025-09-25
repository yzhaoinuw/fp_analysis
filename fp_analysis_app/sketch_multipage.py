# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 23:48:06 2025

@author: yzhao
"""

import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from dash import Dash, dcc, html, Input, Output, no_update

# ========= Dash Setup ==========
app = Dash(__name__)


# ========= Matplotlib Figure Generator ==========
def generate_subplot_image():
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    for i, ax in enumerate(axes.flat):
        x = np.linspace(0, 10, 100)
        y = np.sin(x + i + np.random.rand())  # add variation
        ax.plot(x, y)
        ax.set_title(f"Plot {i+1}")
        ax.axis("off")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{encoded}"


# ========= App Layout ==========
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        # Main Page (always loaded)
        html.Div(
            id="main-page",
            children=[
                html.H1("Main Page"),
                html.P("Your interactive time series visualization would go here."),
                html.A(
                    html.Button("Go to Analysis Page", className="nav-button"),
                    href="/analysis",
                ),
            ],
            style={"display": "none"},
        ),
        # Analysis Page (always loaded)
        html.Div(
            id="analysis-page",
            children=[
                html.H1("Analysis Page"),
                html.A(
                    html.Button("← Back to Main Page", className="nav-button"), href="/"
                ),
                html.Br(),
                html.Br(),
                html.Img(
                    id="analysis-image",
                    style={"width": "100%", "border": "1px solid #ccc"},
                ),
            ],
            style={"display": "none"},
        ),
    ]
)


# ========= Callback: Page Toggle + Dynamic Plot ==========
@app.callback(
    Output("main-page", "style"),
    Output("analysis-page", "style"),
    Output("analysis-image", "src"),
    Input("url", "pathname"),
)
def toggle_pages_and_generate_image(pathname):
    if pathname == "/analysis":
        return {"display": "none"}, {"display": "block"}, generate_subplot_image()
    else:
        return {"display": "block"}, {"display": "none"}, no_update


# ========= Run App ==========
if __name__ == "__main__":
    app.run_server(debug=False)
