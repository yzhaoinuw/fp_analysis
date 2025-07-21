# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 13:24:48 2025

@author: yzhao
"""

from dash import html, register_page

from app_src.components import home_page


register_page(
    __name__,
    path="/",
    page_components=[home_page],
)

layout = html.Div([])
