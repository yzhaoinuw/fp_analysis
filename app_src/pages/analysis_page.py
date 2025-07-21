# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 13:54:32 2025

@author: yzhao
"""

from dash import html, register_page

from app_src.components import analysis_page

register_page(
    __name__,
    path="/analysis",
    page_components=[analysis_page],
)

layout = html.Div([])
    