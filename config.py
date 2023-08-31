# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:48:52 2023

@author: Yue
"""

sleep_score_opacity = 0.5

annotation_config = {
    "type": "rect",
    "xref": "x5",
    "yref": "y5",
    "y0": -1,
    "y1": 2,
    "opacity": sleep_score_opacity,
    "layer": "above",
    "line_width": 0,
}

annotation_color_map = {
    "rgb(102, 178, 255)": 0,
    "rgb(255, 102, 255)": 1,
    "rgb(102, 255, 102)": 2,
}
