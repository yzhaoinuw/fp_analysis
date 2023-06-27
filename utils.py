# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:39:11 2023

@author: Yue
adpated from Shadi Sartipi's mice_3signal_june2023.ipynb
"""

import numpy as np


def windows(data, size, step):
    start = 0
    while (start + size) <= data.shape[0]:
        yield int(start), int(start + size)
        start += step


def segment_signal_without_transition(data, window_size, step):
    segments = []
    for start, end in windows(data, window_size, step):
        if len(data[start:end]) == window_size:
            segments = segments + [data[start:end]]
    return np.array(segments)


def segment_dataset(X, window_size, step):
    win_x = []
    win_x = win_x + [segment_signal_without_transition(X, window_size, step)]
    win_x = np.array(win_x)
    return win_x


def substitute_vector(vector):
    return [0 if element == 3 else element for element in vector]
