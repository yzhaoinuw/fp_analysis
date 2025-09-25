# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:34:40 2025

@author: yzhao
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from event_analysis import Event_Utils, Analyses, Perievent_Plots


DATA_PATH = "../data/"
fp_name = "F268"
fp_file = Path(DATA_PATH) / f"{fp_name}.mat"
fp_data = loadmat(fp_file, squeeze_me=True)
biosignal_names = fp_data["fp_signal_names"]
biosignal_name = "NE2m"
biosignal = fp_data[biosignal_name]
event_file = Path(DATA_PATH) / "Transitions_F268.xlsx"

fp_freq = fp_data["fp_frequency"]
window_len = 120
duration = int(np.ceil(len(biosignal) / fp_freq))
event_utils = Event_Utils(fp_freq, duration, window_len=window_len)
# perievent_labels = Event_Utils.make_perievent_labels(event_file, duration, nsec_before=2, nsec_after=2)
event_time_dict = event_utils.read_events(event_file)
perievent_labels = np.zeros(duration)
perievent_labels[:] = np.nan
perievent_indices_dict = {}
analyses = Analyses(fp_freq=fp_freq)
baseline_window = 30

event = "sws_wake"
event_time = event_time_dict[event]
perievent_windows = event_utils.make_perievent_windows(event_time)
perievent_indices = event_utils.get_perievent_indices(perievent_windows)
perievent_signals = biosignal[perievent_indices]


# %%
event_time_ind = int(np.ceil(perievent_signals.shape[1] / 2))
perievent_signals_scaled = analyses._scale_perieventsignals(perievent_signals)
baseline_mean_values = analyses._get_baseline_means(
    perievent_signals_scaled, baseline_window
)

perievent_signals_normalized = perievent_signals_scaled / baseline_mean_values
perievent_signals_normalized -= perievent_signals_normalized[
    :, event_time_ind : event_time_ind + 1
]

reaction_signals = analyses._get_reaction_signals(perievent_signals_normalized)

reaction_signal_areas = np.mean(reaction_signals, axis=1)
max_peaks = np.max(reaction_signals, axis=1)
first_peak_inds = analyses.find_first_peaks(reaction_signals)
decay_time_array = analyses.compute_decay_time(
    reaction_signals, baseline_mean_values, first_peak_inds
)

fig, ax = plt.subplots(figsize=(10, 6))
plots = Perievent_Plots(
    perievent_signals,
    fp_freq,
    event,
    fp_name=fp_name,
    biosignal_name=biosignal_name,
)
"""
plots.plot_perievent_signals(
    ax=ax,
    perievent_signals=perievent_signals_normalized,
    biosignal_name=biosignal_name+"_normalized",
    first_peak_inds=first_peak_inds
)
"""
plots.plot_distribution(decay_time_array, data_type="Decay Time (s)", ax=ax)
plt.tight_layout()
