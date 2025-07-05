# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 19:26:15 2025

@author: yzhao
"""

import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt


def read_events(event_file, min_time, max_time):
    """
    read in spreadsheet, drop nan, and remove events near the start or end.

    """
    event_time_dict = {}
    df_events = pd.read_excel(event_file)
    for event in df_events.columns:
        df_event = df_events[event]
        df_event = df_event.dropna()
        df_event = df_event[(df_event >= min_time) & (df_event <= max_time)]
        if df_event.empty:
            continue
        event_time_dict[event] = df_event.round().astype(int).tolist()
    return event_time_dict


def make_perievent_windows(event_time, nsec_before=60, nsec_after=60):
    """
    Parameters
    ----------
    event_time : a flat array indicating event time.

    """
    event_time = np.expand_dims(event_time, axis=1)
    window_segment = np.arange(-nsec_before, nsec_after)
    perievent_windows = event_time + window_segment
    return perievent_windows


def get_perievent_indices(perievent_windows):
    window_duration = perievent_windows.shape[1]
    window_segment = np.arange(int(np.ceil(window_duration * fp_freq)))
    perievent_segments = np.ceil(perievent_windows[:, 0:1] * fp_freq) + window_segment
    return perievent_segments.astype(int)


def make_perievent_labels(event_file, duration, nsec_before=60, nsec_after=60):
    min_time = nsec_before
    max_time = duration - nsec_after
    event_time_dict = read_events(event_file, min_time, max_time)
    event_names = []
    perievent_labels = np.zeros(duration)
    perievent_labels[:] = np.nan
    for i, event in enumerate(sorted(event_time_dict.keys())):
        event_names.append(event)
        event_time = event_time_dict[event]
        perievent_windows = make_perievent_windows(
            event_time, nsec_before=nsec_before, nsec_after=nsec_after
        )
        perievent_time = perievent_windows.flatten()
        perievent_labels[perievent_time] = i
    return {"label_names": event_names, "labels": perievent_labels}


def plot_perievent_signals(event, perievent_signals, nsec_before=60, nsec_after=60):
    event_count, seg_len = perievent_signals.shape

    # Time axis in seconds
    t = np.linspace(-nsec_before, nsec_after, seg_len)
    # Plot all signals in one plot
    plt.figure(figsize=(10, 6))
    for i in range(event_count):
        plt.plot(
            t, perievent_signals[i], label=f"Signal {i+1}"
        )  # Offset vertically by 5 units for clarity

    # Add reference lines
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)  # Horizontal at y = 0
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)  # Vertical at x = 0
    plt.xlim(-nsec_before, nsec_after)
    plt.ylim(-10, 10)
    plt.xlabel("Time (s)", fontsize=14, fontweight="bold")
    plt.ylabel(f"{biosignal_name} (dF/F)", fontsize=14, fontweight="bold")
    plt.title(f"{fp_name}_{event}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    return plt


def plot_mean_perievent_signals(
    event, perievent_signals, nsec_before=60, nsec_after=60
):
    perievent_signals_mean = np.mean(perievent_signals, axis=0)
    seg_len = len(perievent_signals_mean)
    plt.figure(figsize=(10, 6))
    t = np.linspace(-nsec_before, nsec_after, seg_len)
    plt.plot(t, perievent_signals_mean)

    # Add reference lines
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)

    # Axis limits and labels
    plt.xlim(-nsec_before, nsec_after)
    plt.ylim(-10, 10)
    plt.xlabel("Time (s)", fontsize=14, fontweight="bold")
    plt.ylabel(f"mean {biosignal_name} (dF/F)", fontsize=14, fontweight="bold")
    plt.title(f"Mean_{fp_name}_{event}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    return plt
    # plt.show()


def plot_perievent_heatmaps(
    event, perievent_signals, fp_freq, nsec_before=60, nsec_after=60
):
    segment_size = int(np.floor(fp_freq))
    time_sec = np.arange(nsec_before + nsec_after)
    start_indices = np.ceil(time_sec * fp_freq).astype(int)
    event_count, _ = perievent_signals.shape
    # Reshape start_indices to be a column vector (N, 1)
    start_indices = start_indices[:, np.newaxis]
    segment_array = np.arange(segment_size)
    # Use broadcasting to add the range_array to each start index
    indices = start_indices + segment_array
    perievent_signals_reshaped = perievent_signals[:, indices]
    perievent_signals_downsampled = np.mean(perievent_signals_reshaped, axis=-1)
    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        perievent_signals_downsampled,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[-nsec_before, nsec_after, 0, event_count],
    )
    event_labels = [f"{i+1}" for i in range(event_count)]
    plt.yticks(np.arange(event_count) + 0.5, event_labels)
    plt.ylabel("Event Index", fontsize=14, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=14, fontweight="bold")
    plt.title(f"{fp_name}_{event}", fontsize=16, fontweight="bold")
    plt.colorbar(im, label="(dF/F)")
    return plt


# %%
if __name__ == "__main__":
    DATA_PATH = "../data/"
    fp_name = "F268"
    fp_file = os.path.join(DATA_PATH, f"{fp_name}.mat")
    fp_data = loadmat(fp_file, squeeze_me=True)
    biosignal_names = fp_data["fp_signal_names"]
    biosignal_name = "NE2m"
    biosignal = fp_data[biosignal_name]

    event_file = os.path.join(DATA_PATH, "Transitions_F268.xlsx")
    fp_freq = fp_data["fp_frequency"]
    nsec_before = 60
    nsec_after = 60
    duration = int(np.ceil(len(biosignal) / fp_freq))
    min_time = nsec_before
    max_time = duration - nsec_after
    # perievent_labels = make_perievent_labels(event_file, duration, nsec_before=2, nsec_after=2)
    event_time_dict = read_events(event_file, min_time, max_time)
    perievent_labels = np.zeros(duration)
    perievent_labels[:] = np.nan
    perievent_indices_dict = {}
    for i, event in enumerate(sorted(event_time_dict.keys())):
        event_time = event_time_dict[event]
        perievent_windows = make_perievent_windows(
            event_time, nsec_before=nsec_before, nsec_after=nsec_after
        )
        perievent_indices_dict[event] = get_perievent_indices(perievent_windows)
        perievent_time = perievent_windows.flatten()
        perievent_labels[perievent_time] = i

    event = "REM_MA"
    perievent_indices = perievent_indices_dict[event]
    perievent_signals = biosignal[perievent_indices]
    plot_perievent_signals(event, perievent_signals)
    plot_mean_perievent_signals(event, perievent_signals)
    plot_perievent_heatmaps(event, perievent_signals, fp_freq)
