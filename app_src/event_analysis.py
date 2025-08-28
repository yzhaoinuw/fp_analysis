# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 19:26:15 2025

@author: yzhao
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class Event_Utils:

    def __init__(self, fp_freq, duration, window_len: int = 120):
        self.fp_freq = fp_freq
        self.duration = duration
        self.nsec_before = self.nsec_after = window_len / 2

    def read_events(self, event_file):
        """
        read in spreadsheet, drop nan, and remove events near the start or end.

        """
        min_time = self.nsec_before
        max_time = self.duration - self.nsec_after
        event_time_dict = {}
        df_events = pd.read_excel(event_file)
        for event in df_events.columns:
            df_event = df_events[event]
            df_event = df_event.dropna()
            df_event = df_event[(df_event >= min_time) & (df_event <= max_time)]
            if df_event.empty:
                continue
            event_time_dict[event] = df_event.round().astype(int).to_numpy()
        return event_time_dict

    def count_events(self, event_time_dict):
        event_count_records = [
            {"Event": event, "Count": len(start_times)}
            for event, start_times in event_time_dict.items()
        ]
        return event_count_records

    def make_perievent_windows(self, event_time):
        """
        Parameters
        ----------
        event_time : a flat array indicating event time.

        """
        event_time = np.expand_dims(event_time, axis=1)
        window_segment = np.arange(-self.nsec_before, self.nsec_after)
        perievent_windows = event_time + window_segment
        return perievent_windows

    def get_perievent_indices(self, perievent_windows):
        window_duration = perievent_windows.shape[1]
        window_segment = np.arange(int(np.ceil(window_duration * self.fp_freq)))
        perievent_segments = (
            np.ceil(perievent_windows[:, 0:1] * self.fp_freq) + window_segment
        )
        return perievent_segments.astype(int)

    def make_perievent_labels(self, event_file, duration):
        max_time = duration - self.nsec_after
        event_time_dict = Event_Utils.read_events(
            event_file, self.nsec_before, max_time
        )
        event_names = []
        perievent_labels = np.zeros(duration)
        perievent_labels[:] = np.nan
        for i, event in enumerate(sorted(event_time_dict.keys())):
            event_names.append(event)
            event_time = event_time_dict[event]
            perievent_windows = Event_Utils.make_perievent_windows(event_time)
            perievent_time = perievent_windows.flatten()
            perievent_labels[perievent_time] = i
        return {"label_names": event_names, "labels": perievent_labels}


class Perievent_Plots:

    def __init__(
        self,
        perievent_signals,
        fp_freq,
        event,
        fp_name="",
        biosignal_name="",
        window_len: int = 120,
    ):
        self.perievent_signals = perievent_signals
        self.fp_freq = fp_freq
        self.event = event
        self.nsec_before = self.nsec_after = window_len / 2
        self.fp_name = fp_name
        self.biosignal_name = biosignal_name

    def plot_perievent_signals(
        self,
        ax=None,
        perievent_signals=None,
        biosignal_name=None,
        first_peak_time_array=None,
        ylim=(-10, 10),
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        if first_peak_time_array is not None:
            first_peak_time_array = first_peak_time_array.flatten()
        if perievent_signals is None:
            perievent_signals = self.perievent_signals
        if biosignal_name is None:
            biosignal_name = self.biosignal_name
        event_count, seg_len = perievent_signals.shape
        # Time axis in seconds
        t = np.linspace(-self.nsec_before, self.nsec_after, seg_len)
        for i in range(event_count):
            (line,) = ax.plot(t, perievent_signals[i], label=f"Signal {i+1}")
            if first_peak_time_array is None:
                continue

            first_peak_time = first_peak_time_array[i]
            if np.isnan(first_peak_time):
                continue
            ax.axvline(
                first_peak_time, color=line.get_color(), linestyle=":", linewidth=2
            )

        # Add reference lines
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)  # Horizontal at y = 0
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)  # Vertical at x = 0
        ax.set_xlim(-self.nsec_before, self.nsec_after)
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{biosignal_name} (dF/F)", fontsize=10, fontweight="bold")
        ax.set_title(f"{self.fp_name}_{self.event}", fontsize=12, fontweight="bold")

    def plot_mean_perievent_signals(
        self,
        ax=None,
        ylim=(-10, 10),
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        perievent_signals_mean = np.mean(self.perievent_signals, axis=0)
        seg_len = len(perievent_signals_mean)
        t = np.linspace(-self.nsec_before, self.nsec_after, seg_len)
        ax.plot(t, perievent_signals_mean)

        # Add reference lines
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)

        # Axis limits and labels
        ax.set_xlim(-self.nsec_before, self.nsec_after)
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax.set_ylabel(
            f"mean {self.biosignal_name} (dF/F)", fontsize=10, fontweight="bold"
        )
        ax.set_title(
            f"{self.fp_name}_{self.event}_Mean", fontsize=12, fontweight="bold"
        )
        # ax.tight_layout()

    def plot_perievent_heatmaps(
        self,
        ax=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        segment_size = int(np.floor(self.fp_freq))
        time_sec = np.arange(self.nsec_before + self.nsec_after)
        start_indices = np.ceil(time_sec * self.fp_freq).astype(int)
        event_count, _ = self.perievent_signals.shape

        # Reshape start_indices to be a column vector (N, 1)
        start_indices = start_indices[:, np.newaxis]
        segment_array = np.arange(segment_size)

        # Compute index map for downsampling
        indices = start_indices + segment_array
        perievent_signals_reshaped = self.perievent_signals[:, indices]
        perievent_signals_downsampled = np.mean(perievent_signals_reshaped, axis=-1)

        # Plot on provided ax
        im = ax.imshow(
            perievent_signals_downsampled,
            aspect="auto",
            cmap="viridis",
            origin="lower",
            extent=[-self.nsec_before, self.nsec_after, 0, event_count],
        )

        # Format axis
        # event_labels = [f"{i+1}" for i in range(event_count)]
        ax.set_yticks(np.arange(event_count) + 0.5)
        event_labels = [
            str(i + 1 * (i // 5 < 1)) if i % 5 == 0 else "" for i in range(event_count)
        ]
        ax.set_yticklabels(event_labels)
        ax.set_ylabel("Event Index", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax.set_title(
            f"{self.fp_name}_{self.event}_Heatmap", fontsize=12, fontweight="bold"
        )

        # Add colorbar to the figure this axis belongs to
        ax.figure.colorbar(im, ax=ax, label=f"{self.biosignal_name} (dF/F)")

    def plot_distribution(self, data, data_type: str, ax=None, ylim=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 3))

        colors = ["blue" if v >= 0 else "red" for v in data]
        x = np.arange(1, len(data) + 1)

        # Scatter of individual data points
        ax.scatter(x, data, color=colors)

        # Mean and std
        mean = np.nanmean(data)
        std = np.nanstd(data)

        # Plot mean line
        ax.axhline(mean, color="black", linestyle="--")

        # Shaded region for ± std
        ax.fill_between(x, mean - std, mean + std, color="gray", alpha=0.5)

        ax.set_xlabel("Event Index")
        ax.set_ylabel(f"Reaction Biosignal {data_type}")
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_xticks(np.arange(1, len(data) + 1, max(1, int(np.ceil(len(data) / 15)))))
        ax.set_title(
            f"{self.fp_name}_{self.event} {data_type}", fontsize=12, fontweight="bold"
        )
        # ax.legend()

        """
        # Assign colors: blue if value >= 0, red if value < 0
        colors = ["blue" if v >= 0 else "red" for v in data]
        x = np.arange(1, len(data) + 1)
        # Make the bar plot
        ax.bar(x, data, color=colors)
        ax.axhline(0, color="black", linewidth=1)  # add baseline at y=0
        ax.set_xlabel("Event Index")
        ax.set_ylabel(f"Reaction Biosignal {data_type}")
        ax.set_xticks(x)
        ax.set_title(f"{fp_name}_{event}_{data_type}", fontsize=12, fontweight="bold")
        """

    def make_perievent_analysis_plots(
        self,
        analysis_result,
        width=4,
        height=3,
        ylim=(-10, 10),
        figure_save_path=None,
    ):
        n_cols = 8
        fig, axes = plt.subplots(1, n_cols, figsize=(width * n_cols, height))
        axes = np.atleast_2d(axes)
        self.plot_perievent_signals(
            ax=axes[0, 0],
            ylim=ylim,
        )
        self.plot_mean_perievent_signals(
            ax=axes[0, 1],
            ylim=ylim,
        )
        self.plot_perievent_heatmaps(
            ax=axes[0, 2],
        )
        perievent_signals_normalized = analysis_result["perievent_signals_normalized"]
        reaction_signal_auc = analysis_result["reaction_signal_auc"]
        max_peak_magnitude = analysis_result["max_peak_magnitude"]
        first_peak_time = analysis_result["first_peak_time"]
        decay_time = analysis_result["decay_time"]

        self.plot_perievent_signals(
            ax=axes[0, 3],
            perievent_signals=perievent_signals_normalized,
            biosignal_name=self.biosignal_name + "_normalized",
            first_peak_time_array=first_peak_time,
            ylim=ylim,
        )
        self.plot_distribution(reaction_signal_auc, data_type="AUC", ax=axes[0, 4])
        self.plot_distribution(
            max_peak_magnitude, data_type="Max Peak Magnitude", ax=axes[0, 5]
        )
        self.plot_distribution(
            first_peak_time,
            data_type="First Peak Time",
            ax=axes[0, 6],
            ylim=(0, 0.5 * window_len),
        )
        self.plot_distribution(
            decay_time,
            data_type="Decay Time",
            ax=axes[0, 7],
            ylim=(0, 0.5 * window_len),
        )

        plt.tight_layout()

        if figure_save_path is not None:
            fig.savefig(figure_save_path, format="png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            return fig
        else:
            return None


class Analyses:
    def __init__(self, fp_freq, baseline_window=30):
        self.fp_freq = fp_freq
        self.baseline_window = baseline_window

    def _scale_perieventsignals(self, perievent_signals):
        signal_min = perievent_signals.min(axis=1, keepdims=True)
        signal_max = perievent_signals.max(axis=1, keepdims=True)
        denom = np.where(
            signal_max - signal_min == 0, 1, signal_max - signal_min
        )  # avoid div by zero
        perievent_signals_scaled = (perievent_signals - signal_min) / denom
        return perievent_signals_scaled

    def _get_baseline_means(self, perievent_signals_scaled):
        event_time_ind = int(np.ceil(perievent_signals_scaled.shape[1] / 2))
        baseline_signals = perievent_signals_scaled[
            :,
            int(
                np.floor(event_time_ind - self.baseline_window * self.fp_freq)
            ) : event_time_ind,
        ]
        baseline_mean_values = np.mean(baseline_signals, axis=1, keepdims=True)
        return baseline_mean_values

    def _get_reaction_signals(self, perievent_signals_normalized):
        event_time_ind = int(np.ceil(perievent_signals_normalized.shape[1] / 2))
        reaction_signals = perievent_signals_normalized[
            :, event_time_ind:
        ]  # perievent signals after the event
        return reaction_signals

    def find_first_peaks(
        self, reaction_signals, distance=None, height=1, prominence=None
    ):
        if distance is None:
            distance = reaction_signals.shape[1] // 20
        first_peak_inds = []
        for signal in reaction_signals:
            peaks, _ = find_peaks(
                signal,
                distance=distance,
                height=height,
                prominence=prominence,
            )
            if peaks.size > 0:
                peak_ind = peaks[0]
                first_peak_inds.append(peak_ind)
            else:
                first_peak_inds.append(np.nan)
        return np.expand_dims(first_peak_inds, axis=1)

    def compute_decay_time(
        self, reaction_signals, baseline_mean_values, first_peak_inds
    ):

        n, l = reaction_signals.shape
        peaks = np.asarray(first_peak_inds).reshape(n)
        # Build a mask of valid positions per row: j > peak_i and X[i, j] < mean_i
        j_idx = np.arange(l)[None, :]  # shape (1, l)
        after_peak = j_idx > peaks[:, None]  # shape (n, l); False if peak is nan
        below_mean = reaction_signals < baseline_mean_values  # shape (n, l)
        cond = after_peak & below_mean  # shape (n, l)

        any_true = cond.any(axis=1)  # (n,)
        first_pos = cond.argmax(axis=1)  # (n,) but undefined where any_true==False

        out = np.full(n, np.nan, dtype=float)  # default NaN
        valid_peak = ~np.isnan(peaks).ravel()  # (n,)

        # Where a peak exists AND we found a position
        sel_found = valid_peak & any_true
        out[sel_found] = first_pos[sel_found]

        # Where a peak exists BUT no position found -> last index
        sel_last = valid_peak & ~any_true
        out[sel_last] = l - 1

        return out / self.fp_freq

    def get_perievent_analyses(self, perievent_signals):
        event_time_ind = int(np.ceil(perievent_signals.shape[1] / 2))
        perievent_signals_scaled = analyses._scale_perieventsignals(perievent_signals)
        baseline_mean_values = analyses._get_baseline_means(perievent_signals_scaled)
        perievent_signals_normalized = perievent_signals_scaled / baseline_mean_values
        perievent_signals_normalized -= perievent_signals_normalized[
            :, event_time_ind : event_time_ind + 1
        ]

        reaction_signals = analyses._get_reaction_signals(perievent_signals_normalized)
        reaction_signal_areas = np.mean(reaction_signals, axis=1)
        max_peaks = np.max(reaction_signals, axis=1)
        first_peak_inds = analyses.find_first_peaks(reaction_signals)
        first_peak_time = np.round(first_peak_inds / self.fp_freq)
        decay_time_array = analyses.compute_decay_time(
            reaction_signals, baseline_mean_values, first_peak_inds
        )

        return {
            "perievent_signals_normalized": perievent_signals_normalized,
            "reaction_signal_auc": reaction_signal_areas,
            "max_peak_magnitude": max_peaks,
            "first_peak_time": first_peak_time,
            "decay_time": decay_time_array,
        }


# %%
if __name__ == "__main__":
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
    baseline_window = 30
    duration = int(np.ceil(len(biosignal) / fp_freq))
    event_utils = Event_Utils(fp_freq, duration, window_len=window_len)
    # perievent_labels = Event_Utils.make_perievent_labels(event_file, duration, nsec_before=2, nsec_after=2)
    event_time_dict = event_utils.read_events(event_file)
    perievent_labels = np.zeros(duration)
    perievent_labels[:] = np.nan
    perievent_indices_dict = {}
    analyses = Analyses(fp_freq=fp_freq, baseline_window=baseline_window)

    for i, event in enumerate(sorted(event_time_dict.keys())):

        event_time = event_time_dict[event]
        perievent_windows = event_utils.make_perievent_windows(event_time)
        perievent_indices = event_utils.get_perievent_indices(perievent_windows)
        perievent_signals = biosignal[perievent_indices]
        perievent_analysis_result = analyses.get_perievent_analyses(perievent_signals)
        plots = Perievent_Plots(
            perievent_signals, fp_freq, event, fp_name, biosignal_name, window_len
        )
        plots.make_perievent_analysis_plots(
            perievent_analysis_result,
        )

        """
        perievent_indices_dict[event] = perievent_indices
        perievent_time = perievent_windows.flatten()
        perievent_labels[perievent_time] = i
        """
