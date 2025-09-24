# -*- coding: utf-8 -*-
"""
Created on Thu May  8 12:58:19 2025

@author: yzhao
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import find_peaks, resample_poly


class Event_Utils:

    def __init__(self, fp_freq, duration, nsec_before=30, nsec_after=60):
        self.fp_freq = fp_freq
        self.duration = duration
        self.nsec_before = nsec_before
        self.nsec_after = nsec_after

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
        return perievent_windows.astype(int)

    def get_perievent_indices(self, perievent_windows):
        window_duration = perievent_windows.shape[1]
        window_segment = np.arange(int(np.ceil(window_duration * self.fp_freq)))
        perievent_segments = (
            np.ceil(perievent_windows[:, 0:1] * self.fp_freq) + window_segment
        )
        return perievent_segments.astype(int)

    def make_perievent_labels(self, event_file):
        event_time_dict = self.read_events(event_file)
        event_names = []
        perievent_labels = np.zeros(self.duration)
        perievent_labels[:] = np.nan
        for i, event in enumerate(sorted(event_time_dict.keys())):
            event_names.append(event)
            event_time = event_time_dict[event]
            perievent_windows = self.make_perievent_windows(event_time)
            perievent_time = perievent_windows.flatten()
            perievent_labels[perievent_time] = i
        return {"label_names": event_names, "labels": perievent_labels}


class Perievent_Plots:

    def __init__(
        self,
        fp_freq,
        event,
        nsec_before=30,
        nsec_after=60,
    ):
        self.fp_freq = fp_freq
        self.event = event
        self.nsec_before = nsec_before
        self.nsec_after = nsec_after

    def plot_perievent_signals(
        self,
        perievent_signals,
        ax=None,
        biosignal_name="",
        first_peak_time_array=None,
        ylim=(-10, 10),
        title=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        if first_peak_time_array is not None:
            first_peak_time_array = first_peak_time_array.flatten()
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
        ax.set_xticks(np.arange(-self.nsec_before, self.nsec_after + 1, 10))
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{biosignal_name} (dF/F)", fontsize=10, fontweight="bold")
        if title is None:
            title = "Perievent Signals"
        ax.set_title(title, fontsize=10, fontweight="bold")

    def plot_mean_perievent_signals(
        self,
        perievent_signals,
        ax=None,
        biosignal_name="",
        ylim=(-10, 10),
        title=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        perievent_signals_mean = np.mean(perievent_signals, axis=0)
        seg_len = len(perievent_signals_mean)
        t = np.linspace(-self.nsec_before, self.nsec_after, seg_len)
        ax.plot(t, perievent_signals_mean)

        # Add reference lines
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)

        # Axis limits and labels
        ax.set_xlim(-self.nsec_before, self.nsec_after)
        ax.set_xticks(np.arange(-self.nsec_before, self.nsec_after + 1, 10))
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"Mean {biosignal_name}(dF/F)", fontsize=10, fontweight="bold")
        if title is None:
            title = "Mean Perievent Signals"
        ax.set_title(title, fontsize=10, fontweight="bold")

    def plot_perievent_heatmaps(
        self,
        perievent_signals,
        ax=None,
        title=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        segment_size = int(np.floor(self.fp_freq))
        time_sec = np.arange(self.nsec_before + self.nsec_after)
        start_indices = np.ceil(time_sec * self.fp_freq).astype(int)
        event_count, _ = perievent_signals.shape

        # Reshape start_indices to be a column vector (N, 1)
        start_indices = start_indices[:, np.newaxis]
        segment_array = np.arange(segment_size)

        # Compute index map for downsampling
        indices = start_indices + segment_array
        perievent_signals_reshaped = perievent_signals[:, indices]
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
        if title is None:
            title = "Heatmap"
        ax.set_title(title, fontsize=10, fontweight="bold")
        # Add colorbar to the figure this axis belongs to
        ax.figure.colorbar(im, ax=ax, label="(dF/F)")

    def plot_distribution(self, data, data_type: str, ax=None, ylim=None, title=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))

        colors = ["blue" if v >= 0 else "red" for v in data]
        x = np.arange(1, len(data) + 1)

        # Scatter of individual data points
        ax.scatter(x, data, color=colors)
        ax.set_xlabel("Event Index")
        ax.set_ylabel(f"{data_type}")
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_xticks(np.arange(1, len(data) + 1, max(1, int(np.ceil(len(data) / 15)))))
        if title is None:
            title = data_type
        ax.set_title(title, fontsize=10, fontweight="bold")

        if not all(np.isnan(data)):
            # Mean and std
            mean = np.nanmean(data)
            std = np.nanstd(data)

            # Plot mean line
            ax.axhline(mean, color="black", linestyle="--")

            # Shaded region for ± std
            ax.fill_between(x, mean - std, mean + std, color="gray", alpha=0.5)

    def _compute_correlation(self, perievent_signals_A, perievent_signals_B):
        assert (
            perievent_signals_A.shape == perievent_signals_B.shape
        ), "A and B must have the same shape."

        # subtract mean per row
        perievent_signals_A -= perievent_signals_A.mean(axis=1, keepdims=True)
        perievent_signals_B -= perievent_signals_B.mean(axis=1, keepdims=True)

        cov = np.sum(perievent_signals_A * perievent_signals_B, axis=1)
        den = np.linalg.norm(perievent_signals_A, axis=1) * np.linalg.norm(
            perievent_signals_B, axis=1
        )
        corr = cov / den
        return corr  # shape (n,)

    def make_perievent_plots(
        self,
        perievent_signals_dict,
        width=4,
        height=3,
        ylim=(-10, 10),
        figure_save_path=None,
    ):
        n_signals = len(perievent_signals_dict)
        assert (
            n_signals <= 2
        ), "More than two biosignals are detected in analysis results."
        n_cols = 3
        fig, axes = plt.subplots(
            n_signals,
            n_cols,
            figsize=(width * n_cols, height * n_signals),
        )
        axes = np.atleast_2d(axes)
        for i, (biosignal_name, perievent_signals) in enumerate(
            perievent_signals_dict.items()
        ):
            self.plot_perievent_signals(
                perievent_signals=perievent_signals,
                ax=axes[i, 0],
                biosignal_name=biosignal_name,
                ylim=ylim,
            )
            self.plot_mean_perievent_signals(
                perievent_signals=perievent_signals,
                ax=axes[i, 1],
                biosignal_name=biosignal_name,
                ylim=ylim,
            )
            self.plot_perievent_heatmaps(
                perievent_signals=perievent_signals,
                ax=axes[i, 2],
            )
        fig.tight_layout()

        if figure_save_path is not None:
            fig.savefig(
                figure_save_path.with_suffix(".png"),
                format="png",
                dpi=100,
                bbox_inches="tight",
            )
            plt.close(fig)
            return fig
        else:
            return None

    def make_perievent_analysis_plots(
        self,
        analysis_result,
        width=4,
        height=3,
        ylim=(-10, 10),
        figure_save_path=None,
    ):
        n_signals = len(analysis_result)
        assert (
            n_signals <= 2
        ), "More than two biosignals are detected in analysis results."
        n_cols = 5
        fig, axes = plt.subplots(
            n_signals,
            n_cols,
            figsize=(width * n_cols, height * n_signals),
        )
        axes = np.atleast_2d(axes)
        for i, (biosignal_name, result) in enumerate(analysis_result.items()):
            perievent_signals_normalized = result["perievent_signals_normalized"]
            reaction_signal_auc = result["reaction_signal_auc"]
            max_peak_magnitude = result["max_peak_magnitude"]
            first_peak_time = result["first_peak_time"]
            decay_time = result["decay_time"]

            self.plot_perievent_signals(
                ax=axes[i, 0],
                perievent_signals=perievent_signals_normalized,
                biosignal_name=biosignal_name + "_normalized",
                first_peak_time_array=first_peak_time,
                ylim=ylim,
                title="Perievent Signals Normalized",
            )
            self.plot_distribution(reaction_signal_auc, data_type="AUC", ax=axes[i, 1])
            self.plot_distribution(
                max_peak_magnitude,
                data_type="Max Peak Magnitude",
                ax=axes[i, 2],
            )
            self.plot_distribution(
                first_peak_time,
                data_type="First Peak Time",
                ax=axes[i, 3],
                ylim=(0, self.nsec_after),
            )
            self.plot_distribution(
                decay_time,
                data_type="Decay Time",
                ax=axes[i, 4],
                ylim=(0, self.nsec_after),
            )

        fig.tight_layout()

        if figure_save_path is not None:
            fig.savefig(
                figure_save_path.with_suffix(".png"),
                format="png",
                dpi=100,
                bbox_inches="tight",
            )
            plt.close(fig)
            return fig
        else:
            return None

    def plot_correlation(
        self,
        perievent_signals_A,
        perievent_signals_B,
        width=4,
        height=3,
        ylim=(-10, 10),
        title=None,
        figure_save_path=None,
    ):
        fig, ax = plt.subplots(figsize=(4, 3))
        corr = self._compute_correlation(
            perievent_signals_A,
            perievent_signals_B,
        )
        self.plot_distribution(
            corr,
            ax=ax,
            data_type="Correlation",
            ylim=(-1, 1),
            title=title,
        )
        plt.tight_layout()

        if figure_save_path is not None:
            fig.savefig(
                figure_save_path.with_suffix(".png"),
                format="png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)
            return fig
        else:
            return None

    def write_spreadsheet(self, results, save_path, index_name="event_index"):
        """
        results: dict
        save_path:  'results.xlsx'
        """
        stats = [
            "reaction_signal_auc",
            "max_peak_magnitude",
            "first_peak_time",
            "decay_time",
        ]
        for biosignal_name, result in results.items():
            data_dict = {k: result[k] for k in stats if k in result}
            df = pd.DataFrame(data_dict)
            df.index = df.index + 1
            df.index.name = index_name
            df.to_excel(save_path, sheet_name=biosignal_name)


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
        event_time_ind = round(self.baseline_window * self.fp_freq)
        baseline_signals = perievent_signals_scaled[
            :,
            :event_time_ind,
        ]
        baseline_mean_values = np.mean(baseline_signals, axis=1, keepdims=True)
        return baseline_mean_values

    def _get_reaction_signals(self, perievent_signals_normalized):
        event_time_ind = round(self.baseline_window * self.fp_freq)
        reaction_signals = perievent_signals_normalized[
            :, event_time_ind:
        ]  # perievent signals after the event
        return reaction_signals

    def _downsample(self, signal, fs, target_fs=10.0):
        up = int(target_fs)  # 10
        down = int(round(fs))  # 1017
        y = resample_poly(signal, up, down)  # anti-aliased by design
        return y

    def _normalize_perievent_signals(self, perievent_signals):
        event_time_ind = round(self.baseline_window * self.fp_freq)
        perievent_signals_scaled = perievent_signals + 1000
        baseline_mean_values = self._get_baseline_means(perievent_signals_scaled)
        perievent_signals_normalized = (
            1000 * perievent_signals_scaled / baseline_mean_values
        )
        perievent_signals_normalized -= perievent_signals_normalized[
            :, event_time_ind : event_time_ind + 1
        ]
        return perievent_signals_normalized

    def find_first_peaks(
        self, reaction_signals, distance=None, height=1, prominence=0.5
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
        return np.array(first_peak_inds)

    def compute_decay_time(
        self, reaction_signals, baseline_mean_values, first_peak_inds
    ):
        # reaction_signals = np.asarray(reaction_signals, dtype=float)
        n, l = reaction_signals.shape
        peaks = np.expand_dims(first_peak_inds, axis=1)
        # Build a mask of valid positions per row: j > peak_i and X[i, j] < mean_i
        j_idx = np.arange(l)[None, :]  # shape (1, l)
        after_peak = j_idx > peaks  # shape (n, l); False if peak is nan
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
        perievent_signals_normalized = self._normalize_perievent_signals(
            perievent_signals
        )
        reaction_signals = self._get_reaction_signals(perievent_signals_normalized)
        reaction_signal_areas = np.mean(reaction_signals, axis=1)
        max_peaks = np.max(reaction_signals, axis=1)
        first_peak_inds = self.find_first_peaks(reaction_signals)
        first_peak_time = np.round(first_peak_inds / self.fp_freq)
        baseline_mean_perievent_signals_normalized = self._get_baseline_means(
            perievent_signals_normalized
        )
        decay_time_array = self.compute_decay_time(
            reaction_signals,
            baseline_mean_perievent_signals_normalized,
            first_peak_inds,
        )

        return {
            "perievent_signals": perievent_signals,
            "perievent_signals_normalized": perievent_signals_normalized,
            "reaction_signal_auc": reaction_signal_areas,
            "max_peak_magnitude": max_peaks,
            "first_peak_time": first_peak_time,
            "decay_time": decay_time_array,
        }


# %%
if __name__ == "__main__":
    DATA_PATH = "../data/"
    FIGURE_DIR = Path(__file__).parent / "assets" / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SPREADSHEET_DIR = Path(__file__).parent / "assets" / "spreadsheets"
    SPREADSHEET_DIR.mkdir(parents=True, exist_ok=True)
    fp_name = "F268"
    fp_file = Path(DATA_PATH) / f"{fp_name}.mat"
    fp_data = loadmat(fp_file, squeeze_me=True)
    biosignal_names = fp_data["fp_signal_names"]
    biosignal_name = "NE2m"
    biosignal = fp_data[biosignal_name]
    event_file = Path(DATA_PATH) / "Transitions_F268.xlsx"

    fp_freq = fp_data["fp_frequency"]
    nsec_after = 60
    nsec_before = 30
    duration = int(np.ceil(len(biosignal) / fp_freq))
    event_utils = Event_Utils(
        fp_freq, duration, nsec_before=nsec_before, nsec_after=nsec_after
    )
    # perievent_labels = Event_Utils.make_perievent_labels(event_file, duration, nsec_before=2, nsec_after=2)
    event_time_dict = event_utils.read_events(event_file)
    perievent_labels = np.zeros(duration)
    perievent_labels[:] = np.nan
    perievent_indices_dict = {}
    analyses = Analyses(fp_freq=fp_freq, baseline_window=nsec_before)

    for i, event in enumerate(sorted(event_time_dict.keys())):

        event_time = event_time_dict[event]
        perievent_windows = event_utils.make_perievent_windows(event_time)
        perievent_indices = event_utils.get_perievent_indices(perievent_windows)

        perievent_signals_dict = {}
        perievent_analysis_dict = {}
        perievent_signals_normalized_array = []
        for biosignal_name in biosignal_names:
            biosignal = fp_data[biosignal_name]
            perievent_signals = biosignal[perievent_indices]
            perievent_signals_dict[biosignal_name] = perievent_signals
            result = analyses.get_perievent_analyses(perievent_signals)
            perievent_analysis_dict[biosignal_name] = result
            perievent_signals_normalized_array.append(
                result["perievent_signals_normalized"]
            )

        figure_name = f"{fp_name}_{event}.png"
        spreadsheet_name = f"{fp_name}_{event}.xlsx"
        figure_save_path = FIGURE_DIR / figure_name
        spreadsheet_save_path = SPREADSHEET_DIR / spreadsheet_name
        plots = Perievent_Plots(
            fp_freq, event, nsec_before=nsec_before, nsec_after=nsec_after
        )
        plots.make_perievent_plots(perievent_signals_dict)
        plots.make_perievent_analysis_plots(
            perievent_analysis_dict,
            # figure_save_path=figure_save_path
        )
        # %%

        plots.plot_correlation(
            perievent_signals_normalized_array[0], perievent_signals_normalized_array[1]
        )
        plots.write_spreadsheet(perievent_analysis_dict, spreadsheet_save_path)

        """
        perievent_indices_dict[event] = perievent_indices
        perievent_time = perievent_windows.flatten()
        perievent_labels[perievent_time] = i
        """
