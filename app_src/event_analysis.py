# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 19:26:15 2025

@author: yzhao
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


class Event_Utils:
    nsec_before = 60
    nsec_after = 60

    @classmethod
    def read_events(cls, event_file, min_time=None, max_time=np.inf):
        """
        read in spreadsheet, drop nan, and remove events near the start or end.

        """
        if min_time is None:
            min_time = cls.nsec_before
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

    @staticmethod
    def count_events(event_time_dict):
        event_count_records = [
            {"Event": event, "Count": len(start_times)}
            for event, start_times in event_time_dict.items()
        ]
        return event_count_records

    @classmethod
    def make_perievent_windows(cls, event_time, nsec_before=None, nsec_after=None):
        """
        Parameters
        ----------
        event_time : a flat array indicating event time.

        """
        if nsec_before is None:
            nsec_before = cls.nsec_before
        if nsec_after is None:
            nsec_after = cls.nsec_after
        event_time = np.expand_dims(event_time, axis=1)
        window_segment = np.arange(-nsec_before, nsec_after)
        perievent_windows = event_time + window_segment
        return perievent_windows

    @staticmethod
    def get_perievent_indices(perievent_windows, fp_freq):
        window_duration = perievent_windows.shape[1]
        window_segment = np.arange(int(np.ceil(window_duration * fp_freq)))
        perievent_segments = (
            np.ceil(perievent_windows[:, 0:1] * fp_freq) + window_segment
        )
        return perievent_segments.astype(int)

    @classmethod
    def make_perievent_labels(
        cls, event_file, duration, nsec_before=None, nsec_after=None
    ):
        if nsec_before is None:
            nsec_before = cls.nsec_before
        if nsec_after is None:
            nsec_after = cls.nsec_after
        max_time = duration - nsec_after
        event_time_dict = Event_Utils.read_events(event_file, nsec_before, max_time)
        event_names = []
        perievent_labels = np.zeros(duration)
        perievent_labels[:] = np.nan
        for i, event in enumerate(sorted(event_time_dict.keys())):
            event_names.append(event)
            event_time = event_time_dict[event]
            perievent_windows = Event_Utils.make_perievent_windows(
                event_time, nsec_before=nsec_before, nsec_after=nsec_after
            )
            perievent_time = perievent_windows.flatten()
            perievent_labels[perievent_time] = i
        return {"label_names": event_names, "labels": perievent_labels}


class Perievent_Plots:
    nsec_before = 60
    nsec_after = 60

    @classmethod
    def plot_perievent_signals(
        cls,
        ax,
        event,
        perievent_signals,
        nsec_before=None,
        nsec_after=None,
        fp_name="",
        biosignal_name="",
        ylim=(-10, 10),
    ):
        if nsec_before is None:
            nsec_before = cls.nsec_before
        if nsec_after is None:
            nsec_after = cls.nsec_after
        event_count, seg_len = perievent_signals.shape
        # Time axis in seconds
        t = np.linspace(-nsec_before, nsec_after, seg_len)
        # plt.figure(figsize=(10, 6))
        for i in range(event_count):
            ax.plot(
                t, perievent_signals[i], label=f"Signal {i+1}"
            )  # Offset vertically by 5 units for clarity

        # Add reference lines
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)  # Horizontal at y = 0
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)  # Vertical at x = 0
        ax.set_xlim(-nsec_before, nsec_after)
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{biosignal_name} (dF/F)", fontsize=10, fontweight="bold")
        ax.set_title(f"{fp_name}_{event}", fontsize=12, fontweight="bold")
        # ax.tight_layout()

    @classmethod
    def plot_mean_perievent_signals(
        cls,
        ax,
        event,
        perievent_signals,
        nsec_before=None,
        nsec_after=None,
        fp_name="",
        biosignal_name="",
        ylim=(-10, 10),
    ):
        if nsec_before is None:
            nsec_before = cls.nsec_before
        if nsec_after is None:
            nsec_after = cls.nsec_after
        perievent_signals_mean = np.mean(perievent_signals, axis=0)
        seg_len = len(perievent_signals_mean)
        t = np.linspace(-nsec_before, nsec_after, seg_len)
        ax.plot(t, perievent_signals_mean)

        # Add reference lines
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)

        # Axis limits and labels
        ax.set_xlim(-nsec_before, nsec_after)
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"mean {biosignal_name} (dF/F)", fontsize=10, fontweight="bold")
        ax.set_title(f"{fp_name}_{event}_Mean", fontsize=12, fontweight="bold")
        # ax.tight_layout()

    @classmethod
    def plot_perievent_heatmaps(
        cls,
        ax,
        event,
        perievent_signals,
        fp_freq,
        nsec_before=None,
        nsec_after=None,
        fp_name="",
        biosignal_name="",
    ):
        if nsec_before is None:
            nsec_before = cls.nsec_before
        if nsec_after is None:
            nsec_after = cls.nsec_after
        segment_size = int(np.floor(fp_freq))
        time_sec = np.arange(nsec_before + nsec_after)
        start_indices = np.ceil(time_sec * fp_freq).astype(int)
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
            extent=[-nsec_before, nsec_after, 0, event_count],
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
        ax.set_title(f"{fp_name}_{event}_Heatmap", fontsize=12, fontweight="bold")

        # Add colorbar to the figure this axis belongs to
        ax.figure.colorbar(im, ax=ax, label=f"{biosignal_name} (dF/F)")

    @staticmethod
    def plot_bars(ax, data, data_type: str):
        # Assign colors: blue if value >= 0, red if value < 0
        colors = ["blue" if v >= 0 else "red" for v in data]
        x = np.arange(1, len(data) + 1)
        # Make the bar plot
        ax.bar(x, data, color=colors)
        ax.axhline(0, color="black", linewidth=1)  # add baseline at y=0
        ax.set_xlabel("Event Index")
        ax.set_ylabel("Reaction Biosignal {data_type}")
        ax.set_xticks(x)
        ax.set_title(f"{fp_name}_{event}_{data_type}", fontsize=12, fontweight="bold")

    """
    @staticmethod
    def make_perievent_plots(
        fp_file,
        biosignal_name,
        event_file,
        nsec_before=60,
        nsec_after=60,
        width=5,
        height=3,
        as_base64=False,
    ):
        n_cols = 3
        fp_name = os.path.basename(fp_file).rstrip(".mat")
        fp_data = loadmat(fp_file, squeeze_me=True)
        # biosignal_names = fp_data["fp_signal_names"]
        biosignal = fp_data[biosignal_name]
        fp_freq = fp_data["fp_frequency"]
        duration = int(np.ceil(len(biosignal) / fp_freq))
        min_time = nsec_before
        max_time = duration - nsec_after
        # perievent_labels = make_perievent_labels(event_file, duration, nsec_before=2, nsec_after=2)
        event_time_dict = Event_Utils.read_events(event_file, min_time, max_time)
        perievent_labels = np.zeros(duration)
        perievent_labels[:] = np.nan
        perievent_indices_dict = {}
        for i, event in enumerate(sorted(event_time_dict.keys())):
            event_time = event_time_dict[event]
            perievent_windows = Event_Utils.make_perievent_windows(
                event_time, nsec_before=nsec_before, nsec_after=nsec_after
            )
            perievent_indices_dict[event] = Event_Utils.get_perievent_indices(
                perievent_windows, fp_freq
            )
            perievent_time = perievent_windows.flatten()
            perievent_labels[perievent_time] = i

        event_count = len(event_time_dict)
        fig, axes = plt.subplots(
            event_count, n_cols, figsize=(width * n_cols, height * event_count)
        )
        axes = np.atleast_2d(axes)
        for col, event in enumerate(event_time_dict.keys()):
            perievent_indices = perievent_indices_dict[event]
            perievent_signals = biosignal[perievent_indices]
            Perievent_Plots.plot_perievent_signals(
                axes[col, 0], event, perievent_signals, fp_name=fp_name
            )
            Perievent_Plots.plot_mean_perievent_signals(
                axes[col, 1], event, perievent_signals, fp_name=fp_name
            )
            Perievent_Plots.plot_perievent_heatmaps(
                axes[col, 2], event, perievent_signals, fp_freq, fp_name=fp_name
            )

        plt.tight_layout()
        if as_base64:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode()
            return f"data:image/png;base64,{encoded}", fig
        else:
            plt.show()
            return None
    """

    @classmethod
    def make_perievent_plots(
        cls,
        perievent_signals,
        fp_name,
        biosignal_name,
        event,
        fp_freq,
        nsec_before=None,
        nsec_after=None,
        width=4,
        height=3,
        ylim=(-10, 10),
        figure_save_path=None,
    ):
        n_cols = 6
        if nsec_before is None:
            nsec_before = cls.nsec_before
        if nsec_after is None:
            nsec_after = cls.nsec_after

        fig, axes = plt.subplots(1, n_cols, figsize=(width * n_cols, height))
        axes = np.atleast_2d(axes)
        Perievent_Plots.plot_perievent_signals(
            axes[0, 0],
            event,
            perievent_signals,
            fp_name=fp_name,
            biosignal_name=biosignal_name,
            ylim=ylim,
        )
        Perievent_Plots.plot_mean_perievent_signals(
            axes[0, 1],
            event,
            perievent_signals,
            fp_name=fp_name,
            biosignal_name=biosignal_name,
            ylim=ylim,
        )
        Perievent_Plots.plot_perievent_heatmaps(
            axes[0, 2],
            event,
            perievent_signals,
            fp_freq,
            fp_name=fp_name,
            biosignal_name=biosignal_name,
        )
        perievent_signals_normalized = Analyses.normalize_perievent_signals(
            perievent_signals, fp_freq
        )
        areas, max_peaks = Analyses.compute_reaction_signal_data(
            perievent_signals_normalized
        )
        Perievent_Plots.plot_perievent_signals(
            axes[0, 3],
            event,
            perievent_signals_normalized,
            fp_name=fp_name,
            biosignal_name=biosignal_name + "_normalized",
            ylim=ylim,
        )
        Perievent_Plots.plot_bars(axes[0, 4], areas, data_type="AUC")
        Perievent_Plots.plot_bars(axes[0, 5], max_peaks, data_type="Max Peak")
        plt.tight_layout()

        if figure_save_path is not None:
            fig.savefig(figure_save_path, format="png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            return fig
        else:
            return None

    @staticmethod
    def _full_extent(ax, pad=0.0):
        """Get the full extent of an axes, including axes labels, tick labels, and
        titles."""
        # For text objects, we need to draw the figure first, otherwise the extents
        # are undefined.
        ax.figure.canvas.draw()
        items = ax.get_xticklabels() + ax.get_yticklabels()
        #    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        items += [ax, ax.title]
        bbox = Bbox.union([item.get_window_extent() for item in items])
        return bbox.expanded(1.0 + pad, 1.0 + pad)

    @staticmethod
    def zip_plots(fig, save_zip_path):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(
            zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zipf:
            for ax in fig.axes:
                title = ax.get_title()
                if not title:  # skip colorbars and such
                    continue
                extent = Perievent_Plots._full_extent(ax).transformed(
                    fig.dpi_scale_trans.inverted()
                )
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", dpi=200, bbox_inches=extent)
                img_buf.seek(0)
                zipf.writestr(f"{title}.png", img_buf.read())

        # Save in-memory zip to the filesystem
        with open(save_zip_path, "wb") as f:
            f.write(zip_buf.getvalue())


class Analyses:
    @staticmethod
    def normalize_perievent_signals(
        perievent_signals,
        fp_freq,
        baseline_window=30,
    ):
        perievent_signals = biosignal[perievent_indices]
        event_time_ind = int(np.ceil(perievent_signals.shape[1] / 2))
        signal_min = perievent_signals.min(axis=1, keepdims=True)
        signal_max = perievent_signals.max(axis=1, keepdims=True)
        denom = np.where(
            signal_max - signal_min == 0, 1, signal_max - signal_min
        )  # avoid div by zero
        perievent_signals_scaled = (perievent_signals - signal_min) / denom
        baseline_signals = perievent_signals_scaled[
            :,
            int(np.floor(event_time_ind - baseline_window * fp_freq)) : event_time_ind,
        ]
        baseline_mean_values = np.mean(baseline_signals, axis=1, keepdims=True)
        perievent_signals_normalized = perievent_signals_scaled / baseline_mean_values
        perievent_signals_normalized -= perievent_signals_normalized[
            :, event_time_ind : event_time_ind + 1
        ]
        return perievent_signals_normalized

    @staticmethod
    def compute_reaction_signal_data(perievent_signals):
        event_time_ind = int(np.ceil(perievent_signals.shape[1] / 2))
        reaction_signals = perievent_signals[
            :, event_time_ind + 1 :
        ]  # perievent signals after the event
        reaction_signal_areas = np.mean(reaction_signals, axis=1)
        max_peaks = np.max(reaction_signals, axis=1)
        return reaction_signal_areas, max_peaks

    """
    def normalize_perievent_signals(
        perievent_signals,
        fp_freq,
        baseline_window=30,
    ):
        perievent_signals = biosignal[perievent_indices]
        event_time_ind = int(np.ceil(perievent_signals.shape[1] / 2))
        perievent_signals_normalized = perievent_signals - perievent_signals[:, event_time_ind:event_time_ind+1]
        baseline_signals = perievent_signals_normalized[:, int(np.floor(event_time_ind-baseline_window*fp_freq)):event_time_ind]
        baseline_mean_values = np.abs(np.mean(baseline_signals, axis=1, keepdims=True))
        perievent_signals_normalized /= baseline_mean_values
        return perievent_signals_normalized
    

    
    @staticmethod    
    def calculate_auc(biosignal, event_time, fp_freq, nsec_before=30):
        
        
        biosignal_normalized = baseline_signals / baseline_mean_values
        baseline_signals_normalized -= baseline_signals_normalized[:, -1:]
    """


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
    nsec_before = 60
    nsec_after = 60
    duration = int(np.ceil(len(biosignal) / fp_freq))
    min_time = nsec_before
    max_time = duration - nsec_after
    # perievent_labels = Event_Utils.make_perievent_labels(event_file, duration, nsec_before=2, nsec_after=2)
    event_time_dict = Event_Utils.read_events(event_file, min_time, max_time)
    perievent_labels = np.zeros(duration)
    perievent_labels[:] = np.nan
    perievent_indices_dict = {}

    for i, event in enumerate(sorted(event_time_dict.keys())):
        event_time = event_time_dict[event]
        perievent_windows = Event_Utils.make_perievent_windows(
            event_time, nsec_before=nsec_before, nsec_after=nsec_after
        )
        perievent_indices = Event_Utils.get_perievent_indices(
            perievent_windows, fp_freq
        )
        perievent_signals = biosignal[perievent_indices]
        # perievent_signals_normalized = Analyses.normalize_perievent_signals(perievent_signals, fp_freq)
        # areas = Analyses.calculate_area(perievent_signals_normalized)

        Perievent_Plots.make_perievent_plots(
            perievent_signals,
            fp_name,
            biosignal_name,
            event,
            fp_freq,
        )

        perievent_indices_dict[event] = perievent_indices
        perievent_time = perievent_windows.flatten()
        perievent_labels[perievent_time] = i
