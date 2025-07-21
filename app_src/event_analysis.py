# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 19:26:15 2025

@author: yzhao
"""

import os
import io
import base64
import zipfile

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


class Event_Utils:
    @staticmethod
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

    @staticmethod
    def count_events(event_time_dict):
        event_count_records = [
            {"Event": event, "Count": len(start_times)}
            for event, start_times in event_time_dict.items()
        ]
        return event_count_records

    @staticmethod
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

    @staticmethod
    def get_perievent_indices(perievent_windows, fp_freq):
        window_duration = perievent_windows.shape[1]
        window_segment = np.arange(int(np.ceil(window_duration * fp_freq)))
        perievent_segments = (
            np.ceil(perievent_windows[:, 0:1] * fp_freq) + window_segment
        )
        return perievent_segments.astype(int)

    @staticmethod
    def make_perievent_labels(event_file, duration, nsec_before=60, nsec_after=60):
        min_time = nsec_before
        max_time = duration - nsec_after
        event_time_dict = Event_Utils.read_events(event_file, min_time, max_time)
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
    @staticmethod
    def plot_perievent_signals(
        ax,
        event,
        perievent_signals,
        nsec_before=60,
        nsec_after=60,
        fp_name="",
        biosignal_name="",
    ):
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
        ax.set_ylim(-10, 10)
        ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{biosignal_name} (dF/F)", fontsize=10, fontweight="bold")
        ax.set_title(f"{fp_name}_{event}", fontsize=12, fontweight="bold")
        # ax.tight_layout()

    @staticmethod
    def plot_mean_perievent_signals(
        ax,
        event,
        perievent_signals,
        nsec_before=60,
        nsec_after=60,
        fp_name="",
        biosignal_name="",
    ):
        perievent_signals_mean = np.mean(perievent_signals, axis=0)
        seg_len = len(perievent_signals_mean)
        t = np.linspace(-nsec_before, nsec_after, seg_len)
        ax.plot(t, perievent_signals_mean)

        # Add reference lines
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)

        # Axis limits and labels
        ax.set_xlim(-nsec_before, nsec_after)
        ax.set_ylim(-10, 10)
        ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"mean {biosignal_name} (dF/F)", fontsize=10, fontweight="bold")
        ax.set_title(f"{fp_name}_{event}_Mean", fontsize=12, fontweight="bold")
        # ax.tight_layout()

    @staticmethod
    def plot_perievent_heatmaps(
        ax, event, perievent_signals, fp_freq, nsec_before=60, nsec_after=60, fp_name=""
    ):
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
        ax.figure.colorbar(im, ax=ax, label="(dF/F)")

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
    """
    Perievent_Plots.make_perievent_plots(
        fp_file, 
        biosignal_name, 
        event_file, 
        nsec_before=60, 
        nsec_after=60,
        width=5,
        height=3,
        as_base64=False
    )
    """

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
        perievent_indices_dict[event] = Event_Utils.get_perievent_indices(
            perievent_windows, fp_freq
        )
        perievent_time = perievent_windows.flatten()
        perievent_labels[perievent_time] = i

    n_cols = 3
    event_count = len(event_time_dict)
    w = 5
    h = 3
    fig, axes = plt.subplots(event_count, n_cols, figsize=(w * n_cols, h * event_count))
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
