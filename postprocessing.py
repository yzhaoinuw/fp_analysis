# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 22:14:49 2024

@author: yzhao
"""

import os

import numpy as np
import pandas as pd
from scipy.io import loadmat


def get_sleep_segments(pred_labels):
    transition_indices = np.flatnonzero(np.diff(pred_labels))
    transition_indices = np.append(transition_indices, len(pred_labels) - 1)

    REM_end_indices = np.flatnonzero(pred_labels[transition_indices] == 2)
    REM_start_indices = REM_end_indices - 1
    REM_end_indices = transition_indices[REM_end_indices]
    REM_start_indices = transition_indices[REM_start_indices] + 1

    wake_end_indices = np.flatnonzero(pred_labels[transition_indices] == 0)
    wake_start_indices = wake_end_indices - 1
    wake_end_indices = transition_indices[wake_end_indices]
    wake_start_indices = transition_indices[wake_start_indices] + 1

    SWS_end_indices = np.flatnonzero(pred_labels[transition_indices] == 1)
    SWS_start_indices = SWS_end_indices - 1
    SWS_end_indices = transition_indices[SWS_end_indices]
    SWS_start_indices = transition_indices[SWS_start_indices] + 1

    df_rem = pd.DataFrame()
    df_rem["pred_labels"] = pd.Series(np.array([2] * REM_end_indices.size))
    df_rem["start"] = pd.Series(REM_start_indices)
    df_rem["end"] = pd.Series(REM_end_indices)

    df_wake = pd.DataFrame()
    df_wake["pred_labels"] = pd.Series(np.array([0] * wake_end_indices.size))
    df_wake["start"] = pd.Series(wake_start_indices)
    df_wake["end"] = pd.Series(wake_end_indices)

    df_SWS = pd.DataFrame()
    df_SWS["pred_labels"] = pd.Series(np.array([1] * SWS_end_indices.size))
    df_SWS["start"] = pd.Series(SWS_start_indices)
    df_SWS["end"] = pd.Series(SWS_end_indices)

    frames = [df_rem, df_wake, df_SWS]
    df = pd.concat(frames)
    df = df.sort_values(by=["end"], ignore_index=True)
    df.at[0, "start"] = 0
    df["duration"] = df["end"] - df["start"] + 1
    return df


def merge_consecutive_pred_labels(df):
    df["group"] = (df["pred_labels"] != df["pred_labels"].shift()).cumsum()
    # Group by 'id' and 'group' and then aggregate
    df_merged = (
        df.groupby(["pred_labels", "group"])
        .agg(
            pred_labels=("pred_labels", "first"),
            start=("start", "min"),
            end=("end", "max"),
            duration=("duration", "sum"),
        )
        .reset_index(drop=True)
    )
    df_merged = df_merged.sort_values(by=["end"], ignore_index=True)
    return df_merged


def evaluate_Wake(
    emg, emg_frequency, start, end, prev_start, prev_end, next_start, next_end
):
    emg_seg = emg[int(start * emg_frequency) : int((end + 1) * emg_frequency)]
    prev_emg_seg = emg[
        int(prev_start * emg_frequency) : int((prev_end + 1) * emg_frequency)
    ]
    next_emg_seg = emg[
        int(next_start * emg_frequency) : int((next_end + 1) * emg_frequency)
    ]
    # check 1: NE increases
    high_emg = (np.percentile(emg_seg, q=75) > np.percentile(prev_emg_seg, q=95)) and (
        np.percentile(emg_seg, q=85) > np.percentile(next_emg_seg, q=95)
    )
    # check 2: NE changes more steeply
    # NE_steep_increase = np.mean(abs(np.diff(next_ne_seg))) > np.mean(abs(np.diff(ne_segment)))
    return high_emg


def modify_Wake(df, emg, emg_frequency):
    """change short Wake (<= 5s) if needed"""
    df_short_Wake = df[(df["pred_labels"] == 0) & (df["duration"] <= 2)]
    for index, row in df_short_Wake.iterrows():
        start, end = row["start"], row["end"]
        prev_start, prev_end = df.loc[index - 1]["start"], df.loc[index - 1]["end"]
        next_start, next_end = df.loc[index + 1]["start"], df.loc[index + 1]["end"]
        if evaluate_Wake(
            emg, emg_frequency, start, end, prev_start, prev_end, next_start, next_end
        ):
            continue

        nearest_seg_duration = 0
        if index >= 1:
            nearest_seg_duration = df.loc[index - 1]["duration"]
            label = df.loc[index - 1]["pred_labels"]
        if index < len(df) - 1:
            if df.loc[index + 1]["duration"] > nearest_seg_duration:
                label = df.loc[index + 1]["pred_labels"]

        df.at[index, "pred_labels"] = label

    return df


def modify_SWS(df):
    """eliminate short SWS (<= 5s)"""
    df_short_SWS = df[(df["pred_labels"] == 1) & (df["duration"] <= 5)]
    for index, row in df_short_SWS.iterrows():
        change = 0
        if index >= 1:
            if df.loc[index - 1]["pred_labels"] == 0:
                change += 1
        else:
            change += 1

        if index < len(df) - 1:
            if df.loc[index + 1]["pred_labels"] == 0:
                change += 1
        else:
            change += 1

        if change == 2:
            df.at[index, "pred_labels"] = 0

    return df


def evaluate_REM(ne, ne_frequency, start, end):
    ne_segment = ne[int(start * ne_frequency) : int(end * ne_frequency)]
    next_ne_seg = ne[int((end + 1) * ne_frequency) : int((end + 10) * ne_frequency)]

    # check 1: NE increases
    NE_increase = np.median(next_ne_seg) > np.percentile(ne_segment, q=85)
    # check 2: NE changes more steeply
    # NE_steep_increase = np.mean(abs(np.diff(next_ne_seg))) > np.mean(abs(np.diff(ne_segment)))
    return NE_increase


def modify_REM(df, ne, ne_frequency):
    """eliminate short REM (< 7s)"""
    df_rem = df[df["pred_labels"] == 2]
    for index, row in df_rem.iterrows():
        rem = True
        start, end = row["start"], row["end"]
        prev_start = df.loc[index - 1]["start"]
        duration = row["duration"]
        """
        if duration <= 7:
            df.at[index, "pred_labels"] = 1
            rem = False
        """

        # if preceded by Wake, make changes
        if rem and df.loc[index - 1]["pred_labels"] == 0:
            if df.loc[index - 1]["duration"] < duration:
                df.at[index - 1, "pred_labels"] = 2
            else:
                df.at[index, "pred_labels"] = 0
                rem = False

        # if the previous segment was modified to REM
        elif df.loc[index - 1]["pred_labels"] == 2:
            start = prev_start
            duration = end - start

        # if proceeded by a SWS, change to
        if rem and df.loc[index + 1]["pred_labels"] == 1:
            if df.loc[index + 1]["duration"] < duration:
                df.at[index + 1, "pred_labels"] = 2

            else:
                df.at[index, "pred_labels"] = 1
                df.at[index - 1, "pred_labels"] = 1
                rem = False
        """ 
        # if NE characteristics do not support REM prediction, change to SWS
        if rem and not evaluate_REM(ne, ne_frequency, start, end):
            df.at[index, "pred_labels"] = 1
        """

    return df


def edit_sleep_scores(sleep_scores, df):
    pred_labels_post = sleep_scores.copy()
    for index, row in df.iterrows():
        start, end = row["start"], row["end"]
        label = row["pred_labels"]
        pred_labels_post[start : end + 1] = label
    return pred_labels_post


def postprocess_pred_labels(pred_labels, data, return_table=False):
    ne_frequency = data.get("ne_frequency").item()
    ne = data.get("ne").flatten()
    emg_frequency = data.get(
        "eeg_frequency"
    ).item()  # eeg and emg have the same frequency
    emg = data.get("emg").flatten()
    df = get_sleep_segments(pred_labels)
    df = modify_Wake(df, emg, emg_frequency)
    df = merge_consecutive_pred_labels(df)
    df = modify_SWS(df)
    df = merge_consecutive_pred_labels(df)
    df = modify_REM(df, ne, ne_frequency)
    df = merge_consecutive_pred_labels(df)
    # df = modify_REM(df, ne, ne_frequency)
    # df = merge_consecutive_pred_labels(df)
    pred_labels_post = edit_sleep_scores(pred_labels, df)
    if return_table:
        return pred_labels_post, df
    return pred_labels_post


# %%
if __name__ == "__main__":
    data_path = ".\\user_test_files\\"
    mat_file = "arch_387_sdreamer_3class_augment10.mat"
    mat = loadmat(os.path.join(data_path, mat_file))
    pred_labels = mat.get("pred_labels").flatten()
    pred_labels_post, df = postprocess_pred_labels(
        pred_labels, data=mat, return_table=True
    )
    df.to_csv("arch_387_table_mod.csv")
