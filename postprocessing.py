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


def modify_SWS(df, pred_labels):
    """eliminate short SWS (< 5s)"""
    pred_labels_post = pred_labels.copy()
    df_short_SWS = df[(df["pred_labels"] == 1) & (df["duration"] <= 5)]
    for index, row in df_short_SWS.iterrows():
        if index < 1 or index >= len(df):
            continue
        # if df.loc[index-1]["pred_labels"] == 0 and df.loc[index+1]["pred_labels"] == 0:
        df.at[index, "pred_labels"] = 0
        start, end = row["start"], row["end"]
        pred_labels_post[start : end + 1] = 0
    return pred_labels_post, df


def modify_REM(df, pred_labels):
    """eliminate short REM (< 7s)"""
    pred_labels_post = pred_labels.copy()
    df_rem = df[df["pred_labels"] == 2]
    for index, row in df_rem.iterrows():
        rem = True
        start, end = row["start"], row["end"]
        prev_start, prev_end = df.loc[index - 1]["start"], df.loc[index - 1]["end"]
        next_start, next_end = df.loc[index + 1]["start"], df.loc[index + 1]["end"]
        if row["duration"] < 7:
            df.at[index, "pred_labels"] = 1
            pred_labels_post[start : end + 1] = 1
            rem = False

        # if preceded by Wake, change to Wake
        if rem and df.loc[index - 1]["pred_labels"] == 0:
            if df.loc[index - 1]["duration"] < row["duration"]:
                df.at[index - 1, "pred_labels"] = 2
                pred_labels_post[prev_start : prev_end + 1] = 2
            else:
                df.at[index, "pred_labels"] = 0
                pred_labels_post[start : end + 1] = 0
                rem = False

        # if proceeded by a SWS, change to
        if rem and df.loc[index + 1]["pred_labels"] == 1:
            if df.loc[index + 1]["duration"] < row["duration"]:
                df.at[index + 1, "pred_labels"] = 2
                pred_labels_post[next_start : next_end + 1] = 2
            # if the previous segment was modified to REM
            elif df.loc[index - 1]["pred_labels"] == 2:
                df.at[index, "pred_labels"] = 0
                pred_labels_post[start : end + 1] = 0
            else:
                df.at[index, "pred_labels"] = 1
                pred_labels_post[start : end + 1] = 1
    return pred_labels_post, df


def postprocess_pred_labels(pred_labels, return_table=False):
    df = get_sleep_segments(pred_labels)
    pred_labels_post, df = modify_SWS(df, pred_labels)
    df = merge_consecutive_pred_labels(df)
    pred_labels_post, df = modify_REM(df, pred_labels_post)
    df = merge_consecutive_pred_labels(df)
    if return_table:
        return pred_labels_post, df
    return pred_labels_post


# %%
if __name__ == "__main__":
    data_path = ".\\user_test_files\\"
    mat_file = "arch_387_sdreamer_3class_augment10.mat"
    mat = loadmat(os.path.join(data_path, mat_file))
    pred_labels = mat.get("pred_labels").flatten()
    pred_labels_post, df = postprocess_pred_labels(pred_labels, return_table=True)
    df.to_csv("arch_387_table_mod.csv")
