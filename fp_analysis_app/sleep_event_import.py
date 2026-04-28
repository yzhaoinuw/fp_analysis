from collections import defaultdict

import numpy as np


SLEEP_BOUT_REQUIRED_COLUMNS = {"sleep_scores", "start", "end", "duration"}

ZERO_BASED_STATE_MAP = {
    0: "wake",
    1: "nrem",
    2: "rem",
    3: "ma",
}

ONE_BASED_STATE_MAP = {
    1: "wake",
    2: "nrem",
    3: "rem",
    4: "ma",
}


def _normalize_columns(df):
    return {str(column).strip().lower(): column for column in df.columns}


def is_sleep_bout_table(df):
    normalized_columns = _normalize_columns(df)
    return SLEEP_BOUT_REQUIRED_COLUMNS.issubset(normalized_columns.keys())


def _get_sleep_bout_columns(df):
    normalized_columns = _normalize_columns(df)
    return {
        column_name: normalized_columns[column_name]
        for column_name in SLEEP_BOUT_REQUIRED_COLUMNS
    }


def _infer_state_map(sleep_scores):
    unique_scores = {
        int(score)
        for score in sleep_scores
        if not np.isnan(score)
    }
    if unique_scores.issubset(ONE_BASED_STATE_MAP):
        return ONE_BASED_STATE_MAP
    if unique_scores.issubset(ZERO_BASED_STATE_MAP):
        return ZERO_BASED_STATE_MAP
    raise ValueError(
        "Unsupported sleep state codes in spreadsheet. Expected only "
        "0-3 or 1-4 in the 'sleep_scores' column."
    )


def sleep_bout_table_to_event_time_dict(df, min_time, max_time):
    columns = _get_sleep_bout_columns(df)
    df_sleep = (
        df[
            [
                columns["sleep_scores"],
                columns["start"],
                columns["end"],
                columns["duration"],
            ]
        ]
        .rename(columns={value: key for key, value in columns.items()})
        .dropna(subset=["sleep_scores", "start"])
        .copy()
    )

    if df_sleep.empty:
        return {}

    df_sleep["sleep_scores"] = df_sleep["sleep_scores"].astype(int)
    df_sleep["start"] = df_sleep["start"].round().astype(int)
    df_sleep["end"] = df_sleep["end"].round().astype(int)
    df_sleep = df_sleep.sort_values(["start", "end"]).reset_index(drop=True)

    state_map = _infer_state_map(df_sleep["sleep_scores"].to_numpy())
    event_times = defaultdict(list)

    for index in range(len(df_sleep) - 1):
        current_state = state_map[df_sleep.at[index, "sleep_scores"]]
        next_state = state_map[df_sleep.at[index + 1, "sleep_scores"]]
        if current_state == next_state:
            continue

        transition_time = df_sleep.at[index + 1, "start"]
        if transition_time < min_time or transition_time > max_time:
            continue

        event_name = f"{current_state}_{next_state}"
        event_times[event_name].append(transition_time)

    return {
        event_name: np.asarray(times, dtype=int)
        for event_name, times in event_times.items()
        if times
    }
