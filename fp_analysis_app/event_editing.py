import numpy as np

EVENT_TIME_NAMES_MAT_FIELD = "fp_event_names"
EVENT_TIME_VALUES_MAT_FIELD = "fp_event_times"


def selected_data_to_x_span(selected_data):
    if not selected_data:
        return None

    selected_range = selected_data.get("range") or {}
    x_range = selected_range.get("x")
    if x_range and len(x_range) >= 2:
        x0, x1 = float(x_range[0]), float(x_range[1])
        return min(x0, x1), max(x0, x1)

    x_values = [
        float(point["x"])
        for point in selected_data.get("points", [])
        if point.get("x") is not None
    ]
    if not x_values:
        return None
    return min(x_values), max(x_values)


def remove_event_times_in_span(event_time_dict, x_span):
    if not event_time_dict or not x_span:
        return event_time_dict or {}, 0

    x0, x1 = sorted(float(value) for value in x_span)
    filtered_events = {}
    removed_count = 0

    for event_name, event_times in event_time_dict.items():
        event_times = np.asarray(event_times)
        if event_times.size == 0:
            continue

        remove_mask = (event_times >= x0) & (event_times <= x1)
        removed_count += int(np.count_nonzero(remove_mask))
        remaining_times = event_times[~remove_mask]
        if remaining_times.size:
            filtered_events[event_name] = remaining_times

    return filtered_events, removed_count


def is_remove_event_key(keyboard_event):
    return bool(
        keyboard_event
        and keyboard_event.get("key") in {"Delete", "Backspace"}
    )


def event_time_dict_to_store_data(event_time_dict):
    return {
        event_name: np.asarray(event_times).tolist()
        for event_name, event_times in normalize_event_time_dict(event_time_dict).items()
    }


def store_data_to_event_time_dict(store_data):
    return normalize_event_time_dict(store_data)


def normalize_event_time_dict(event_time_dict):
    normalized = {}
    for event_name, event_times in (event_time_dict or {}).items():
        event_times = np.asarray(event_times, dtype=float).ravel()
        event_times = event_times[np.isfinite(event_times)]
        if event_times.size:
            normalized[str(event_name)] = event_times
    return normalized


def event_time_dicts_equal(left, right):
    left = normalize_event_time_dict(left)
    right = normalize_event_time_dict(right)
    if set(left.keys()) != set(right.keys()):
        return False
    return all(np.array_equal(left[event], right[event]) for event in left)


def copy_event_time_dict(event_time_dict):
    return {
        event_name: event_times.copy()
        for event_name, event_times in normalize_event_time_dict(event_time_dict).items()
    }


def event_time_dict_to_mat_arrays(event_time_dict):
    event_time_dict = normalize_event_time_dict(event_time_dict)
    event_names = np.asarray(list(event_time_dict.keys()), dtype=object)
    event_times = np.empty((len(event_names),), dtype=object)
    for index, event_name in enumerate(event_names):
        event_times[index] = event_time_dict[str(event_name)]
    return event_names, event_times


def event_time_dict_from_mat_arrays(event_names, event_times):
    if event_names is None or event_times is None:
        return {}

    names = np.asarray(event_names, dtype=object).ravel()
    times = np.asarray(event_times, dtype=object).ravel()
    if names.size == 0 or times.size == 0:
        return {}

    event_time_dict = {}
    for raw_name, raw_times in zip(names, times):
        name_array = np.asarray(raw_name).squeeze()
        event_name = str(name_array.item() if name_array.shape == () else name_array)
        time_array = np.asarray(raw_times, dtype=float).ravel()
        time_array = time_array[np.isfinite(time_array)]
        if time_array.size:
            event_time_dict[event_name] = time_array
    return event_time_dict


def mat_has_saved_event_time_dict(mat):
    return (
        EVENT_TIME_NAMES_MAT_FIELD in mat
        and EVENT_TIME_VALUES_MAT_FIELD in mat
    )


def event_time_dict_from_mat(mat):
    if not mat_has_saved_event_time_dict(mat):
        return {}
    return event_time_dict_from_mat_arrays(
        mat.get(EVENT_TIME_NAMES_MAT_FIELD),
        mat.get(EVENT_TIME_VALUES_MAT_FIELD),
    )


def write_event_time_dict_to_mat(mat, event_time_dict):
    event_names, event_times = event_time_dict_to_mat_arrays(event_time_dict)
    mat[EVENT_TIME_NAMES_MAT_FIELD] = event_names
    mat[EVENT_TIME_VALUES_MAT_FIELD] = event_times
    return mat
