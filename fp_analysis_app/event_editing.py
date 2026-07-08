import numpy as np


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
        for event_name, event_times in (event_time_dict or {}).items()
    }


def store_data_to_event_time_dict(store_data):
    return {
        event_name: np.asarray(event_times)
        for event_name, event_times in (store_data or {}).items()
        if len(event_times)
    }
