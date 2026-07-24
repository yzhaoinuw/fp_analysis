from pathlib import Path

from fp_analysis_app.event_analysis import Perievent_Plots
from fp_analysis_app.export_settings import (
    get_analysis_export_dir,
    write_analysis_description_file,
)


SIGNAL_ANALYSIS_EXPORT_TYPES = (
    "mean_trace",
    "auc",
    "positive_peak_value",
    "negative_peak_value",
    "first_peak_time",
    "decay_time",
)

CROSS_CORRELATION_ANALYSIS_EXPORT_TYPES = (
    "cross_correlation",
    "strongest_cross_correlation_lag",
)

ANALYSIS_EXPORT_TYPE_ORDER = (
    *SIGNAL_ANALYSIS_EXPORT_TYPES,
    *CROSS_CORRELATION_ANALYSIS_EXPORT_TYPES,
)

ANALYSIS_EXPORT_TYPE_LABELS = {
    "mean_trace": "Mean trace",
    "auc": "AUC",
    "positive_peak_value": "Positive peak value",
    "negative_peak_value": "Negative peak value",
    "first_peak_time": "First peak time",
    "decay_time": "Decay time",
    "cross_correlation": "Mean cross-correlation",
    "strongest_cross_correlation_lag": "Strongest cross-correlation lag",
}


def get_analysis_type_labels(analysis_types):
    return [
        ANALYSIS_EXPORT_TYPE_LABELS[analysis_type]
        for analysis_type in ANALYSIS_EXPORT_TYPE_ORDER
        if analysis_type in analysis_types
    ]


def get_signal_export_workbook_name(
    analysis_type,
    signal_name,
    baseline_window,
    analysis_window,
):
    if analysis_type == "mean_trace":
        return f"{signal_name}_bw{baseline_window}_aw{analysis_window}.xlsx"
    if analysis_type == "auc":
        return f"{signal_name}_auc_bw{baseline_window}_aw{analysis_window}.xlsx"
    if analysis_type == "positive_peak_value":
        return (
            f"{signal_name}_positive_peak_value_"
            f"bw{baseline_window}_aw{analysis_window}.xlsx"
        )
    if analysis_type == "negative_peak_value":
        return (
            f"{signal_name}_negative_peak_value_"
            f"bw{baseline_window}_aw{analysis_window}.xlsx"
        )
    if analysis_type == "first_peak_time":
        return (
            f"{signal_name}_first_peak_time_"
            f"bw{baseline_window}_aw{analysis_window}.xlsx"
        )
    if analysis_type == "decay_time":
        return f"{signal_name}_decay_time_bw{baseline_window}_aw{analysis_window}.xlsx"
    raise ValueError(f"Unsupported signal analysis export type: {analysis_type}")


def get_cross_correlation_workbook_name(sig_a, sig_b, baseline_window, analysis_window):
    return (
        f"{sig_a}_{sig_b}_cross_correlation_"
        f"bw{baseline_window}_aw{analysis_window}.xlsx"
    )


def get_strongest_cross_correlation_workbook_name(
    sig_a,
    sig_b,
    baseline_window,
    analysis_window,
):
    return (
        f"{sig_a}_{sig_b}_strongest_cross_correlation_time_lag_"
        f"bw{baseline_window}_aw{analysis_window}.xlsx"
    )


def get_available_analysis_types(export_payload):
    available_types = []
    signal_event_exports = export_payload.get("signal_event_exports", {})
    for analysis_type in SIGNAL_ANALYSIS_EXPORT_TYPES:
        if signal_event_exports.get(analysis_type):
            available_types.append(analysis_type)

    if export_payload.get("cross_correlation_event_exports"):
        available_types.append("cross_correlation")
    if export_payload.get("strongest_cross_correlation_event_exports"):
        available_types.append("strongest_cross_correlation_lag")
    return available_types


def build_analysis_type_checklist_options(export_payload):
    available_types = set(get_available_analysis_types(export_payload))
    return [
        {
            "label": ANALYSIS_EXPORT_TYPE_LABELS[analysis_type],
            "value": analysis_type,
            "disabled": analysis_type not in available_types,
        }
        for analysis_type in ANALYSIS_EXPORT_TYPE_ORDER
        if analysis_type in available_types
    ]


def get_analysis_type_checklist_values(options, remembered_analysis_types=None):
    available_values = [
        option["value"] for option in options if not option.get("disabled")
    ]
    remembered_values = [
        analysis_type
        for analysis_type in (remembered_analysis_types or [])
        if analysis_type in available_values
    ]
    return remembered_values or available_values


def write_analysis_workbooks(
    primary_dir,
    fallback_dir,
    export_payload,
    selected_analysis_types,
):
    selected_analysis_types = [
        analysis_type
        for analysis_type in ANALYSIS_EXPORT_TYPE_ORDER
        if analysis_type in set(selected_analysis_types)
    ]
    if not selected_analysis_types:
        raise ValueError("At least one analysis type must be selected.")

    primary_dir = Path(primary_dir)
    fallback_dir = Path(fallback_dir)
    selected_signals = tuple(export_payload["selected_signals"])
    baseline_window = export_payload["baseline_window"]
    analysis_window = export_payload["analysis_window"]

    config_dir = get_analysis_export_dir(
        base_dir=primary_dir,
        selected_signals=selected_signals,
        baseline_window=baseline_window,
        analysis_window=analysis_window,
    )
    fallback_config_dir = get_analysis_export_dir(
        base_dir=fallback_dir,
        selected_signals=selected_signals,
        baseline_window=baseline_window,
        analysis_window=analysis_window,
    )

    def export_all_workbooks(target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        write_analysis_description_file(
            export_dir=target_dir,
            mat_filepath=export_payload["mat_filepath"],
            selected_signals=selected_signals,
            baseline_window=baseline_window,
            analysis_window=analysis_window,
            event_names=export_payload["event_names"],
            saved_analysis_types=get_analysis_type_labels(selected_analysis_types),
        )

        signal_event_exports = export_payload.get("signal_event_exports", {})
        for analysis_type in selected_analysis_types:
            if analysis_type not in SIGNAL_ANALYSIS_EXPORT_TYPES:
                continue
            for sig, event_sheet_dfs in signal_event_exports.get(
                analysis_type,
                {},
            ).items():
                workbook_save_path = target_dir / get_signal_export_workbook_name(
                    analysis_type=analysis_type,
                    signal_name=sig,
                    baseline_window=baseline_window,
                    analysis_window=analysis_window,
                )
                if analysis_type == "mean_trace":
                    Perievent_Plots.export_mean_trace_workbook(
                        workbook_save_path=workbook_save_path,
                        event_sheet_dfs=event_sheet_dfs,
                    )
                else:
                    Perievent_Plots.export_occurrence_value_workbook(
                        workbook_save_path=workbook_save_path,
                        event_sheet_dfs=event_sheet_dfs,
                        index_column="event_index",
                    )

        if (
            "cross_correlation" in selected_analysis_types
            and export_payload.get("cross_correlation_event_exports")
        ):
            sig_a, sig_b = selected_signals
            workbook_save_path = target_dir / get_cross_correlation_workbook_name(
                sig_a,
                sig_b,
                baseline_window,
                analysis_window,
            )
            Perievent_Plots.export_cross_correlation_workbook(
                workbook_save_path=workbook_save_path,
                event_sheet_dfs=export_payload["cross_correlation_event_exports"],
            )

        if (
            "strongest_cross_correlation_lag" in selected_analysis_types
            and export_payload.get("strongest_cross_correlation_event_exports")
        ):
            sig_a, sig_b = selected_signals
            workbook_save_path = (
                target_dir
                / get_strongest_cross_correlation_workbook_name(
                    sig_a,
                    sig_b,
                    baseline_window,
                    analysis_window,
                )
            )
            Perievent_Plots.export_strongest_cross_correlation_workbook(
                workbook_save_path=workbook_save_path,
                event_sheet_dfs=export_payload[
                    "strongest_cross_correlation_event_exports"
                ],
            )

    try:
        export_all_workbooks(config_dir)
        return (
            config_dir,
            "Analysis spreadsheets saved next to the input MAT file in "
            f"'{config_dir}'.",
        )
    except OSError:
        export_all_workbooks(fallback_config_dir)
        return (
            fallback_config_dir,
            "Could not save analysis spreadsheets next to the input MAT file. "
            "Saved them to the app spreadsheet folder instead: "
            f"'{fallback_config_dir}'.",
        )
