from pathlib import Path


def build_analysis_config_dirname(
    selected_signals,
    baseline_window,
    analysis_window,
):
    sorted_signals = sorted(str(signal) for signal in selected_signals)
    signal_key = "_".join(sorted_signals)
    return f"{signal_key}_bw{baseline_window}_aw{analysis_window}"


def get_analysis_export_dir(
    base_dir,
    selected_signals,
    baseline_window,
    analysis_window,
):
    base_dir = Path(base_dir)
    config_dirname = build_analysis_config_dirname(
        selected_signals=selected_signals,
        baseline_window=baseline_window,
        analysis_window=analysis_window,
    )
    return base_dir / config_dirname


def build_analysis_description_text(
    mat_filepaths,
    export_dir,
    selected_signals,
    baseline_window,
    analysis_window,
    event_names,
):
    mat_paths = [Path(path) for path in mat_filepaths]
    export_dir = Path(export_dir)
    event_names = [str(event_name) for event_name in event_names]

    lines = [
        "Analysis export description",
        f"Export folder: {export_dir}",
        (
            "Selected signals (sorted folder key): "
            f"{', '.join(sorted(selected_signals))}"
        ),
        f"Baseline window (s): {baseline_window}",
        f"Analysis window (s): {analysis_window}",
        f"Event types: {', '.join(event_names)}",
        "Source MAT paths:",
    ]
    lines.extend(f"- {mat_path}" for mat_path in mat_paths)
    return "\n".join(lines) + "\n"


def _read_existing_mat_paths(description_path):
    if not description_path.exists():
        return []

    lines = description_path.read_text(encoding="utf-8").splitlines()
    try:
        start_index = lines.index("Source MAT paths:") + 1
    except ValueError:
        return []

    mat_paths = []
    for line in lines[start_index:]:
        if not line.startswith("- "):
            continue
        mat_paths.append(Path(line[2:]))
    return mat_paths


def write_analysis_description_file(
    export_dir,
    mat_filepath,
    selected_signals,
    baseline_window,
    analysis_window,
    event_names,
):
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    description_path = export_dir / "data_description.txt"
    mat_filepath = Path(mat_filepath)
    existing_mat_paths = _read_existing_mat_paths(description_path)
    combined_mat_paths = []
    for path in [*existing_mat_paths, mat_filepath]:
        if path not in combined_mat_paths:
            combined_mat_paths.append(path)
    description_text = build_analysis_description_text(
        mat_filepaths=combined_mat_paths,
        export_dir=export_dir,
        selected_signals=selected_signals,
        baseline_window=baseline_window,
        analysis_window=analysis_window,
        event_names=event_names,
    )
    description_path.write_text(description_text, encoding="utf-8")
    return description_path
