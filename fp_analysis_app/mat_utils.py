from __future__ import annotations

import numpy as np


def get_fp_signal_names(mat) -> list[str]:
    """Return FP signal names as a list, including single-name MAT payloads."""

    signal_names = mat.get("fp_signal_names", [])
    if isinstance(signal_names, str):
        return [signal_names]
    if isinstance(signal_names, np.ndarray):
        if signal_names.ndim == 0:
            return [str(signal_names.item())]
        return signal_names.tolist()
    return list(signal_names)


def get_visualization_signal_names_and_frequency(mat):
    """
    Return signal names plus the matching frequency for figure construction.

    Falls back to the NE signal and NE frequency when FP signal metadata is
    absent.
    """

    signal_names = get_fp_signal_names(mat)
    frequency = mat.get("fp_frequency")

    if not signal_names:
        if "ne" not in mat:
            raise KeyError(
                "MAT data must include 'fp_signal_names' or an 'ne' signal "
                "for visualization."
            )
        signal_names = ["ne"]
        frequency = mat.get("ne_frequency")

    if frequency is None:
        raise KeyError(
            "MAT data must include 'fp_frequency' or 'ne_frequency' for "
            "visualization."
        )

    return signal_names, float(np.asarray(frequency).item())
