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
