# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:55:12 2023

@author: yzhao
"""


import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat

import torch
import torch.utils.data

from msda_v1.utils import (
    rolling_window,
    run_test,
    edit_one,
    edit_two,
    edit_three,
    find_ma,
)


code_size_map = {100: 128, 200: 96, 400: 64, 300: 96, 600: 32, 500: 32}


def infer(data, num_class=3, output_path=None, batch_size=16, signaling=100):
    Fs = 512
    fs = 10
    if output_path is None:
        output_path = "./data_prediction"
    output_path += f"_msda_{num_class}class.mat"

    trial_eeg = data["trial_eeg"]
    trial_emg = data["trial_emg"]
    trial_ne = data["trial_ne"]

    trial_ne = signal.resample(trial_ne, fs, axis=1)

    eeg, emg, ne = (
        trial_eeg.reshape([-1, Fs, 1]),
        trial_emg.reshape([-1, Fs, 1]),
        trial_ne.reshape([-1, fs, 1]),
    )

    eeg_segment = rolling_window(
        eeg, 128, 64
    )  # shape (Time, 1, (data.shape[1] - window) // step + 1, 128)
    emg_segment = rolling_window(
        emg, 128, 64
    )  # shape (Time, 1, (data.shape[1] - window) // step + 1, 128)
    fft = np.abs(np.fft.fft(eeg.squeeze(-1), axis=1))

    eeg = torch.from_numpy(eeg_segment)
    ne = torch.from_numpy(ne)
    emg = torch.from_numpy(emg_segment)
    fft = torch.from_numpy(fft)
    test_dataset = torch.utils.data.TensorDataset(
        eeg.float(), ne.float(), emg.float(), fft.float()
    )

    predictions, confidence = run_test(3, batch_size, test_dataset, signaling)
    final_predictions, final_confidence = edit_one(predictions, confidence)
    final_predictions[0] = 0
    final_predictions = edit_three(edit_two(final_predictions))

    if num_class == 4:
        predictions_4class, confidence_4class = run_test(
            4, batch_size, test_dataset, signaling
        )
        p = np.zeros((len(final_predictions)))
        for i in range(len(final_predictions)):
            if predictions_4class[i] == 1 and confidence_4class[i] > 0.70:
                p[i] = 0
            else:
                p[i] = final_predictions[i]
        final_predictions = np.array(find_ma(p))

    results = {
        "pred_labels": final_predictions,
        "confidence": final_confidence,
        "trial_eeg": trial_eeg,
        "trial_emg": trial_emg,
        "trial_ne": trial_ne,
    }

    savemat(output_path, results)
    return (final_predictions, final_confidence)

if __name__ == "__main__":
    data = loadmat("C:\\Users\\yzhao\\python_projects\\sleep_scoring\\data.mat")
    final_predictions, final_confidence = infer(data, num_class=4)
