# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:55:12 2023

@author: yzhao
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat

import torch
import torch.utils.data

from models.msda_v1.utils import (
    rolling_window,
    run_test,
    edit_one,
    edit_two,
    edit_three,
    find_ma,
)
from preprocessing import reshape_sleep_data


code_size_map = {100: 128, 200: 96, 400: 64, 300: 96, 600: 32, 500: 32}


def infer(data: dict, model_path, output_path, num_class=3, batch_size=16):
    Fs = 512
    fs = 10
    model_path += "msda_v1/checkpoints/"
    eeg, emg = reshape_sleep_data(data)
    num_class = data["num_class"]
    # ne = data.get("ne")
    ne = None
    eeg, emg = (eeg.reshape([-1, Fs, 1]), emg.reshape([-1, Fs, 1]))
    eeg_segment = rolling_window(
        eeg, 128, 64
    )  # shape (Time, 1, (data.shape[1] - window) // step + 1, 128)
    emg_segment = rolling_window(
        emg, 128, 64
    )  # shape (Time, 1, (data.shape[1] - window) // step + 1, 128)
    fft = np.abs(np.fft.fft(eeg.squeeze(-1), axis=1))

    eeg = torch.from_numpy(eeg_segment)

    emg = torch.from_numpy(emg_segment)
    fft = torch.from_numpy(fft)

    if ne is not None:
        has_ne = True
        signaling = 100
        ne = signal.resample(ne, fs, axis=1)
        ne = ne.reshape([-1, fs, 1])
        ne = torch.from_numpy(ne)
        test_dataset = torch.utils.data.TensorDataset(
            eeg.float(), ne.float(), emg.float(), fft.float()
        )

    else:
        num_class = 3  # only supports three-class prediction without NE
        ne = np.nan
        has_ne = False
        signaling = 200
        test_dataset = torch.utils.data.TensorDataset(
            eeg.float(), emg.float(), fft.float()
        )
    predictions, confidence = run_test(
        model_path, 3, has_ne, batch_size, test_dataset, signaling
    )

    final_predictions, final_confidence = edit_one(predictions, confidence)
    final_predictions[0] = 0
    final_predictions = edit_three(edit_two(final_predictions))

    if num_class == 4:
        predictions_4class, confidence_4class = run_test(
            model_path, 4, has_ne, batch_size, test_dataset, signaling
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
        "num_class": num_class,
        "eeg_frequency": data["eeg_frequency"],
        "ne_frequency": data["ne_frequency"],
        "eeg": data["eeg"],
        "emg": data["emg"],
        "ne": data["ne"],
    }

    output_path = os.path.splitext(output_path)[0] + f"_msda_{num_class}class.mat"
    savemat(output_path, results)
    return final_predictions, final_confidence, output_path


if __name__ == "__main__":
    mat_file = "./user_test_files/arch_387.mat"
    data = loadmat(mat_file)
    model_path = "./models/"
    final_predictions, final_confidence, output_path = infer(data, model_path, mat_file)
