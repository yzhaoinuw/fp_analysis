# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:09:21 2023

@author: yzhao
"""

import os
from scipy.io import loadmat, savemat

import run_inference_msda
import run_inference_sdreamer
from postprocessing import postprocess_pred_labels


MODEL_PATH = "./models/"


def run_inference(
    data, model_choice="sdreamer", num_class=3, postprocess=True, output_path=None
):
    num_class = 3
    predictions, confidence = run_inference_sdreamer.infer(data, MODEL_PATH)

    if postprocess:
        predictions = postprocess_pred_labels(predictions, data)

    results = {
        "pred_labels": predictions,
        "confidence": confidence,
        "num_class": 3,
        "eeg_frequency": data["eeg_frequency"],
        "ne_frequency": data["ne_frequency"],
        "eeg": data["eeg"],
        "emg": data["emg"],
        "ne": data["ne"],
    }
    if output_path is not None:
        output_path = (
            os.path.splitext(output_path)[0] + f"_sdreamer_{num_class}class.mat"
        )
        savemat(output_path, results)
    return predictions, confidence, output_path


if __name__ == "__main__":
    model_choice = "sdreamer"
    mat_file = "./user_test_files/arch_387.mat"
    data = loadmat(mat_file)
    predictions, confidence, output_path = run_inference(
        data, model_choice, num_class=3
    )
