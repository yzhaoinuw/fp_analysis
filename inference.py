# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:09:21 2023

@author: yzhao
"""

import os
from scipy.io import loadmat, savemat

import run_inference_sdreamer
from postprocessing import postprocess_pred_labels


MODEL_PATH = "./models/sdreamer/checkpoints/"


def run_inference(
    mat, model_choice="sdreamer", num_class=3, postprocess=True, output_path=None
):
    # num_class = 3
    predictions, confidence = run_inference_sdreamer.infer(mat, MODEL_PATH)
    mat["pred_labels"] = predictions
    mat["confidence"] = confidence
    # mat["num_class"] = 3
    if postprocess:
        predictions = postprocess_pred_labels(mat)
        mat["pred_labels"] = predictions

    if output_path is not None:
        output_path = (
            os.path.splitext(output_path)[0] + f"_sdreamer_{num_class}class.mat"
        )
        savemat(output_path, mat)
    return mat, output_path


if __name__ == "__main__":
    model_choice = "sdreamer"
    data_path = "C:/Users/yzhao/python_projects/sleep_scoring/610Hz data/"
    mat_file = os.path.join(data_path, "20240808_3_FP_Temp_BS_rep.mat")
    mat = loadmat(mat_file)
    mat, output_path = run_inference(mat, model_choice, postprocess=False)
