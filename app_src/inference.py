# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:09:21 2023

@author: yzhao
"""

import os
from scipy.io import loadmat, savemat

import app_src.run_inference_ne as run_inference_ne
import app_src.run_inference_sdreamer as run_inference_sdreamer
from app_src.postprocessing import postprocess_pred_labels


def run_inference(
    mat,
    model_path="./models/sdreamer/checkpoints/",
    num_class=3,
    postprocess=False,
    output_path=None,
    save_inference=False,
):
    # num_class = 3
    ne = mat.get("ne")
    ne_tag = ""
    post_tag = ""
    if ne is not None and len(ne) != 0:
        ne_tag = "_ne"
        predictions, confidence = run_inference_ne.infer(mat, model_path)
    else:
        predictions, confidence = run_inference_sdreamer.infer(mat, model_path)

    mat["pred_labels"] = predictions
    mat["confidence"] = confidence
    if postprocess:
        post_tag = "_post"
        predictions = postprocess_pred_labels(mat)
        mat["pred_labels"] = predictions

    if output_path is not None:
        output_path = (
            os.path.splitext(output_path)[0] + f"_sdreamer{ne_tag}{post_tag}.mat"
        )
        if save_inference:
            savemat(output_path, mat)
    return mat, output_path


if __name__ == "__main__":
    data_path = "../user_test_files/"
    mat_file = os.path.join(data_path, "20241113_1_263_2_259_24h_test/bin_1.mat")
    mat = loadmat(mat_file)
    mat, output_path = run_inference(
        mat, model_path="../models/sdreamer/checkpoints/", postprocess=False
    )
