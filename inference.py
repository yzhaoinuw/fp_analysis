# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:09:21 2023

@author: yzhao
"""


from scipy.io import loadmat

import run_inference_msda
import run_inference_sdreamer


MODEL_PATH = "./model_save_states/"


def run_inference(data, model_choice="msda", num_class=None, output_path=None):
    if model_choice == "msda":
        if num_class is None:
            num_class = 3

        predictions, confidence, output_path = run_inference_msda.infer(
            data, MODEL_PATH, num_class=num_class, output_path=output_path
        )

    else:
        predictions, confidence, output_path = run_inference_sdreamer.infer(
            data, MODEL_PATH, output_path
        )
    return predictions, confidence, output_path


if __name__ == "__main__":
    model_choice = "msda"
    data = loadmat("data.mat")
    predictions, confidence, output_path = run_inference(
        data, model_choice, num_class=3
    )
