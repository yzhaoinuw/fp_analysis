# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:28:04 2023

@author: Yue
adpated from Shadi Sartipi's mice_3signal_june2023.ipynb
"""


from scipy import signal
from scipy.io import savemat, loadmat
import numpy as np
from numpy.random import seed

seed(1)

from utils import segment_dataset
from model import Sleep_Scoring_Model


def run_inference(data, model_path=None, output_path=None):
    if model_path is None:
        model_path = "./weighteegxnexemg-[3. 5. 7.].h5"
    if output_path is None:
        output_path = "./results.mat"

    fs = 10
    eeg = data["trial_eeg"]
    emg = data["trial_emg"]
    ne = data["trial_ne"]

    ne_resample = signal.resample(ne, fs, axis=1)

    test_eeg7, test_emg7, test_ne7 = (
        np.expand_dims(eeg, axis=-1),
        np.expand_dims(emg, axis=-1),
        np.expand_dims(ne_resample, axis=-1),
    )

    test7 = np.zeros((test_eeg7.shape[0], 7, 128, 1))
    for tr in range(test_eeg7.shape[0]):
        temp1 = np.squeeze(test_eeg7[tr, :, :])
        temp3 = segment_dataset(temp1, 128, 64)
        temp4 = temp3.reshape(7, 128, 1)
        test7[tr, :, :, :] = temp4

    test7_emg = np.zeros((test_emg7.shape[0], 7, 128, 1))
    for tr in range(test_emg7.shape[0]):
        temp1 = np.squeeze(test_emg7[tr, :, :])
        temp3 = segment_dataset(temp1, 128, 64)
        temp4 = temp3.reshape(7, 128, 1)
        test7_emg[tr, :, :, :] = temp4

    EEG = test7
    EMG = test7_emg
    NE = test_ne7

    model = Sleep_Scoring_Model(model_path)
    pred_labels, probs = model.infer(EEG, NE, EMG)
    final_labels = pred_labels
    for i in range(1, len(pred_labels) - 1):
        if pred_labels[i] == 1 and pred_labels[i - 1] == 0 and pred_labels[i + 1] == 0:
            final_labels[i] = 0
        if pred_labels[i] == 2 and pred_labels[i - 1] == 0:
            final_labels[i] = 0

    results = {
        "pred_labels": final_labels,
        "scores": probs,
        "eeg": eeg,
        "emg": emg,
        "ne": ne,
        # "pred_beforcorrecting": pred_labels,
    }
    savemat(output_path, results)


if __name__ == "__main__":
    data = loadmat("C:\\Users\\Yue\\python_projects\\sleep_scoring\\data.mat")
    run_inference(data)
