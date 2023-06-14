# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:28:04 2023

@author: Yue
adpated from Shadi Sartipi's mice_3signal_june2023.ipynb
"""

import logging

from scipy import signal
from scipy.io import savemat, loadmat
import numpy as np
from numpy.random import seed
seed(1)
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Bidirectional,
    Conv1D,
    TimeDistributed,
    Concatenate,
    Input,
    Flatten,
    MaxPooling1D,
    Dropout
)

from utils import segment_dataset
from model import load_model

log = logging.getLogger(__name__)

######################################
# variable path
# this should be the same as your data.mat directory in your google drive
# as an example
path = "C:\\Users\\Yue\\python_projects\\sleep_scoring\\"
model_path = path + "weighteegxnexemg-[3. 5. 7.].h5"
# testmodel_mice is the name of the folder which can be changed based on the name of the folder you put the data.mat and
# weighteegxnexemg-[3. 5. 7.].h5


def evaluate_model(
    testX1, testX2, testX3
):
    model = load_model(model_path)
    pred = model.predict([testX1, testX2, testX3])
    pred_labels = np.argmax(pred, axis=1)
    probs = np.max(pred, axis=1)
    return pred_labels, probs

#%%

Fs = 512
FS = 1017
fs = 10

eeg = loadmat(path + "data.mat")["trial_eeg"]
emg = loadmat(path + "data.mat")["trial_emg"]
ne = loadmat(path + "data.mat")["trial_ne"]

Ne = np.zeros((ne.shape[0], fs))
for i in range(ne.shape[0]):
    temp = ne[i, :]
    temp = np.squeeze(temp)
    temp = signal.resample(temp, fs)
    Ne[i, :] = temp

test_eeg7, test_emg7, test_ne7 = (
    eeg.reshape([-1, Fs, 1]),
    emg.reshape([-1, Fs, 1]),
    Ne.reshape([-1, fs, 1]),
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


##############################################for test the model
pred_labels, probs  = evaluate_model(EEG, NE, EMG)
final_labels = pred_labels
for i in range(1, len(pred_labels) - 1):
    if pred_labels[i] == 1 and pred_labels[i - 1] == 0 and pred_labels[i + 1] == 0:
        final_labels[i] = 0
    if pred_labels[i] == 2 and pred_labels[i - 1] == 0:
        final_labels[i] = 0

################################

mdict = {
    "pred_labels": final_labels,
    "score": probs,
    "eeg": EEG,
    "emg": EMG,
    "ne": NE,
    "pred_beforcorrecting": pred_labels,
}
#savemat(path + "/finalresults.mat", mdict)
