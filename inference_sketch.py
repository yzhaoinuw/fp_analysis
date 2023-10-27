# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:55:12 2023

@author: yzhao
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np

from scipy import signal
from scipy.io import loadmat

import torch
import torch.utils.data

from msda_v1.utils import rolling_window, test, edit_one, edit_two, edit_three

# %%
num_class = 3
data = loadmat("./data.mat")
model_path = f"./msda_{num_class}class_v1.pth"
output_path = "./results.mat"

if model_path is None:
    model_path = f"./msda_{num_class}class_v1.pth"
if output_path is None:
    output_path = "./results.mat"

Fs = 512
FS = 1017
fs = 10
eeg = data["trial_eeg"]
emg = data["trial_emg"]
ne = data["trial_ne"]

ne_resample = signal.resample(ne, fs, axis=1)

test_eeg, test_emg, test_ne = (
    eeg.reshape([-1, Fs, 1]),
    emg.reshape([-1, Fs, 1]),
    ne_resample.reshape([-1, fs, 1]),
)

# Compute the rolling window
eeg_segment = rolling_window(test_eeg, 128, 64)
emg_segment = rolling_window(test_emg, 128, 64)
test_fft = np.abs(np.fft.fft(test_eeg.squeeze(-1), axis=1))

# %%

flow1 = torch.from_numpy(eeg_segment)
flow2 = torch.from_numpy(test_ne)
flow3 = torch.from_numpy(emg_segment)
flow4 = torch.from_numpy(test_fft)
half_batch = 100
target_dataset = torch.utils.data.TensorDataset(flow1, flow2, flow3, flow4)

# %%

model_MA = "msda_version1_MA_by_Shadi.pth"
signaling = 100
test_loader = torch.utils.data.DataLoader(
    dataset=target_dataset, batch_size=half_batch, shuffle=False
)

result_pred1 = test(model_path, test_loader, signaling)

# %%

pred1 = []
for i in range(len(result_pred1)):
    temp = np.array(result_pred1[i].data.numpy())
    pred1.append(temp)
pred1 = np.array(pred1)
pred1 = np.concatenate(pred1)

pred_final1 = np.argmax(pred1, axis=1)
score_final1 = np.max(pred1, axis=1)
"""
#######################################final results for three class ##################################
final_label1, final_score1 = edit_one(pred_final1, score_final1)
final_label1[0] = 0
final_label1 = edit_three(edit_two(final_label1))
"""
