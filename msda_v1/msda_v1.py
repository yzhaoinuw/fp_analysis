import warnings

warnings.filterwarnings("ignore")

import numpy as np
from numpy.fft import fft

from scipy import signal
from scipy.io import loadmat

import torch
import torch.utils.data

from msda_v1.utils import *


######################################INSERT INPUT###################


Fs = 512
FS = 1017
fs = 10
##################################
data = loadmat("../data.mat")
eeg = data["trial_eeg"]
emg = data["trial_emg"]
ne = data["trial_ne"]
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


test_fft7 = np.zeros((test_eeg7.shape[0], Fs))
for tr in range(test_eeg7.shape[0]):
    temp1 = np.squeeze(test_eeg7[tr, :, :])
    test_fft7[tr, :] = np.abs(fft(temp1))
EEG = test7.copy()
EMG = test7_emg.copy()
NE = test_ne7.copy()
FFT = test_fft7.copy()
target_flow1 = torch.from_numpy(EEG)
target_flow2 = torch.from_numpy(NE)
target_flow3 = torch.from_numpy(EMG)
target_flow4 = torch.from_numpy(FFT)
half_batch = 100
target_dataset = torch.utils.data.TensorDataset(
    target_flow1, target_flow2, target_flow3, target_flow4
)
"""
##################################################END INSERT DATA##########################
model = "msda_version1_thressclass_by_Shadi.pth"
model_MA = "msda_version1_MA_by_Shadi.pth"
signaling = 100
test_loader = torch.utils.data.DataLoader(
    dataset=target_dataset, batch_size=half_batch, shuffle=False
)


result_pred1 = test(model, test_loader, signaling)

pred1 = []
for i in range(len(result_pred1)):
    temp = np.array(result_pred1[i].data.numpy())
    pred1.append(temp)
pred1 = np.array(pred1)
pred1 = np.concatenate(pred1)

pred_final1 = np.argmax(pred1, axis=1)
score_final1 = np.max(pred1, axis=1)

#######################################final results for three class ##################################
final_label1, final_score1 = edit_one(pred_final1, score_final1)
final_label1[0] = 0
final_label1 = edit_three(edit_two(final_label1))

# outputs are final_label1 and final_score1 ####################################0 wake; 1 sws; 2rem
##########################################################################


result_ma1 = test2(model_MA, test_loader, signaling)

pred_ma1 = []
for i in range(len(result_ma1)):
    temp = np.array(result_ma1[i].data.numpy())
    pred1.append(temp)
pred_ma1 = np.array(pred_ma1)
pred_ma1 = np.concatenate(pred_ma1)

p2 = np.argmax(pred_ma1, axis=1)
scorep = np.max(pred_ma1, axis=1)

p1 = final_label1.copy()
p = np.zeros((len(p1)))

for i in range(len(p1)):
    if p2[i] == 1 and scorep[i] > 0.70:
        p[i] = 0
    else:
        p[i] = p1[i]


MA_results = find_ma(p)

###############################final result is MA_results the labels, 0 wake; 1 sws; 2 rem;3 MA ###################################
"""
