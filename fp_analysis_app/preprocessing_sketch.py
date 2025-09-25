# -*- coding: utf-8 -*-
"""
Created on Tue May 20 08:54:33 2025

@author: yzhao
"""

import os
import math

import numpy as np
from scipy import stats
from scipy import signal
from scipy.io import loadmat


def trim_missing_labels(filt, trim="b"):
    first = 0
    trim = trim.upper()
    if "F" in trim:
        for i in filt:
            if i == -1 or np.isnan(i):
                first = first + 1
            else:
                break
    last = len(filt)
    if "B" in trim:
        for i in filt[::-1]:
            if i == -1 or np.isnan(i):
                last = last - 1
            else:
                break
    return filt[first:last]


data_path = "..\\data\\"
mat_file = "M1 (1).mat"
filename = os.path.splitext(os.path.basename(mat_file))[0]
mat = loadmat(os.path.join(data_path, mat_file), squeeze_me=True)

segment_size = (512,)
standardize = (False,)
has_labels = True
signal = mat["fiber_c"]

if standardize:
    signal = stats.zscore(signal)

freq = mat["fp_frequency"]

# clip the last non-full second and take the shorter duration of the two
end_time = math.floor(signal.size / freq)

# if sampling rate is much higher than 512, downsample using poly resample
if math.ceil(eeg_freq) != segment_size and math.floor(eeg_freq) != segment_size:
    down, up = (
        Fraction(eeg_freq / segment_size).limit_denominator(100).as_integer_ratio()
    )
    print(f"file has sampling frequency of {eeg_freq}.")
    eeg = signal.resample_poly(eeg, up, down)
    emg = signal.resample_poly(emg, up, down)
    eeg_freq = segment_size

time_sec = np.arange(end_time)
start_indices = np.ceil(time_sec * eeg_freq).astype(int)

# Reshape start_indices to be a column vector (N, 1)
start_indices = start_indices[:, np.newaxis]
segment_array = np.arange(segment_size)
# Use broadcasting to add the range_array to each start index
indices = start_indices + segment_array

eeg_reshaped = eeg[indices]
emg_reshaped = emg[indices]

if has_labels:
    sleep_scores = mat["sleep_scores"]
    sleep_scores = trim_missing_labels(sleep_scores, trim="b")  # trim trailing zeros
