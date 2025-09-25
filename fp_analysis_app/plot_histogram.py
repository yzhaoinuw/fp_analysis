# -*- coding: utf-8 -*-
"""
Created on Tue May 20 08:38:01 2025

@author: yzhao
"""

import os

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


data_path = "..\\data\\"
mat_file = "M1 (1).mat"
filename = os.path.splitext(os.path.basename(mat_file))[0]
mat = loadmat(os.path.join(data_path, mat_file), squeeze_me=True)
# %%

labels = mat.get("sleep_scores")
signal = mat.get("fiber_c")

bins = 50
# Make sure both arrays are 1D and same length
assert signal.ndim == labels.ndim == 1
assert signal.size == labels.size

# Mask out NaN labels
valid_mask = ~np.isnan(labels)
signal = signal[valid_mask]
labels = labels[valid_mask]

# Get unique labels
unique_labels = np.unique(labels)

# Plot histogram for each label
plt.figure(figsize=(10, 6))
for lbl in unique_labels:
    mask = labels == lbl
    plt.hist(signal[mask], bins=bins, alpha=0.5, label=f"Label {lbl}")

plt.xlabel("Signal Value")
plt.ylabel("Frequency")
plt.title("Histogram of Signal Grouped by Label")
plt.legend()
plt.tight_layout()
plt.show()
