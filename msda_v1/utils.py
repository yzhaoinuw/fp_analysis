# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:41:54 2023

@author: yzhao
adpated from Shadi Sartipi's msda_version1_utils_byShadi.py
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import torch
import torch.utils.data

from tqdm import tqdm

from msda_v1.models import DSN, DSN2


num_workers = os.cpu_count()


def run_test(num_class, batch_size, test_dataset, signaling):
    ###################
    # params          #
    ###################
    code_size_map = {100: 128, 200: 96, 400: 64, 300: 96, 600: 32, 500: 32}
    model_path = f"./msda_{num_class}class_v1.pth"
    code_size = code_size_map[signaling]
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    if num_class == 3:
        model = DSN(code_size=code_size)
    else:
        model = DSN2(code_size=code_size)

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    predictions = []
    confidence = []
    with tqdm(total=len(test_dataset), unit=" seconds of signal") as pbar:
        with torch.no_grad():
            for batch, (eeg_signal, ne_signal, emg_signal, fft_signal) in enumerate(
                test_loader, 1
            ):
                output = model(
                    eeg_signal,
                    ne_signal,
                    emg_signal,
                    fft_signal,
                    mode="source",
                    signaling=signaling,
                )
                probs = output[3]
                preds = torch.argmax(probs, axis=1)
                conf = torch.max(probs, axis=1).values
                predictions.extend(preds.tolist())
                confidence.extend(conf.tolist())
                pbar.update(batch_size)
            pbar.set_postfix({"Batch": batch})

    predictions = np.array(predictions)
    confidence = np.array(confidence)
    return (predictions, confidence)


##################################################EDIT Labels#######################################
def edit_one(pred, score):
    high_score = np.where(score > 0.90)[0]
    # print(len(high_score))
    final_label = pred.copy()
    final_score = score.copy()
    for i in range(len(high_score) - 1):
        if pred[high_score[i]] == pred[high_score[i + 1]]:
            final_label[high_score[i] : high_score[i + 1]] = pred[high_score[i]]
            final_score[high_score[i] : high_score[i + 1]] = 0.91
    return final_label, final_score


def edit_two(L):
    label = L.copy()
    diff_label = np.diff(label)
    idx_0 = [0]
    diff_label = np.concatenate((idx_0, diff_label), axis=0)
    idx = np.where(diff_label != 0)[0]
    idx = np.concatenate((idx_0, idx), axis=0)
    wake = []
    wake_t = []
    sws = []
    sws_t = []
    rem = []
    rem_t = []
    for i in range(len(idx) - 1):
        a1 = idx[i]
        a2 = idx[i + 1] - 1
        dur = [a1, a2]
        len_dur = a2 - a1 + 1
        label_temp = label[idx[i]]
        if label_temp == 0:
            wake.append(dur)
            wake_t.append(len_dur)
        if label_temp == 1:
            sws.append(dur)
            sws_t.append(len_dur)
        if label_temp == 2:
            rem.append(dur)
            rem_t.append(len_dur)
    wake = np.array(wake)
    wake_t = np.array(wake_t)
    sws = np.array(sws)
    sws_t = np.array(sws_t)
    rem = np.array(rem)
    rem_t = np.array(rem_t)
    idx_ma = np.where(sws_t < 5)[0]
    mask = (sws_t >= 4) & (sws_t < 11)
    idx_wake = np.where(mask)[0]
    # print(idx_wake)

    for i in range(len(idx_ma)):
        temp = sws[idx_ma[i]]
        a1 = label[temp[0] - 1]
        a2 = label[temp[1] + 1]
        if a1 == a2 == 0:
            label[temp[0] : temp[1] + 1] = 0

    return label


def edit_three(L):
    label = L.copy()
    diff_label = np.diff(label)
    idx_0 = [0]
    diff_label = np.concatenate((idx_0, diff_label), axis=0)
    idx = np.where(diff_label != 0)[0]
    idx = np.concatenate((idx_0, idx), axis=0)
    wake = []
    wake_t = []
    sws = []
    sws_t = []
    rem = []
    rem_t = []
    for i in range(len(idx) - 1):
        a1 = idx[i]
        a2 = idx[i + 1] - 1
        dur = [a1, a2]
        len_dur = a2 - a1 + 1
        label_temp = label[idx[i]]
        if label_temp == 0:
            wake.append(dur)
            wake_t.append(len_dur)
        if label_temp == 1:
            sws.append(dur)
            sws_t.append(len_dur)
        if label_temp == 2:
            rem.append(dur)
            rem_t.append(len_dur)
    wake = np.array(wake)
    wake_t = np.array(wake_t)
    sws = np.array(sws)
    sws_t = np.array(sws_t)
    rem = np.array(rem)
    rem_t = np.array(rem_t)
    idx_ma = np.where(sws_t < 4)[0]
    mask = (sws_t >= 4) & (sws_t < 11)
    idx_wake = np.where(mask)[0]
    # print(idx_wake)

    for i in range(len(idx_ma)):
        temp = sws[idx_ma[i]]
        a1 = label[temp[0] - 1]
        a2 = label[temp[1] + 1]
        if a1 == a2 == 2:
            label[temp[0] : temp[1] + 1] = 2

    return label


def find_ma(L):
    label = L.copy()
    diff_label = np.diff(label)
    idx_0 = [0]
    diff_label = np.concatenate((idx_0, diff_label), axis=0)
    idx = np.where(diff_label != 0)[0]
    idx = np.concatenate((idx_0, idx), axis=0)
    wake = []
    wake_t = []
    sws = []
    sws_t = []
    rem = []
    rem_t = []
    for i in range(len(idx) - 1):
        a1 = idx[i]
        a2 = idx[i + 1] - 1
        dur = [a1, a2]
        len_dur = a2 - a1 + 1
        label_temp = label[idx[i]]
        if label_temp == 0:
            wake.append(dur)
            wake_t.append(len_dur)
        if label_temp == 1:
            sws.append(dur)
            sws_t.append(len_dur)
        if label_temp == 2:
            rem.append(dur)
            rem_t.append(len_dur)
    wake = np.array(wake)
    wake_t = np.array(wake_t)
    sws = np.array(sws)
    sws_t = np.array(sws_t)
    rem = np.array(rem)
    rem_t = np.array(rem_t)
    idx_ma = np.where(wake_t < 15)[0]

    for i in range(len(idx_ma)):
        temp = wake[idx_ma[i]]
        a1 = label[temp[0] - 1]
        a2 = label[temp[1] + 1]
        if a1 == a2 == 1:
            # if wake_t[idx_ma[i]]==1:
            #   label[temp[0]]=3
            # else:
            label[temp[0] : temp[1] + 1] = 3
    return label


def rolling_window(data, window, step):
    shape = (
        data.shape[2],  # 1
        data.shape[0],  # 16946
        (data.shape[1] - window) // step + 1,  # 7
        window,  # 128
    )
    strides = (step * data.strides[1],) + data.strides

    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides).swapaxes(
        0, 1
    )
