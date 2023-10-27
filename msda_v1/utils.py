# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:41:54 2023

@author: yzhao
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np

import torch
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from msda_v1.models import DSN, DSN2


def test(model, dataloader, signaling):
    ###################
    # params          #
    ###################
    # mcf1s = MulticlassF1Score(num_classes=3, average='macro')
    cuda = False
    cudnn.benchmark = False

    if signaling == 100:
        my_net = DSN(code_size=128)
    elif signaling == 200:
        my_net = DSN(code_size=96)
    elif signaling == 400:
        my_net = DSN(code_size=64)
    elif signaling == 300:
        my_net = DSN(code_size=96)
    elif signaling == 600:
        my_net = DSN(code_size=32)
    elif signaling == 500:
        my_net = DSN(code_size=32)

    # my_net=DSN()
    model_root = model
    my_net.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
    my_net.eval()
    # my_net.cuda()
    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    pred_label = []
    result_pred = []

    while i < len_dataloader:
        data_input = next(data_iter)
        img1, img2, img3, img4 = data_input

        batch_size = len(img1)

        input_img1 = torch.FloatTensor(batch_size, 1, 7, 128)
        input_img2 = torch.FloatTensor(batch_size, 10, 1)
        input_img3 = torch.FloatTensor(batch_size, 1, 7, 128)
        input_img4 = torch.FloatTensor(batch_size, 512)

        if cuda:
            img1 = img1.cuda()
            img2 = img2.cuda()
            img3 = img3.cuda()
            img4 = img4.cuda()
            input_img1 = input_img1.cuda()
            input_img2 = input_img2.cuda()
            input_img3 = input_img3.cuda()
            input_img4 = input_img4.cuda()

        input_img1.resize_as_(input_img1).copy_(img1)
        input_img2.resize_as_(input_img2).copy_(img2)
        input_img3.resize_as_(input_img3).copy_(img3)
        input_img4.resize_as_(input_img4).copy_(img4)
        inputv_img1 = Variable(input_img1)
        inputv_img2 = Variable(input_img2)
        inputv_img3 = Variable(input_img3)
        inputv_img4 = Variable(input_img4)

        result = my_net(
            inputv_img1,
            inputv_img2,
            inputv_img3,
            inputv_img4,
            mode="source",
            signaling=signaling,
        )
        pred = result[3].data.max(1, keepdim=True)[1]

        # f_1+=multiclass_f1_score(pred.squeeze(), classv_label, num_classes=3, average="macro")

        n_total += batch_size
        result_pred.append(result[3].data.cpu())

        i += 1

    return result_pred


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
    print(idx_wake)

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
    print(idx_wake)

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


#######################################################End EDIT###############################################


def test2(model, dataloader, signaling):
    ###################
    # params          #
    ###################

    cuda = False
    cudnn.benchmark = False

    if signaling == 100:
        my_net = DSN2(code_size=128)
    elif signaling == 200:
        my_net = DSN2(code_size=96)
    elif signaling == 400:
        my_net = DSN2(code_size=64)
    elif signaling == 300:
        my_net = DSN2(code_size=96)
    elif signaling == 600:
        my_net = DSN2(code_size=32)
    elif signaling == 500:
        my_net = DSN2(code_size=32)

    # my_net=DSN()
    model_root = model
    my_net.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
    my_net.eval()
    # my_net.cuda()
    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    pred_label = []
    result_pred = []

    while i < len_dataloader:
        data_input = next(data_iter)
        img1, img2, img3, img4 = data_input

        batch_size = len(img1)

        input_img1 = torch.FloatTensor(batch_size, 1, 7, 128)
        input_img2 = torch.FloatTensor(batch_size, 10, 1)
        input_img3 = torch.FloatTensor(batch_size, 1, 7, 128)
        input_img4 = torch.FloatTensor(batch_size, 512)

        if cuda:
            img1 = img1.cuda()
            img2 = img2.cuda()
            img3 = img3.cuda()
            img4 = img4.cuda()
            input_img1 = input_img1.cuda()
            input_img2 = input_img2.cuda()
            input_img3 = input_img3.cuda()
            input_img4 = input_img4.cuda()

        input_img1.resize_as_(input_img1).copy_(img1)
        input_img2.resize_as_(input_img2).copy_(img2)
        input_img3.resize_as_(input_img3).copy_(img3)
        input_img4.resize_as_(input_img4).copy_(img4)
        inputv_img1 = Variable(input_img1)
        inputv_img2 = Variable(input_img2)
        inputv_img3 = Variable(input_img3)
        inputv_img4 = Variable(input_img4)

        result = my_net(
            inputv_img1,
            inputv_img2,
            inputv_img3,
            inputv_img4,
            mode="source",
            signaling=signaling,
        )
        pred = result[3].data.max(1, keepdim=True)[1]

        # f_1+=multiclass_f1_score(pred.squeeze(), classv_label, num_classes=3, average="macro")

        n_total += batch_size
        result_pred.append(result[3].data.cpu())

        i += 1

    return result_pred


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
