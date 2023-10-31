# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:50:11 2023

@author: yzhao
"""

import os
import argparse

from tqdm import tqdm
import numpy as np
from scipy.io import loadmat, savemat

import torch
from torch.utils.data import DataLoader

from models.seq import n2nBaseLineNE
from configs.model_config_dict import model_config_dict
from data_provider.data_loader_test import LongSequenceLoader


num_workers = os.cpu_count()


def build_args(model_name="model_A", **kwargs):
    parser = argparse.ArgumentParser(description="Transformer family for sleep scoring")
    args = parser.parse_args()
    parser_dict = vars(args)

    model_config = model_config_dict[model_name]
    for k, v in model_config.items():
        parser_dict[k] = v

    for k, v in kwargs.items():
        parser_dict[k] = v

    return args


def infer(data, output_path=None):
    args = build_args()
    num_class = args.c_out
    batch_size = args.batch_size
    if args.use_gpu:
        if args.use_multi_gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")

    if output_path is None:
        output_path = "./data_prediction"
    output_path += f"_sdreamer_{num_class}class.mat"

    model = n2nBaseLineNE.Model(args)
    ckpt_path = args.reload_ckpt
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    eeg_data = data["trial_eeg"]  # (16946, 512)
    emg_data = data["trial_emg"]  # (16946, 512)
    ne_data = data["trial_ne"]  # (16946, 1017)

    test_dataset = LongSequenceLoader(
        eeg_data, emg_data, ne_data, n_sequences=args.n_sequences, useNorm=args.useNorm
    )

    data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    model.eval()
    with tqdm(total=eeg_data.shape[0], unit=" seconds of signal") as pbar:
        with torch.no_grad():
            all_pred = []
            all_prob = []
            for batch, (traces, nes) in enumerate(data_loader, 1):
                traces = traces.to(device)  # [1, 64, 2, 1, 512]
                nes = nes.to(device)  # # [1, 64, 1, 1, 1017]

                out_dict = model(traces, nes, label=None)
                out = out_dict["out"]

                prob = torch.max(torch.softmax(out, dim=1), dim=1).values
                all_prob.append(prob.detach().cpu())

                pred = np.argmax(out.detach().cpu(), axis=1)
                all_pred.append(pred)
                pbar.update(batch_size*args.n_sequences)
            pbar.set_postfix({"Number of batches": batch})

    all_pred = np.concatenate(all_pred)
    all_prob = np.concatenate(all_prob)

    results = {
        "pred_labels": all_pred,
        "confidence": all_prob,
        "trial_eeg": eeg_data,
        "trial_emg": emg_data,
        "trial_ne": ne_data,
    }

    savemat(output_path, results)
    return all_pred, all_prob, output_path


if __name__ == "__main__":
    data = loadmat("C:\\Users\\yzhao\\python_projects\\sleep_scoring\\data.mat")
    all_pred, all_prob, output_path = infer(data)
