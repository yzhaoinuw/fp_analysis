# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:17:16 2024

@author: yzhao
"""

import os
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat, savemat

from models.sdreamer import n2nSeqNewMoE2
from preprocessing import reshape_sleep_data


# %%
def slice_data(data, n_sequences=64):
    n = len(data)
    n_to_crop = n % n_sequences
    if n_to_crop != 0:
        data = data[:-n_to_crop]
    assert (n - n_to_crop) % n_sequences == 0
    n_new_seq = (n - n_to_crop) // n_sequences
    data = data.reshape(
        (n_new_seq, n_sequences, data.shape[1], data.shape[2], data.shape[3])
    )
    return data


class SequenceDataset(Dataset):
    def __init__(self, eeg, emg, n_sequences=64):
        eeg = eeg[:, np.newaxis, :]
        emg = emg[:, np.newaxis, :]
        data = np.stack((eeg, emg), axis=1)
        data = torch.from_numpy(data)
        mean, std = torch.mean(data, dim=0), torch.std(data, dim=0)
        norm_data = (data - mean) / std
        # data_sliced = slice_data(data)
        self.traces = slice_data(norm_data)
        # self.traces = torch.cat([data_sliced, norm_data_sliced], dim=3)

    def __len__(self):
        return self.traces.size(0)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        return trace


# %%

config = dict(
    model="SeqNewMoE2",
    data="Seq",
    isNE=False,
    features="ALL",
    n_sequences=64,
    useNorm=True,
    seq_len=512,
    patch_len=16,
    stride=8,
    padding_patch="end",
    subtract_last=0,
    decomposition=0,
    kernel_size=25,
    individual=0,
    mix_type=0,
    c_out=3,
    d_model=128,
    n_heads=8,
    e_layers=2,
    ca_layers=1,
    seq_layers=3,
    d_ff=512,
    dropout=0.1,
    path_drop=0.0,
    activation="glu",
    norm_type="layernorm",
    output_attentions=False,
)


def build_args(**kwargs):
    parser = argparse.ArgumentParser(description="Transformer family for sleep scoring")
    args = parser.parse_args()
    parser_dict = vars(args)

    for k, v in config.items():
        parser_dict[k] = v
    for k, v in kwargs.items():
        parser_dict[k] = v
    return args


# %%
def infer(data, model_path, output_path, batch_size=32):
    args = build_args()
    num_class = args.c_out
    output_path = os.path.splitext(output_path)[0] + f"_sdreamer_{num_class}class.mat"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = n2nSeqNewMoE2.Model(args)
    model = model.to(device)
    checkpoint_path = (
        model_path
        + "sdreamer/checkpoints/SeqNewMoE2_Seq_ftALL_pl16_ns64_dm128_el2_dff512_eb0_bs64_f1.pth.tar"
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    eeg_reshaped, emg_reshaped = reshape_sleep_data(data)
    dataset = SequenceDataset(eeg_reshaped, emg_reshaped)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    model.eval()
    with torch.no_grad():
        all_pred = []
        all_prob = []

        with tqdm(total=eeg_reshaped.shape[0], unit=" seconds of signal") as pbar:
            for batch, traces in enumerate(data_loader, 1):
                traces = traces.to(device)  # [batch_size, 64, 2, 1, 512]
                out_dict = model(traces, label=None)
                out = out_dict["out"]

                prob = torch.max(torch.softmax(out, dim=1), dim=1).values
                all_prob.append(prob.detach().cpu())

                pred = np.argmax(out.detach().cpu(), axis=1)
                all_pred.append(pred)

                pbar.update(batch_size * args.n_sequences)
            pbar.set_postfix({"Number of batches": batch})

        all_pred = np.concatenate(all_pred)
        all_prob = np.concatenate(all_prob)

    results = {
        "pred_labels": all_pred,
        "confidence": all_prob,
        "num_class": num_class,
        "eeg_frequency": data["eeg_frequency"],
        "ne_frequency": data["ne_frequency"],
        "eeg": data["eeg"],
        "emg": data["emg"],
        "ne": data["ne"],
    }

    savemat(output_path, results)
    return all_pred, all_prob, output_path


if __name__ == "__main__":
    model_path = "./models/"
    mat_file = "./user_test_files/arch_387.mat"
    data = loadmat(mat_file)
    all_pred, all_prob, output_path = infer(data, model_path, mat_file)
