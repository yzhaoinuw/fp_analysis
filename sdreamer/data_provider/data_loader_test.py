import torch
import numpy as np
from torch.utils.data import Dataset


class LongSequenceLoader(Dataset):
    def __init__(self, eeg_data, emg_data, ne_data, n_sequcnes=16, useNorm=False):
        eeg_data = eeg_data.reshape(
            (eeg_data.shape[0], 1, eeg_data.shape[1])
        )  # (16946, 1, 512)
        emg_data = emg_data.reshape(
            (emg_data.shape[0], 1, emg_data.shape[1])
        )  # (16946, 1, 512)
        ne_data = ne_data.reshape((ne_data.shape[0], 1, ne_data.shape[1]))
        trace_data = np.concatenate((eeg_data, emg_data), axis=1)  # (16946, 2, 512)

        trace_data = torch.from_numpy(trace_data).float()
        ne_data = torch.from_numpy(ne_data).float()
        trace_data = trace_data.unsqueeze(2)  # (16946, 2, 1, 512)
        ne_data = ne_data.unsqueeze(2)  # (16946, 1, 1, 1017)

        trace_mean, trace_std = torch.mean(trace_data, dim=0), torch.std(
            trace_data, dim=0
        )
        trace_data_norm = (trace_data - trace_mean) / trace_std
        ne_mean, ne_std = torch.mean(ne_data, dim=0), torch.std(ne_data, dim=0)
        ne_data_norm = (ne_data - ne_mean) / ne_std

        train_data = slice_trace_wNE(
            trace_data, ne_data, trace_data_norm, ne_data_norm, n_sequcnes
        )

        self.traces = torch.cat([train_data[0], train_data[2]], dim=3)
        self.ne = torch.cat([train_data[1], train_data[3]], dim=3)

        self.traces = (
            self.traces[:, :, :, :1] if not useNorm else self.traces[:, :, :, -1:]
        )
        self.ne = self.ne[:, :, :, :1] if not useNorm else self.ne[:, :, :, -1:]

    def __len__(self):
        return self.traces.size(0)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        ne = self.ne[idx]
        return trace, ne


def slice_trace_wNE(trace, ne, norm, norm_ne, n_sequences):
    n = len(trace)
    n_to_crop = n % n_sequences
    if n_to_crop != 0:
        trace = trace[:-n_to_crop]
        ne = ne[:-n_to_crop]
        norm = norm[:-n_to_crop]
        norm_ne = norm_ne[:-n_to_crop]
    assert (n - n_to_crop) % n_sequences == 0
    n_new_seq = (n - n_to_crop) // n_sequences
    trace = trace.reshape(
        (n_new_seq, n_sequences, trace.shape[1], trace.shape[2], trace.shape[3])
    )
    ne = ne.reshape((n_new_seq, n_sequences, ne.shape[1], ne.shape[2], ne.shape[3]))
    norm = norm.reshape(
        (n_new_seq, n_sequences, norm.shape[1], norm.shape[2], norm.shape[3])
    )
    norm_ne = norm_ne.reshape(
        (n_new_seq, n_sequences, norm_ne.shape[1], norm_ne.shape[2], norm_ne.shape[3])
    )
    return [trace, ne, norm, norm_ne]
