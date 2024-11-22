import torch
from torch import nn


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class cls_head(nn.Module):
    def __init__(self, hidden_size, c_out=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, c_out),
        )
        self.mlp.apply(init_weights)

    def forward(self, x):
        x = self.mlp(x)
        return x


class SeqPooler2(nn.Module):
    def __init__(self, hidden_size, useRaw=False):
        super().__init__()
        self.pool_head = (
            nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh())
            if not useRaw
            else nn.Identity()
        )
        self.pool_head.apply(init_weights)

    def forward(self, hidden_states):
        hidden_states = torch.cat(
            (
                hidden_states[:, : hidden_states.shape[1] // 2],
                hidden_states[:, hidden_states.shape[1] // 2 :],
            ),
            dim=-1,
        )
        pooled_output = self.pool_head(hidden_states)
        return pooled_output
