import torch.nn as nn


def _build_mlp(input_dim, hidden_layers, activation, dropout):
    activations = {"ReLU": nn.ReLU, "SiLU": nn.SiLU, "GELU": nn.GELU}
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activations[activation]())
        layers.append(nn.Dropout(dropout))
        last_dim = hidden_dim
    return nn.Sequential(*layers), last_dim


class ResidualDecoder(nn.Module):
    def __init__(self, embed_dim, horizon, mlp_units, activation="ReLU", dropout=0.0):
        super().__init__()
        self.mlp, last_dim = _build_mlp(embed_dim, mlp_units, activation, dropout)
        self.forecast_head = nn.Linear(last_dim, horizon)

    def forward(self, emb):
        return self.forecast_head(self.mlp(emb))
