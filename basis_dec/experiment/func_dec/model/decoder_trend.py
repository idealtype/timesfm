import torch
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


class TrendDecoder(nn.Module):
    def __init__(self, embed_dim, horizon, n_knots, mlp_units, activation="ReLU", dropout=0.0):
        super().__init__()
        s = torch.arange(n_knots, dtype=torch.float32) * (horizon - 1) / (n_knots - 1)
        t = torch.arange(horizon, dtype=torch.float32).unsqueeze(1)
        self.register_buffer("A", (t >= s).to(torch.float32))

        self.mlp, last_dim = _build_mlp(embed_dim, mlp_units, activation, dropout)
        self.forecast_head = nn.Linear(last_dim, n_knots + 1)

    def forward(self, emb):
        out = self.forecast_head(self.mlp(emb))
        intercept = out[:, :1]
        delta = out[:, 1:]
        return intercept + delta @ self.A.T, delta
