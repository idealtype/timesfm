import torch
import torch.nn as nn


N_FOURIER_TERMS = {"daily": 6, "weekly": 4, "yearly": 8}


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


class SeasonalDecoder(nn.Module):
    def __init__(self, embed_dim, mlp_units, activation="ReLU", dropout=0.0):
        super().__init__()
        self.mlp_daily, last_d = _build_mlp(embed_dim, mlp_units, activation, dropout)
        self.mlp_weekly, last_w = _build_mlp(embed_dim, mlp_units, activation, dropout)
        self.mlp_yearly, last_y = _build_mlp(embed_dim, mlp_units, activation, dropout)

        self.forecast_head_daily = nn.Linear(last_d, 2 * N_FOURIER_TERMS["daily"])
        self.forecast_head_weekly = nn.Linear(last_w, 2 * N_FOURIER_TERMS["weekly"])
        self.forecast_head_yearly = nn.Linear(last_y, 2 * N_FOURIER_TERMS["yearly"])

    def forward(self, emb, daily_basis, weekly_basis, yearly_basis):
        coef_d = self.forecast_head_daily(self.mlp_daily(emb))
        coef_w = self.forecast_head_weekly(self.mlp_weekly(emb))
        coef_y = self.forecast_head_yearly(self.mlp_yearly(emb))

        s_d = torch.bmm(daily_basis, coef_d.unsqueeze(-1)).squeeze(-1)
        s_w = torch.bmm(weekly_basis, coef_w.unsqueeze(-1)).squeeze(-1)
        s_y = torch.bmm(yearly_basis, coef_y.unsqueeze(-1)).squeeze(-1)

        return s_d + s_w + s_y
