import torch
import torch.nn as nn

from .decoder_residual import ResidualDecoder
from .decoder_seasonal import SeasonalDecoder
from .decoder_trend import TrendDecoder


class FuncDecModel(nn.Module):
    def __init__(self, cfg: dict, load_backbone: bool = False):
        super().__init__()
        self.embed_dim = cfg["embed_dim"]
        self.context_len = cfg["context_len"]
        self.backbone = None

        if load_backbone:
            from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
            from timesfm.torch.util import revin, update_running_stats

            pretrained = TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch", torch_compile=False
            )
            self.backbone = pretrained.model
            self.backbone.eval()
            self.patch_len = self.backbone.p
            self._revin = revin
            self._update_running_stats = update_running_stats
            for p in self.backbone.parameters():
                p.requires_grad = False

        self._init_decoders(cfg)

    def _init_decoders(self, cfg: dict):
        horizon = cfg["horizon"]
        self.horizon = horizon
        n_knots = cfg["n_knots"][str(horizon)]
        mlp_units = cfg["mlp_units"]

        self.decoder_t = TrendDecoder(
            embed_dim=self.embed_dim,
            horizon=horizon,
            n_knots=n_knots,
            mlp_units=mlp_units["trend"],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
        )
        self.decoder_s = SeasonalDecoder(
            embed_dim=self.embed_dim,
            mlp_units=mlp_units["seasonal"],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
        )
        self.decoder_r = ResidualDecoder(
            embed_dim=self.embed_dim,
            horizon=horizon,
            mlp_units=mlp_units["residual"],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
        )

    def reset_decoders(self, cfg: dict):
        try:
            device = next(self.decoder_t.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self._init_decoders(cfg)
        self.decoder_t.to(device)
        self.decoder_s.to(device)
        self.decoder_r.to(device)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def forward(self, emb, daily_basis, weekly_basis, yearly_basis):
        trend_n, delta = self.decoder_t(emb)
        seasonal_n = self.decoder_s(emb, daily_basis, weekly_basis, yearly_basis)
        residual_n = self.decoder_r(emb)
        pred_n = trend_n + seasonal_n + residual_n
        return pred_n, {"trend": trend_n, "seasonal": seasonal_n, "residual": residual_n, "delta": delta}

    def _encode(self, context, masks):
        if self.backbone is None:
            raise NotImplementedError("load_backbone=False로 초기화됨. 추론용 별도 스크립트 사용.")

        batch_size, context_len = context.shape
        if context_len != self.context_len:
            raise ValueError(f"context length mismatch: got {context_len}, expected {self.context_len}")
        if context_len % self.patch_len != 0:
            raise ValueError(f"context_len={context_len} must be divisible by patch_len={self.patch_len}")
        patched_inputs = context.reshape(batch_size, -1, self.patch_len)
        patched_masks = masks.reshape(batch_size, -1, self.patch_len)
        num_patches = patched_inputs.shape[1]

        device = context.device
        n = torch.zeros(batch_size, device=device)
        mu = torch.zeros(batch_size, device=device)
        sigma = torch.zeros(batch_size, device=device)
        patch_mu = []
        patch_sigma = []

        for patch_idx in range(num_patches):
            (n, mu, sigma), _ = self._update_running_stats(
                n,
                mu,
                sigma,
                patched_inputs[:, patch_idx],
                patched_masks[:, patch_idx],
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)

        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)
        normed_inputs = self._revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

        with torch.no_grad():
            (_, output_embeddings, _, _), _ = self.backbone(normed_inputs, patched_masks)

        return output_embeddings[:, -1, :], context_mu[:, -1:], context_sigma[:, -1:]
