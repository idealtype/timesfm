from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
MONASH_TSF_ROOT = "/home/sia2/project/4.28basis/basis_dec/data/monash"

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, MONASH_TSF_ROOT)

from model.decomp_funcdec import FuncDecModel  # noqa: E402
from model.decoder_seasonal import N_FOURIER_TERMS  # noqa: E402
from prepare import iter_rows, read_meta  # noqa: E402


FREQ_DAYS = {
    "hourly": 1 / 24,
    "half_hourly": 1 / 48,
    "10_minutes": 1 / 144,
    "minutely": 1 / 1440,
    "4_seconds": 1 / 21600,
    "daily": 1.0,
    "weekly": 7.0,
    "monthly": 30.4375,
    "quarterly": 91.3125,
    "yearly": 365.25,
}
SEASONALITY_PERIODS = {"daily": 1.0, "weekly": 7.0, "yearly": 365.25}
REVIN_TOL = 1e-6
DEFAULT_CONFIG = {
    "context_len": 512,
    "embed_dim": 1280,
    "n_knots": {"96": 10, "192": 15, "336": 22, "720": 40},
    "mlp_units": {"trend": [512, 512], "seasonal": [512, 512], "residual": [512, 512]},
    "activation": "ReLU",
    "dropout": 0.0,
}


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def find_tsf_path(dataset_name):
    candidates = [
        os.path.join(MONASH_TSF_ROOT, f"{dataset_name}.tsf"),
        os.path.join(MONASH_TSF_ROOT, f"{dataset_name}_without_missing_values.tsf"),
        os.path.join(MONASH_TSF_ROOT, f"{dataset_name}_with_missing_values.tsf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"TSF file not found for dataset={dataset_name}")


def build_config(args):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        {
            "context_len": int(args.context_len),
            "horizon": int(args.horizon),
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "device": args.device,
        }
    )
    if str(cfg["horizon"]) not in cfg["n_knots"]:
        raise ValueError(f"Unsupported horizon for n_knots: {cfg['horizon']}")
    return cfg


def build_fourier_basis(freq, context_len, horizon, device):
    fd = FREQ_DAYS[freq]
    context_span = context_len * fd
    mask = {}
    for stype in ["daily", "weekly", "yearly"]:
        period = SEASONALITY_PERIODS[stype]
        mask[stype] = bool((fd < period) and (context_span >= period))

    t = torch.arange(horizon, dtype=torch.float32, device=device)
    bases = {}
    for stype in ["daily", "weekly", "yearly"]:
        n = N_FOURIER_TERMS[stype]
        basis = torch.zeros(horizon, 2 * n, dtype=torch.float32, device=device)
        if mask[stype]:
            p_steps = SEASONALITY_PERIODS[stype] / fd
            for k in range(n):
                basis[:, 2 * k] = torch.sin(2 * math.pi * (k + 1) * t / p_steps)
                basis[:, 2 * k + 1] = torch.cos(2 * math.pi * (k + 1) * t / p_steps)
        bases[stype] = basis
    return bases, mask


class TsfWindowDataset(Dataset):
    def __init__(self, meta, context_len, horizon):
        self.context_len = context_len
        self.horizon = horizon
        self.samples = []
        stride = max(1, context_len // 4)
        for row in iter_rows(meta):
            values = row.values.astype(np.float32)
            max_start = len(values) - context_len - horizon
            if max_start < 0:
                continue
            for win_start in range(0, max_start + 1, stride):
                self.samples.append((row.sid, win_start, values[win_start : win_start + context_len + horizon].copy()))
        if not self.samples:
            raise ValueError(f"No evaluation windows for horizon={horizon}, context_len={context_len}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, win_start, values = self.samples[idx]
        context = torch.from_numpy(values[: self.context_len])
        future = torch.from_numpy(values[self.context_len :])
        masks = torch.zeros(self.context_len, dtype=torch.bool)
        return context, masks, future, sid, win_start


def load_checkpoint(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    decoder_keys = [k for k in state if k.startswith("decoder_")]
    if not decoder_keys:
        raise ValueError(f"No decoder parameters found in checkpoint: {checkpoint_path}")
    incompatible = model.load_state_dict(state, strict=False)
    missing_decoders = [k for k in incompatible.missing_keys if k.startswith("decoder_")]
    unexpected = [k for k in incompatible.unexpected_keys if not k.startswith("backbone.")]
    if missing_decoders or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing_decoders[:5]} unexpected={unexpected[:5]}")
    print(f"Loaded checkpoint: {checkpoint_path}")


@torch.no_grad()
def evaluate(model, loader, bases, horizon, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_values = 0
    for context, masks, future, _sid, _win_start in loader:
        context = context.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)

        emb, mu, sigma = model._encode(context, masks)
        batch_size = emb.shape[0]
        daily_basis = bases["daily"].unsqueeze(0).expand(batch_size, -1, -1)
        weekly_basis = bases["weekly"].unsqueeze(0).expand(batch_size, -1, -1)
        yearly_basis = bases["yearly"].unsqueeze(0).expand(batch_size, -1, -1)
        pred_n, _decomp = model(emb, daily_basis, weekly_basis, yearly_basis)

        denom = torch.where(sigma < REVIN_TOL, torch.ones_like(sigma), sigma)
        future_n = (future - mu) / denom
        total_mse += torch.sum((pred_n - future_n) ** 2).item()
        total_mae += torch.sum(torch.abs(pred_n - future_n)).item()
        n_values += future_n.numel()
    return total_mse / n_values, total_mae / n_values, n_values // horizon


def main():
    args = parse_args()
    cfg = build_config(args)
    device = torch.device(cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tsf_path = find_tsf_path(args.dataset)
    meta = read_meta(Path(tsf_path))
    if meta.frequency not in FREQ_DAYS:
        raise ValueError(f"Unsupported frequency: {meta.frequency}")

    model = FuncDecModel(cfg, load_backbone=True).to(device)
    load_checkpoint(model, args.checkpoint, device)

    bases, mask = build_fourier_basis(meta.frequency, cfg["context_len"], cfg["horizon"], device)
    dataset = TsfWindowDataset(meta, cfg["context_len"], cfg["horizon"])
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0 if len(dataset) <= 64 else 2,
        pin_memory=device.type == "cuda",
    )

    mse, mae, n_windows = evaluate(model, loader, bases, cfg["horizon"], device)
    result = {
        "dataset": args.dataset,
        "tsf_path": tsf_path,
        "frequency": meta.frequency,
        "context_len": cfg["context_len"],
        "horizon": cfg["horizon"],
        "checkpoint": args.checkpoint,
        "space": "normalized",
        "mse": float(mse),
        "mae": float(mae),
        "n_windows": int(n_windows),
        "seasonality_mask": mask,
    }

    out_path = os.path.join(RESULTS_DIR, f"zeroshot_{args.dataset}_h{cfg['horizon']}.json")
    with open(out_path, "w") as f:
        json.dump(to_jsonable(result), f, indent=2)

    print(f"{args.dataset} h={cfg['horizon']} zero-shot")
    print(f"  windows={n_windows} freq={meta.frequency} mask={mask}")
    print(f"  MSE={mse:.4f} MAE={mae:.4f}")
    print(f"  saved={out_path}")


if __name__ == "__main__":
    main()
