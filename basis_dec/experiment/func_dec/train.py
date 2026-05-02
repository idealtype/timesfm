from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CACHE_DIR = os.path.join(EXP_DIR, "dataset_cache")
MONASH_TSF_ROOT = "/home/sia2/project/4.28basis/basis_dec/data/monash"
MONASH_CACHE_ROOT = "/home/sia2/project/4.22prophet/timesfm/prophet_ans/data/monash/cache"
CKPT_DIR = os.path.join(EXP_DIR, "checkpoints")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
RUNS_DIR = os.path.join(EXP_DIR, "runs")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, EXP_DIR)

from model.decomp_funcdec import FuncDecModel  # noqa: E402
from timesfm.configs import ResidualBlockConfig  # noqa: E402
from timesfm.torch.dense import ResidualBlock  # noqa: E402


DEFAULT_CONFIG = {
    "context_len": 512,
    "embed_dim": 1280,
    "horizons": [96, 192, 336, 720],
    "n_knots": {"96": 10, "192": 15, "336": 22, "720": 40},
    "n_fourier_terms": {"daily": 6, "weekly": 4, "yearly": 8},
    "mlp_units": {"trend": [512, 512], "seasonal": [512, 512], "residual": [512, 512]},
    "activation": "ReLU",
    "dropout": 0.0,
    "learning_rate": 1e-3,
    "max_steps": 5000,
    "batch_size": 256,
    "val_split": 0.2,
    "lambda_orth": 0.1,
    "lambda_sparse": 0.01,
    "lambda_res": 1.0,
    "loss": "MAE",
    "device": "cuda:0",
    "monash_cache_root": MONASH_CACHE_ROOT,
    "dataset_cache_dir": DATASET_CACHE_DIR,
    "datasets": None,
    "eval_tfm_zeroshot": True,
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


def _variance(x):
    return torch.var(x.float(), unbiased=False).item()


def compute_decomposition_metrics(futures, preds, decomps):
    trend = decomps["trend"].float()
    seasonal = decomps["seasonal"].float()
    residual = decomps["residual"].float()
    pred = preds.float()
    future = futures.float()
    total_component_var = _variance(trend) + _variance(seasonal) + _variance(residual) + 1e-12
    pred_var = _variance(pred) + 1e-12
    future_var = _variance(future) + 1e-12
    metrics = {
        "trend_var_share_components": _variance(trend) / total_component_var,
        "seasonal_var_share_components": _variance(seasonal) / total_component_var,
        "residual_var_share_components": _variance(residual) / total_component_var,
        "residual_var_share_pred": _variance(residual) / pred_var,
        "pred_var_share_future": pred_var / future_var,
        "mean_abs_component_corr": float(
            torch.stack(
                [
                    batch_corr(trend, seasonal).sqrt(),
                    batch_corr(trend, residual).sqrt(),
                    batch_corr(seasonal, residual).sqrt(),
                ]
            )
            .mean()
            .item()
        ),
        "residual_to_signal_mse": float(F.mse_loss(residual, future - trend - seasonal).item()),
    }
    return metrics


def _short_float(value):
    text = f"{value:.0e}" if value < 0.01 else f"{value:g}"
    return text.replace("+", "").replace("e-0", "e-").replace("e+0", "e")


def _dataset_label(datasets):
    if datasets is None:
        return "all"
    names = [name.removesuffix("_dataset") for name in datasets]
    label = "-".join(names)
    if len(label) <= 64:
        return label
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]
    return f"{names[0]}-plus{len(names) - 1}-{digest}"


def make_run_dir(cfg):
    ds_label = _dataset_label(cfg["datasets"])
    fourier = cfg["n_fourier_terms"]
    hp_label = (
        f"lr{_short_float(cfg['learning_rate'])}"
        f"_lo{_short_float(cfg['lambda_orth'])}"
        f"_ls{_short_float(cfg['lambda_sparse'])}"
        f"_lres{_short_float(cfg['lambda_res'])}"
        f"_fd{fourier['daily']}fw{fourier['weekly']}fy{fourier['yearly']}"
        f"_{cfg['loss'].lower()}"
        f"_bs{cfg['batch_size']}"
        f"_st{cfg['max_steps']}"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RUNS_DIR, f"{timestamp}_{ds_label}_{hp_label}")


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lambda_orth", type=float, default=None)
    parser.add_argument("--lambda_sparse", type=float, default=None)
    parser.add_argument("--lambda_res", type=float, default=None)
    parser.add_argument("--loss", choices=["MSE", "MAE"], default=None)
    parser.add_argument("--val_split", type=float, default=None)
    parser.add_argument("--fourier_daily", type=int, default=None)
    parser.add_argument("--fourier_weekly", type=int, default=None)
    parser.add_argument("--fourier_yearly", type=int, default=None)
    parser.add_argument("--skip_tfm_zeroshot", action="store_true")
    args = parser.parse_args()

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key in [
        "horizons",
        "datasets",
        "max_steps",
        "learning_rate",
        "batch_size",
        "device",
        "lambda_orth",
        "lambda_sparse",
        "lambda_res",
        "loss",
        "val_split",
    ]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    cfg["horizon"] = int(cfg["horizons"][0])
    if not 0.0 <= cfg["val_split"] < 1.0:
        raise ValueError("--val_split must be in [0, 1).")
    if args.skip_tfm_zeroshot:
        cfg["eval_tfm_zeroshot"] = False
    for name in ["daily", "weekly", "yearly"]:
        value = getattr(args, f"fourier_{name}")
        if value is not None:
            cfg["n_fourier_terms"][name] = value
    return cfg


def discover_backbone_caches(horizon, monash_cache_root, dataset_cache_dir=DATASET_CACHE_DIR, datasets=None):
    found = []
    if not os.path.isdir(monash_cache_root):
        print(f"[warn] missing monash cache root: {monash_cache_root}")
        return found
    dataset_names = sorted(datasets) if datasets is not None else sorted(os.listdir(monash_cache_root))
    for dataset_name in dataset_names:
        backbone_dir = os.path.join(monash_cache_root, dataset_name)
        if not os.path.isdir(backbone_dir):
            print(f"[warn] skip {dataset_name} h={horizon}: missing backbone dataset dir")
            continue
        ds_cache_dir = os.path.join(dataset_cache_dir, dataset_name)
        raw_path = os.path.join(ds_cache_dir, f"raw_futures_h{horizon}.pt")
        basis_path = os.path.join(ds_cache_dir, f"fourier_basis_h{horizon}.pt")
        if not (os.path.exists(raw_path) and os.path.exists(basis_path)):
            print(f"[warn] skip {dataset_name} h={horizon}: missing func_dec dataset cache")
            continue
        pattern = os.path.join(backbone_dir, f"backbone_emb_c*_h{horizon}_stride*.pt")
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"[warn] skip {dataset_name} h={horizon}: missing backbone cache")
            continue
        found.append((dataset_name, ds_cache_dir, matches[0]))
    return found


def find_tsf_path(dataset_name):
    for suffix in ("", "_without_missing_values", "_with_missing_values"):
        path = Path(MONASH_TSF_ROOT) / f"{dataset_name}{suffix}.tsf"
        if path.exists():
            return path
    return None


def read_tsf_series(dataset_name):
    path = find_tsf_path(dataset_name)
    if path is None:
        print(f"[warn] missing tsf: {dataset_name}")
        return {}
    attr_names = []
    rows = {}
    in_data = False
    with path.open("r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not in_data:
                if line.startswith("@attribute"):
                    parts = line.split(maxsplit=2)
                    if len(parts) == 3:
                        attr_names.append(parts[1])
                in_data = line == "@data"
                continue
            parts = line.split(":")
            if len(parts) < len(attr_names) + 1:
                continue
            attrs = dict(zip(attr_names, parts[: len(attr_names)], strict=False))
            sid = str(attrs.get("series_name", f"series_{len(rows)}"))
            values = []
            for value in parts[-1].split(","):
                try:
                    values.append(float(value))
                except ValueError:
                    continue
            rows[sid] = np.asarray(values, dtype=np.float32)
    return rows


def _limit_fourier_basis(basis, n_terms):
    keep_cols = 2 * int(n_terms)
    if keep_cols >= basis.shape[1]:
        return basis
    limited = basis.clone()
    limited[:, keep_cols:] = 0.0
    return limited


class MonashFuncDecDataset(Dataset):
    def __init__(self, cache_pairs, horizon, split, val_split, dataset_cache_dir, n_fourier_terms=None):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'.")
        n_fourier_terms = n_fourier_terms or DEFAULT_CONFIG["n_fourier_terms"]
        self.samples = []
        self.datasets_used = []

        for dataset_name, ds_cache_dir, backbone_path in cache_pairs:
            backbone_cache = torch.load(backbone_path, map_location="cpu", weights_only=False)
            futures_cache = torch.load(
                os.path.join(ds_cache_dir, f"raw_futures_h{horizon}.pt"),
                map_location="cpu",
                weights_only=False,
            )
            basis_cache = torch.load(
                os.path.join(ds_cache_dir, f"fourier_basis_h{horizon}.pt"),
                map_location="cpu",
                weights_only=False,
            )

            valid_idx = futures_cache["valid_mask"].nonzero(as_tuple=True)[0]
            win_starts = backbone_cache["win_starts"][valid_idx]
            samples_ds = []
            for idx, i in enumerate(valid_idx):
                i_int = int(i.item())
                emb = backbone_cache["embeddings"][i_int].float()
                future_n = futures_cache["futures_n"][i_int].float()
                if not (torch.isfinite(emb).all() and torch.isfinite(future_n).all()):
                    continue
                samples_ds.append(
                    {
                        "dataset_name": dataset_name,
                        "emb": emb,
                        "future_n": future_n,
                        "daily_basis": _limit_fourier_basis(
                            basis_cache["daily_basis"].float(), n_fourier_terms["daily"]
                        ),
                        "weekly_basis": _limit_fourier_basis(
                            basis_cache["weekly_basis"].float(), n_fourier_terms["weekly"]
                        ),
                        "yearly_basis": _limit_fourier_basis(
                            basis_cache["yearly_basis"].float(), n_fourier_terms["yearly"]
                        ),
                        "win_start": int(win_starts[idx]),
                    }
                )

            samples_ds.sort(key=lambda item: item["win_start"])
            if len(samples_ds) <= 1:
                selected = samples_ds if split == "train" else []
            else:
                split_at = int(len(samples_ds) * (1 - val_split))
                split_at = max(1, min(split_at, len(samples_ds) - 1))
                selected = samples_ds[:split_at] if split == "train" else samples_ds[split_at:]
            if selected:
                self.datasets_used.append(dataset_name)
                self.samples.extend(selected)

        if split == "train" and not self.samples:
            raise ValueError(f"No train samples for horizon={horizon}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["emb"], s["future_n"], s["daily_basis"], s["weekly_basis"], s["yearly_basis"]

    def dataset_indices(self):
        indices = {}
        for idx, sample in enumerate(self.samples):
            indices.setdefault(sample["dataset_name"], []).append(idx)
        return indices


class DirectDataset(Dataset):
    def __init__(self, source: MonashFuncDecDataset):
        self.source = source

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        emb, future_n = self.source[idx][:2]
        return emb, future_n


class TFMRandomDirectHead(nn.Module):
    def __init__(self, embed_dim, horizon):
        super().__init__()
        self.head = ResidualBlock(
            ResidualBlockConfig(
                input_dims=embed_dim,
                hidden_dims=embed_dim,
                output_dims=horizon,
                use_bias=False,
                activation="swish",
            )
        )

    def forward(self, emb):
        return self.head(emb)


def batch_corr(x, y):
    x_c = x - x.mean(1, keepdim=True)
    y_c = y - y.mean(1, keepdim=True)
    r = (x_c * y_c).sum(1) / (x_c.norm(dim=1) * y_c.norm(dim=1) + 1e-8)
    return (r**2).mean()


def prediction_loss(pred, target, cfg):
    if cfg["loss"] == "MAE":
        return F.l1_loss(pred, target)
    return F.mse_loss(pred, target)


def compute_losses(model, batch, cfg, device):
    emb, future_n, db, wb, yb = [t.to(device, non_blocking=True) for t in batch]
    pred_n, decomp = model(emb, db, wb, yb)
    trend_n = decomp["trend"]
    seasonal_n = decomp["seasonal"]
    residual_n = decomp["residual"]

    L_pred = prediction_loss(pred_n, future_n, cfg)
    L_orth = batch_corr(trend_n, seasonal_n) + batch_corr(trend_n, residual_n) + batch_corr(seasonal_n, residual_n)
    L_sparse = decomp["delta"].abs().mean()
    L_res = prediction_loss(residual_n, future_n - trend_n.detach() - seasonal_n.detach(), cfg)
    total = (
        L_pred
        + cfg["lambda_orth"] * L_orth
        + cfg["lambda_sparse"] * L_sparse
        + cfg["lambda_res"] * L_res
    )
    return total, L_pred, L_orth, L_sparse, L_res


def train_funcdec(model, train_loader, optimizer, scheduler, cfg, device):
    model.train()
    params = list(model.decoder_t.parameters()) + list(model.decoder_s.parameters()) + list(model.decoder_r.parameters())
    history = {"total": [], "pred": [], "orth": [], "sparse": [], "res": []}
    step = 0
    while step < cfg["max_steps"]:
        for batch in train_loader:
            total, L_pred, L_orth, L_sparse, L_res = compute_losses(model, batch, cfg, device)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            history["total"].append(float(total.item()))
            history["pred"].append(float(L_pred.item()))
            history["orth"].append(float(L_orth.item()))
            history["sparse"].append(float(L_sparse.item()))
            history["res"].append(float(L_res.item()))
            step += 1
            if step == 1 or step % 500 == 0 or step == cfg["max_steps"]:
                print(
                    f"Step {step}/{cfg['max_steps']} | pred={L_pred.item():.4f} "
                    f"orth={L_orth.item():.4f} sparse={L_sparse.item():.4f} res={L_res.item():.4f}",
                    flush=True,
                )
            if step >= cfg["max_steps"]:
                break
    return history


def train_tfm_rdh(head, direct_loader, cfg, device):
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_steps"])
    head.train()
    losses = []
    step = 0
    while step < cfg["max_steps"]:
        for emb, future_n in direct_loader:
            emb = emb.to(device, non_blocking=True)
            future_n = future_n.to(device, non_blocking=True)
            pred_n = head(emb)
            loss = prediction_loss(pred_n, future_n, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(float(loss.item()))
            step += 1
            if step == 1 or step % 500 == 0 or step == cfg["max_steps"]:
                print(f"TFM_RDH Step {step}/{cfg['max_steps']} | loss={loss.item():.4f}", flush=True)
            if step >= cfg["max_steps"]:
                break
    return losses


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_values = 0
    all_preds = []
    all_futures = []
    all_decomps = []
    for batch in val_loader:
        emb, future_n, db, wb, yb = [t.to(device, non_blocking=True) for t in batch]
        pred_n, decomp = model(emb, db, wb, yb)
        total_mse += torch.sum((pred_n - future_n) ** 2).item()
        total_mae += torch.sum(torch.abs(pred_n - future_n)).item()
        n_values += future_n.numel()
        all_preds.append(pred_n.cpu())
        all_futures.append(future_n.cpu())
        all_decomps.append({k: v.cpu() for k, v in decomp.items()})
    preds = torch.cat(all_preds, dim=0)
    futures = torch.cat(all_futures, dim=0)
    decomps = {key: torch.cat([d[key] for d in all_decomps], dim=0) for key in all_decomps[0]}
    return total_mse / n_values, total_mae / n_values, preds, futures, decomps


@torch.no_grad()
def evaluate_tfm_rdh(head, val_loader, device):
    head.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_values = 0
    for batch in val_loader:
        emb = batch[0].to(device, non_blocking=True)
        future_n = batch[1].to(device, non_blocking=True)
        pred_n = head(emb)
        total_mse += torch.sum((pred_n - future_n) ** 2).item()
        total_mae += torch.sum(torch.abs(pred_n - future_n)).item()
        n_values += future_n.numel()
    return total_mse / n_values, total_mae / n_values


def _val_indices_for_cache(backbone_cache, futures_cache, val_split):
    valid_idx = futures_cache["valid_mask"].nonzero(as_tuple=True)[0]
    win_starts = backbone_cache["win_starts"][valid_idx]
    finite = []
    for idx, i in enumerate(valid_idx):
        i_int = int(i.item())
        emb = backbone_cache["embeddings"][i_int].float()
        future_n = futures_cache["futures_n"][i_int].float()
        if torch.isfinite(emb).all() and torch.isfinite(future_n).all():
            finite.append((int(win_starts[idx]), i_int))
    finite.sort(key=lambda item: item[0])
    if len(finite) <= 1:
        return []
    split_at = int(len(finite) * (1 - val_split))
    split_at = max(1, min(split_at, len(finite) - 1))
    return [i for _ws, i in finite[split_at:]]


@torch.no_grad()
def evaluate_tfm_zeroshot(bl_model, cache_pairs, horizon, cfg):
    from timesfm.configs import ForecastConfig

    context_len = int(cfg["context_len"])
    for _dataset_name, _ds_cache_dir, backbone_path in cache_pairs:
        cache = torch.load(backbone_path, map_location="cpu", weights_only=False)
        context_len = max(context_len, int(cache["context_len"]))
    max_horizon = int(math.ceil(max(horizon, 128) / 128) * 128)
    bl_model.compile(
        ForecastConfig(
            max_context=context_len,
            max_horizon=max_horizon,
            per_core_batch_size=64,
            force_flip_invariance=True,
            normalize_inputs=True,
            infer_is_positive=False,
        )
    )
    total_mse = 0.0
    total_mae = 0.0
    n_values = 0
    batch_contexts = []
    batch_future = []
    batch_mu = []
    batch_sigma = []

    def flush():
        nonlocal total_mse, total_mae, n_values
        if not batch_contexts:
            return
        point_forecast, _ = bl_model.forecast(horizon, batch_contexts)
        pred = torch.as_tensor(point_forecast, dtype=torch.float32)
        future_n = torch.stack(batch_future)
        mu = torch.as_tensor(batch_mu, dtype=torch.float32).view(-1, 1)
        sigma = torch.as_tensor(batch_sigma, dtype=torch.float32).view(-1, 1)
        denom = torch.where(sigma >= 1e-6, sigma, torch.ones_like(sigma))
        pred_n = (pred - mu) / denom
        total_mse += torch.sum((pred_n - future_n) ** 2).item()
        total_mae += torch.sum(torch.abs(pred_n - future_n)).item()
        n_values += future_n.numel()
        batch_contexts.clear()
        batch_future.clear()
        batch_mu.clear()
        batch_sigma.clear()

    for dataset_name, ds_cache_dir, backbone_path in cache_pairs:
        series = read_tsf_series(dataset_name)
        if not series:
            continue
        backbone_cache = torch.load(backbone_path, map_location="cpu", weights_only=False)
        futures_cache = torch.load(
            os.path.join(ds_cache_dir, f"raw_futures_h{horizon}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        ctx = int(backbone_cache["context_len"])
        for i in _val_indices_for_cache(backbone_cache, futures_cache, cfg["val_split"]):
            sid = backbone_cache["col_ids"][i]
            if sid not in series:
                continue
            ws = int(backbone_cache["win_starts"][i])
            context = series[sid][ws : ws + ctx]
            if len(context) != ctx:
                continue
            batch_contexts.append(context.astype(np.float32))
            batch_future.append(futures_cache["futures_n"][i].float())
            batch_mu.append(float(backbone_cache["mu"][i]))
            batch_sigma.append(float(backbone_cache["sigma"][i]))
            if len(batch_contexts) == 64:
                flush()
    flush()
    if n_values == 0:
        return None, None
    return total_mse / n_values, total_mae / n_values


def _smooth(values):
    steps = len(values)
    if steps < 2:
        return np.asarray(values), np.arange(steps)
    w = min(100, max(1, steps // 5))
    if w <= 1:
        return np.asarray(values), np.arange(steps)
    smoothed = np.convolve(values, np.ones(w) / w, mode="valid")
    return smoothed, np.arange(w - 1, steps)


def plot_losses(loss_history, horizon, results_dir=RESULTS_DIR):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key in zip(axes.ravel(), ["pred", "orth", "sparse", "res"], strict=True):
        values = loss_history[key]
        ax.plot(values, alpha=0.45, linewidth=0.8)
        smooth, x = _smooth(values)
        ax.plot(x, smooth, linewidth=1.8)
        ax.set_title(key)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"FuncDec training losses (h={horizon})")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"training_losses_h{horizon}.png"), dpi=150)
    plt.close(fig)


def plot_single_loss(values, title, out_path):
    if not values:
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(values, alpha=0.45, linewidth=0.8)
    smooth, x = _smooth(values)
    ax.plot(x, smooth, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_decomposition_samples(futures, preds, decomps, horizon, label, out_path, n_samples=6):
    if futures.numel() == 0:
        return
    n = min(n_samples, futures.shape[0])
    indices = np.linspace(0, futures.shape[0] - 1, num=n, dtype=int)
    x = np.arange(horizon)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for row, (ax, idx) in enumerate(zip(axes, indices, strict=True)):
        ax.plot(x, futures[idx].numpy(), label="GT", color="black", linestyle="--", linewidth=1.8)
        ax.plot(x, preds[idx].numpy(), label="Pred", color="#d62728", linewidth=1.5)
        ax.plot(x, decomps["trend"][idx].numpy(), label="Trend", color="#1f77b4", linewidth=1.1)
        ax.plot(x, decomps["seasonal"][idx].numpy(), label="Seasonal", color="#2ca02c", linewidth=1.0)
        ax.plot(x, decomps["residual"][idx].numpy(), label="Residual", color="#ff7f0e", linewidth=1.0, alpha=0.7)
        ax.grid(True, alpha=0.25)
        if row == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=5)
    axes[-1].set_xlabel("Forecast step")
    fig.suptitle(f"{label} decomposition samples (h={horizon})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_decomposition_samples_by_dataset(futures, preds, decomps, horizon, dataset_indices, out_dir, n_samples=6):
    os.makedirs(out_dir, exist_ok=True)
    for dataset_name, indices in sorted(dataset_indices.items()):
        if not indices:
            continue
        idx = torch.tensor(indices, dtype=torch.long)
        plot_decomposition_samples(
            futures.index_select(0, idx),
            preds.index_select(0, idx),
            {k: v.index_select(0, idx) for k, v in decomps.items()},
            horizon,
            label=f"FuncDec {dataset_name}",
            out_path=os.path.join(out_dir, f"decomposition_{dataset_name}_h{horizon}.png"),
            n_samples=n_samples,
        )


def plot_summary(results, run_dir=EXP_DIR):
    if not results:
        return
    horizons = [r["horizon"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(horizons, [r["funcdec_val_mse"] for r in results], "o-", label="FuncDec")
    axes[0].plot(horizons, [r["tfm_rdh_val_mse"] for r in results], "^--", label="TFM_RDH")
    axes[0].plot(horizons, [r["tfm_zeroshot_val_mse"] for r in results], "s--", label="TFM_ZS")
    axes[0].set_title("MSE")
    axes[1].plot(horizons, [r["funcdec_val_mae"] for r in results], "o-", label="FuncDec")
    axes[1].plot(horizons, [r["tfm_rdh_val_mae"] for r in results], "^--", label="TFM_RDH")
    axes[1].plot(horizons, [r["tfm_zeroshot_val_mae"] for r in results], "s--", label="TFM_ZS")
    axes[1].set_title("MAE")
    for ax in axes:
        ax.set_xlabel("Horizon")
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("FuncDec horizon comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "horizon_comparison.png"), dpi=150)
    plt.close(fig)


def write_benchmark(results, cfg, run_dir=EXP_DIR):
    path = os.path.join(run_dir, "benchmark_results.txt")
    lines = [
        "=" * 72,
        f"  FuncDec Benchmark  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  steps={cfg['max_steps']}  loss={cfg['loss']}",
        "=" * 72,
    ]
    for r in results:
        lines.append(
            f"  h={r['horizon']}:  FuncDec MSE={r['funcdec_val_mse']:.4f} "
            f"MAE={r['funcdec_val_mae']:.4f} | TFM_RDH MSE={r['tfm_rdh_val_mse']:.4f} "
            f"MAE={r['tfm_rdh_val_mae']:.4f} | TFM_ZS MSE={r['tfm_zeroshot_val_mse']:.4f} "
            f"MAE={r['tfm_zeroshot_val_mae']:.4f}"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_single_horizon(cfg, model, device, tfm_zeroshot_model=None):
    horizon = cfg["horizon"]
    cache_pairs = discover_backbone_caches(
        horizon,
        cfg["monash_cache_root"],
        cfg["dataset_cache_dir"],
        cfg["datasets"],
    )
    if not cache_pairs:
        print(f"[skip] h={horizon}: no caches")
        return None

    model.reset_decoders(cfg)
    model.to(device)

    train_ds = MonashFuncDecDataset(
        cache_pairs, horizon, "train", cfg["val_split"], cfg["dataset_cache_dir"], cfg["n_fourier_terms"]
    )
    val_ds = MonashFuncDecDataset(
        cache_pairs, horizon, "val", cfg["val_split"], cfg["dataset_cache_dir"], cfg["n_fourier_terms"]
    )
    if len(val_ds) == 0:
        print(f"[skip] h={horizon}: no validation samples")
        return None

    nw_train = 0 if len(train_ds) <= cfg["batch_size"] else 4
    nw_val = 0 if len(val_ds) <= cfg["batch_size"] else 2
    pm = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=nw_train,
        pin_memory=pm,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=nw_val,
        pin_memory=pm,
    )
    print(f"h={horizon} | train={len(train_ds)} val={len(val_ds)} datasets={sorted(set(train_ds.datasets_used))}")

    decoder_params = list(model.decoder_t.parameters()) + list(model.decoder_s.parameters()) + list(model.decoder_r.parameters())
    optimizer = torch.optim.AdamW(decoder_params, lr=cfg["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_steps"])

    loss_history = train_funcdec(model, train_loader, optimizer, scheduler, cfg, device)
    plot_losses(loss_history, horizon, cfg["results_dir"])
    plot_single_loss(
        loss_history["total"],
        f"FuncDec total loss (h={horizon})",
        os.path.join(cfg["results_dir"], f"training_total_loss_h{horizon}.png"),
    )
    with open(os.path.join(cfg["results_dir"], f"training_losses_h{horizon}.json"), "w") as f:
        json.dump(to_jsonable(loss_history), f, indent=2)

    ckpt = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
    torch.save(ckpt, os.path.join(cfg["ckpt_dir"], f"funcdec_h{horizon}.pt"))

    april_mse, april_mae, preds, futures, decomps = evaluate(model, val_loader, device)
    decomp_metrics = compute_decomposition_metrics(futures, preds, decomps)
    with open(os.path.join(cfg["results_dir"], f"decomposition_metrics_h{horizon}.json"), "w") as f:
        json.dump(to_jsonable(decomp_metrics), f, indent=2)
    plot_decomposition_samples(
        futures,
        preds,
        decomps,
        horizon,
        label="FuncDec",
        out_path=os.path.join(cfg["results_dir"], f"decomposition_h{horizon}.png"),
    )
    plot_decomposition_samples_by_dataset(
        futures,
        preds,
        decomps,
        horizon,
        val_ds.dataset_indices(),
        os.path.join(cfg["results_dir"], "decomposition_by_dataset", f"h{horizon}"),
    )

    direct_ds = DirectDataset(train_ds)
    direct_loader = DataLoader(
        direct_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=nw_train,
        pin_memory=pm,
    )
    tfm_rdh = TFMRandomDirectHead(cfg["embed_dim"], horizon).to(device)
    tfm_rdh_losses = train_tfm_rdh(tfm_rdh, direct_loader, cfg, device)
    plot_single_loss(
        tfm_rdh_losses,
        f"TFM_RDH {cfg['loss']} loss (h={horizon})",
        os.path.join(cfg["results_dir"], f"tfm_rdh_loss_h{horizon}.png"),
    )
    with open(os.path.join(cfg["results_dir"], f"tfm_rdh_losses_h{horizon}.json"), "w") as f:
        json.dump(to_jsonable(tfm_rdh_losses), f, indent=2)
    torch.save(tfm_rdh.state_dict(), os.path.join(cfg["ckpt_dir"], f"tfm_rdh_h{horizon}.pt"))
    tfm_mse, tfm_mae = evaluate_tfm_rdh(tfm_rdh, val_loader, device)
    if cfg["eval_tfm_zeroshot"] and tfm_zeroshot_model is not None:
        tfm_zs_mse, tfm_zs_mae = evaluate_tfm_zeroshot(tfm_zeroshot_model, cache_pairs, horizon, cfg)
        if tfm_zs_mse is None:
            tfm_zs_mse, tfm_zs_mae = float("nan"), float("nan")
    else:
        tfm_zs_mse, tfm_zs_mae = float("nan"), float("nan")

    print(f"\n  h={horizon}:")
    print(f"    FuncDec  | MSE: {april_mse:.4f} | MAE: {april_mae:.4f}")
    print(f"    TFM_RDH  | MSE: {tfm_mse:.4f}  | MAE: {tfm_mae:.4f}")
    print(f"    TFM_ZS   | MSE: {tfm_zs_mse:.4f}  | MAE: {tfm_zs_mae:.4f}")

    return {
        "horizon": horizon,
        "funcdec_val_mse": float(april_mse),
        "funcdec_val_mae": float(april_mae),
        "tfm_rdh_val_mse": float(tfm_mse),
        "tfm_rdh_val_mae": float(tfm_mae),
        "tfm_zeroshot_val_mse": float(tfm_zs_mse),
        "tfm_zeroshot_val_mae": float(tfm_zs_mae),
        "decomposition_metrics": decomp_metrics,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "datasets_used": sorted(set(train_ds.datasets_used + val_ds.datasets_used)),
    }


def main():
    cfg = get_config()
    device = torch.device(cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu")
    run_dir = make_run_dir(cfg)
    cfg["run_dir"] = run_dir
    cfg["ckpt_dir"] = os.path.join(run_dir, "checkpoints")
    cfg["results_dir"] = os.path.join(run_dir, "results")
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    print(f"run_dir={run_dir}")

    config_to_save = dict(cfg)
    config_to_save.pop("horizon", None)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(to_jsonable(config_to_save), f, indent=2)

    model = FuncDecModel(cfg, load_backbone=False).to(device)
    tfm_zeroshot_model = None
    if cfg["eval_tfm_zeroshot"]:
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

        tfm_zeroshot_model = TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
            torch_compile=False,
        )

    results = []
    for h in cfg["horizons"]:
        cfg["horizon"] = int(h)
        r = run_single_horizon(cfg, model, device, tfm_zeroshot_model)
        if r:
            results.append(r)

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(to_jsonable(results), f, indent=2)
    plot_summary(results, run_dir)
    write_benchmark(results, cfg, run_dir)


if __name__ == "__main__":
    main()
