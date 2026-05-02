from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


MONASH_CACHE_ROOT = "/home/sia2/project/4.22prophet/timesfm/prophet_ans/data/monash/cache"
MONASH_TSF_ROOT = "/home/sia2/project/4.28basis/basis_dec/data/monash"
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CACHE_DIR = os.path.join(EXP_DIR, "dataset_cache")

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
N_FOURIER_TERMS = {"daily": 6, "weekly": 4, "yearly": 8}
SEASONALITY_PERIODS = {"daily": 1.0, "weekly": 7.0, "yearly": 365.25}
HORIZONS = [96, 192, 336, 720]
REVIN_TOL = 1e-6

sys.path.insert(0, MONASH_TSF_ROOT)
from prepare import iter_rows, read_meta  # noqa: E402


def find_tsf_path(dataset_name: str) -> str | None:
    candidates = [
        os.path.join(MONASH_TSF_ROOT, f"{dataset_name}.tsf"),
        os.path.join(MONASH_TSF_ROOT, f"{dataset_name}_without_missing_values.tsf"),
        os.path.join(MONASH_TSF_ROOT, f"{dataset_name}_with_missing_values.tsf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    print(f"[warning] missing tsf: {dataset_name}")
    return None


def build_seasonality_mask(freq: str, context_len: int, out_path: str) -> dict:
    if os.path.exists(out_path):
        return torch.load(out_path, weights_only=False)

    fd = FREQ_DAYS[freq]
    context_span = context_len * fd
    mask = {}
    for stype in ["daily", "weekly", "yearly"]:
        period = SEASONALITY_PERIODS[stype]
        mask[stype] = bool((fd < period) and (context_span >= period))

    data = {
        "yearly": mask["yearly"],
        "weekly": mask["weekly"],
        "daily": mask["daily"],
        "freq": freq,
        "context_len": context_len,
    }
    torch.save(data, out_path)
    return data


def build_fourier_basis(freq: str, context_len: int, horizon: int, out_path: str) -> dict:
    if os.path.exists(out_path):
        return torch.load(out_path, weights_only=False)

    mask_path = os.path.join(os.path.dirname(out_path), "seasonality_mask.pt")
    mask_data = build_seasonality_mask(freq, context_len, mask_path)
    fd = FREQ_DAYS[freq]
    t = torch.arange(horizon, dtype=torch.float32)
    save_data = {
        "mask": {
            "daily": bool(mask_data["daily"]),
            "weekly": bool(mask_data["weekly"]),
            "yearly": bool(mask_data["yearly"]),
        },
        "freq": freq,
        "horizon": int(horizon),
    }

    for stype in ["daily", "weekly", "yearly"]:
        n = N_FOURIER_TERMS[stype]
        basis = torch.zeros(horizon, 2 * n)
        if mask_data[stype]:
            p_steps = SEASONALITY_PERIODS[stype] / fd
            for k in range(n):
                basis[:, 2 * k] = torch.sin(2 * math.pi * (k + 1) * t / p_steps)
                basis[:, 2 * k + 1] = torch.cos(2 * math.pi * (k + 1) * t / p_steps)
        save_data[f"{stype}_basis"] = basis

    torch.save(save_data, out_path)
    return save_data


def build_raw_futures(dataset_name: str, horizon: int, backbone_cache_path: str, out_path: str) -> dict | None:
    if os.path.exists(out_path):
        return torch.load(out_path, weights_only=False)

    tsf_path = find_tsf_path(dataset_name)
    if tsf_path is None:
        return None

    cache = torch.load(backbone_cache_path, weights_only=False)
    meta = read_meta(Path(tsf_path))
    series_dict = {row.sid: row.values for row in iter_rows(meta)}

    col_ids = cache["col_ids"]
    win_starts = cache["win_starts"]
    context_len = int(cache["context_len"])
    n_samples = len(col_ids)
    futures_n = torch.zeros(n_samples, horizon)
    valid_mask = torch.zeros(n_samples, dtype=torch.bool)

    for i in range(n_samples):
        sid = col_ids[i]
        ws = int(win_starts[i])
        mu = float(cache["mu"][i])
        sig = float(cache["sigma"][i])
        denom = sig if sig >= REVIN_TOL else 1.0

        if sid not in series_dict:
            continue
        series = series_dict[sid]
        start = ws + context_len
        end = ws + context_len + horizon
        if end > len(series):
            continue

        future = series[start:end].astype(np.float32)
        futures_n[i] = torch.from_numpy((future - mu) / denom)
        valid_mask[i] = True

    data = {
        "futures_n": futures_n,
        "valid_mask": valid_mask,
        "context_len": context_len,
        "horizon": horizon,
    }
    torch.save(data, out_path)
    return data


def find_backbone_cache(dataset_name: str, context_len: int | None, horizon: int) -> str | None:
    if context_len is None:
        pattern = os.path.join(MONASH_CACHE_ROOT, dataset_name, f"backbone_emb_c*_h{horizon}_stride*.pt")
    else:
        pattern = os.path.join(MONASH_CACHE_ROOT, dataset_name, f"backbone_emb_c{context_len}_h{horizon}_stride*.pt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"[warning] missing backbone cache: {dataset_name} h{horizon}")
        return None
    return matches[0]


def iter_dataset_h96_caches(dataset_filter: set[str] | None):
    pattern = os.path.join(MONASH_CACHE_ROOT, "*", "backbone_emb_c*_h96_stride*.pt")
    for path in sorted(glob.glob(pattern)):
        dataset_name = os.path.basename(os.path.dirname(path))
        if dataset_filter is not None and dataset_name not in dataset_filter:
            continue
        yield dataset_name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
    dataset_filter = set(args.datasets) if args.datasets is not None else None
    summaries = []

    for dataset_name, h96_cache_path in iter_dataset_h96_caches(dataset_filter):
        h96_cache = torch.load(h96_cache_path, weights_only=False)
        freq = h96_cache["frequency"]
        context_len = int(h96_cache["context_len"])
        ds_cache_dir = os.path.join(DATASET_CACHE_DIR, dataset_name)
        os.makedirs(ds_cache_dir, exist_ok=True)

        mask_path = os.path.join(ds_cache_dir, "seasonality_mask.pt")
        mask_data = build_seasonality_mask(freq, context_len, mask_path)
        horizon_counts = {}

        for horizon in args.horizons:
            fourier_path = os.path.join(ds_cache_dir, f"fourier_basis_h{horizon}.pt")
            build_fourier_basis(freq, context_len, horizon, fourier_path)

            backbone_cache_path = find_backbone_cache(dataset_name, context_len, horizon)
            if backbone_cache_path is None:
                horizon_counts[horizon] = None
                continue

            raw_path = os.path.join(ds_cache_dir, f"raw_futures_h{horizon}.pt")
            raw_data = build_raw_futures(dataset_name, horizon, backbone_cache_path, raw_path)
            horizon_counts[horizon] = None if raw_data is None else int(raw_data["valid_mask"].sum().item())

        summaries.append((dataset_name, freq, mask_data, horizon_counts))

    for dataset_name, freq, mask_data, horizon_counts in summaries:
        mask = {k: bool(mask_data[k]) for k in ["daily", "weekly", "yearly"]}
        counts = ", ".join(f"h{h}={count}" for h, count in horizon_counts.items())
        print(f"{dataset_name}: freq={freq} mask={mask} valid={counts}")


if __name__ == "__main__":
    main()
