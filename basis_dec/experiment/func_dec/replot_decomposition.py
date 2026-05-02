from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
from torch.utils.data import DataLoader


EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RUNS_DIR = os.path.join(EXP_DIR, "runs")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, EXP_DIR)

from model.decomp_funcdec import FuncDecModel  # noqa: E402
from train import (  # noqa: E402
    DEFAULT_CONFIG,
    MonashFuncDecDataset,
    discover_backbone_caches,
    evaluate,
    plot_decomposition_samples,
    plot_decomposition_samples_by_dataset,
)


def latest_run_dir():
    runs = [path for path in glob.glob(os.path.join(RUNS_DIR, "*")) if os.path.isdir(path)]
    if not runs:
        raise FileNotFoundError(f"No run directories found under {RUNS_DIR}")
    return max(runs, key=os.path.getmtime)


def load_config(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        saved = json.load(f)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(saved)
    cfg["run_dir"] = run_dir
    cfg["ckpt_dir"] = os.path.join(run_dir, "checkpoints")
    cfg["results_dir"] = os.path.join(run_dir, "results")
    return cfg


def replot_horizon(cfg, horizon, device):
    cfg["horizon"] = int(horizon)
    ckpt_path = os.path.join(cfg["ckpt_dir"], f"funcdec_h{horizon}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[skip] h={horizon}: missing checkpoint {ckpt_path}")
        return

    cache_pairs = discover_backbone_caches(
        int(horizon),
        cfg["monash_cache_root"],
        cfg["dataset_cache_dir"],
        cfg["datasets"],
    )
    if not cache_pairs:
        print(f"[skip] h={horizon}: no caches")
        return

    model = FuncDecModel(cfg, load_backbone=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)

    val_ds = MonashFuncDecDataset(
        cache_pairs,
        int(horizon),
        "val",
        cfg["val_split"],
        cfg["dataset_cache_dir"],
        cfg["n_fourier_terms"],
    )
    if len(val_ds) == 0:
        print(f"[skip] h={horizon}: no validation samples")
        return

    pm = device.type == "cuda"
    nw_val = 0 if len(val_ds) <= cfg["batch_size"] else 2
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=nw_val,
        pin_memory=pm,
    )
    mse, mae, preds, futures, decomps = evaluate(model, val_loader, device)
    plot_decomposition_samples(
        futures,
        preds,
        decomps,
        int(horizon),
        label="FuncDec",
        out_path=os.path.join(cfg["results_dir"], f"decomposition_h{horizon}.png"),
    )
    out_dir = os.path.join(cfg["results_dir"], "decomposition_by_dataset", f"h{horizon}")
    plot_decomposition_samples_by_dataset(
        futures,
        preds,
        decomps,
        int(horizon),
        val_ds.dataset_indices(),
        out_dir,
    )
    print(f"h={horizon} | val={len(val_ds)} | MSE={mse:.4f} MAE={mae:.4f} | saved={out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    cfg = load_config(run_dir)
    if args.horizons is not None:
        cfg["horizons"] = args.horizons
    if args.device is not None:
        cfg["device"] = args.device

    device = torch.device(cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"run_dir={run_dir}")
    for horizon in cfg["horizons"]:
        replot_horizon(cfg, int(horizon), device)


if __name__ == "__main__":
    main()
