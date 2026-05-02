"""
Monash TSF 데이터 준비 스크립트.

사용법:
  python prepare.py --mode raw
  python prepare.py --mode prophet --dataset m3_monthly_dataset --n_workers 4
  python prepare.py --mode backbone --dataset m3_monthly_dataset --horizon 96
  python prepare.py --mode all --dataset m3_monthly_dataset --max_series 20
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import multiprocessing
import os
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from prophet_config import (
    DEFAULT_PROPHET_CONFIG,
    FREQUENCY_CONTEXT,
    FREQUENCY_PANDAS,
    FREQUENCY_TIMESFM,
    HORIZONS,
    START_DATE_FALLBACK,
)


ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "cache"
DIAG_ROOT = ROOT / "diagnostics"
PLOT_SAMPLE_RATIO = 0.01
PLOT_RANDOM_SEED = 42
PLOT_MAX_POINTS = 720


@dataclass(frozen=True)
class TsfMeta:
    key: str
    path: Path
    relation: str
    frequency: str | None
    horizon: int | None
    missing: bool
    equallength: bool
    attributes: list[tuple[str, str]]


@dataclass(frozen=True)
class SeriesRow:
    sid: str
    start_timestamp: str | None
    values: np.ndarray
    missing_count: int


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def dataset_key(path: Path) -> str:
    name = path.stem
    for suffix in ("_without_missing_values", "_with_missing_values"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def read_meta(path: Path) -> TsfMeta:
    attrs: list[tuple[str, str]] = []
    raw: dict[str, str] = {}
    with path.open("r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line == "@data":
                break
            if line.startswith("@attribute"):
                parts = line.split(maxsplit=2)
                if len(parts) == 3:
                    attrs.append((parts[1], parts[2]))
                continue
            if line.startswith("@"):
                parts = line.split(maxsplit=1)
                raw[parts[0][1:]] = parts[1] if len(parts) > 1 else ""
    horizon = raw.get("horizon")
    return TsfMeta(
        key=dataset_key(path),
        path=path,
        relation=raw.get("relation", path.stem),
        frequency=raw.get("frequency"),
        horizon=int(horizon) if horizon and horizon.isdigit() else None,
        missing=parse_bool(raw.get("missing")),
        equallength=parse_bool(raw.get("equallength")),
        attributes=attrs,
    )


def selected_metas() -> tuple[list[TsfMeta], list[dict]]:
    files = sorted(ROOT.glob("*.tsf"))
    metas = [read_meta(path) for path in files]
    without_keys = {
        dataset_key(meta.path)
        for meta in metas
        if meta.path.stem.endswith("_without_missing_values")
    }

    selected: list[TsfMeta] = []
    excluded: list[dict] = []
    seen: set[str] = set()
    for meta in metas:
        stem = meta.path.stem
        if stem.endswith("_with_missing_values"):
            excluded.append({"file": meta.path.name, "reason": "with_missing_values"})
            continue
        if meta.key in without_keys and not stem.endswith("_without_missing_values"):
            excluded.append({"file": meta.path.name, "reason": "paired_without_missing_values_exists"})
            continue
        if not meta.frequency:
            excluded.append({"file": meta.path.name, "reason": "missing_frequency"})
            continue
        if meta.frequency not in FREQUENCY_CONTEXT:
            excluded.append({"file": meta.path.name, "reason": f"unsupported_frequency={meta.frequency}"})
            continue
        if meta.key in seen:
            excluded.append({"file": meta.path.name, "reason": f"duplicate_key={meta.key}"})
            continue
        selected.append(meta)
        seen.add(meta.key)
    return selected, excluded


def iter_rows(meta: TsfMeta) -> Iterable[SeriesRow]:
    in_data = False
    attr_names = [name for name, _ in meta.attributes]
    with meta.path.open("r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not in_data:
                in_data = line == "@data"
                continue
            parts = line.split(":")
            if len(parts) < len(attr_names) + 1:
                continue
            values_text = parts[-1]
            attr_values = parts[: len(attr_names)]
            attrs = dict(zip(attr_names, attr_values, strict=False))
            raw_values = pd.Series(values_text.split(","))
            numeric = pd.to_numeric(raw_values, errors="coerce")
            missing_count = int(numeric.isna().sum())
            values = numeric.dropna()
            yield SeriesRow(
                sid=str(attrs.get("series_name", f"series_{len(attr_values)}")),
                start_timestamp=attrs.get("start_timestamp"),
                values=values.to_numpy(dtype=np.float32),
                missing_count=missing_count,
            )


def series_dates(row: SeriesRow, freq: str) -> pd.DatetimeIndex:
    start = row.start_timestamp or START_DATE_FALLBACK
    if " " in start:
        date_part, time_part = start.split(maxsplit=1)
        start = f"{date_part} {time_part.replace('-', ':')}"
    start_ts = pd.to_datetime(start, errors="coerce")
    if pd.isna(start_ts):
        start_ts = pd.to_datetime(START_DATE_FALLBACK)
    if getattr(start_ts, "tzinfo", None) is not None:
        start_ts = start_ts.tz_localize(None)
    return pd.date_range(start_ts, periods=len(row.values), freq=FREQUENCY_PANDAS[freq])


def auto_n_changepoints(train_len: int) -> int:
    return max(5, min(50, train_len // 50))


def build_config(train_len: int, freq: str) -> dict:
    cfg = DEFAULT_PROPHET_CONFIG.copy()
    cfg["n_changepoints"] = auto_n_changepoints(train_len)
    return cfg


def build_prophet(cfg: dict):
    from prophet import Prophet

    model = Prophet(
        growth=cfg["growth"],
        n_changepoints=cfg["n_changepoints"],
        changepoint_range=cfg["changepoint_range"],
        changepoint_prior_scale=cfg["changepoint_prior_scale"],
        seasonality_mode=cfg["seasonality_mode"],
        seasonality_prior_scale=cfg["seasonality_prior_scale"],
        yearly_seasonality=cfg["yearly_seasonality"],
        weekly_seasonality=cfg["weekly_seasonality"],
        daily_seasonality=cfg["daily_seasonality"],
        uncertainty_samples=cfg["uncertainty_samples"],
    )
    for seasonality in cfg.get("custom_seasonalities", []):
        model.add_seasonality(**seasonality)
    return model


def compute_diagnostics(y: np.ndarray, fcst: pd.DataFrame) -> dict:
    residuals = y - fcst["yhat"].values
    if len(residuals) > 1 and np.std(residuals[:-1]) > 0 and np.std(residuals[1:]) > 0:
        acf1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    else:
        acf1 = 0.0
    trend_var_ratio = float(np.var(fcst["trend"].values) / (np.var(y) + 1e-8))
    seasonal_var_ratio = float(np.var(fcst["additive_terms"].values) / (np.var(y) + 1e-8))
    return {
        "series_len": int(len(y)),
        "residual_acf1": acf1,
        "trend_var_ratio": trend_var_ratio,
        "seasonal_var_ratio": seasonal_var_ratio,
        "approved": bool(acf1 < 0.3 and trend_var_ratio < 1.0),
    }


def paths_for(meta: TsfMeta) -> dict[str, Path]:
    cache_dir = CACHE_ROOT / meta.key
    diag_dir = DIAG_ROOT / meta.key
    (diag_dir / "plots").mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {"cache": cache_dir, "diag": diag_dir}


def raw_summary(meta: TsfMeta) -> dict:
    lengths = []
    missing_values = 0
    for row in iter_rows(meta):
        lengths.append(len(row.values))
        missing_values += row.missing_count
    context_len = FREQUENCY_CONTEXT[meta.frequency]
    usable = {
        str(h): sum(1 for n in lengths if n >= context_len + h)
        for h in HORIZONS
    }
    return {
        "dataset": meta.key,
        "file": meta.path.name,
        "relation": meta.relation,
        "frequency": meta.frequency,
        "pandas_freq": FREQUENCY_PANDAS[meta.frequency],
        "timesfm_freq": FREQUENCY_TIMESFM[meta.frequency],
        "source_horizon": meta.horizon,
        "horizons": HORIZONS,
        "missing": meta.missing,
        "equallength": meta.equallength,
        "series_count": len(lengths),
        "min_len": int(min(lengths)) if lengths else 0,
        "median_len": float(np.median(lengths)) if lengths else 0.0,
        "max_len": int(max(lengths)) if lengths else 0,
        "context_len": context_len,
        "stride": context_len,
        "usable_series_by_horizon": usable,
        "skipped_short_by_horizon": {h: len(lengths) - usable[str(h)] for h in HORIZONS},
        "missing_values_after_parse": missing_values,
        "split": "all series points are training data; no temporal test split",
    }


def run_raw(metas: list[TsfMeta], excluded: list[dict], write_manifest: bool = True) -> None:
    DIAG_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for meta in metas:
        summary = raw_summary(meta)
        paths = paths_for(meta)
        with (paths["diag"] / "raw_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        summaries.append(summary)
        print(
            f"[raw] {meta.key}: freq={meta.frequency} series={summary['series_count']} "
            f"len={summary['min_len']}/{summary['median_len']}/{summary['max_len']} "
            f"context={summary['context_len']}",
            flush=True,
        )
    if write_manifest:
        with (DIAG_ROOT / "selected_datasets.json").open("w") as f:
            json.dump({"selected": summaries, "excluded": excluded}, f, indent=2)
    print(f"[raw] selected={len(summaries)} excluded={len(excluded)}", flush=True)


def fit_one(args: tuple[str, str, str, str, str | None, np.ndarray, bool]) -> tuple[str, dict]:
    dataset_key_, cache_dir, freq, sid, start_timestamp, values, should_plot = args
    from prophet.serialize import model_to_json

    logging.getLogger("cmdstanpy").disabled = True
    row = SeriesRow(sid=sid, start_timestamp=start_timestamp, values=values, missing_count=0)
    dates = series_dates(row, freq)
    prophet_df = pd.DataFrame({"ds": dates, "y": values.astype(np.float64)})
    cfg = build_config(len(values), freq)
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            model = build_prophet(cfg)
            model.fit(prophet_df)
    fcst = model.predict(prophet_df[["ds"]])
    holidays = fcst["holidays"].values if "holidays" in fcst.columns else np.zeros(len(fcst))
    trend = fcst["trend"].values
    seasonal = fcst["additive_terms"].values - holidays
    residual = holidays + (values - fcst["yhat"].values)
    metrics = compute_diagnostics(values, fcst)
    stem = safe_name(sid)
    np.savez(
        Path(cache_dir) / f"prophet_decomp_{stem}.npz",
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        approved=np.array(metrics["approved"]),
    )
    with (Path(cache_dir) / f"prophet_model_{stem}.json").open("w") as f:
        f.write(model_to_json(model))
    return sid, metrics


def run_prophet(meta: TsfMeta, n_workers: int, max_series: int, overwrite: bool) -> None:
    paths = paths_for(meta)
    rows_iter = iter_rows(meta)
    rows = list(islice(rows_iter, max_series)) if max_series else list(rows_iter)
    if overwrite:
        for file in paths["cache"].glob("prophet_*"):
            file.unlink()
    pending = [
        row for row in rows
        if not (paths["cache"] / f"prophet_decomp_{safe_name(row.sid)}.npz").exists()
    ]
    if not pending:
        print(f"[prophet] {meta.key}: all selected series already cached", flush=True)
        return
    rng = np.random.default_rng(PLOT_RANDOM_SEED)
    plot_count = max(1, int(np.ceil(len(rows) * PLOT_SAMPLE_RATIO)))
    plot_ids = set(rng.choice([row.sid for row in rows], size=min(plot_count, len(rows)), replace=False).tolist())
    worker_count = n_workers or (os.cpu_count() or 4)
    worker_count = max(1, min(worker_count, len(pending)))
    args = [
        (meta.key, str(paths["cache"]), meta.frequency, row.sid, row.start_timestamp, row.values, row.sid in plot_ids)
        for row in pending
    ]
    print(f"[prophet] {meta.key}: pending={len(args)} workers={worker_count}", flush=True)
    results = []
    with multiprocessing.Pool(processes=worker_count) as pool:
        for done, result in enumerate(pool.imap_unordered(fit_one, args), start=1):
            results.append(result)
            if done == 1 or done % 10 == 0 or done == len(args):
                print(f"[prophet] {meta.key}: fitted={done}/{len(args)}", flush=True)
    metrics_path = paths["diag"] / "fit_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    for sid, item in results:
        metrics[sid] = item
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[prophet] metrics saved: {metrics_path}", flush=True)


class BackboneDataset:
    def __init__(self, meta: TsfMeta, horizon: int):
        self.context_len = FREQUENCY_CONTEXT[meta.frequency]
        self.horizon = horizon
        self.stride = self.context_len
        self.samples: list[tuple[int, str, int]] = []
        self.series: list[np.ndarray] = []
        self.skipped_short: list[str] = []
        for row_idx, row in enumerate(iter_rows(meta)):
            values = row.values.astype(np.float32, copy=False)
            self.series.append(values)
            max_start = len(values) - self.context_len - horizon
            if max_start < 0:
                self.skipped_short.append(row.sid)
                continue
            for start in range(0, max_start + 1, self.stride):
                self.samples.append((row_idx, row.sid, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row_idx, sid, start = self.samples[idx]
        values = self.series[row_idx]
        return values[start:start + self.context_len], row_idx, sid, start


def collate_backbone(batch):
    import torch

    contexts, row_indices, series_ids, starts = zip(*batch)
    context = torch.from_numpy(np.stack(contexts, axis=0))
    return {
        "contexts": context,
        "masks": torch.zeros_like(context, dtype=torch.bool),
        "row_indices": torch.tensor(row_indices, dtype=torch.long),
        "series_ids": list(series_ids),
        "win_starts": torch.tensor(starts, dtype=torch.long),
    }


def resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def run_backbone(meta: TsfMeta, horizon: int, batch_size: int, num_workers: int, device_name: str) -> None:
    import sys
    import torch
    from torch.utils.data import DataLoader

    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}: {horizon}")
    paths = paths_for(meta)
    context_len = FREQUENCY_CONTEXT[meta.frequency]
    if context_len % 32 != 0:
        raise ValueError("context_len은 TimesFM patch_len=32의 배수여야 합니다.")
    cache_path = paths["cache"] / f"backbone_emb_c{context_len}_h{horizon}_stride{context_len}.pt"
    if cache_path.exists():
        print(f"[backbone] Already exists: {cache_path}", flush=True)
        return
    missing_decomp = [
        row.sid
        for row in iter_rows(meta)
        if len(row.values) >= context_len + horizon
        and not (paths["cache"] / f"prophet_decomp_{safe_name(row.sid)}.npz").exists()
    ]
    if missing_decomp:
        raise FileNotFoundError(
            f"{meta.key}: Prophet decomp 캐시 누락 {len(missing_decomp)}개 "
            f"(예: {missing_decomp[:5]}). --mode prophet 먼저 실행하세요."
        )

    project_root = ROOT.parents[2]
    src_dir = str(project_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    from timesfm.torch.util import revin, update_running_stats

    dataset = BackboneDataset(meta, horizon)
    if len(dataset) == 0:
        raise ValueError(f"{meta.key}: 생성 가능한 train window가 없습니다.")
    device = resolve_device(device_name)
    print(
        f"[backbone] {meta.key}: context={context_len} horizon={horizon} "
        f"samples={len(dataset)} skipped_short={len(dataset.skipped_short)} device={device}",
        flush=True,
    )
    pretrained = TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False,
    )
    backbone = pretrained.model.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    patch_len = backbone.p
    embed_dim = backbone.md
    num_patches = context_len // patch_len
    if patch_len != 32 or embed_dim != 1280:
        raise ValueError(f"TimesFM shape 불일치: patch_len={patch_len}, embed_dim={embed_dim}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_backbone,
    )
    embeddings = torch.empty((len(dataset), embed_dim), dtype=torch.float32)
    mu_all = torch.empty((len(dataset), 1), dtype=torch.float32)
    sigma_all = torch.empty((len(dataset), 1), dtype=torch.float32)
    row_indices_all = torch.empty((len(dataset),), dtype=torch.long)
    win_starts_all = torch.empty((len(dataset),), dtype=torch.long)
    series_ids_all: list[str] = []

    def encode_context_batch(context: torch.Tensor, masks: torch.Tensor):
        batch_n = context.shape[0]
        patched_inputs = context.reshape(batch_n, -1, patch_len)
        patched_masks = masks.reshape(batch_n, -1, patch_len)
        n = torch.zeros(batch_n, device=device)
        mu = torch.zeros(batch_n, device=device)
        sigma = torch.zeros(batch_n, device=device)
        patch_mu, patch_sigma = [], []
        for patch_idx in range(num_patches):
            (n, mu, sigma), _ = update_running_stats(
                n, mu, sigma, patched_inputs[:, patch_idx], patched_masks[:, patch_idx],
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)
        normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
        with torch.no_grad():
            (_, output_embeddings, _, _), _ = backbone(normed_inputs, patched_masks)
        return output_embeddings[:, -1, :], context_mu[:, -1:], context_sigma[:, -1:]

    write_idx = 0
    for batch_idx, batch in enumerate(loader, start=1):
        context = batch["contexts"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        last_emb, last_mu, last_sigma = encode_context_batch(context, masks)
        next_idx = write_idx + context.shape[0]
        embeddings[write_idx:next_idx] = last_emb.detach().cpu().float()
        mu_all[write_idx:next_idx] = last_mu.detach().cpu().float()
        sigma_all[write_idx:next_idx] = last_sigma.detach().cpu().float()
        row_indices_all[write_idx:next_idx] = batch["row_indices"].cpu()
        win_starts_all[write_idx:next_idx] = batch["win_starts"].cpu()
        series_ids_all.extend(batch["series_ids"])
        write_idx = next_idx
        if batch_idx == 1 or batch_idx % 25 == 0 or write_idx == len(dataset):
            print(f"[backbone] batches={batch_idx} samples={write_idx}/{len(dataset)}", flush=True)

    torch.save(
        {
            "embeddings": embeddings,
            "mu": mu_all,
            "sigma": sigma_all,
            "row_indices": row_indices_all,
            "series_ids": series_ids_all,
            "col_ids": series_ids_all,
            "win_starts": win_starts_all,
            "context_len": int(context_len),
            "horizon": int(horizon),
            "stride": int(context_len),
            "dataset": meta.key,
            "frequency": meta.frequency,
            "timesfm_freq": FREQUENCY_TIMESFM[meta.frequency],
            "skipped_short": dataset.skipped_short,
            "split": "train_only",
        },
        cache_path,
    )
    print(f"[backbone] saved: {cache_path}", flush=True)


def choose_dataset(metas: list[TsfMeta], key: str | None) -> list[TsfMeta]:
    if not key:
        return metas
    selected = [meta for meta in metas if meta.key == key or meta.path.stem == key]
    if not selected:
        known = ", ".join(meta.key for meta in metas[:10])
        raise ValueError(f"unknown dataset={key}. examples: {known}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["raw", "prophet", "backbone", "all"], default="raw")
    parser.add_argument("--dataset", type=str, default="", help="dataset key, e.g. m3_monthly_dataset")
    parser.add_argument("--horizon", type=int, default=0, help="backbone mode only; 0 means all horizons")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0 등")
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--max_series", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--all_datasets",
        action="store_true",
        help="prophet/backbone/all을 전체 Monash 데이터셋에 실행할 때 명시적으로 지정",
    )
    args = parser.parse_args()

    logging.getLogger("cmdstanpy").disabled = True
    metas, excluded = selected_metas()
    selected = choose_dataset(metas, args.dataset or None)
    if args.mode in ("prophet", "backbone", "all") and not args.dataset and not args.all_datasets:
        raise ValueError(
            "Monash 전체 Prophet/backbone 실행은 매우 큽니다. "
            "--dataset DATASET_KEY를 지정하거나 전체 실행 의도면 --all_datasets를 추가하세요."
        )
    if args.mode in ("raw", "all"):
        run_raw(selected, excluded, write_manifest=not args.dataset)
    if args.mode in ("prophet", "all"):
        for meta in selected:
            run_prophet(meta, args.n_workers, args.max_series, args.overwrite)
    if args.mode in ("backbone", "all"):
        horizons = [args.horizon] if args.horizon else HORIZONS
        for meta in selected:
            for horizon in horizons:
                run_backbone(meta, horizon, args.batch_size, args.num_workers, args.device)


if __name__ == "__main__":
    main()
