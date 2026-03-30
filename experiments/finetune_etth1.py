"""TimesFM + N-HiTS Head 파인튜닝: ETTh1으로 학습, ETTh2로 평가."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.nhits_head import TimesFMWithNHiTSHead
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.configs import ForecastConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

CONTEXT_LEN = 512
HORIZONS = [96, 192, 336, 720]
COVARIATE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
BATCH_SIZE = 32
EPOCHS = 30
LR_BACKBONE = 1e-5
LR_HEAD = 5e-4
UNFREEZE_LAST_N = 0
TRAIN_SPLIT = 0.8  # ETTh1 내 train/val 비율


# ---------------------------------------------------------------------------
# 데이터셋
# ---------------------------------------------------------------------------

class ETTDataset(Dataset):
    """슬라이딩 윈도우로 (context, covariates, future) 샘플 생성."""

    def __init__(self, df, context_len, horizon, covariate_cols, target_col="OT"):
        self.context_len = context_len
        self.horizon = horizon
        self.covariate_cols = covariate_cols

        self.target = df[target_col].values.astype(np.float32)
        self.covariates = {
            col: df[col].values.astype(np.float32) for col in covariate_cols
        }

        total_len = context_len + horizon
        self.n_samples = len(self.target) - total_len + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ctx_end = idx + self.context_len
        fut_end = ctx_end + self.horizon

        context = torch.tensor(self.target[idx:ctx_end])
        future = torch.tensor(self.target[ctx_end:fut_end])
        masks = torch.zeros(self.context_len, dtype=torch.bool)

        covs = {
            col: torch.tensor(self.covariates[col][idx:fut_end])
            for col in self.covariate_cols
        }

        return context, masks, covs, future


def collate_fn(batch):
    """커스텀 collate: covariate dict를 배치로 묶기."""
    contexts, masks, covs_list, futures = zip(*batch)
    contexts = torch.stack(contexts)
    masks = torch.stack(masks)
    futures = torch.stack(futures)

    cov_cols = covs_list[0].keys()
    covs = {col: torch.stack([c[col] for c in covs_list]) for col in cov_cols}

    return contexts, masks, covs, futures


def load_ett(name="ETTh1"):
    """ETT 데이터셋 다운로드."""
    url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
    return pd.read_csv(url)


# ---------------------------------------------------------------------------
# 학습/평가
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for context, masks, covs, future in loader:
        context = context.to(device)
        masks = masks.to(device)
        future = future.to(device)
        covs = {k: v.to(device) for k, v in covs.items()}

        pred = model(context, masks, covs)
        loss = torch.nn.functional.mse_loss(pred, future)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * context.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse = 0
    total_mae = 0
    n = 0
    all_preds = []
    all_futures = []

    for context, masks, covs, future in loader:
        context = context.to(device)
        masks = masks.to(device)
        future = future.to(device)
        covs = {k: v.to(device) for k, v in covs.items()}

        pred = model(context, masks, covs)
        total_mse += ((pred - future) ** 2).sum().item()
        total_mae += (pred - future).abs().sum().item()
        n += future.numel()

        all_preds.append(pred.cpu())
        all_futures.append(future.cpu())

    mse = total_mse / n
    mae = total_mae / n
    preds = torch.cat(all_preds)
    futures = torch.cat(all_futures)
    return mse, mae, preds, futures


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def evaluate_baseline(baseline_model, test_ds, horizon):
    """기존 TimesFM(AR)으로 ETTh2 테스트 데이터 예측."""
    all_preds = []
    all_futures = []

    # 배치 단위로 context를 모아서 baseline 예측
    contexts_np = []
    futures_np = []
    for i in range(len(test_ds)):
        context, _, _, future = test_ds[i]
        contexts_np.append(context.numpy())
        futures_np.append(future.numpy())

    # baseline forecast (배치)
    batch_size = 64
    for start in range(0, len(contexts_np), batch_size):
        batch_ctx = contexts_np[start:start + batch_size]
        point_forecast, _ = baseline_model.forecast(horizon, batch_ctx)
        all_preds.append(point_forecast)

    all_preds = np.concatenate(all_preds, axis=0)
    all_futures = np.array(futures_np)

    mse = np.mean((all_preds - all_futures) ** 2)
    mae = np.mean(np.abs(all_preds - all_futures))
    return mse, mae, torch.tensor(all_preds, dtype=torch.float32), torch.tensor(all_futures, dtype=torch.float32)


def plot_results(preds, futures, title, save_name, baseline_preds=None):
    """예측 vs Ground Truth vs Baseline 비교 그래프 (샘플 4개)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    n_samples = preds.shape[0]
    sample_ids = np.random.choice(n_samples, min(4, n_samples), replace=False)

    for ax, idx in zip(axes.flat, sample_ids):
        ax.plot(futures[idx].numpy(), label="Ground Truth", color="green", linestyle="--", linewidth=2)
        if baseline_preds is not None:
            ax.plot(baseline_preds[idx].numpy(), label="TimesFM (AR)", color="blue", alpha=0.7)
        ax.plot(preds[idx].numpy(), label="N-HiTS Head (Direct)", color="red")
        mse_i = ((preds[idx] - futures[idx]) ** 2).mean().item()
        ax.set_title(f"Sample {idx} (MSE={mse_i:.3f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, save_name)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"시각화 저장: {save_path}")


def plot_decomposition(model, test_ds, horizon, device, save_name):
    """N-HiTS 헤드의 성분 분해 시각화 (샘플 4개): GT, Total, Trend, Seasonal, Residual."""
    model.eval()
    n_samples = len(test_ds)
    sample_ids = np.random.choice(n_samples, min(4, n_samples), replace=False)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    for ax, idx in zip(axes, sample_ids):
        context, masks, covs, future = test_ds[idx]
        context = context.unsqueeze(0).to(device)
        masks = masks.unsqueeze(0).to(device)
        covs_batch = {k: v.unsqueeze(0).to(device) for k, v in covs.items()}
        future = future.numpy()

        with torch.no_grad():
            pred, decomp = model(context, masks, covs_batch, return_decomposition=True)

        pred_np = pred[0].cpu().numpy()
        trend_np = decomp["trend"][0].cpu().numpy()
        seasonal_np = decomp["seasonal"][0].cpu().numpy()
        residual_np = pred_np - trend_np - seasonal_np
        t = np.arange(horizon)

        ax.plot(t, future, label="Ground Truth", color="green", linestyle="--", linewidth=2)
        ax.plot(t, pred_np, label="Total Prediction", color="black", linewidth=1.5)
        ax.plot(t, trend_np, label="Trend", color="#e74c3c", linewidth=1.2)
        ax.plot(t, seasonal_np, label="Seasonal", color="#3498db", linewidth=1.2)
        ax.plot(t, residual_np, label="Residual", color="gray", alpha=0.4, linewidth=1)

        ax.set_title(f"Sample {idx}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"N-HiTS Decomposition (h={horizon})", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, save_name)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"분해 시각화 저장: {save_path}")


def plot_training_curve(train_losses, val_losses):
    """학습 곡선 그래프."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "training_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"학습 곡선 저장: {save_path}")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def run_single_horizon(horizon, df_h1, df_h2, device, baseline_model):
    """하나의 horizon에 대해 학습 + 평가 수행."""
    print(f"\n{'='*60}")
    print(f"  Horizon = {horizon}")
    print(f"{'='*60}")

    # --- 1. 데이터 준비 (랜덤 분할) ---
    full_ds = ETTDataset(df_h1, CONTEXT_LEN, horizon, COVARIATE_COLS)
    n_total = len(full_ds)
    indices = np.random.RandomState(42).permutation(n_total)
    split_idx = int(n_total * TRAIN_SPLIT)
    train_ds = Subset(full_ds, indices[:split_idx])
    val_ds = Subset(full_ds, indices[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    print(f"Train 샘플: {len(train_ds)}, Val 샘플: {len(val_ds)}")

    # --- 2. 모델 ---
    print("모델 로딩 중...")
    model = TimesFMWithNHiTSHead(
        horizon=horizon,
        context_len=CONTEXT_LEN,
        covariate_cols=COVARIATE_COLS,
        unfreeze_last_n=UNFREEZE_LAST_N,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"학습 파라미터: {trainable:,} / 전체: {total:,} ({100*trainable/total:.1f}%)")

    # --- 3. 학습 ---
    param_groups = model.get_param_groups(lr_backbone=LR_BACKBONE, lr_head=LR_HEAD)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_losses = []
    val_losses = []
    best_val_mse = float("inf")
    model_path = os.path.join(SCRIPT_DIR, f"best_model_h{horizon}.pt")

    print(f"\n학습 시작 (epochs={EPOCHS})...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mse, val_mae, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), model_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f}")

    print(f"\n학습 완료! Best Val MSE: {best_val_mse:.4f}")
    plot_training_curve(train_losses, val_losses)
    # 학습 곡선 rename
    os.rename(
        os.path.join(SCRIPT_DIR, "training_curve.png"),
        os.path.join(SCRIPT_DIR, f"training_curve_h{horizon}.png"),
    )

    # --- 4. ETTh2 평가 ---
    test_ds = ETTDataset(df_h2, CONTEXT_LEN, horizon, COVARIATE_COLS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # N-HiTS Head (Direct) 평가
    model.load_state_dict(torch.load(model_path, weights_only=True))
    test_mse, test_mae, test_preds, test_futures = evaluate(model, test_loader, device)

    # Baseline: 기존 TimesFM (AR) 평가
    print("  Baseline (TimesFM AR) 예측 중...")
    max_horizon = max(horizon, 128)  # TimesFM 최소 output_patch_len
    max_horizon = ((max_horizon - 1) // 128 + 1) * 128  # 128의 배수로
    baseline_model.compile(ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=max_horizon,
        per_core_batch_size=64,
        force_flip_invariance=True,
        infer_is_positive=False,
    ))
    bl_mse, bl_mae, bl_preds, _ = evaluate_baseline(baseline_model, test_ds, horizon)

    print(f"\n  ETTh2 테스트 결과 (h={horizon}):")
    print(f"    TimesFM (AR)         | MSE: {bl_mse:.4f} | MAE: {bl_mae:.4f}")
    print(f"    N-HiTS Head (Direct) | MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")

    plot_results(
        test_preds, test_futures,
        f"ETTh2 (h={horizon}) | AR MSE={bl_mse:.4f} vs Direct MSE={test_mse:.4f}",
        f"etth2_test_h{horizon}.png",
        baseline_preds=bl_preds,
    )

    # 성분 분해 시각화
    plot_decomposition(model, test_ds, horizon, device, f"etth2_decomp_h{horizon}.png")

    return {
        "horizon": horizon,
        "val_mse": best_val_mse,
        "test_mse": test_mse, "test_mae": test_mae,
        "bl_mse": bl_mse, "bl_mae": bl_mae,
    }


def plot_summary(results):
    """전체 horizon별 성능 비교 그래프 (AR vs Direct)."""
    horizons = [r["horizon"] for r in results]
    test_mses = [r["test_mse"] for r in results]
    test_maes = [r["test_mae"] for r in results]
    bl_mses = [r["bl_mse"] for r in results]
    bl_maes = [r["bl_mae"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(horizons, bl_mses, "s--", color="blue", linewidth=2, markersize=8, label="TimesFM (AR)")
    ax1.plot(horizons, test_mses, "o-", color="red", linewidth=2, markersize=8, label="N-HiTS Head (Direct)")
    ax1.set_xlabel("Horizon")
    ax1.set_ylabel("MSE")
    ax1.set_title("ETTh2 Test MSE")
    ax1.set_xticks(horizons)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(horizons, bl_maes, "s--", color="blue", linewidth=2, markersize=8, label="TimesFM (AR)")
    ax2.plot(horizons, test_maes, "o-", color="red", linewidth=2, markersize=8, label="N-HiTS Head (Direct)")
    ax2.set_xlabel("Horizon")
    ax2.set_ylabel("MAE")
    ax2.set_title("ETTh2 Test MAE")
    ax2.set_xticks(horizons)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("TimesFM AR vs N-HiTS Direct — Horizon 비교")
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "horizon_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n비교 그래프 저장: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("데이터 로딩 중...")
    df_h1 = load_ett("ETTh1")
    df_h2 = load_ett("ETTh2")

    # Baseline 모델 (기존 TimesFM AR) 한 번만 로드
    print("Baseline 모델 (TimesFM AR) 로딩 중...")
    baseline_model = TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", torch_compile=False,
    )

    results = []
    for horizon in HORIZONS:
        result = run_single_horizon(horizon, df_h1, df_h2, device, baseline_model)
        results.append(result)

    # --- 최종 비교표 ---
    print(f"\n{'='*60}")
    print("  최종 결과 (ETTh2 Test)")
    print(f"{'='*60}")
    print(f"  {'Horizon':>8s}  {'AR MSE':>10s}  {'AR MAE':>10s}  {'Direct MSE':>10s}  {'Direct MAE':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in results:
        print(f"  {r['horizon']:>8d}  {r['bl_mse']:>10.4f}  {r['bl_mae']:>10.4f}  {r['test_mse']:>10.4f}  {r['test_mae']:>10.4f}")

    plot_summary(results)


if __name__ == "__main__":
    main()
