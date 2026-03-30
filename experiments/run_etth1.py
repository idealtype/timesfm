"""ETTh1 데이터셋으로 TimesFM 2.5 예측 (covariate 포함 vs 미포함 비교)."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.configs import ForecastConfig


def main():
    # --- 1. ETTh1 데이터 로드 ---
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    df = pd.read_csv(url)
    print(f"데이터 shape: {df.shape}")
    print(f"컬럼: {df.columns.tolist()}")
    # 타겟: OT (Oil Temperature)
    # 외인변수: HUFL, HULL, MUFL, MULL, LUFL, LULL (전력 부하 관련)

    horizon = 96
    context_len = 512
    covariate_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

    # 마지막 (context_len + horizon) 구간 사용
    total_len = context_len + horizon
    segment = df.iloc[-total_len:]

    target = segment["OT"].values
    context = target[:context_len]
    future = target[context_len:]

    # --- 2. 모델 로드 ---
    print("모델 로딩 중...")
    model = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

    # (A) covariate 없이 예측
    fc_no_cov = ForecastConfig(
        max_context=512,
        max_horizon=128,
        per_core_batch_size=1,
        force_flip_invariance=True,
        infer_is_positive=False,
        use_continuous_quantile_head=True,
        fix_quantile_crossing=True,
    )
    model.compile(fc_no_cov)
    print("예측 중 (covariate 없이)...")
    pred_no_cov, _ = model.forecast(horizon, [context])
    pred_no_cov = pred_no_cov[0]

    # (B) covariate 포함 예측
    fc_cov = ForecastConfig(
        max_context=512,
        max_horizon=128,
        per_core_batch_size=1,
        force_flip_invariance=True,
        infer_is_positive=False,
        use_continuous_quantile_head=True,
        fix_quantile_crossing=True,
        return_backcast=True,  # covariate 모드에 필요
    )
    model.compile(fc_cov)

    # dynamic_numerical_covariates: 각 covariate는 [context + horizon] 길이
    dynamic_covs = {}
    for col in covariate_cols:
        full_series = segment[col].values.tolist()  # context_len + horizon 길이
        dynamic_covs[col] = [full_series]

    # (B) xreg → TimesFM
    print("예측 중 (xreg → TimesFM)...")
    result_xreg_first = model.forecast_with_covariates(
        inputs=[context.tolist()],
        dynamic_numerical_covariates=dynamic_covs,
        xreg_mode="xreg + timesfm",
    )
    pred_xreg_first = result_xreg_first[0][0][-horizon:]

    # (C) TimesFM → xreg
    print("예측 중 (TimesFM → xreg)...")
    result_tsfm_first = model.forecast_with_covariates(
        inputs=[context.tolist()],
        dynamic_numerical_covariates=dynamic_covs,
        xreg_mode="timesfm + xreg",
    )
    pred_tsfm_first = result_tsfm_first[0][0][-horizon:]

    # --- 3. 평가 ---
    def mae(actual, predicted):
        return np.mean(np.abs(actual - predicted))

    def mse(actual, predicted):
        return np.mean((actual - predicted) ** 2)

    print(f"\n===== 결과 (horizon={horizon}) =====")
    print(f"{'':20s} {'MAE':>10s} {'MSE':>10s}")
    print(f"{'No covariate':20s} {mae(future, pred_no_cov):10.3f} {mse(future, pred_no_cov):10.3f}")
    print(f"{'xreg → TimesFM':20s} {mae(future, pred_xreg_first):10.3f} {mse(future, pred_xreg_first):10.3f}")
    print(f"{'TimesFM → xreg':20s} {mae(future, pred_tsfm_first):10.3f} {mse(future, pred_tsfm_first):10.3f}")

    # --- 4. 시각화 ---
    fig, ax = plt.subplots(figsize=(14, 5))
    t_ctx = np.arange(context_len)
    t_fut = np.arange(context_len, context_len + horizon)

    ax.plot(t_ctx[-200:], context[-200:], label="Context", color="blue")
    ax.plot(t_fut, future, label="Ground Truth", color="green", linestyle="--", linewidth=2)
    ax.plot(t_fut, pred_no_cov, label="No covariate", color="red", alpha=0.8)
    ax.plot(t_fut, pred_xreg_first, label="xreg → TimesFM", color="orange", linewidth=2)
    ax.plot(t_fut, pred_tsfm_first, label="TimesFM → xreg", color="purple", linewidth=2)

    ax.set_xlabel("Time Step (hourly)")
    ax.set_ylabel("Oil Temperature (OT)")
    ax.set_title("ETTh1: TimesFM 2.5 — No cov vs xreg→TimesFM vs TimesFM→xreg")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "etth1_results.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n시각화 저장: {save_path}")


if __name__ == "__main__":
    main()
