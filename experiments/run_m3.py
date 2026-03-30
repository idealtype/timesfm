"""M3 Monthly 데이터셋으로 TimesFM 2.5 예측 실행."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasetsforecast.m3 import M3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.configs import ForecastConfig


def main():
    # --- 1. M3 Monthly 데이터 로드 ---
    df, *_ = M3.load(".", "Monthly")
    horizon = 18  # M3 Monthly의 공식 예측 길이

    # 시리즈별로 분리
    series_ids = df["unique_id"].unique()
    print(f"총 시리즈 수: {len(series_ids)}, 예측 horizon: {horizon}")

    # context(학습용)와 future(평가용) 분리
    contexts = []
    futures = []
    for sid in series_ids:
        vals = df[df["unique_id"] == sid]["y"].values
        contexts.append(vals[:-horizon])
        futures.append(vals[-horizon:])

    # --- 2. 모델 로드 ---
    print("모델 로딩 중...")
    model = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

    fc = ForecastConfig(
        max_context=512,
        max_horizon=128,
        per_core_batch_size=64,
        force_flip_invariance=True,
        infer_is_positive=True,
        use_continuous_quantile_head=True,
        fix_quantile_crossing=True,
    )
    model.compile(fc)

    # --- 3. 예측 ---
    print("예측 중...")
    point_forecasts, quantile_forecasts = model.forecast(horizon, contexts)

    # --- 4. 평가 (sMAPE) ---
    def smape(actual, predicted):
        return 100 * np.mean(
            2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-8)
        )

    smapes = []
    for i in range(len(futures)):
        s = smape(futures[i], point_forecasts[i])
        smapes.append(s)

    print(f"\n===== 결과 =====")
    print(f"평균 sMAPE: {np.mean(smapes):.2f}%")
    print(f"중앙값 sMAPE: {np.median(smapes):.2f}%")

    # --- 5. 시각화 (샘플 4개) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    sample_ids = np.random.choice(len(series_ids), 4, replace=False)

    for ax, idx in zip(axes.flat, sample_ids):
        ctx = contexts[idx]
        fut = futures[idx]
        pred = point_forecasts[idx]

        t_ctx = np.arange(len(ctx))
        t_fut = np.arange(len(ctx), len(ctx) + horizon)

        ax.plot(t_ctx[-60:], ctx[-60:], label="Context", color="blue")
        ax.plot(t_fut, fut, label="Ground Truth", color="green", linestyle="--")
        ax.plot(t_fut, pred, label="TimesFM", color="red")
        ax.set_title(f"{series_ids[idx]} (sMAPE={smapes[idx]:.1f}%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"TimesFM 2.5 on M3 Monthly | Mean sMAPE: {np.mean(smapes):.2f}%")
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, "m3_results.png")
    plt.savefig(save_path, dpi=150)
    print(f"시각화 저장: {save_path}")


if __name__ == "__main__":
    main()
