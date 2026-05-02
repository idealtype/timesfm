"""Prophet/TimesFM 준비 설정 - Monash TSF 묶음."""

DATASET_NAME = "Monash"

HORIZONS = [96, 192, 336, 720]
HORIZON = max(HORIZONS)

FREQUENCY_CONTEXT = {
    "weekly": 256,
    "monthly": 64,
    "quarterly": 64,
    "yearly": 64,
    "daily": 512,
    "hourly": 512,
    "half_hourly": 512,
    "10_minutes": 512,
    "4_seconds": 512,
    "minutely": 512,
}

FREQUENCY_PANDAS = {
    "weekly": "W",
    "monthly": "MS",
    "quarterly": "QS",
    "yearly": "YS",
    "daily": "D",
    "hourly": "h",
    "half_hourly": "30min",
    "10_minutes": "10min",
    "4_seconds": "4s",
    "minutely": "min",
}

FREQUENCY_TIMESFM = {
    "weekly": 1,
    "monthly": 1,
    "quarterly": 2,
    "yearly": 2,
    "daily": 0,
    "hourly": 0,
    "half_hourly": 0,
    "10_minutes": 0,
    "4_seconds": 0,
    "minutely": 0,
}

START_DATE_FALLBACK = "2000-01-01"

DEFAULT_PROPHET_CONFIG = {
    "growth": "linear",
    "n_changepoints": 5,
    "changepoint_range": 1.0,
    "changepoint_prior_scale": 0.05,
    "seasonality_mode": "additive",
    "seasonality_prior_scale": 10.0,
    "yearly_seasonality": "auto",
    "weekly_seasonality": "auto",
    "daily_seasonality": "auto",
    "custom_seasonalities": [],
    "uncertainty_samples": 0,
}
