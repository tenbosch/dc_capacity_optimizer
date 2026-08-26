"""Forecasting, outlier detection, and DC capacity scoring."""
import numpy as np


def linear_forecast(series, horizon=4):
    """Simple linear regression forecast with confidence intervals."""
    y = series.dropna().values.astype(float)
    if len(y) < 3:
        return None, None, None
    x = np.arange(len(y))
    # Fit linear model: y = mx + b
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]
    # Forecast future points
    future_x = np.arange(len(y), len(y) + horizon)
    forecast = slope * future_x + intercept
    # Confidence interval (based on residual std)
    fitted = slope * x + intercept
    residual_std = np.std(y - fitted)
    ci_upper = forecast + 1.96 * residual_std
    ci_lower = forecast - 1.96 * residual_std
    return forecast, ci_lower, ci_upper


def exponential_smoothing(series, alpha=0.3, horizon=4):
    """Simple exponential smoothing forecast."""
    y = series.dropna().values.astype(float)
    if len(y) < 3:
        return None, None, None
    # Compute smoothed values
    smoothed = np.zeros(len(y))
    smoothed[0] = y[0]
    for i in range(1, len(y)):
        smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i - 1]
    # Forecast is the last smoothed value projected forward
    last_smooth = smoothed[-1]
    forecast = np.full(horizon, last_smooth)
    # Confidence grows with horizon
    residual_std = np.std(y - smoothed)
    ci_factors = np.array([1.96 * residual_std * np.sqrt(h + 1) for h in range(horizon)])
    ci_upper = forecast + ci_factors
    ci_lower = forecast - ci_factors
    return forecast, ci_lower, ci_upper


def detect_outliers_iqr(series, multiplier=1.5):
    """Detect outliers using IQR method. Returns boolean mask."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper), lower, upper, q1, q3, iqr


def compute_capacity_score(row):
    """Compute a 0-100 capacity efficiency score for a DC."""
    score = 100
    if row.get("overstock_pct", 0) > 30:
        score -= 25
    elif row.get("overstock_pct", 0) > 15:
        score -= 10
    if row.get("dead_stock_pct", 0) > 10:
        score -= 20
    elif row.get("dead_stock_pct", 0) > 5:
        score -= 10
    if row.get("median_doh", 0) > 60:
        score -= 15
    elif row.get("median_doh", 0) > 30:
        score -= 5
    return max(0, min(100, score))
