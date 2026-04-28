from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denominator = np.where(np.abs(np.asarray(y_true)) < 1e-8, 1e-8, np.abs(np.asarray(y_true)))
    mape = float(np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / denominator)))
    directional_accuracy = float(
        np.mean(
            np.sign(np.diff(np.asarray(y_true), prepend=np.asarray(y_true)[0]))
            == np.sign(np.diff(np.asarray(y_pred), prepend=np.asarray(y_pred)[0]))
        )
    )
    return {"rmse": rmse, "mae": mae, "mape": mape, "directional_accuracy": directional_accuracy}
