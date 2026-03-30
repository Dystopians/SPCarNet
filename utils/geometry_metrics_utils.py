import math

import numpy as np


def depth_metrics(pred: np.ndarray, gt: np.ndarray):
    diff = pred - gt
    abs_diff = np.abs(diff)
    sq_diff = diff * diff
    abs_rel = abs_diff / np.clip(gt, 1e-8, None)
    rmse = math.sqrt(float(np.mean(sq_diff)))
    ratio = np.maximum(pred / np.clip(gt, 1e-8, None), gt / np.clip(pred, 1e-8, None))
    d1 = float(np.mean(ratio < 1.25))
    d2 = float(np.mean(ratio < (1.25**2)))
    d3 = float(np.mean(ratio < (1.25**3)))
    return {
        "count": int(pred.shape[0]),
        "mae": float(np.mean(abs_diff)),
        "rmse": float(rmse),
        "abs_rel": float(np.mean(abs_rel)),
        "median_abs_rel": float(np.median(abs_rel)),
        "delta_1.25": d1,
        "delta_1.25^2": d2,
        "delta_1.25^3": d3,
    }


def normal_metrics_from_abs_cos(cos_abs: np.ndarray):
    cos_abs = np.clip(cos_abs, 0.0, 1.0)
    ang = np.degrees(np.arccos(cos_abs))
    return {
        "count": int(cos_abs.shape[0]),
        "mean_abs_cos": float(np.mean(cos_abs)),
        "mean_ang_deg": float(np.mean(ang)),
        "median_ang_deg": float(np.median(ang)),
        "pct_ang_lt_11.25": float(np.mean(ang < 11.25)),
        "pct_ang_lt_22.5": float(np.mean(ang < 22.5)),
        "pct_ang_lt_30": float(np.mean(ang < 30.0)),
    }
