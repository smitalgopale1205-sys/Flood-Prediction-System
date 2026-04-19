"""
utils.py — Shared utility functions for the Flood Prediction System.
Used by both train_model.py (training) and app.py (inference).
"""

import numpy as np
import pandas as pd
from config import (
    NORM_RANGES, SCORE_WEIGHTS, LAND_COVER_RISK, SOIL_TYPE_RISK,
    RISK_THRESHOLD
)


def normalize(series_or_val, lo: float, hi: float):
    """
    Min-max normalize a value or pandas Series to [0, 1].
    Works for both single float (app inference) and Series (training).
    """
    if isinstance(series_or_val, pd.Series):
        return (series_or_val.clip(lo, hi) - lo) / (hi - lo)
    else:
        return max(0.0, min(1.0, (series_or_val - lo) / (hi - lo)))


def compute_flood_risk_score(data) -> float | pd.Series:
    """
    Compute the domain-knowledge flood risk score (0–100).

    Accepts either:
      - a dict of single values  →  returns a float  (used by app.py)
      - a pandas DataFrame       →  returns a Series (used by train_model.py)

    Score formula (weights from hydrology literature):
        score = (Σ weighted_normalized_features) × LC_multiplier × ST_multiplier
        scaled to 0–100, threshold at 50 → High Risk
    """
    is_df = isinstance(data, pd.DataFrame)

    def _get(key):
        return data[key] if is_df else data[key]

    raw = (
        normalize(_get("Rainfall"),          *NORM_RANGES["Rainfall"])          * SCORE_WEIGHTS["Rainfall"]
      + normalize(_get("Water_Level"),        *NORM_RANGES["Water_Level"])        * SCORE_WEIGHTS["Water_Level"]
      + normalize(_get("River_Discharge"),    *NORM_RANGES["River_Discharge"])    * SCORE_WEIGHTS["River_Discharge"]
      + normalize(_get("Humidity"),           *NORM_RANGES["Humidity"])           * SCORE_WEIGHTS["Humidity"]
      + (1 - normalize(_get("Elevation"),     *NORM_RANGES["Elevation"]))         * SCORE_WEIGHTS["Elevation"]
      + _get("Historical_Floods")                                                  * SCORE_WEIGHTS["Historical_Floods"]
      + normalize(_get("Population_Density"), *NORM_RANGES["Population_Density"]) * SCORE_WEIGHTS["Population_Density"]
      + (1 - _get("Infrastructure"))                                               * SCORE_WEIGHTS["Infrastructure"]
    )

    lc_mult = (_get("Land_Cover").map(LAND_COVER_RISK)
               if is_df else LAND_COVER_RISK.get(_get("Land_Cover"), 1.0))
    st_mult = (_get("Soil_Type").map(SOIL_TYPE_RISK)
               if is_df else SOIL_TYPE_RISK.get(_get("Soil_Type"), 1.0))

    score = raw * lc_mult * st_mult

    if is_df:
        return (score / score.max()) * 100
    else:
        # For single-value inference, use the same max scaling factor
        # (raw max ≈ 0.98 based on dataset analysis)
        return min(100.0, score * 100 / 0.98)


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add flood_risk_score and Flood_Risk_Binary columns to df.
    Called by train_model.py during dataset preparation.
    """
    df = df.copy()
    df["flood_risk_score"]  = compute_flood_risk_score(df)
    df["Flood_Risk_Binary"] = (df["flood_risk_score"] > RISK_THRESHOLD).astype(int)
    return df


def risk_level(prob: float) -> tuple[str, str]:
    """
    Map a flood probability to a (label, emoji) tuple.
    Used by app.py for result display.
    """
    if prob < 0.35:
        return "Low Risk",    "🟢"
    elif prob < 0.65:
        return "Medium Risk", "🟡"
    else:
        return "High Risk",   "🔴"


def prepare_input_row(
    rainfall, humidity, water_level, river_discharge,
    elevation, pop_density, hist_floods, lc_enc, st_enc
) -> np.ndarray:
    """
    Build a 2D numpy array for model.predict() / predict_proba().
    Feature order must match config.FEATURES exactly.
    Called by app.py.
    """
    return np.array([[
        rainfall, humidity, water_level, river_discharge,
        elevation, pop_density, hist_floods, lc_enc, st_enc
    ]])
