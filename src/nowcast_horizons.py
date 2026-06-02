"""
Prévision « coupure dans les H prochaines heures » avec les modèles Lacor.

Priorité :
  1. Modèles horizons entraînés (`models/nowcast_horizons/`) — cible explicite
     « coupure dans les H h », mêmes features que le nowcast ;
  2. Repli : modèle nowcast horaire + agrégation 1−∏(1−p) sur les heures futures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.features.build_features import apply_feature_engineering_single

logger = logging.getLogger(__name__)

HORIZONS = (1, 3, 6)
ROOT = Path(__file__).resolve().parents[1]


def union_outage_probability(probas: np.ndarray) -> float:
    """P(au moins une coupure) à partir des probabilités horaires marginales."""
    p = np.clip(np.asarray(probas, dtype=float), 0.0, 1.0)
    if p.size == 0:
        return 0.0
    return float(1.0 - np.prod(1.0 - p))


def predict_proba_bundle(bundle: dict, row: pd.DataFrame) -> float:
    """Probabilité calibrée pour une ligne et un bundle horizon_model.joblib."""
    feats = bundle.get("features") or []
    mdl = bundle.get("model")
    cal = bundle.get("calibrator")
    if mdl is None or not feats or row.empty:
        return 0.0
    X = row.reindex(columns=feats)
    for col in feats:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0.0)
    p = float(mdl.predict_proba(X)[:, 1][0])
    if cal is not None:
        try:
            p = float(np.asarray(cal.predict([p])).ravel()[0])
        except Exception:  # noqa: BLE001
            pass
    return max(0.0, min(1.0, p))


def _row_at_ref(df: pd.DataFrame, ref_ts: pd.Timestamp) -> pd.DataFrame:
    dt = pd.to_datetime(df["datetime"])
    ref_ts = pd.Timestamp(ref_ts)
    exact = df.loc[dt == ref_ts]
    if not exact.empty:
        return exact.tail(1)
    return df.loc[dt <= ref_ts].tail(1)


def predict_horizons_at_ref(
    df: pd.DataFrame,
    ref_ts: pd.Timestamp,
    horizon_models: dict[int, dict],
) -> dict[int, float]:
    """Prédit P(coupure dans les H h) à partir de l'état à `ref_ts`."""
    row = _row_at_ref(df, ref_ts)
    if row.empty:
        return {}
    out: dict[int, float] = {}
    for h in sorted(horizon_models):
        bundle = horizon_models[h]
        out[h] = predict_proba_bundle(bundle, row)
    return out


def predict_horizons_from_dataset(
    df: pd.DataFrame,
    ref_ts: pd.Timestamp,
    feature_cols: list[str],
    predict_proba: Callable[[pd.DataFrame], np.ndarray],
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[int, float]:
    """Repli : agrège le nowcast horaire sur les heures futures."""
    dt = pd.to_datetime(df["datetime"])
    ref_ts = pd.Timestamp(ref_ts)
    out: dict[int, float] = {}
    for h in horizons:
        mask = (dt > ref_ts) & (dt <= ref_ts + pd.Timedelta(hours=h))
        chunk = df.loc[mask]
        if chunk.empty:
            continue
        X = chunk.reindex(columns=feature_cols)
        for col in feature_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.fillna(0.0)
        out[h] = union_outage_probability(predict_proba(X))
    return dict(sorted(out.items()))


def predict_horizons(
    df: pd.DataFrame,
    ref_ts: pd.Timestamp,
    feature_cols: list[str],
    predict_proba: Callable[[pd.DataFrame], np.ndarray],
    horizon_models: dict[int, dict] | None = None,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[int, float]:
    """Point d'entrée : modèles horizons si présents, sinon repli nowcast."""
    if horizon_models:
        probs = predict_horizons_at_ref(df, ref_ts, horizon_models)
        if probs:
            return probs
    return predict_horizons_from_dataset(
        df, ref_ts, feature_cols, predict_proba, horizons=horizons,
    )


def _meteo_forecast_path(hospital_key: str) -> Path:
    return ROOT / "data" / "raw" / f"meteo_forecast_{hospital_key}.csv"


def _load_meteo_forecast(hospital_key: str) -> pd.DataFrame | None:
    path = _meteo_forecast_path(hospital_key)
    if not path.exists():
        return None
    try:
        fc = pd.read_csv(path)
        fc["datetime"] = pd.to_datetime(fc["datetime"])
        return fc.sort_values("datetime")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prévision météo illisible pour %s : %s", hospital_key, exc)
        return None


def extend_raw_window(
    raw: pd.DataFrame,
    hospital_key: str,
    max_hours: int,
) -> pd.DataFrame:
    """Extension pour le repli marginal (météo prévue + conso plate)."""
    raw = raw.copy()
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    ref_ts = raw["datetime"].max()
    last = raw.iloc[-1]
    fc = _load_meteo_forecast(hospital_key)
    meteo_cols = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "wind_gusts_10m", "precipitation", "surface_pressure",
        "shortwave_radiation", "weathercode",
    ]
    rows = []
    for h in range(1, max_hours + 1):
        ts = ref_ts + pd.Timedelta(hours=h)
        row = {c: last.get(c, 0.0) for c in raw.columns if c != "datetime"}
        row["datetime"] = ts
        if fc is not None and not fc.empty:
            fdt = pd.to_datetime(fc["datetime"])
            idx = (fdt - ts).abs().idxmin()
            near = fc.loc[idx]
            for c in meteo_cols:
                if c in near.index and pd.notna(near[c]):
                    row[c] = near[c]
        rows.append(row)
    if not rows:
        return raw
    extra = pd.DataFrame(rows)
    out = pd.concat([raw, extra], ignore_index=True)
    out["is_outage"] = 0
    out.loc[out["datetime"] > ref_ts, "is_outage"] = 0
    return out


def predict_horizons_realtime(
    raw_window: pd.DataFrame,
    hospital_key: str,
    feature_cols: list[str],
    predict_proba: Callable[[pd.DataFrame], np.ndarray],
    horizon_models: dict[int, dict] | None = None,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[int, float]:
    """Temps réel : fenêtre passée feature-engineered ; horizons ou repli."""
    raw = raw_window.copy()
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    if "is_outage" not in raw.columns:
        raw["is_outage"] = 0
    feats = apply_feature_engineering_single(raw)
    ref_ts = pd.to_datetime(raw["datetime"]).max()

    if horizon_models:
        probs = predict_horizons_at_ref(feats, ref_ts, horizon_models)
        if probs:
            return probs

    max_h = max(horizons) if horizons else 6
    extended = extend_raw_window(raw, hospital_key, max_h)
    extended["is_outage"] = 0
    feats_ext = apply_feature_engineering_single(extended)
    return predict_horizons_from_dataset(
        feats_ext, ref_ts, feature_cols, predict_proba, horizons=horizons,
    )
