"""Cibles « coupure dans les H prochaines heures » (partagées entraînement / tests)."""

from __future__ import annotations

import pandas as pd


def build_horizon_target(is_outage: pd.Series, horizon: int) -> pd.Series:
    """1 si au moins une coupure survient dans les `horizon` heures suivantes."""
    s = is_outage.astype(int)
    fwd = pd.concat([s.shift(-k) for k in range(1, horizon + 1)], axis=1)
    return (fwd.max(axis=1) >= 1).astype("float")
