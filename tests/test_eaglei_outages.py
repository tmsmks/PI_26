"""Fusion EAGLE-I → profils horaires NYC."""

import numpy as np
import pandas as pd

from src.data.eaglei_outages import (
    OUTAGE_SOURCE_EAGLEI,
    OUTAGE_SOURCE_SYNTHETIC,
    align_eaglei_calendar,
    apply_eaglei_outages,
)
from src.data.ingest_eaglei import output_path
from src.utils.hospitals import get_target_source


def test_align_eaglei_calendar_shifts_year():
    eaglei = pd.DataFrame({
        "datetime": pd.date_range("2023-06-01", periods=24, freq="h"),
        "is_outage": [0, 1] * 12,
    })
    out = align_eaglei_calendar(eaglei, 2022)
    assert out["datetime"].dt.year.unique().tolist() == [2022]


def test_apply_eaglei_replaces_outage(tmp_path, monkeypatch):
    import src.data.eaglei_outages as mod
    import src.data.ingest_eaglei as ing

    county = "kings_ny"
    path = tmp_path / f"eaglei_{county}.csv"
    eaglei = pd.DataFrame({
        "datetime": pd.date_range("2022-01-01", periods=48, freq="h"),
        "is_outage": [1] * 24 + [0] * 24,
        "customers_out_frac": np.linspace(0, 0.2, 48),
    })
    eaglei.to_csv(path, index=False)

    monkeypatch.setattr(ing, "RAW_DIR", tmp_path)
    monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
    monkeypatch.setattr(mod, "output_path", lambda k: tmp_path / f"eaglei_{k}.csv")

    hourly = pd.DataFrame({
        "datetime": pd.date_range("2022-01-01", periods=48, freq="h"),
        "total_load_kw": 100.0,
        "is_outage": 0,
        "grid_available": 1,
        "generators_kw": 0.0,
    })
    out, source = apply_eaglei_outages(hourly, county)
    assert source == OUTAGE_SOURCE_EAGLEI
    assert out["is_outage"].sum() == 24
    assert out["outage_source"].iloc[0] == OUTAGE_SOURCE_EAGLEI
    assert "county_customers_out_frac" in out.columns


def test_apply_eaglei_fallback_without_file(tmp_path, monkeypatch):
    import src.data.eaglei_outages as mod
    import src.data.ingest_eaglei as ing

    monkeypatch.setattr(ing, "RAW_DIR", tmp_path)
    monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
    monkeypatch.setattr(mod, "output_path", lambda k: tmp_path / f"eaglei_{k}.csv")

    hourly = pd.DataFrame({
        "datetime": pd.date_range("2022-01-01", periods=10, freq="h"),
        "total_load_kw": 50.0,
        "is_outage": 1,
    })
    out, source = apply_eaglei_outages(hourly, "missing_county")
    assert source == OUTAGE_SOURCE_SYNTHETIC
    assert out["is_outage"].sum() == 1


def test_nyc_target_source_is_county_network():
    assert get_target_source("nyc_bellevue") == "county_network"
