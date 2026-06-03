"""Tests du score de risque réseau + météo (heuristique)."""

import pandas as pd

from src.grid_outage_risk import assess_outage_risk


def _synthetic_context() -> dict:
    grid = pd.DataFrame({
        "datetime": pd.date_range("2022-06-01", periods=24, freq="h"),
        "em_total_load_mw": [100.0] * 20 + [180.0, 185.0, 190.0, 195.0],
        "em_carbon_intensity_gco2_kwh": [400.0] * 23 + [650.0],
        "em_fossil_pct": [70.0] * 24,
    })
    wx = pd.DataFrame({
        "datetime": [pd.Timestamp("2022-06-02 12:00:00")],
        "temperature_2m": [34.0],
        "precipitation": [5.0],
        "wind_speed_10m": [30.0],
        "cape": [800.0],
    })
    return {
        "zone_meta": {"zone": "UG", "source": "api", "country_name": "Uganda"},
        "grid": grid,
        "weather": wx,
        "fetched_at": "2022-06-02T12:00:00Z",
    }


def test_assess_outage_risk_high_stress():
    out = assess_outage_risk(_synthetic_context(), hospital_key="lacor_uganda")
    assert out["score"] >= 0.4
    assert out["level"] in ("MOYEN", "ÉLEVÉ")
    assert len(out["factors"]) == 4
    assert out["zone"] == "UG"


def test_assess_outage_risk_low_stress():
    grid = pd.DataFrame({
        "datetime": pd.date_range("2022-06-01", periods=12, freq="h"),
        "em_total_load_mw": [50.0] * 12,
        "em_carbon_intensity_gco2_kwh": [300.0] * 12,
        "em_renewable_pct": [60.0] * 12,
    })
    wx = pd.DataFrame({
        "datetime": [pd.Timestamp("2022-06-01 12:00:00")],
        "temperature_2m": [22.0],
        "precipitation": [0.0],
        "wind_speed_10m": [5.0],
    })
    ctx = {"zone_meta": {"zone": "GB"}, "grid": grid, "weather": wx}
    out = assess_outage_risk(ctx, hospital_key="st_thomas_nhs")
    assert out["score"] < 0.5
    assert out["level"] in ("FAIBLE", "MOYEN")
