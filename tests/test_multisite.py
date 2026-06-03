"""
Tests de l'expérience multi-sites (EAGLE-I + LOSO).

Couvre les transformations DÉTERMINISTES (aucun réseau, aucun brut EAGLE-I
requis) :
  - ingest_eaglei.to_hourly : 15 min → horaire (max), réindex année, cible p90
  - ingest_eaglei._normalize_fips : codes comté sur 5 chiffres
  - multisite_experiment.build_exog_features : 29 features exogènes, sans NaN
  - multisite_experiment.run_loso : leave-one-site-out sur jeu synthétique

Lancer : python -m pytest tests/test_multisite.py -q
"""

import numpy as np
import pandas as pd

from src.data.ingest_eaglei import _normalize_fips, _resolve_eaglei_columns, to_hourly
from src.models.multisite_experiment import (
    EXOG_FEATURES,
    build_exog_features,
    run_loso,
)
from src.utils.hospitals import (
    HOSPITAL_DISPLAY,
    TARGET_SOURCE_META,
    get_target_source,
    hospital_label,
)


def test_target_source_taxonomy():
    # Lacor = seule vérité terrain.
    assert get_target_source("lacor_uganda") == "real"
    # NHS ERIC et NYC LL84 = étiquettes synthétiques.
    assert get_target_source("st_thomas_nhs") == "synthetic"
    assert get_target_source("nyc_bellevue") == "county_network"
    # africa_grid = profil cloné, aucune étiquette.
    assert get_target_source("kenyatta_kenya") == "cloned"


def test_target_source_explicit_override():
    # Un override explicite dans l'info prime sur la dérivation par data_source.
    info = {"data_source": "africa_grid", "target_source": "real"}
    assert get_target_source("kenyatta_kenya", info) == "real"


def test_every_hospital_has_known_target_source():
    # Tout site du catalogue a une provenance reconnue (pas de trou).
    for key, info in HOSPITAL_DISPLAY.items():
        assert get_target_source(key, info) in TARGET_SOURCE_META


def test_every_hospital_has_flag_in_label():
    for info in HOSPITAL_DISPLAY.values():
        flag = (info.get("flag") or "").strip()
        assert flag, info.get("name")
        assert flag in hospital_label(info)


def test_normalize_fips_pads_to_five():
    s = pd.Series(["4013", "4013.0", 4013, "06037"])
    out = _normalize_fips(s)
    assert list(out) == ["04013", "04013", "04013", "06037"]


def _toy_eaglei_snapshots(year: int = 2023, seed: int = 0) -> pd.DataFrame:
    """Snapshots 15 min sur ~2 jours avec un pic de coupure marqué."""
    rng = np.random.RandomState(seed)
    ts = pd.date_range(f"{year}-06-01", periods=4 * 48, freq="15min")
    base = rng.randint(0, 50, size=len(ts))
    # Pic de coupure sur une fenêtre → garantit des positifs au-dessus du p90.
    base[40:60] += 5000
    return pd.DataFrame({"run_start_time": ts, "customers_out": base})


def test_resolve_eaglei_columns_accepts_sum_alias():
    time_col, out_col = _resolve_eaglei_columns(
        pd.Index(["fips_code", "county", "state", "sum", "run_start_time"])
    )
    assert time_col == "run_start_time"
    assert out_col == "sum"


def test_to_hourly_accepts_sum_column():
    raw = _toy_eaglei_snapshots().rename(columns={"customers_out": "sum"})
    out = to_hourly(raw, fips="04013", year=2023, max_customers=100_000, quantile=0.90)
    assert len(out) == 8760
    assert out["customers_out"].max() >= 5000


def test_to_hourly_resamples_and_fills_year():
    raw = _toy_eaglei_snapshots()
    out = to_hourly(raw, fips="04013", year=2023, max_customers=100_000, quantile=0.90)

    # Année complète réindexée à l'heure.
    assert len(out) == 8760
    assert {"datetime", "customers_out", "customers_out_frac", "is_outage", "fips"} <= set(out.columns)
    # Agrégation par max : la valeur horaire ≥ la moyenne des snapshots.
    assert out["customers_out"].max() >= 5000
    # Fraction bornée [0, 1].
    assert out["customers_out_frac"].between(0, 1).all()
    # Heures hors données → 0 coupure (pas de NaN).
    assert not out["customers_out"].isna().any()


def test_to_hourly_target_is_sparse():
    """La binarisation au p90 ne marque qu'une petite fraction d'heures."""
    raw = _toy_eaglei_snapshots()
    out = to_hourly(raw, fips="04013", year=2023, max_customers=100_000, quantile=0.90)
    rate = out["is_outage"].mean()
    # Très majoritairement des zéros (peu d'heures avec données dans ce jouet).
    assert 0.0 < rate < 0.05


def _toy_site_frame(site: str, year: int, n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dt = pd.date_range(f"{year}-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "datetime": dt,
        "is_outage": rng.binomial(1, 0.1, size=n),
        "temperature_2m": rng.uniform(10, 35, size=n),
        "relative_humidity_2m": rng.uniform(20, 95, size=n),
        "wind_speed_10m": rng.uniform(0, 40, size=n),
        "precipitation": rng.exponential(0.5, size=n),
        "surface_pressure": rng.uniform(990, 1030, size=n),
        "shortwave_radiation": rng.uniform(0, 800, size=n),
    })
    feats = build_exog_features(df)
    feats["site"] = site
    return feats[["site", "datetime", "is_outage", *EXOG_FEATURES]]


def test_build_exog_features_has_29_columns_no_nan():
    df = _toy_site_frame("A", 2023)
    # Les 29 features sont présentes et numériques, sans NaN.
    assert len(EXOG_FEATURES) == 29
    assert set(EXOG_FEATURES) <= set(df.columns)
    assert not df[EXOG_FEATURES].isna().any().any()
    # Missing weather columns (cloud_cover, cape…) sont créées à 0.
    assert (df["weathercode"] == 0).all()


def test_run_loso_leaves_one_site_out():
    data = pd.concat([
        _toy_site_frame("A", 2023, seed=1),
        _toy_site_frame("B", 2023, seed=2),
        _toy_site_frame("C", 2023, seed=3),
    ], ignore_index=True)

    rows = run_loso(data)
    assert len(rows) == 3
    sites = {r["site"] for r in rows}
    assert sites == {"A", "B", "C"}
    for r in rows:
        assert r["n_test"] == 500
        # Toutes les métriques attendues sont présentes et finies.
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier"):
            assert key in r
        assert 0.0 <= r["accuracy"] <= 1.0
