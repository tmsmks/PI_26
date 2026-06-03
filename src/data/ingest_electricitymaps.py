"""
Ingestion Electricity Maps : charge réseau, intensité carbone et mix
de production en temps réel pour la zone électrique locale de chaque
hôpital.

──────────────────────────────────────────────────────────────────────
Pourquoi c'est utile pour la prédiction de coupures
──────────────────────────────────────────────────────────────────────
L'API Electricity Maps n'expose pas la consommation interne d'un
hôpital (il n'existe aucune API publique pour ça), mais elle expose
en temps réel l'état du **réseau électrique local** auquel l'hôpital
est physiquement raccordé. Ce contexte réseau est la cause racine
n° 1 des coupures externes :
    - charge réseau (total load, MW) ↗   → stress, risque de délestage
    - mix très fossile / peu d'inertie    → réseau plus instable
    - chute brutale d'une production      → fréquence du réseau ↘
    - intensité carbone qui bouge fort    → reconfiguration du réseau

Combiné aux signaux météo / catastrophes / pollution déjà en place,
ça donne au modèle une vision « état réseau » qui était jusque-là
absente.

──────────────────────────────────────────────────────────────────────
Endpoints exploités (API v4, voir src/utils/config.py)
──────────────────────────────────────────────────────────────────────
    GET /v4/zone                      → résolution lat/lon → zone
    GET /v4/total-load/latest         → charge totale en cours (MW)
    GET /v4/total-load/history        → 24 h horaires de charge
    GET /v4/carbon-intensity/latest   → intensité carbone (gCO2/kWh)
    GET /v4/carbon-intensity/history  → 24 h horaires d'intensité
    GET /v4/electricity-mix/history   → 24 h de mix de production
                                        (% renouvelable, % fossile…)

──────────────────────────────────────────────────────────────────────
Authentification
──────────────────────────────────────────────────────────────────────
Header `auth-token: <TOKEN>` requis sauf pour `/v4/zones` (liste).
Le token est lu dans la variable d'env `ELECTRICITY_MAPS_TOKEN`.
Si la variable n'est pas définie, l'ingestion est ignorée
silencieusement (le pipeline reste fonctionnel).

Inscription gratuite (academia / Home Assistant) :
    https://www.electricitymaps.com/free-tier-api

──────────────────────────────────────────────────────────────────────
Sortie
──────────────────────────────────────────────────────────────────────
Un CSV horaire par hôpital : `data/raw/electricitymaps_<hospital>.csv`
contenant les colonnes (toutes préfixées `em_` pour la fusion auto) :

    datetime, em_zone, em_total_load_mw, em_carbon_intensity_gco2_kwh,
    em_renewable_pct, em_fossil_pct, em_low_carbon_pct, hospital
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from src.utils.config import (
    ELECTRICITYMAPS_BASE,
    ELECTRICITYMAPS_TOKEN_ENV,
    HOSPITAL_ELECTRICITY_ZONES,
    HOSPITAL_LOCATIONS,
    RAW_DIR,
)
from src.utils.http import http_get
from src.utils.io import save_csv

logger = logging.getLogger(__name__)

# Endpoints utilisés (suffixes joints à ELECTRICITYMAPS_BASE)
EP_ZONE = "/v4/zone"
EP_TOTAL_LOAD_LATEST = "/v4/total-load/latest"
EP_TOTAL_LOAD_HISTORY = "/v4/total-load/history"
EP_TOTAL_LOAD_RANGE = "/v4/total-load/past-range"
EP_CARBON_LATEST = "/v4/carbon-intensity/latest"
EP_CARBON_HISTORY = "/v4/carbon-intensity/history"
EP_CARBON_RANGE = "/v4/carbon-intensity/past-range"
EP_MIX_HISTORY = "/v4/electricity-mix/history"
EP_MIX_RANGE = "/v4/electricity-mix/past-range"

DEFAULT_TIMEOUT = 30


# ──────────────────────────────────────────────────────────────────────
# Helpers HTTP
# ──────────────────────────────────────────────────────────────────────
def _get_token() -> str | None:
    """Récupère le token depuis l'env, retourne None si absent."""
    token = os.environ.get(ELECTRICITYMAPS_TOKEN_ENV)
    if not token:
        logger.warning(
            "Variable d'env %s absente — ingestion Electricity Maps ignorée.",
            ELECTRICITYMAPS_TOKEN_ENV,
        )
    return token


def _call(endpoint: str, token: str, params: dict[str, Any]) -> dict | None:
    """Appel GET unique vers l'API Electricity Maps.

    Retourne le dict JSON, ou None en cas d'erreur (4xx/5xx) après
    avoir loggé l'échec. Le token absent → erreur applicative.
    """
    url = ELECTRICITYMAPS_BASE + endpoint
    headers = {"auth-token": token}
    try:
        resp = http_get(url, headers=headers, params=params, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        logger.warning("Electricity Maps %s : erreur réseau (%s)", endpoint, exc)
        return None

    if resp.status_code == 401:
        logger.error("Electricity Maps : token invalide ou expiré (401).")
        return None
    if resp.status_code == 403:
        logger.warning(
            "Electricity Maps %s : zone hors plan (%s) — ignorée.",
            endpoint, params.get("zone"),
        )
        return None
    if not resp.ok:
        logger.warning(
            "Electricity Maps %s : HTTP %d (%s)",
            endpoint, resp.status_code, resp.text[:200],
        )
        return None

    try:
        return resp.json()
    except ValueError:
        logger.warning("Electricity Maps %s : réponse non-JSON.", endpoint)
        return None


# ──────────────────────────────────────────────────────────────────────
# Résolution de la zone réseau locale d'un hôpital
# ──────────────────────────────────────────────────────────────────────
def resolve_zone(
    hospital: str,
    token: str,
    lat: float | None = None,
    lon: float | None = None,
) -> str | None:
    """Zone EM la plus fine : `/v4/zone` au point GPS, repli mapping statique."""
    from src.utils.em_zone import resolve_zone_precise

    res = resolve_zone_precise(hospital, lat=lat, lon=lon, token=token)
    return res.get("zone")


# ──────────────────────────────────────────────────────────────────────
# Parsing : conversion d'une réponse "history" en DataFrame
# ──────────────────────────────────────────────────────────────────────
def _history_to_df(
    payload: dict,
    value_keys: list[str],
    rename: dict[str, str],
    *,
    fallback_scalar: str = "value",
) -> pd.DataFrame:
    """Convertit une réponse `/history` ou `/past-range` en DataFrame.

    Les payloads d'Electricity Maps suivent toujours la même forme :
        {"zone": "FR", "history": [{"datetime": "...", <champs>}, ...]}
    Depuis 2024+, la charge et le carbone arrivent souvent sous la clé
    générique `value` plutôt que `totalLoad` / `carbonIntensity`.
    """
    items = payload.get("history") or payload.get("data") or []
    if not items:
        return pd.DataFrame()

    out_col = next(iter(rename.values()))
    rows: list[dict] = []
    for entry in items:
        ts = entry.get("datetime")
        if not ts:
            continue
        val = None
        for k in value_keys:
            if k in entry and entry[k] is not None:
                val = entry[k]
                break
        if val is None and fallback_scalar in entry:
            val = entry.get(fallback_scalar)
        rows.append({"datetime": ts, out_col: val})

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    return df


# ──────────────────────────────────────────────────────────────────────
# Trois fetchers, un par signal
# ──────────────────────────────────────────────────────────────────────
def fetch_total_load_history(zone: str, token: str) -> pd.DataFrame:
    """24 h glissantes de charge réseau totale (MW)."""
    payload = _call(EP_TOTAL_LOAD_HISTORY, token, {"zone": zone})
    if not payload:
        return pd.DataFrame()
    return _history_to_df(
        payload,
        value_keys=["totalLoad", "total_load", "value"],
        rename={"totalLoad": "em_total_load_mw"},
    )


def fetch_carbon_intensity_history(zone: str, token: str) -> pd.DataFrame:
    """24 h glissantes d'intensité carbone (gCO2eq/kWh)."""
    payload = _call(EP_CARBON_HISTORY, token, {"zone": zone})
    if not payload:
        return pd.DataFrame()
    return _history_to_df(
        payload,
        value_keys=["carbonIntensity", "carbon_intensity", "value"],
        rename={"carbonIntensity": "em_carbon_intensity_gco2_kwh"},
    )


def fetch_electricity_mix_history(zone: str, token: str) -> pd.DataFrame:
    """24 h glissantes du mix de production (% renouvelable / fossile / bas-carbone)."""
    payload = _call(EP_MIX_HISTORY, token, {"zone": zone})
    if not payload:
        return pd.DataFrame()
    items = payload.get("history") or payload.get("data") or []
    if not items:
        return pd.DataFrame()

    # Le mix expose des dictionnaires nested ; on extrait juste les agrégats
    # haut niveau (renewablePercentage, fossilFreePercentage…). Si l'API
    # retourne plutôt les pourcentages détaillés, on les agrège côté client.
    rows = []
    for entry in items:
        ts = entry.get("datetime")
        if not ts:
            continue
        renew = entry.get("renewablePercentage")
        fossil = entry.get("fossilPercentage")
        low_carbon = entry.get("fossilFreePercentage") or entry.get("lowCarbonPercentage")

        # Fallback : reconstruire depuis le breakdown si les agrégats manquent
        breakdown = entry.get("powerProductionBreakdown") or {}
        if breakdown and (renew is None or fossil is None):
            total = sum(v for v in breakdown.values() if isinstance(v, (int, float)) and v > 0)
            if total:
                renewables = sum(
                    breakdown.get(k, 0) or 0
                    for k in ("solar", "wind", "hydro", "biomass", "geothermal")
                )
                fossils = sum(
                    breakdown.get(k, 0) or 0
                    for k in ("coal", "gas", "oil")
                )
                renew = renew if renew is not None else round(100 * renewables / total, 2)
                fossil = fossil if fossil is not None else round(100 * fossils / total, 2)

        rows.append({
            "datetime": ts,
            "em_renewable_pct": renew,
            "em_fossil_pct": fossil,
            "em_low_carbon_pct": low_carbon,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    return df


# ──────────────────────────────────────────────────────────────────────
# Endpoints "latest" pour debug / app temps réel
# ──────────────────────────────────────────────────────────────────────
def fetch_total_load_latest(zone: str, token: str) -> dict | None:
    return _call(EP_TOTAL_LOAD_LATEST, token, {"zone": zone})


def fetch_carbon_intensity_latest(zone: str, token: str) -> dict | None:
    return _call(EP_CARBON_LATEST, token, {"zone": zone})


# ──────────────────────────────────────────────────────────────────────
# Endpoints "past-range" pour mode TRAIN (historique sur une fenêtre)
# Disponibilité dépend du plan : fonctionnera pour les utilisateurs
# disposant d'un plan commercial. Sinon retourne 403 et on tombe en
# mode "history 24h" via run_live().
# ──────────────────────────────────────────────────────────────────────
def fetch_total_load_range(zone: str, token: str, start: datetime, end: datetime) -> pd.DataFrame:
    payload = _call(
        EP_TOTAL_LOAD_RANGE,
        token,
        {"zone": zone, "start": start.isoformat(), "end": end.isoformat()},
    )
    if not payload:
        return pd.DataFrame()
    return _history_to_df(
        payload,
        value_keys=["totalLoad", "total_load", "value"],
        rename={"totalLoad": "em_total_load_mw"},
    )


def fetch_carbon_intensity_range(zone: str, token: str, start: datetime, end: datetime) -> pd.DataFrame:
    payload = _call(
        EP_CARBON_RANGE,
        token,
        {"zone": zone, "start": start.isoformat(), "end": end.isoformat()},
    )
    if not payload:
        return pd.DataFrame()
    return _history_to_df(
        payload,
        value_keys=["carbonIntensity", "carbon_intensity", "value"],
        rename={"carbonIntensity": "em_carbon_intensity_gco2_kwh"},
    )


# ──────────────────────────────────────────────────────────────────────
# Orchestration : fusion des 3 signaux en un seul DataFrame par hôpital
# ──────────────────────────────────────────────────────────────────────
def _merge_signals(load_df: pd.DataFrame, carbon_df: pd.DataFrame, mix_df: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les 3 séries horaires sur la colonne `datetime`."""
    frames = [df for df in (load_df, carbon_df, mix_df) if not df.empty]
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for other in frames[1:]:
        merged = merged.merge(other, on="datetime", how="outer")
    return merged.sort_values("datetime").reset_index(drop=True)


def _ingest_one_hospital(hospital: str, coords: dict, token: str, mode: str) -> pd.DataFrame | None:
    """Pipeline complet pour un hôpital : zone → 3 signaux → merge."""
    zone = resolve_zone(hospital, token, coords.get("lat"), coords.get("lon"))
    if not zone:
        logger.warning("Pas de zone Electricity Maps pour %s — ignoré.", hospital)
        return None

    logger.info("Electricity Maps %-22s → zone=%s", hospital, zone)

    if mode == "live":
        load_df = fetch_total_load_history(zone, token)
        carbon_df = fetch_carbon_intensity_history(zone, token)
        mix_df = fetch_electricity_mix_history(zone, token)
    else:
        # mode "train" : on tente une fenêtre passée plus large
        end = datetime.now(timezone.utc).replace(microsecond=0)
        start = end - timedelta(days=10)
        load_df = fetch_total_load_range(zone, token, start, end)
        carbon_df = fetch_carbon_intensity_range(zone, token, start, end)
        mix_df = pd.DataFrame()  # past-range non couvert ici, on ignore

        # Repli si plan trop limité : on récupère au moins les 24 h
        if load_df.empty:
            load_df = fetch_total_load_history(zone, token)
        if carbon_df.empty:
            carbon_df = fetch_carbon_intensity_history(zone, token)
        if mix_df.empty:
            mix_df = fetch_electricity_mix_history(zone, token)

    merged = _merge_signals(load_df, carbon_df, mix_df)
    if merged.empty:
        logger.warning("Electricity Maps %s : aucune donnée récupérée.", hospital)
        return None

    merged["em_zone"] = zone
    merged["hospital"] = hospital
    return merged


def run_live(window_hours: int = 24) -> None:
    """Mode temps réel : récupère ~24 h glissantes pour chaque hôpital.

    `window_hours` est conservé pour cohérence avec les autres ingesters
    mais l'API Electricity Maps `/history` renvoie systématiquement les
    24 dernières heures.
    """
    token = _get_token()
    if not token:
        return

    from src.utils.em_zone import resolve_and_cache_zone

    for hospital, coords in HOSPITAL_LOCATIONS.items():
        resolve_and_cache_zone(hospital, token=token)
        df = _ingest_one_hospital(hospital, coords, token, mode="live")
        if df is None or df.empty:
            continue
        save_csv(df, RAW_DIR / f"electricitymaps_{hospital}.csv")

    logger.info("Ingestion Electricity Maps (live) terminée.")


def run(year: int | None = None) -> None:
    """Mode batch : utilisé par le pipeline d'entraînement.

    Avec un plan commercial, tente de récupérer une fenêtre passée
    de 10 jours via `/past-range`. Sinon, retombe sur la fenêtre 24 h
    de `/history` (utile pour tester sans plan payant).
    """
    token = _get_token()
    if not token:
        return

    from src.utils.em_zone import resolve_and_cache_zone

    for hospital, coords in HOSPITAL_LOCATIONS.items():
        resolve_and_cache_zone(hospital, token=token)
        df = _ingest_one_hospital(hospital, coords, token, mode="train")
        if df is None or df.empty:
            continue
        save_csv(df, RAW_DIR / f"electricitymaps_{hospital}.csv")

    logger.info("Ingestion Electricity Maps (train) terminée.")


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run_live()
