"""
Résolution de la zone Electricity Maps la plus fine pour un hôpital.

Priorité : coordonnées exactes de l'hôpital (`/v4/zone?lat&lon`) — polygone EM
au point GPS. Repli : mapping statique `HOSPITAL_ELECTRICITY_ZONES` si l'API
échoue ou le token est absent.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import (
    ELECTRICITYMAPS_BASE,
    ELECTRICITYMAPS_TOKEN_ENV,
    HOSPITAL_ELECTRICITY_ZONES,
    HOSPITAL_LOCATIONS,
    RAW_DIR,
)
from src.utils.hospitals import HOSPITAL_DISPLAY

logger = logging.getLogger(__name__)

EP_ZONE = "/v4/zone"
ZONE_CACHE_PATH = RAW_DIR / "em_zone_resolution.json"


def _get_token() -> str | None:
    import os
    return os.environ.get(ELECTRICITYMAPS_TOKEN_ENV) or None


def _call_zone_api(lat: float, lon: float, token: str) -> dict[str, Any] | None:
    from src.utils.http import http_get

    url = ELECTRICITYMAPS_BASE + EP_ZONE
    try:
        resp = http_get(
            url,
            headers={"auth-token": token},
            params={"lat": lat, "lon": lon},
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Electricity Maps /zone : %s", exc)
        return None
    if not resp.ok:
        logger.warning("Electricity Maps /zone : HTTP %s", resp.status_code)
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def resolve_zone_precise(
    hospital_key: str,
    *,
    lat: float | None = None,
    lon: float | None = None,
    token: str | None = None,
    use_static_fallback: bool = True,
) -> dict[str, Any]:
    """Résout la zone EM pour un hôpital.

    Retourne un dict :
      zone, lat, lon, source ('api' | 'static' | 'missing'),
      country_name, zone_name (si dispo), raw (réponse API optionnelle).
    """
    info = HOSPITAL_DISPLAY.get(hospital_key, {})
    lat = lat if lat is not None else info.get("lat")
    lon = lon if lon is not None else info.get("lon")
    token = token or _get_token()

    out: dict[str, Any] = {
        "hospital": hospital_key,
        "lat": lat,
        "lon": lon,
        "zone": None,
        "source": "missing",
        "country_name": None,
        "zone_name": None,
    }

    if token and lat is not None and lon is not None:
        payload = _call_zone_api(float(lat), float(lon), token)
        if payload:
            zone = payload.get("zone") or payload.get("zoneKey")
            out.update(
                zone=zone,
                source="api",
                country_name=payload.get("countryName") or payload.get("country"),
                zone_name=payload.get("zoneName") or payload.get("zoneParent"),
                raw=payload,
            )
            return out

    if use_static_fallback:
        static = HOSPITAL_ELECTRICITY_ZONES.get(hospital_key)
        if static:
            out.update(zone=static, source="static")
            return out

    return out


def load_zone_cache() -> dict[str, dict]:
    if not ZONE_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(ZONE_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_zone_cache_entry(hospital_key: str, resolution: dict[str, Any]) -> None:
    cache = load_zone_cache()
    entry = {
        k: resolution[k]
        for k in ("zone", "lat", "lon", "source", "country_name", "zone_name")
        if k in resolution
    }
    entry["resolved_at"] = datetime.now(timezone.utc).isoformat()
    cache[hospital_key] = entry
    ZONE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ZONE_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def resolve_and_cache_zone(hospital_key: str, token: str | None = None) -> dict[str, Any]:
    """Résout + persiste la zone pour un hôpital."""
    res = resolve_zone_precise(hospital_key, token=token)
    if res.get("zone"):
        save_zone_cache_entry(hospital_key, res)
    return res


def resolve_all_hospital_zones(token: str | None = None) -> dict[str, dict]:
    """Résout les zones pour tous les sites `ingest_geo`."""
    token = token or _get_token()
    if not token:
        logger.warning("Pas de token EM — résolution de zones ignorée.")
        return {}
    results: dict[str, dict] = {}
    for hospital_key in HOSPITAL_LOCATIONS:
        results[hospital_key] = resolve_and_cache_zone(hospital_key, token=token)
    return results
