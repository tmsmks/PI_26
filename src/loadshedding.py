"""
Délestage programmé (load-shedding) en TEMPS RÉEL via EskomSePush.

Pourquoi
────────
En Afrique du Sud, la cause n°1 des coupures n'est pas la météo ni la
consommation locale : c'est le **délestage programmé** (load-shedding) d'Eskom
et des municipalités. C'est un signal *causal direct* et *anticipé* (le stade
et son calendrier sont publiés à l'avance).

Ce module interroge EskomSePush (`/status`) pour récupérer, pour le bloc
pertinent (ex. `capetown` pour Groote Schuur), le **stade actuel** et les
**prochains changements de stade** programmés.

⚠️ Couverture : Afrique du Sud uniquement. Ce signal ne s'applique pas aux
autres hôpitaux et n'est PAS une feature du modèle (entraîné sur Lacor, en
Ouganda) — c'est un **contexte temps réel** affiché tel quel.

Pré-requis : variable d'env `ESKOM_SEPUSH_TOKEN` (token gratuit, 50 appels/j).
"""

from __future__ import annotations

import logging
import os

import requests

from src.utils.config import (
    ESKOM_SEPUSH_BASE,
    ESKOM_SEPUSH_STATUS_BLOCK,
    ESKOM_SEPUSH_TOKEN_ENV,
)

logger = logging.getLogger(__name__)


def is_supported(hospital_key: str) -> bool:
    """True si l'hôpital a un bloc de statut de délestage exploité."""
    return hospital_key in ESKOM_SEPUSH_STATUS_BLOCK


def fetch_status(token: str, timeout: int = 20) -> dict | None:
    """Appelle EskomSePush `/status`. Renvoie le payload `status` brut
    ({eskom:{...}, capetown:{...}, ...}) ou None en cas d'échec."""
    try:
        resp = requests.get(
            f"{ESKOM_SEPUSH_BASE}/status",
            headers={"token": token},
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("EskomSePush injoignable (%s)", exc)
        return None
    if resp.status_code == 403:
        logger.warning("EskomSePush 403 : token invalide ou quota dépassé.")
        return None
    if resp.status_code != 200:
        logger.warning("EskomSePush statut %s", resp.status_code)
        return None
    try:
        return resp.json().get("status")
    except Exception as exc:  # noqa: BLE001
        logger.warning("EskomSePush réponse illisible (%s)", exc)
        return None


def _stage_int(raw) -> int | None:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def loadshedding_for_hospital(hospital_key: str, token: str | None = None) -> dict | None:
    """Contexte délestage pour un hôpital sud-africain.

    Renvoie ``{block, name, stage, stage_label, severity, updated, next:[
    {stage, start}]}`` ou ``None`` si non supporté, token absent, ou API en
    échec. ``severity`` ∈ {"none","low","high"} pour piloter l'affichage.
    """
    block = ESKOM_SEPUSH_STATUS_BLOCK.get(hospital_key)
    if block is None:
        return None
    token = token or os.environ.get(ESKOM_SEPUSH_TOKEN_ENV)
    if not token:
        return None

    status = fetch_status(token)
    if not status or block not in status:
        return None

    info = status[block] or {}
    stage = _stage_int(info.get("stage"))
    next_stages = []
    for ns in info.get("next_stages", []) or []:
        s = _stage_int(ns.get("stage"))
        if s is None:
            continue
        next_stages.append({"stage": s, "start": ns.get("stage_start_timestamp")})

    if stage is None:
        severity = "none"
    elif stage <= 0:
        severity = "none"
    elif stage <= 2:
        severity = "low"
    else:
        severity = "high"

    label = "Aucun délestage" if (stage or 0) <= 0 else f"Stade {stage}"
    return {
        "block": block,
        "name": info.get("name", block),
        "stage": stage if stage is not None else 0,
        "stage_label": label,
        "severity": severity,
        "updated": info.get("stage_updated"),
        "next": next_stages,
    }
