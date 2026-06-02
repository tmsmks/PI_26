"""
Client HTTP partagé avec retry/backoff exponentiel.

Toutes les ingestions réseau (`src/data/ingest_*.py`) frappent des APIs
publiques sujettes à des erreurs transitoires (429 rate-limit, 5xx, coupures
réseau). `requests.get` n'effectue AUCUNE nouvelle tentative par défaut : un
hoquet ponctuel faisait perdre toute une source (même si le pipeline
l'encapsule dans un try/except au niveau orchestration).

`http_get` ré-essaie automatiquement sur 429/5xx et erreurs de connexion,
avec backoff exponentiel, et applique un timeout par défaut. Drop-in pour
`requests.get(url, params=..., headers=..., timeout=...)`.
"""

from __future__ import annotations

import logging

import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 est une dépendance de requests ; le chemin a varié selon versions
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    from requests.packages.urllib3.util.retry import Retry  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.5  # 0.5s, 1s, 2s… entre les tentatives
RETRY_STATUS = (429, 500, 502, 503, 504)


def make_retry_session(
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    status_forcelist: tuple[int, ...] = RETRY_STATUS,
) -> requests.Session:
    """Crée une `requests.Session` montée avec une stratégie de retry."""
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION: requests.Session | None = None


def get_session() -> requests.Session:
    """Session partagée (créée paresseusement, réutilise les connexions)."""
    global _SESSION
    if _SESSION is None:
        _SESSION = make_retry_session()
    return _SESSION


def http_get(url: str, *, timeout: float = DEFAULT_TIMEOUT, **kwargs) -> requests.Response:
    """GET avec retry/backoff (429/5xx, erreurs de connexion) + timeout par défaut.

    Remplacement direct de `requests.get`. Accepte `params`, `headers`,
    `stream`, etc. via `**kwargs`.
    """
    return get_session().get(url, timeout=timeout, **kwargs)
