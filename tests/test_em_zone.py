"""Résolution de zone Electricity Maps (repli statique)."""

from src.utils.em_zone import resolve_zone_precise


def test_resolve_zone_static_fallback(monkeypatch):
    monkeypatch.delenv("ELECTRICITY_MAPS_TOKEN", raising=False)
    res = resolve_zone_precise("lacor_uganda", use_static_fallback=True)
    assert res["zone"] == "UG"
    assert res["source"] == "static"
