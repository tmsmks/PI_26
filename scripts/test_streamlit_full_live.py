#!/usr/bin/env python3
"""Batterie live http://localhost:8502 — plusieurs familles d'hôpitaux + onglets."""

from __future__ import annotations

import json
import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8502"

HOSPITALS = [
    ("Lacor", "lacor", {"target": "réelles", "banner": None}),
    ("Groote Schuur", "africa", {"target": "étiquetée", "banner": "estimé"}),
    ("St Thomas", "eric", {"target": "coupures simulées", "banner": "ERIC"}),
    ("Bellevue", "nyc", {"target": "coupures simulées", "banner": "LL84"}),
    ("Manchester", "eric", {"target": "coupures simulées", "banner": "ERIC"}),
    ("Kenyatta", "africa", {"target": "étiquetée", "banner": "estimé"}),
    ("NYU", "nyc", {"target": "coupures simulées", "banner": "LL84"}),
]


def _select(page, text: str) -> None:
    page.locator('[data-testid="stSelectbox"]').first.locator('[data-baseweb="select"]').click()
    time.sleep(0.25)
    page.keyboard.press("Control+a")
    page.keyboard.type(text, delay=30)
    time.sleep(0.4)
    page.locator('[role="option"]').first.click()
    page.wait_for_load_state("networkidle", timeout=90000)
    time.sleep(0.9)


def _tab(page, name: str) -> None:
    page.get_by_role("tab", name=re.compile(re.escape(name))).click()
    time.sleep(0.7)


def _test_one(page, search: str, family: str, expect: dict) -> dict:
    _select(page, search)
    body = page.inner_text("body")
    r = {
        "search": search,
        "family": family,
        "ok": True,
        "issues": [],
    }
    if expect["target"] == "réelles" and "Coupures réelles" not in body:
        r["issues"].append("badge cible réelles absent")
    if expect["target"] == "coupures simulées" and "coupures simulées" not in body.lower():
        r["issues"].append("badge coupures simulées absent")
    if expect["target"] == "étiquetée" and "étiquetée" not in body:
        r["issues"].append("badge cloné absent")
    if expect["banner"] == "ERIC" and "ERIC NHS" not in body:
        r["issues"].append("bandeau ERIC absent")
    if expect["banner"] == "LL84" and "LL84" not in body:
        r["issues"].append("bandeau NYC absent")
    if expect["banner"] == "estimé" and "estimé" not in body.lower():
        r["issues"].append("bandeau africa absent")

    _tab(page, "Prochaine coupure")
    b = page.inner_text("body")
    if family == "africa" and "Temps réel" not in b:
        r["issues"].append("radio temps réel absent (africa)")
    if family == "eric" and "Historique 2022" not in b:
        r["issues"].append("mode historique absent (eric)")

    _tab(page, "Analyse historique")
    btn = page.get_by_role("button", name=re.compile(r"Analyser le risque"))
    if btn.count():
        btn.first.click()
        time.sleep(2.5)
        b2 = page.inner_text("body")
        if "Synthèse du risque" not in b2:
            r["issues"].append("analyse : pas de synthèse risque")
        if "LightGBM" not in b2 and "heuristique" not in b2.lower():
            r["issues"].append("analyse : note durée absente")
        if family != "lacor" and "illustratif" not in b2.lower() and "Profil" not in b2:
            r["issues"].append("analyse : avertissement illustratif absent")
    else:
        r["issues"].append("bouton analyser absent")

    _tab(page, "Prévisions J+7")
    b3 = page.inner_text("body")
    if "Pas de prévisions météo" in b3:
        r["forecast_skip"] = "météo absente"
    else:
        proj = page.get_by_role("button", name=re.compile(r"Projeter"))
        if proj.count():
            proj.first.click()
            time.sleep(3.0)
            b3 = page.inner_text("body")
            if "Pic de risque prévu" not in b3:
                r["issues"].append("J+7 : projection sans résumé pic")
            if "Durée est." not in b3:
                r["issues"].append("J+7 : colonne durée absente")
        else:
            r["issues"].append("J+7 : bouton projeter absent")

    _tab(page, "Simulation")
    sim = page.get_by_role("button", name=re.compile(r"Simuler", re.I))
    if sim.count():
        sim.first.scroll_into_view_if_needed()
        sim.first.click()
        time.sleep(3.0)
        if "Synthèse du risque" not in page.inner_text("body"):
            r["issues"].append("simulation : pas de résultat")
    else:
        r["issues"].append("simulation : bouton absent")

    if r["issues"]:
        r["ok"] = False
    return r


def main() -> int:
    report = {"url": URL, "results": []}
    with sync_playwright() as p:
        page = p.chromium.launch(headless=True).new_page(viewport={"width": 1500, "height": 1200})
        page.goto(URL, wait_until="networkidle", timeout=120000)
        cap = [l for l in page.inner_text("body").split("\n") if "sélectionnables" in l]
        report["selector_caption"] = cap
        for search, family, expect in HOSPITALS:
            report["results"].append(_test_one(page, search, family, expect))

    report["n_ok"] = sum(1 for r in report["results"] if r["ok"])
    report["n_fail"] = len(report["results"]) - report["n_ok"]
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "reports" / "streamlit_full_live_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
