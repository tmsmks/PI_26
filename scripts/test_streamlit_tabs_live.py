#!/usr/bin/env python3
"""Test live : onglets + actions clés pour Lacor, 1 hôpital africa, et vérif sélecteur ERIC/NYC."""

from __future__ import annotations

import json
import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8502"
TIMEOUT_MS = 90_000

SAMPLES = [
    "🇺🇬 Lacor Hospital — Gulu, Ouganda",
    "🇿🇦 Groote Schuur Hospital — Le Cap, Afrique du Sud",
    "🇬🇧 St Thomas' Hospital — Londres, Royaume-Uni",
    "🇺🇸 NYC Health + Hospitals / Bellevue — Manhattan, USA",
]


def _goto(page):
    page.goto(URL, wait_until="networkidle", timeout=TIMEOUT_MS)
    page.wait_for_selector("text=Prédiction de coupures", timeout=TIMEOUT_MS)


def _select(page, label: str):
    page.locator('[data-testid="stSelectbox"]').first.locator('[data-baseweb="select"]').click()
    time.sleep(0.35)
    page.locator('[role="option"]').filter(has_text=label).first.click()
    page.wait_for_load_state("networkidle", timeout=TIMEOUT_MS)
    time.sleep(1.0)


def _tab(page, name: str):
    page.get_by_role("tab", name=re.compile(re.escape(name))).click()
    time.sleep(0.8)


def _body(page) -> str:
    return page.inner_text("body")


def _sidebar(page) -> str:
    side = page.locator('[data-testid="stSidebar"]')
    return side.inner_text() if side.count() else ""


def test_hospital(page, label: str) -> dict:
    _select(page, label)
    body = _body(page)
    side = _sidebar(page)
    r = {
        "hospital": label,
        "sidebar_model_dup": side.count("Modèle : LightGBM"),
        "sidebar_duration": ("durée" in side.lower()),
        "sidebar_eagle": ("EAGLE" in side or "LOSO" in side),
        "target_badge": sum(
            1
            for t in (
                "Coupures réelles observées",
                "coupures simulées",
                "Aucune coupure étiquetée",
            )
            if t in body
        ),
        "africa_banner": "profil de consommation estimé" in body.lower(),
        "eric_in_selector": "ERIC" in body or "NHS" in body,
    }

    # Onglet Prochaine coupure
    _tab(page, "Prochaine coupure")
    b1 = _body(page)
    r["next_has_realtime_radio"] = "Temps réel" in b1
    r["next_default_mode"] = "Historique 2022" in b1
    r["next_illustratif"] = "illustratif" in b1.lower() or "Indicatif" in b1

    # Onglet Analyse — bouton analyser
    _tab(page, "Analyse historique")
    btn = page.get_by_role("button", name=re.compile(r"Analyser le risque"))
    if btn.count():
        btn.first.click()
        time.sleep(2.5)
        b2 = _body(page)
        r["predict_ran"] = "Synthèse du risque" in b2 or "Probabilité" in b2
        r["predict_duration_note"] = "LightGBM" in b2 or "heuristique" in b2.lower()
        r["predict_shap_or_factors"] = "SHAP" in b2 or "Facteurs explicatifs" in b2
    else:
        r["predict_ran"] = False

    # Onglet J+7
    _tab(page, "Prévisions J+7")
    b3 = _body(page)
    r["forecast_meteo_missing"] = "Pas de prévisions météo" in b3
    proj = page.get_by_role("button", name=re.compile(r"Projeter le risque"))
    if proj.count() and "Pas de prévisions" not in b3:
        proj.first.click()
        time.sleep(3.0)
        b3 = _body(page)
        r["forecast_projected"] = "Pic de risque prévu" in b3
        r["forecast_duration_col"] = "Durée est." in b3
    else:
        r["forecast_projected"] = False

    # Simulation
    _tab(page, "Simulation manuelle")
    sim = page.get_by_role("button", name=re.compile(r"Simuler", re.I))
    if sim.count():
        sim.first.scroll_into_view_if_needed()
        sim.first.click()
        time.sleep(3.5)
        b4 = _body(page)
        r["simulate_ok"] = "Synthèse du risque" in b4
    else:
        r["simulate_ok"] = False

    return r


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        _goto(page)
        page.locator('[data-testid="stSelectbox"]').first.locator('[data-baseweb="select"]').click()
        time.sleep(0.3)
        all_opts = [page.locator('[role="option"]').nth(i).inner_text() for i in range(page.locator('[role="option"]').count())]
        page.keyboard.press("Escape")
        time.sleep(0.2)

        report = {
            "url": URL,
            "selector_options": all_opts,
            "n_options": len(all_opts),
            "has_eric_nyc": any("ERIC" in o or "NYC" in o or "NHS" in o for o in all_opts),
            "deep_tests": [],
        }
        for s in SAMPLES:
            if s in all_opts:
                report["deep_tests"].append(test_hospital(page, s))

        browser.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
