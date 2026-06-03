#!/usr/bin/env python3
"""Test live : onglets + actions clés (Lacor, africa_grid, ERIC, NYC).

UI : 3 onglets ; Prochaine coupure → Réseau & météo (live) | Modèle 1/3/6 h (replay).
Tests replay : ouvrir le sous-onglet replay pour « Date / Heure de référence ».
"""

from __future__ import annotations

import json
import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8503"
TIMEOUT_MS = 90_000

# Recherche clavier dans le sélecteur « Établissement » (libellés dynamiques).
SAMPLES = [
    ("Lacor", "lacor"),
    ("Groote Schuur", "africa"),
    ("St Thomas", "eric"),
    ("Bellevue", "nyc"),
]


def _goto(page):
    page.goto(URL, wait_until="networkidle", timeout=TIMEOUT_MS)
    page.wait_for_selector("text=Prédiction de coupures", timeout=TIMEOUT_MS)


def _select(page, search: str):
    page.locator('[data-testid="stSelectbox"]').first.locator('[data-baseweb="select"]').click()
    time.sleep(0.35)
    page.keyboard.press("Control+a")
    page.keyboard.type(search, delay=30)
    time.sleep(0.4)
    page.locator('[role="option"]').first.click()
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


def _target_signals(body: str, family: str) -> bool:
    """Badge / sous-titre de provenance cible visible."""
    if family == "lacor":
        return "Coupures réelles observées" in body
    if family == "nyc":
        return (
            "comté" in body.lower()
            or "EAGLE-I" in body
            or "réseau comté" in body.lower()
        )
    if family == "eric":
        return "coupures simulées" in body.lower() or "Charge réelle" in body
    if family == "africa":
        return (
            "cloné" in body.lower()
            or "étiquetée" in body.lower()
            or "Aucune coupure" in body
        )
    return True


def test_hospital(page, search: str, family: str) -> dict:
    _select(page, search)
    body = _body(page)
    side = _sidebar(page)
    r = {
        "search": search,
        "family": family,
        "sidebar_model_dup": side.count("Modèle : LightGBM"),
        "sidebar_duration": "durée" in side.lower(),
        "sidebar_eagle": "EAGLE" in side or "LOSO" in side,
        "target_ok": _target_signals(body, family),
        "africa_clone_caption": (
            "profil" in body.lower() and "cloné" in body.lower()
            if family == "africa"
            else None
        ),
        "eric_caption": "ERIC" in body if family == "eric" else None,
        "nyc_caption": "LL84" in body if family == "nyc" else None,
    }

    _tab(page, "Prochaine coupure")
    b1 = _body(page)
    r["next_no_realtime_radio"] = "Temps réel" not in b1
    r["next_has_replay_ui"] = (
        "Date de référence" in b1 and "Heure de référence" in b1
    )
    r["next_has_risk_section"] = "Risque à venir" in b1 or "Coupure" in b1
    r["next_illustratif"] = "illustratif" in b1.lower() or "Score illustratif" in b1

    _tab(page, "Historique")
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

    r["no_forecast_tab"] = page.get_by_role("tab", name=re.compile(r"Prévisions J")).count() == 0
    r["has_three_main_tabs"] = all(
        page.get_by_role("tab", name=re.compile(re.escape(t))).count() > 0
        for t in ("Prochaine coupure", "Historique", "Simulation")
    )

    _tab(page, "Simulation")
    sim = page.get_by_role("button", name=re.compile(r"Simuler", re.I))
    if sim.count():
        sim.first.scroll_into_view_if_needed()
        sim.first.click()
        time.sleep(3.5)
        b4 = _body(page)
        r["simulate_ok"] = "Synthèse du risque" in b4
    else:
        r["simulate_ok"] = False

    r["ok"] = (
        r["target_ok"]
        and r["next_no_realtime_radio"]
        and r["next_has_replay_ui"]
        and r["no_forecast_tab"]
        and r["has_three_main_tabs"]
        and r.get("predict_ran", True)
        and r.get("simulate_ok", True)
    )
    return r


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        _goto(page)
        page.locator('[data-testid="stSelectbox"]').first.locator('[data-baseweb="select"]').click()
        time.sleep(0.3)
        opts = page.locator('[role="option"]')
        all_opts = [opts.nth(i).inner_text() for i in range(opts.count())]
        page.keyboard.press("Escape")
        time.sleep(0.2)

        report = {
            "url": URL,
            "selector_options_sample": all_opts[:5],
            "n_options": len(all_opts),
            "has_eric_nyc": any(
                "ERIC" in o or "NYC" in o or "NHS" in o or "comté" in o.lower()
                for o in all_opts
            ),
            "deep_tests": [test_hospital(page, s, f) for s, f in SAMPLES],
            "n_ok": 0,
            "n_fail": 0,
        }
        report["n_ok"] = sum(1 for t in report["deep_tests"] if t.get("ok"))
        report["n_fail"] = len(report["deep_tests"]) - report["n_ok"]
        browser.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
