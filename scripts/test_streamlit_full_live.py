#!/usr/bin/env python3
"""Batterie live — 3 onglets ; sous-onglet replay pour date/heure (pas « Temps réel »)."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8503"

HOSPITALS = [
    ("Lacor", "lacor", {"target": "réelles", "banner": None}),
    ("Groote Schuur", "africa", {"target": "cloné", "banner": "clone"}),
    ("St Thomas", "eric", {"target": "synthétique", "banner": "ERIC"}),
    ("Bellevue", "nyc", {"target": "comté", "banner": "LL84"}),
    ("Manchester", "eric", {"target": "synthétique", "banner": "ERIC"}),
    ("Kenyatta", "africa", {"target": "cloné", "banner": "clone"}),
    ("NYU", "nyc", {"target": "comté", "banner": "LL84"}),
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


def _check_target(body: str, expect: str) -> bool:
    if expect == "réelles":
        return "Coupures réelles" in body
    if expect == "comté":
        return "comté" in body.lower() or "EAGLE-I" in body
    if expect == "synthétique":
        return "coupures simulées" in body.lower() or "Charge réelle" in body
    if expect == "cloné":
        return "cloné" in body.lower() or "étiquetée" in body.lower()
    return True


def _test_one(page, search: str, family: str, expect: dict) -> dict:
    _select(page, search)
    body = page.inner_text("body")
    r = {
        "search": search,
        "family": family,
        "ok": True,
        "issues": [],
    }
    if not _check_target(body, expect["target"]):
        r["issues"].append(f"badge cible « {expect['target']} » absent")

    if expect["banner"] == "ERIC" and "ERIC" not in body:
        r["issues"].append("mention ERIC absente")
    if expect["banner"] == "LL84" and "LL84" not in body:
        r["issues"].append("mention LL84 absente")
    if expect["banner"] == "clone" and "cloné" not in body.lower():
        r["issues"].append("mention profil cloné absente")

    _tab(page, "Prochaine coupure")
    b = page.inner_text("body")
    if "Temps réel" in b:
        r["issues"].append("radio / libellé « Temps réel » encore présent (obsolète)")
    if "Date de référence" not in b or "Heure de référence" not in b:
        r["issues"].append("contrôles replay date/heure absents")

    _tab(page, "Historique")
    btn = page.get_by_role("button", name=re.compile(r"Analyser le risque"))
    if btn.count():
        btn.first.click()
        time.sleep(2.5)
        b2 = page.inner_text("body")
        if "Synthèse du risque" not in b2:
            r["issues"].append("historique : pas de synthèse risque")
        if "LightGBM" not in b2 and "heuristique" not in b2.lower():
            r["issues"].append("historique : note durée absente")
        if family not in ("lacor",) and "illustratif" not in b2.lower() and "Score illustratif" not in b2:
            r["issues"].append("historique : avertissement illustratif absent")
    else:
        r["issues"].append("bouton analyser absent")

    if page.get_by_role("tab", name=re.compile(r"Prévisions J")).count():
        r["issues"].append("onglet Prévisions J+7 encore présent")

    for tab_name in ("Prochaine coupure", "Historique", "Simulation"):
        if page.get_by_role("tab", name=re.compile(re.escape(tab_name))).count() == 0:
            r["issues"].append(f"onglet principal « {tab_name} » absent")

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
    out = Path(__file__).resolve().parents[1] / "reports" / "streamlit_full_live_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
