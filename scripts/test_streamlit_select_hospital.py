#!/usr/bin/env python3
"""Sélection d'un hôpital par recherche clavier dans le select Streamlit."""

import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8502"
SEARCH = sys.argv[2] if len(sys.argv) > 2 else "St Thomas"


def select_by_search(page, text: str) -> str:
    sel = page.locator('[data-testid="stSelectbox"]').first
    sel.locator('[data-baseweb="select"]').click()
    time.sleep(0.3)
    page.keyboard.type(text, delay=40)
    time.sleep(0.5)
    opt = page.locator('[role="option"]').filter(has_text=text).first
    label = opt.inner_text().strip()
    opt.click()
    page.wait_for_load_state("networkidle", timeout=90000)
    time.sleep(1.0)
    return label


def snapshot(page) -> dict:
    body = page.inner_text("body")
    side = page.locator('[data-testid="stSidebar"]').inner_text()
    return {
        "target_badge": sum(
            1
            for t in (
                "Coupures réelles observées",
                "coupures simulées",
                "Aucune coupure étiquetée",
            )
            if t in body
        ),
        "eric_banner": "ERIC NHS" in body or "données ERIC" in body.lower(),
        "nyc_banner": "NYC" in body and "LL84" in body,
        "africa_banner": "profil de consommation estimé" in body.lower(),
        "blocking": any(
            x in body.lower()
            for x in ("impossible de charger", "aucune donnée disponible", "données introuvables")
        ),
        "sidebar_model_lines": side.count("Modèle :"),
        "illustratif_tab": "illustratif" in body.lower() or "Score illustratif" in body,
    }


def main():
    tests = [
        ("St Thomas", "eric UK"),
        ("Bellevue", "nyc US"),
        ("Lacor", "lacor"),
        ("Groote Schuur", "africa ZA"),
        ("Manchester", "eric UK"),
    ]
    out = []
    with sync_playwright() as p:
        page = p.chromium.launch(headless=True).new_page(viewport={"width": 1500, "height": 1100})
        page.goto(URL, wait_until="networkidle", timeout=120000)
        cap = [l for l in page.inner_text("body").split("\n") if "sélectionnables" in l]
        for search, kind in tests:
            try:
                label = select_by_search(page, search)
                snap = snapshot(page)
                snap.update({"search": search, "kind": kind, "selected": label, "ok": not snap["blocking"]})
                out.append(snap)
            except Exception as e:
                out.append({"search": search, "kind": kind, "ok": False, "error": str(e)})
    import json
    print(json.dumps({"caption": cap, "tests": out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
