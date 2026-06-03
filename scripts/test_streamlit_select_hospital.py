#!/usr/bin/env python3
"""Sélection d'hôpitaux par recherche clavier (sélecteur « Établissement »)."""

from __future__ import annotations

import json
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8503"


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
        "target_marker": any(
            m in body
            for m in (
                "Coupures réelles observées",
                "coupures simulées",
                "Aucune coupure étiquetée",
                "Coupures réseau comté",
                "comté",
            )
        ),
        "eric_banner": "ERIC" in body,
        "nyc_banner": "LL84" in body,
        "africa_clone": "cloné" in body.lower(),
        "no_realtime_ui": "Temps réel" not in body,
        "blocking": any(
            x in body.lower()
            for x in (
                "impossible de charger",
                "aucune donnée disponible",
                "données introuvables",
            )
        ),
        "sidebar_model_lines": side.count("Modèle :"),
        "illustratif_hint": "illustratif" in body.lower() or "Score illustratif" in body,
    }


def main():
    tests = [
        ("St Thomas", "eric"),
        ("Bellevue", "nyc"),
        ("Lacor", "lacor"),
        ("Groote Schuur", "africa"),
        ("Manchester", "eric"),
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
                snap.update({
                    "search": search,
                    "kind": kind,
                    "selected": label,
                    "ok": not snap["blocking"] and snap["no_realtime_ui"],
                })
                out.append(snap)
            except Exception as e:
                out.append({"search": search, "kind": kind, "ok": False, "error": str(e)})
    print(json.dumps({"caption": cap, "tests": out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
