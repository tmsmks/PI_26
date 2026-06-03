#!/usr/bin/env python3
"""Test live Streamlit (http://localhost:8502) — parcourt les hôpitaux du sélecteur."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8502"
TIMEOUT_MS = 90_000


def _wait_app(page) -> None:
    page.goto(URL, wait_until="networkidle", timeout=TIMEOUT_MS)
    page.wait_for_selector("text=Prédiction de coupures", timeout=TIMEOUT_MS)


def _hospital_select(page):
    """Premier select Streamlit sous le label Hôpital."""
    labels = page.locator("label").filter(has_text=re.compile(r"^Hôpital$"))
    if labels.count() == 0:
        return page.locator('[data-testid="stSelectbox"]').first
    block = labels.first.locator("xpath=ancestor::div[contains(@class,'stSelectbox')][1]")
    if block.count():
        return block
    return page.locator('[data-testid="stSelectbox"]').first


def _open_select_and_options(page, select_locator) -> list[str]:
    select_locator.locator('[data-baseweb="select"]').click()
    time.sleep(0.4)
    opts = page.locator('[role="option"]')
    names = [opts.nth(i).inner_text().strip() for i in range(opts.count())]
    page.keyboard.press("Escape")
    time.sleep(0.2)
    return names


def _select_hospital(page, select_locator, label: str) -> None:
    select_locator.locator('[data-baseweb="select"]').click()
    time.sleep(0.3)
    page.locator('[role="option"]').filter(has_text=label).first.click()
    page.wait_for_load_state("networkidle", timeout=TIMEOUT_MS)
    time.sleep(1.2)


def _page_issues(page) -> dict:
    alerts = page.locator('[data-testid="stAlert"]')
    texts = []
    for i in range(alerts.count()):
        t = alerts.nth(i).inner_text().strip()
        if t:
            texts.append(t[:500])
    errors = [t for t in texts if "error" in page.locator('[data-testid="stAlert"]').nth(texts.index(t)).get_attribute("class") or ""]
    # Classify by emoji / keywords
    blocking = []
    warnings = []
    infos = []
    for t in texts:
        low = t.lower()
        if any(x in low for x in ("impossible", "introuvable", "aucune donnée", "erreur", "échec")):
            blocking.append(t)
        elif "⚠" in t or "illustratif" in low or "synthétique" in low:
            warnings.append(t)
        else:
            infos.append(t)
    body = page.inner_text("body")
    has_tabs = all(
        x in body
        for x in (
            "Prochaine coupure",
            "Analyse historique",
            "Prévisions J+7",
            "Simulation manuelle",
        )
    )
    has_hero = "Prédiction de coupures" in body
    has_sources = "Sources utilisées pour" in body
    has_target_badge = any(
        k in body
        for k in (
            "Coupures réelles observées",
            "coupures simulées",
            "Aucune coupure étiquetée",
        )
    )
    return {
        "alert_count": len(texts),
        "blocking": blocking,
        "warnings": warnings[:3],
        "infos_sample": infos[:2],
        "has_tabs": has_tabs,
        "has_hero": has_hero,
        "has_sources": has_sources,
        "has_target_badge": has_target_badge,
        "sidebar_duration": "Modèle de **durée**" in body or "Durée : heuristique" in body,
        "em_panel": "Electricity Maps" in body or "snapshot" in body.lower(),
    }


def main() -> int:
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        try:
            _wait_app(page)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"App inaccessible: {e}"}, ensure_ascii=False))
            browser.close()
            return 1

        sel = _hospital_select(page)
        try:
            options = _open_select_and_options(page, sel)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"Sélecteur hôpital: {e}"}, ensure_ascii=False))
            browser.close()
            return 1

        if not options:
            print(json.dumps({"ok": False, "error": "Aucune option dans le sélecteur"}, ensure_ascii=False))
            browser.close()
            return 1

        for opt in options:
            entry = {"hospital_option": opt, "ok": True}
            try:
                _select_hospital(page, sel, opt)
                issues = _page_issues(page)
                entry.update(issues)
                if issues["blocking"]:
                    entry["ok"] = False
            except Exception as e:
                entry["ok"] = False
                entry["error"] = str(e)
            results.append(entry)

        browser.close()

    out_path = Path(__file__).resolve().parents[1] / "reports" / "streamlit_hospital_live_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "url": URL,
        "n_hospitals": len(results),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_fail": sum(1 for r in results if not r.get("ok")),
        "hospitals": results,
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
