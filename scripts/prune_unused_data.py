#!/usr/bin/env python3
"""
Supprime les artefacts data/ non utilisés par l'app ni le modèle Lacor.

Usage :
  python scripts/prune_unused_data.py          # exécution
  python scripts/prune_unused_data.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import RAW_DIR
from src.utils.hospitals import HOSPITAL_DISPLAY

# Comtés EAGLE-I horaires utiles : NYC (ingestion LL84) + option --multisite.
NYC_EAGLEI_KEYS = {
    info["eaglei_county_key"]
    for info in HOSPITAL_DISPLAY.values()
    if info.get("eaglei_county_key")
}

MULTISITE_EAGLEI_KEYS = {
    "maricopa_az",
    "harris_tx",
    "cook_il",
    "miamidade_fl",
    "losangeles_ca",
    "king_wa",
    "orleans_la",
}

SUMMARY_ARTIFACTS = [
    RAW_DIR / "eric" / "eric_summary.csv",
    RAW_DIR / "nyc_ll84" / "nyc_summary.csv",
    RAW_DIR / "nyc_ll84" / "hospitals_filtered.csv",
]


def _collect_paths() -> list[Path]:
    paths: list[Path] = []

    src = RAW_DIR / "eaglei_source"
    if src.is_dir():
        paths.extend(src.rglob("*"))

    for pattern in (
        "electricitymaps_*.csv",
        "meteo_forecast_*.csv",
        "eaglei_meteo_*.csv",
    ):
        paths.extend(RAW_DIR.glob(pattern))

    for p in RAW_DIR.glob("eaglei_*.csv"):
        key = p.stem.removeprefix("eaglei_")
        if key not in NYC_EAGLEI_KEYS and key not in MULTISITE_EAGLEI_KEYS:
            continue
        if key in MULTISITE_EAGLEI_KEYS:
            paths.append(p)

    for key, info in HOSPITAL_DISPLAY.items():
        if not info.get("ui_hidden"):
            continue
        for pattern in (
            f"meteo_{key}.csv",
            f"meteo_forecast_{key}.csv",
            f"electricitymaps_{key}.csv",
        ):
            paths.append(RAW_DIR / pattern)

    paths.extend(SUMMARY_ARTIFACTS)
    paths.append(ROOT / "data" / ".DS_Store")

    return sorted({p for p in paths if p.is_file()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = _collect_paths()
    total = sum(p.stat().st_size for p in targets)
    print(f"{len(targets)} fichier(s) — {total / 1e6:.1f} Mo")

    for path in targets:
        rel = path.relative_to(ROOT)
        if args.dry_run:
            print(f"  [dry-run] {rel}")
        else:
            path.unlink()
            print(f"  supprimé  {rel}")

    if not args.dry_run and (RAW_DIR / "eaglei_source").exists():
        try:
            (RAW_DIR / "eaglei_source").rmdir()
        except OSError:
            pass

    if args.dry_run:
        print("\nRelancer sans --dry-run pour appliquer.")
    else:
        print("\nPuis régénérer sans em_* :")
        print("  python -c \"from src.data.preprocessing import run; run()\"")
        print("  python -c \"from src.features.build_features import run; run()\"")


if __name__ == "__main__":
    main()
