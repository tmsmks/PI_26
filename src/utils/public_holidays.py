"""
Jours fériés 2022 par pays (ISO3) pour ``is_public_holiday``.

Le modèle Lacor est entraîné avec le calendrier Ouganda. Les autres sites du
pipeline reçoivent le calendrier de leur pays quand il est défini ici ;
sinon ``is_public_holiday = 0`` (pas de calendrier inventé).
"""

from __future__ import annotations

import pandas as pd

from src.utils.config import UGANDA_PUBLIC_HOLIDAYS_2022
from src.utils.hospitals import HOSPITAL_DISPLAY

# Angleterre / pays de Galles — jours fériés 2022 (simplifié)
_GBR_HOLIDAYS_2022 = [
    "2022-01-01",
    "2022-04-15",
    "2022-04-18",
    "2022-05-02",
    "2022-06-02",
    "2022-06-03",
    "2022-08-29",
    "2022-09-19",
    "2022-12-26",
    "2022-12-27",
]

# Fériés fédéraux USA 2022 (simplifié)
_USA_HOLIDAYS_2022 = [
    "2022-01-01",
    "2022-01-17",
    "2022-02-21",
    "2022-05-30",
    "2022-06-20",
    "2022-07-04",
    "2022-09-05",
    "2022-10-10",
    "2022-11-11",
    "2022-11-24",
    "2022-12-25",
]

HOLIDAY_DATES_BY_COUNTRY: dict[str, list[str]] = {
    "UGA": UGANDA_PUBLIC_HOLIDAYS_2022,
    "GBR": _GBR_HOLIDAYS_2022,
    "USA": _USA_HOLIDAYS_2022,
}


def country_for_hospital(hospital_key: str) -> str:
    return str(HOSPITAL_DISPLAY.get(hospital_key, {}).get("country", ""))


def holiday_dates_for_country(country_iso3: str) -> pd.DatetimeIndex | None:
    raw = HOLIDAY_DATES_BY_COUNTRY.get(country_iso3)
    if not raw:
        return None
    return pd.to_datetime(raw).normalize()


def is_public_holiday_mask(datetimes: pd.Series, country_iso3: str) -> pd.Series:
    """Série 0/1 alignée sur ``datetimes``."""
    holidays = holiday_dates_for_country(country_iso3)
    if holidays is None or len(holidays) == 0:
        return pd.Series(0, index=datetimes.index, dtype=int)
    days = pd.to_datetime(datetimes).dt.normalize()
    return days.isin(holidays).astype(int)


def is_public_holiday_at(ts: pd.Timestamp, country_iso3: str) -> int:
    holidays = holiday_dates_for_country(country_iso3)
    if holidays is None:
        return 0
    day = pd.to_datetime(ts).normalize()
    return int(day in holidays)


def representative_date_2022(month: int, day_of_week: int) -> pd.Timestamp:
    """Premier jour de ``month`` en 2022 qui correspond au jour de semaine."""
    for day in range(1, 29):
        ts = pd.Timestamp(year=2022, month=month, day=day)
        if ts.dayofweek == day_of_week:
            return ts.normalize()
    return pd.Timestamp(year=2022, month=month, day=1).normalize()


def is_public_holiday_for_hospital(
    hospital_key: str,
    *,
    month: int,
    day_of_week: int,
    datetime: pd.Timestamp | None = None,
) -> int:
    """Férié pour un site (simulation ou point unique)."""
    country = country_for_hospital(hospital_key)
    if datetime is not None:
        return is_public_holiday_at(datetime, country)
    return is_public_holiday_at(representative_date_2022(month, day_of_week), country)
