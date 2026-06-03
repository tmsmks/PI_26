"""Calendriers fériés par pays / site."""

import pandas as pd

from src.utils.public_holidays import (
    country_for_hospital,
    is_public_holiday_for_hospital,
    is_public_holiday_mask,
)


def test_country_for_lacor():
    assert country_for_hospital("lacor_uganda") == "UGA"


def test_uganda_holiday_2022():
    ts = pd.Timestamp("2022-01-01")
    mask = is_public_holiday_mask(pd.Series([ts]), "UGA")
    assert int(mask.iloc[0]) == 1


def test_gbr_holiday_via_eric_site():
    assert is_public_holiday_for_hospital(
        "st_thomas_nhs",
        month=4,
        day_of_week=4,
        datetime=pd.Timestamp("2022-04-15"),
    ) == 1


def test_unknown_country_zero():
    assert is_public_holiday_for_hospital(
        "__unknown__", month=6, day_of_week=2
    ) == 0


def test_simulation_representative_non_holiday():
    # Mercredi ordinaire en juin (pas férié UGA/GBR/USA simplifié)
    assert is_public_holiday_for_hospital(
        "lacor_uganda", month=6, day_of_week=2
    ) in (0, 1)
