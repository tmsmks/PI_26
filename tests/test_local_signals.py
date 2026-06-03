"""Granularité locale des signaux par hôpital."""

from src.utils.hospitals import HOSPITAL_DISPLAY
from src.utils.local_signals import eaglei_county_key, local_signal_profile


def test_lacor_is_building_local():
    p = local_signal_profile("lacor_uganda")
    assert p["consumption"].startswith("Bâtiment")
    assert p["outage"].startswith("Bâtiment")
    assert p["eaglei_county_key"] is None


def test_nyc_hospitals_map_to_borough_counties():
    from src.utils.hospitals import get_target_source

    expected = {
        "nyc_bellevue": "new_york_ny",
        "nyc_nyu_tisch": "new_york_ny",
        "nyc_nyp_brooklyn": "kings_ny",
        "nyc_elmhurst": "queens_ny",
        "nyc_lincoln": "bronx_ny",
    }
    for hkey, county in expected.items():
        info = HOSPITAL_DISPLAY[hkey]
        assert eaglei_county_key(hkey, info) == county
        assert get_target_source(hkey, info) == "county_network"
        p = local_signal_profile(hkey, info)
        assert "Comté US" in p["outage"]


def test_africa_grid_not_local_consumption():
    p = local_signal_profile("kenyatta_kenya", HOSPITAL_DISPLAY["kenyatta_kenya"])
    assert "cloné" in p["consumption"].lower()
    assert p["eaglei_county_key"] is None
