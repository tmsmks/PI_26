"""
Contenu statique de présentation pour l'app Streamlit (app.py).

Pur : libellés lisibles des features, catégories d'affichage, catalogue des
sources de données, et helpers de catégorisation. Aucune dépendance Streamlit
→ testable et importable isolément. Extrait d'app.py (#10) pour alléger le
monolithe.
"""

FEATURE_LABELS = {
    # ── Énergie & consommation ──
    "solar_ratio": "Part du solaire dans la charge",
    "solar_pv_kw": "Production solaire (kW)",
    "solar_available": "Solaire disponible",
    "total_load_kw": "Consommation totale (kW)",
    "sterilization_kw": "Stérilisation (kW)",
    "base_load_kw": "Charge de base (kW)",
    "base_load_ratio": "Ratio charge de base",
    "load_rolling_6h": "Charge moyenne (6h)",
    "load_rolling_24h": "Charge moyenne (24h)",
    "load_std_24h": "Variabilité de la charge (24h)",
    "load_diff_1h": "Variation de charge (1h)",
    "load_diff_24h": "Variation de charge (24h)",
    "load_pct_change_1h": "Variation relative (1h)",
    "peak_ratio": "Ratio pic / moyenne",
    # ── Historique coupures (réseau) ──
    "hours_since_last_outage": "Heures depuis dernière coupure",
    "last_outage_duration_h": "Durée dernière coupure (h)",
    "outage_frequency_7d": "Fréquence coupures (7 jours)",
    "avg_outage_duration_7d": "Durée moy. coupures (7 jours)",
    "outage_trend_7d": "Tendance coupures (7 jours)",
    # ── Temporel ──
    "hour": "Heure de la journée",
    "hour_sin": "Cycle horaire (sin)",
    "hour_cos": "Cycle horaire (cos)",
    "month": "Mois",
    "month_sin": "Cycle mensuel (sin)",
    "month_cos": "Cycle mensuel (cos)",
    "day_of_week": "Jour de la semaine",
    "is_weekend": "Week-end",
    "is_public_holiday": "Jour férié",
    # ── Météo ──
    "temperature_2m": "Température (°C)",
    "relative_humidity_2m": "Humidité relative (%)",
    "dew_point_2m": "Point de rosée (°C)",
    "wind_speed_10m": "Vitesse du vent (km/h)",
    "wind_gusts_10m": "Rafales de vent (km/h)",
    "precipitation": "Précipitations (mm)",
    "surface_pressure": "Pression (hPa)",
    "shortwave_radiation": "Rayonnement solaire (W/m²)",
    "cape": "Énergie convective (CAPE)",
    "weathercode": "Code météo",
    "temp_humidity_interaction": "Interaction temp × humidité",
    "wind_precipitation_interaction": "Interaction vent × pluie",
    "heat_stress": "Stress thermique",
    "cloud_cover_pct": "Couverture nuageuse (%)",
    "visibility_m": "Visibilité (m)",
    "evapotranspiration": "Évapotranspiration",
    "rain_intensity": "Intensité de la pluie",
    "thermal_amplitude_24h": "Amplitude thermique (24h)",
    "humidity_change_3h": "Variation humidité (3h)",
    "pressure_change_3h": "Variation pression (3h)",
    # ── Contexte ──
    "grid_availability_ratio": "Disponibilité réseau",
    "grid_availability_rolling_6h": "Disponibilité réseau (6h)",
    "recent_outages_6h": "Coupures récentes (6h)",
    "recent_outages_24h": "Coupures récentes (24h)",
    "generators_kw": "Générateur (kW)",
    "generator_active": "Générateur actif",
    "generator_ratio": "Part du générateur",
    "cloud_cover": "Couverture nuageuse",
    "visibility": "Visibilité",
    "et0_fao_evapotranspiration": "Évapotranspiration FAO",
}


# Catégorisation des features pour affichage groupé / coloration.
# Chaque catégorie : (nom affiché, emoji, couleur hex, ordre)
FEATURE_CATEGORIES = {
    "energy":   {"label": "Énergie & consommation", "emoji": "🔋", "color": "#3498db"},
    "outage":   {"label": "Historique coupures",    "emoji": "⚡", "color": "#c0392b"},
    "time":     {"label": "Temporel",                "emoji": "🕐", "color": "#9b59b6"},
    "meteo":    {"label": "Météo",                   "emoji": "🌤️", "color": "#e67e22"},
    "other":    {"label": "Autre",                   "emoji": "▫️", "color": "#95a5a6"},
}


def get_feature_category(feat: str) -> str:
    """Retourne la clé de catégorie d'une feature à partir de son nom."""
    if feat in {
        "hours_since_last_outage", "last_outage_duration_h",
        "outage_frequency_7d", "avg_outage_duration_7d", "outage_trend_7d",
        "recent_outages_6h", "recent_outages_24h", "grid_availability_ratio",
        "grid_availability_rolling_6h",
    }:
        return "outage"
    if feat in {
        "hour", "hour_sin", "hour_cos", "month", "month_sin", "month_cos",
        "day_of_week", "is_weekend", "is_public_holiday",
    }:
        return "time"
    if feat in {
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "wind_speed_10m", "wind_gusts_10m", "precipitation",
        "surface_pressure", "shortwave_radiation", "cape", "weathercode",
        "temp_humidity_interaction", "wind_precipitation_interaction",
        "heat_stress", "cloud_cover_pct", "cloud_cover", "visibility_m",
        "visibility", "evapotranspiration", "et0_fao_evapotranspiration",
        "rain_intensity", "thermal_amplitude_24h",
        "humidity_change_3h", "pressure_change_3h",
        "solar_available",
    }:
        return "meteo"
    if feat in {
        "solar_ratio", "solar_pv_kw", "total_load_kw", "sterilization_kw",
        "base_load_kw", "base_load_ratio", "load_rolling_6h",
        "load_rolling_24h", "load_std_24h", "load_diff_1h", "load_diff_24h",
        "load_pct_change_1h", "peak_ratio", "generators_kw",
        "generator_active", "generator_ratio",
    }:
        return "energy"
    return "other"


def feature_label(feat: str) -> str:
    """Retourne le label humain d'une feature, fallback = nom brut."""
    return FEATURE_LABELS.get(feat, feat)


# Sources actives du pipeline (affichage panneau « Sources & facteurs »).
# role :
#   - model_pipeline → familles de features du modèle Lacor (tous les onglets)
#   - context_app → affichées dans l'app mais exclues des features (em_* hors modèle)
DATA_SOURCES = [
    {"name": "Lacor Hospital — consommation 2022 (terrain)",
     "icon": "🏥", "type": "Hospitalier", "role": "model_pipeline",
     "desc": "Seule source avec coupures réelles observées — entraîne les modèles Lacor",
     "key": True},
    {"name": "ERIC NHS (UK)",
     "icon": "🇬🇧", "type": "Hospitalier", "role": "model_pipeline",
     "desc": "10 hôpitaux, profils horaires dérivés des relevés ERIC (comparaison multi-sites)"},
    {"name": "NYC LL84 (USA)",
     "icon": "🇺🇸", "type": "Hospitalier", "role": "model_pipeline",
     "desc": "5 hôpitaux NYC, profils horaires dérivés du registre LL84"},
    {"name": "Open-Meteo Archive",
     "icon": "🌦️", "type": "Météo historique", "role": "model_pipeline",
     "desc": "Météo horaire 2022 (entraînement + analyse historique)"},
    {"name": "Open-Meteo Forecast",
     "icon": "🔮", "type": "Météo prévision", "role": "context_app",
     "desc": "Optionnel — ingest_openmeteo_forecast.py, hors run_pipeline par défaut"},
    {"name": "Electricity Maps API",
     "icon": "⚡", "type": "Réseau zone", "role": "context_app",
     "desc": "Zone résolue au lat/lon hôpital + score risque (réseau + météo) — hors modèle Lacor"},
    {"name": "EskomSePush",
     "icon": "🇿🇦", "type": "Délestage (RSA)", "role": "context_app",
     "desc": "Délestage programmé Cape Town — contexte uniquement (non testable sur Lacor)"},
    {"name": "EAGLE-I (comtés US)",
     "icon": "🌍", "type": "Validation inter-sites", "role": "context_app",
     "desc": "Coupures réseau réelles — expérience leave-one-site-out (onglet Validation)"},
]

# Signaux retirés du projet (preuves conservées pour le rapport)
REMOVED_DATA_SOURCES_NOTE = (
    "Signaux **testés puis retirés** (n'amélioraient pas la prédiction sur Lacor) : "
    "GDELT (médias), qualité de l'air, GDACS, USGS, NOAA Storm. "
    "Voir `models/external_signal_experiment.json`."
)

def source_role_model_pipeline(src: dict) -> bool:
    return src.get("role") == "model_pipeline"


def source_role_context_app(src: dict) -> bool:
    return src.get("role") == "context_app"
