"""
Configuration centralisée du projet.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FEATURES_DIR = ROOT_DIR / "data" / "features"
MODELS_DIR = ROOT_DIR / "models"

# ── API Météo ───────────────────────────────────────────────────────
METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"
METEO_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "surface_pressure",
    "shortwave_radiation",
    "cloud_cover",
    "visibility",
    "et0_fao_evapotranspiration",
    "cape",
    "weathercode",
]

# ── Coordonnées des hôpitaux ────────────────────────────────────────
# Dérivé automatiquement du catalogue UI dans `src/utils/hospitals.py`.
# Utilisé par les ingestions géolocalisées (Open-Meteo, Forecast, Electricity Maps).
from src.utils.hospitals import HOSPITAL_DISPLAY, build_hospital_locations  # noqa: E402

HOSPITAL_LOCATIONS = build_hospital_locations()

# ── Open-Meteo Forecast (prédictions futures pour l'app Streamlit) ──
METEO_FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
METEO_FORECAST_DAYS = 7

# ── Electricity Maps API (charge & mix réseau temps réel) ──────────
# API commerciale (token gratuit pour usage perso/recherche, payant pour
# usage pro). Couverture mondiale, granularité horaire (5 min/15 min sur
# certaines zones).
# Documentation : https://static.electricitymaps.com/api/docs/index.html
#
# Endpoints clés exploités ici (préfixe API_BASE) :
#   /v4/zone                      → résolution lat/lon → zone réseau
#   /v4/total-load/latest         → charge totale instantanée du réseau (MW)
#   /v4/total-load/history        → charge horaire des 24 dernières heures
#   /v4/carbon-intensity/latest   → intensité carbone instantanée (gCO2/kWh)
#   /v4/carbon-intensity/history  → intensité carbone horaire 24 h
#   /v4/electricity-mix/history   → mix de production (renouvelable/fossile)
#
# Pourquoi c'est utile pour la prédiction de coupure :
#   - La charge totale (total_load) reflète le STRESS du réseau local
#     auquel l'hôpital est connecté → forte corrélation avec le risque
#     de délestage et de blackout en cascade.
#   - Le mix de production (% renouvelable, % fossile) traduit la
#     stabilité de l'alimentation : plus de fossile → réseau plus
#     pilotable mais plus cher ; plus de solaire/éolien → variabilité
#     accrue, dépendance météo.
#   - L'intensité carbone est un indicateur composite (mix + flux) qui
#     bouge avant une instabilité (déconnexion d'une centrale, etc.).
ELECTRICITYMAPS_BASE = "https://api.electricitymap.org"
ELECTRICITYMAPS_TOKEN_ENV = "ELECTRICITY_MAPS_TOKEN"

# Mapping hôpital → zone réseau électrique locale (key Electricity Maps).
# Quand un hôpital est dans un pays mono-zone on prend le code ISO2 ; pour
# les pays découpés en sous-réseaux (USA, Canada, Australie…) on prend le
# code du Balancing Authority / ISO le plus proche géographiquement.
#   NYC           → US-NY-NYIS  (NY-ISO)
#   UK NHS        → GB           (réseau National Grid unique pour GB)
#   Lacor         → UG, etc.
HOSPITAL_ELECTRICITY_ZONES = {
    "lacor_uganda":            "UG",
    "kenyatta_kenya":          "KE",
    "tikur_ethiopia":          "ET",
    "groote_schuur_sa":        "ZA",
    "dhaka_bangladesh":        "BD",
    "fann_senegal":            "SN",
    "parirenyatwa_zimbabwe":   "ZW",
    "muhimbili_tanzania":      "TZ",
    "luth_nigeria":            "NG",
    "korle_bu_ghana":          "GH",
    "ibn_sina_morocco":        "MA",
    "kasr_alainy_egypt":       "EG",
    "chuk_rwanda":             "RW",
    "st_thomas_nhs":           "GB",
    "guys_nhs":                "GB",
    "addenbrookes_nhs":        "GB",
    "manchester_nhs":          "GB",
    "kings_college_nhs":       "GB",
    "john_radcliffe_nhs":      "GB",
    "leeds_general_nhs":       "GB",
    "birmingham_heartlands_nhs": "GB",
    "newcastle_rvi_nhs":       "GB",
    "royal_devon_nhs":         "GB",
    "nyc_bellevue":            "US-NY-NYIS",
    "nyc_nyu_tisch":           "US-NY-NYIS",
    "nyc_nyp_brooklyn":        "US-NY-NYIS",
    "nyc_elmhurst":            "US-NY-NYIS",
    "nyc_lincoln":             "US-NY-NYIS",
}

# ── EskomSePush (délestage programmé Afrique du Sud) ────────────────
# Cause DIRECTE des coupures en Afrique du Sud : le délestage (load-shedding)
# d'Eskom et des municipalités est planifié et publié. EskomSePush expose ce
# stade en temps réel + le calendrier à venir.
#   Inscription / token gratuit (50 appels/jour) : https://eskomsepush.gumroad.com/l/api
#   Doc API : https://documenter.getpostman.com/view/1296288/UzQuNk3E
# Header requis : `token: <API_KEY>`.
#
# Honnêteté : ce signal ne concerne QUE les sites sud-africains. Il ne peut pas
# être testé sur Lacor (Ouganda, réseau UMEME) faute de calendrier équivalent.
# Il est donc exposé comme CONTEXTE temps réel dans l'app (pas une feature du
# modèle entraîné sur Lacor).
ESKOM_SEPUSH_BASE = "https://developer.sepush.co.za/business/2.0"
ESKOM_SEPUSH_TOKEN_ENV = "ESKOM_SEPUSH_TOKEN"
# Bloc de statut national pertinent par hôpital (Cape Town publie son propre
# stade, souvent inférieur à celui d'Eskom). Les sites non listés ⇒ pas de
# délestage programmé public exploité.
ESKOM_SEPUSH_STATUS_BLOCK = {
    "groote_schuur_sa": "capetown",
}

# ── EAGLE-I (coupures réelles par comté, USA) ───────────────────────
# EAGLE-I (Environment for Analysis of Geo-Located Energy Information,
# Oak Ridge National Laboratory) publie les coupures d'électricité réelles
# par comté américain : nombre de clients coupés (`customers_out`) toutes
# les 15 min, plus un fichier MCC (Max Customer Count) par comté pour
# normaliser. C'est une SOURCE DE VÉRITÉ TERRAIN multi-sites — utilisée
# pour valider la généralisation inter-sites (cf. multisite_experiment.py)
# et lever la limite « un seul site vérité terrain » (Lacor).
#
# Téléchargement (figshare, libre, ~quelques centaines de Mo/an) :
#   https://doi.org/10.6084/m9.figshare.24237376  (EAGLE-I 2014-2023)
# Déposer les CSV annuels nationaux dans EAGLEI_SOURCE_DIR :
#   data/raw/eaglei_source/eaglei_outages_<YYYY>.csv   (snapshots 15 min)
#   data/raw/eaglei_source/MCC.csv                     (max clients/comté)
# Le module src/data/ingest_eaglei.py filtre les comtés des hôpitaux,
# rééchantillonne à l'heure et écrit data/raw/eaglei_<site>.csv.
EAGLEI_SOURCE_DIR = RAW_DIR / "eaglei_source"
EAGLEI_DOWNLOAD_DOI = "https://doi.org/10.6084/m9.figshare.24237376"
EAGLEI_YEAR = 2023  # année EAGLE-I de référence pour l'expérience multi-sites
# Seuil de binarisation de la cible par site : une heure est « en coupure »
# si le nb de clients coupés dépasse le quantile p90 propre au site
# (⇒ ~10 % de positifs/site, comparable au taux réel de Lacor ≈ 9.7 %).
EAGLEI_OUTAGE_QUANTILE = 0.90

# ── Fichiers de données brutes ──────────────────────────────────────
LACOR_FILE = RAW_DIR / "lacor_hospital.xlsx"

RANDOM_SEED = 42
TEST_SIZE = 0.2

# ── Colonnes exclues du modèle (centralisé) ─────────────────────────
# Réutilisé par `train_baseline.py` (pour `prepare_data`) et par `app.py`
# (pour `get_feature_columns`). Toute modification ici se propage des
# deux côtés.
COLS_TO_DROP = [
    "datetime",
    "is_outage",
    # Colonnes avec fuite directe (connues uniquement pendant la coupure)
    "grid_availability_ratio",
    # grid_available ≈ inverse exact de is_outage (grid_available=1 ⇒ 0 coupure
    # sur 131 362 lignes ; is_outage=1 ⇒ grid_available=0 à 100 %). C'est la
    # cible déguisée : fuite franche, à exclure comme grid_availability_ratio.
    "grid_available",
    "generators_kw",
    "generator_active",
    "generator_ratio",
    "grid_availability_rolling_6h",
    "recent_outages_6h",
    "recent_outages_24h",
    # Legacy : plus calculée depuis juin 2026, peut subsister dans d'anciens CSV
    "storm_risk",
    # Colonnes brutes météo redondantes avec les features dérivées
    "cloud_cover",
    "visibility",
    "et0_fao_evapotranspiration",
    # NOTE : la consommation en kW ABSOLUS (total_load_kw, solar_pv_kw,
    # base_load_kw, sterilization_kw, load_rolling_*, load_std_24h,
    # load_diff_*) est CONSERVÉE comme feature. Elle est le signal le plus
    # prédictif sur Lacor ; le projet assume un modèle pilote MONO-SITE
    # (Lacor), où l'échelle absolue est cohérente. Elle avait été retirée pour
    # tenter une généralisation multi-sites — abandonnée faute de données
    # réelles (conso + coupures subies) sur d'autres hôpitaux.
]

# ── Signaux externes exclus du modèle (cf. #3 train-serve skew) ──────
# La charge réseau Electricity Maps (`em_*`) est INGÉRÉE et affichée comme
# contexte (et sert la prévision temps réel via l'API live), mais reste
# EXCLUE des features du modèle : elle n'est pas disponible à l'échelle du
# site et créerait un décalage entraînement/service. Les autres signaux
# externes historiques (GDELT, GDACS, USGS, qualité de l'air, NOAA) ont été
# retirés du projet : testés, ils n'annonçaient pas les coupures et
# dégradaient le modèle (cf. models/external_signal_experiment.json).
EXTERNAL_SIGNAL_PREFIXES = (
    "em_",
)


def is_external_signal(col: str) -> bool:
    """True si la colonne dérive d'un signal externe exclu du modèle."""
    return col.startswith(EXTERNAL_SIGNAL_PREFIXES)


def drop_external_signal_columns(columns) -> list:
    """Retourne `columns` sans les colonnes de signaux externes."""
    return [c for c in columns if not is_external_signal(c)]

# ── Jours fériés Uganda 2022 (Lacor) ─────────────────────────────────
# Centralisé pour éviter la duplication dans build_features.py et app.py.
UGANDA_PUBLIC_HOLIDAYS_2022 = [
    "2022-01-01",  # New Year
    "2022-01-26",  # NRM Liberation Day
    "2022-02-16",  # Archbishop Janani Luwum Day
    "2022-03-08",  # International Women's Day
    "2022-04-15",  # Good Friday
    "2022-04-18",  # Easter Monday
    "2022-05-01",  # Labour Day
    "2022-05-02",  # Eid al-Fitr (approx.)
    "2022-06-03",  # Martyrs' Day
    "2022-06-09",  # National Heroes' Day
    "2022-07-09",  # Eid al-Adha (approx.)
    "2022-10-09",  # Independence Day
    "2022-12-25",  # Christmas
    "2022-12-26",  # Boxing Day
]

# ── Performance tuning ───────────────────────────────────────────────
# Ces valeurs servent de défauts globaux et peuvent être surchargées
# via CLI dans run_pipeline.py.
CV_FOLDS = 5
FAST_MODE = False
GRID_SCALE = "full"  # "compact" | "full"
SHAP_SAMPLE_SIZE = 5000
