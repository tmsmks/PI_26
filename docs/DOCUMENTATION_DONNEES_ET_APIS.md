# Documentation des données et APIs

Ce document décrit le pipeline de données orchestré par `run_pipeline.py`
(modes `train` et `live`) : sources ingérées, fusion multi-hôpitaux, jeu de
features et artefacts de modélisation.

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Source 1 — Lacor Hospital (dataset principal de la cible)](#2-source-1--lacor-hospital-dataset-principal-de-la-cible)
3. [Source 2 — NHS ERIC (10 hôpitaux UK)](#3-source-2--nhs-eric-10-hôpitaux-uk)
4. [Source 3 — NYC LL84 (5 hôpitaux USA)](#4-source-3--nyc-ll84-5-hôpitaux-usa)
5. [Source 4 — Open-Meteo (Archive + Forecast)](#5-source-4--open-meteo-archive--forecast)
6. [Source 5 — Electricity Maps (réseau local)](#6-source-5--electricity-maps-réseau-local)
7. [Source 6 — EskomSePush (délestage RSA, contexte app)](#7-source-6--eskomsepush-délestage-rsa-contexte-app)
8. [Schéma de fusion des données](#8-schéma-de-fusion-des-données)
9. [Dictionnaire des variables](#9-dictionnaire-des-variables)
10. [Modes train / live et fenêtres temporelles](#10-modes-train--live-et-fenêtres-temporelles)

---

## 1. Vue d'ensemble

Le pipeline agrège **6 familles de sources actives** pour prédire les coupures
d'électricité dans les hôpitaux. La cible (`is_outage`) provient exclusivement
du dataset **Lacor Hospital** (relevés terrain). Les autres sources
construisent des profils de charge comparés (ERIC, NYC) et le contexte météo /
réseau.

| Catégorie | Sources principales | Usage |
|-----------|---------------------|-------|
| Consommation hospitalière | Lacor (terrain), ERIC NHS (UK), NYC LL84 (USA) | Variable cible (Lacor) + profils de charge (16 sites) |
| Météorologie historique | Open-Meteo Archive | Features du modèle (nowcast + horizons) |
| Météorologie prévisionnelle | Open-Meteo Forecast | Onglet « Prévisions J+7 » + fenêtre temps réel |
| Réseau électrique local | Electricity Maps API | Contexte + estimation conso (`africa_grid`) ; colonnes `em_*` exclues du modèle |
| Délestage programmé (RSA) | EskomSePush API | Contexte temps réel (Groote Schuur) ; hors modèle Lacor |

Le pipeline s'exécute via `run_pipeline.py` (CLI : `--mode {train,live}`,
`--window-days`, `--fast`, `--grid-scale`, `--cv-folds`,
`--shap-sample-size`, `--no-full-artifacts`, `--scope`).

```
ingest_consumption  →  ingest_meteo  →  ingest_eric  →  ingest_nyc_ll84
        ↓                ↓                ↓                  ↓
        ingest_openmeteo_forecast  →  ingest_electricitymaps
                            ↓
                  preprocessing (fusion multi-hôpitaux)
                            ↓
                  build_features (feature engineering)
                            ↓
                  train_baseline (nowcast) → train_horizons (1/3/6 h)
```

Les modèles **horizons 1/3/6 h** (onglet « Prochaine coupure », mêmes features
que le nowcast) sont entraînés à l’étape 5 de `run_pipeline.py` via
`src/models/train_horizons.py` (voir
[`DOCUMENTATION_MODELE_ET_PREDICTIONS.md`](DOCUMENTATION_MODELE_ET_PREDICTIONS.md)).

Toutes les ingestions sont enveloppées dans un `try/except` : si une source
externe est indisponible, le pipeline continue avec les autres signaux et
l'entraînement final ne plante pas.

---

## 2. Source 1 — Lacor Hospital (dataset principal de la cible)

### Description

Le dataset principal est celui du **St. Mary's Hospital Lacor** situé à
Gulu, dans le nord de l'Ouganda. C'est un hôpital de 482 lits alimenté par
un mix réseau / solaire / générateur diesel. **C'est la seule source qui
contient la variable cible `is_outage`** ; tous les autres hôpitaux sont
donc utilisés en train comme contexte de profil de charge.

### Métadonnées

| Attribut | Valeur |
|----------|--------|
| **Source** | Zenodo |
| **DOI** | `10.5281/zenodo.7466652` |
| **Format** | Excel (.xlsx), feuille "Sheet1" |
| **Résolution** | 15 minutes |
| **Période** | 1er janvier 2022 → 31 décembre 2022 |
| **Volume brut** | 35 040 lignes × 7 colonnes |
| **Volume horaire** | 8 760 lignes × 8 colonnes |
| **Taux de coupures (horaire)** | ~9.7 % des heures |
| **Fichier local** | `data/raw/lacor_hospital.xlsx` |
| **Script** | `src/data/ingest_consumption.py` |

### Colonnes brutes

| Colonne originale | Colonne renommée | Type | Description |
|-------------------|-----------------|------|-------------|
| `Unnamed: 0` | `datetime` | datetime | Horodatage (15 min) |
| `Solar PV kW` | `solar_pv_kw` | float | Production solaire photovoltaïque (kW) |
| `Total load kW` | `total_load_kw` | float | Consommation électrique totale (kW) |
| `Generators kW` | `generators_kw` | float | Production des générateurs diesel (kW) |
| `Sterilization kW` | `sterilization_kw` | float | Consommation stérilisation (kW) |
| `Base load kW` | `base_load_kw` | float | Charge de base (kW) |
| `Grid avail` | `grid_available` | int (0/1) | 1 = réseau disponible, 0 = coupure |

### Variable cible

```
is_outage = 1 - grid_available
```

- `is_outage = 1` → coupure de réseau en cours
- `is_outage = 0` → réseau fonctionnel

### Rééchantillonnage 15 min → 1 h

Effectué par `src/data/preprocessing.py` (`resample_lacor_hourly`) :

| Variable | Règle d'agrégation horaire |
|----------|----------------------------|
| `solar_pv_kw`, `total_load_kw`, `generators_kw`, `sterilization_kw`, `base_load_kw` | Moyenne |
| `grid_available` | Moyenne → renommée `grid_availability_ratio` (∈ [0, 1]) |
| `is_outage` | Max → 1 si au moins une coupure dans l'heure |

Résultat : **8 760 lignes horaires**.

---

## 3. Source 2 — NHS ERIC (10 hôpitaux UK)

### Description

**ERIC** (Estates Returns Information Collection) est une collecte annuelle
**obligatoire** de tous les NHS Trusts en Angleterre. Elle contient les
données d'utilités (électricité, gaz, eau), les coûts et la surface de
chaque site hospitalier.

| Attribut | Valeur |
|----------|--------|
| **Source** | NHS England Digital |
| **URL officielle** | https://digital.nhs.uk/data-and-information/publications/statistical/estates-returns-information-collection |
| **Édition de référence** | ERIC 2022-23 (publiée le 14 décembre 2023) |
| **Couverture** | ~1 200 sites hospitaliers en Angleterre |
| **Énergie totale NHS** | 11.1 TWh (2022-23) |
| **Coût moyen électricité** | ~£115/MWh |
| **Script** | `src/data/ingest_eric.py` |
| **Répertoire local** | `data/raw/eric/` |

### Stratégie d'accès

Le site NHS Digital bloque les accès programmatiques (HTTP 403). Le script
`ingest_eric.py` adopte donc une double stratégie :

1. **Si `data/raw/eric/eric_site_level.csv` existe** : chargement direct.
2. **Sinon** : génération d'un dataset réaliste à partir des statistiques
   agrégées publiées (consommations annuelles + ratios kWh/m² ERIC).

### Hôpitaux référencés (10 sites)

| Site | Code | Trust | Ville | Lits | Surface (m²) | Électricité (GWh/an) |
|------|------|-------|-------|------|---------------|----------------------|
| St Thomas' Hospital | RJ121 | Guy's & St Thomas' | London | 840 | 150 000 | 82 |
| Guy's Hospital | RJ122 | Guy's & St Thomas' | London | 400 | 82 000 | 48 |
| John Radcliffe Hospital | RTH01 | Oxford Uni. Hospitals | Oxford | 832 | 120 000 | 62 |
| Addenbrooke's Hospital | RGT01 | Cambridge Uni. Hospitals | Cambridge | 1 000 | 160 000 | 78 |
| Manchester Royal Infirmary | R0A01 | Manchester Uni. | Manchester | 752 | 115 000 | 58 |
| Leeds General Infirmary | RR801 | Leeds Teaching | Leeds | 700 | 100 000 | 52 |
| Birmingham Heartlands | RQ301 | Uni. Hospitals Birmingham | Birmingham | 660 | 95 000 | 46 |
| Royal Victoria Infirmary | RA701 | Newcastle Hospitals | Newcastle | 900 | 130 000 | 68 |
| Royal Devon & Exeter | RA401 | Royal Devon Uni. | Exeter | 600 | 80 000 | 38 |
| King's College Hospital | RXH01 | King's College | London | 950 | 140 000 | 72 |

### Génération des profils horaires (8 760 h × 10 sites)

Le script génère **8 760 heures** de données par hôpital en modélisant :

| Composante | Formule | Description |
|------------|---------|-------------|
| **Cycle journalier** | `0.85 + 0.15 × sin(π(h−7)/13)` si 7h-20h, sinon `0.60 + 0.10 × sin(πh/24)` | Pic 10h-14h, creux nocturne |
| **Saisonnalité** | `1.0 + 0.15 × cos(2π(m−1)/12)` | Consommation plus haute en hiver (chauffage UK) |
| **Week-end** | `× 0.82` si samedi/dimanche | Réduction d'activité |
| **Bruit** | `N(1.0, 0.05)` | Variabilité stochastique |
| **Coupures UK** | `P(outage) ≈ 0.0016 × peak_stress × winter_stress` | Fiabilité 99.5%, pic en hiver |

Variables générées : `datetime`, `total_load_kw`, `solar_pv_kw`,
`base_load_kw`, `sterilization_kw`, `is_outage`, `grid_available`,
`generators_kw`, `site_code`, `site_name`.

> ⚠️ Le `is_outage` synthétique sur les sites NHS est volontairement très
> peu fréquent (≈ 0.5 %). En pratique, la cible utile pour l'entraînement
> reste celle de Lacor.

---

## 4. Source 3 — NYC LL84 (5 hôpitaux USA)

### Description

**NYC Local Law 84** est une obligation déclarative de la ville de New York
imposant aux propriétaires de bâtiments > 25 000 ft² de publier leur
consommation d'énergie annuelle. Le dataset est public sur OpenData NYC
(`data.cityofnewyork.us`, dataset `5zyy-y8am`, ~120 hôpitaux NYC publiés).

| Attribut | Valeur |
|----------|--------|
| **Source** | OpenData NYC — LL84 Energy & Water Data |
| **Script** | `src/data/ingest_nyc_ll84.py` |
| **Répertoire local** | `data/raw/nyc_ll84/` |
| **Volume** | 8 760 h × 5 sites |

### Hôpitaux référencés (5 sites)

| Site | Code interne | Surface (m²) | Conso annuelle (kWh) |
|------|--------------|--------------|----------------------|
| Bellevue Hospital Center | `nyc_bellevue` | 211 475 | 52 960 248 |
| NYU Langone Tisch Hospital | `nyc_nyu_tisch` | 64 040 | 45 139 152 |
| NewYork-Presbyterian Brooklyn Methodist | `nyc_nyp_brooklyn` | 126 587 | 32 396 762 |
| Elmhurst Hospital Center | `nyc_elmhurst` | 89 366 | 30 507 199 |
| Lincoln Medical Center | `nyc_lincoln` | 110 874 | 31 236 421 |

### Désagrégation horaire

Le script applique un profil horaire NYC-spécifique (climatisation Con
Edison, pic estival) à la consommation annuelle déclarée pour produire
8 760 lignes par hôpital. La météo locale (Open-Meteo) est jointe ensuite
côté app via `load_nyc_features`.

---

## 5. Source 4 — Open-Meteo (Archive + Forecast)

### 5.1. Open-Meteo Archive (météo historique)

| Attribut | Valeur |
|----------|--------|
| **URL** | `https://archive-api.open-meteo.com/v1/archive` |
| **Authentification** | Aucune (clé optionnelle) |
| **Rate limit** | 10 000 requêtes/jour (gratuit) |
| **Script** | `src/data/ingest_meteo.py` |
| **Fichier de sortie** | `data/raw/meteo_<hospital_key>.csv` |

13 variables horaires demandées (`METEO_HOURLY_VARS` dans
`src/utils/config.py`) :

```
temperature_2m, relative_humidity_2m, dew_point_2m,
wind_speed_10m, wind_gusts_10m, precipitation,
surface_pressure, shortwave_radiation, cloud_cover,
visibility, et0_fao_evapotranspiration, cape, weathercode
```

### 5.2. Open-Meteo Forecast (prévisions J+7)

| Attribut | Valeur |
|----------|--------|
| **URL** | `https://api.open-meteo.com/v1/forecast` |
| **Horizon** | `METEO_FORECAST_DAYS = 7` jours |
| **Script** | `src/data/ingest_openmeteo_forecast.py` |
| **Fichier de sortie** | `data/raw/meteo_forecast_<hospital_key>.csv` |
| **Usage** | Onglet « Prévisions J+7 » de l'app Streamlit |

### Sites interrogés

**28 sites** géolocalisés dans `HOSPITAL_LOCATIONS` (`ingest_geo: True`) :
Lacor, 12 sites `africa_grid` (dont 1 masqué dans l'UI), 10 NHS ERIC, 5 NYC LL84.
L'app affiche **27 hôpitaux** (`ui_hidden` exclu). Un fichier `meteo_<hospital_key>.csv`
par site ingéré.

### Fusion avec les données de consommation

Côté preprocessing, la jointure se fait par hôpital sur l'horodatage
exact (résolution horaire alignée). Côté app (mode live / forecast), la
jointure se fait via `pd.merge_asof` avec une tolérance de 24 h.

---

## 6. Source 5 — Electricity Maps (réseau local)

API commerciale (token gratuit pour usage perso/recherche, payant pour
usage pro). Couverture mondiale, granularité horaire.

| Attribut | Valeur |
|----------|--------|
| **URL** | `https://api.electricitymap.org` |
| **Token** | Variable d'env. `ELECTRICITY_MAPS_TOKEN` |
| **Script** | `src/data/ingest_electricitymaps.py` (`run` train + `run_live` live) |
| **Fichier de sortie** | `data/raw/electricitymaps_<hospital_key>.csv` |

### Endpoints exploités

| Endpoint | Usage |
|----------|-------|
| `/v4/zone` | Résolution lat/lon → zone réseau |
| `/v4/total-load/latest` | Charge totale instantanée (MW) |
| `/v4/total-load/history` | Charge horaire des 24 dernières heures |
| `/v4/carbon-intensity/latest` | Intensité carbone (gCO₂/kWh) |
| `/v4/carbon-intensity/history` | Intensité carbone horaire 24 h |
| `/v4/electricity-mix/history` | Mix de production (renouv. / fossile) |

### Mapping zone par hôpital (`HOSPITAL_ELECTRICITY_ZONES`)

| Hôpital | Zone Electricity Maps |
|---------|------------------------|
| Lacor | `UG` |
| Kenyatta | `KE` |
| Tikur Anbessa | `ET` |
| Groote Schuur | `ZA` |
| Dhaka | `BD` |
| Fann | `SN` |
| Parirenyatwa | `ZW` |
| Muhimbili | `TZ` |
| LUTH | `NG` |
| Korle Bu | `GH` |
| Ibn Sina | `MA` |
| Kasr Al Ainy | `EG` |
| CHUK | `RW` |
| NHS UK (5 sites) | `GB` |

> Les sites NYC LL84 ne sont pas dans `HOSPITAL_ELECTRICITY_ZONES` et
> n'ont donc pas de fichier `electricitymaps_*.csv`. L'app gère ce cas en
> affichant un message « Electricity Maps non disponible ».

### Colonnes finales

Préfixées `em_*` :

| Colonne | Description |
|---------|-------------|
| `em_zone` | Code zone (string) |
| `em_total_load_mw` | Charge totale instantanée (MW) |
| `em_carbon_intensity_gco2_kwh` | Intensité carbone (gCO₂/kWh) |
| `em_renewable_pct` | % renouvelable du mix |
| `em_fossil_pct` | % fossile du mix |
| `em_low_carbon_pct` | % bas carbone |

Le bandeau temps réel de l'app (« État réseau local ») affiche les 24 h
glissantes : MW courants, stress vs moyenne 24 h, intensité carbone, mix.
Il propose aussi une **estimation de la conso hôpital** :

```
hospital_load_kw_est = avg_load_kw × (em_total_load_mw_now / em_total_load_mw_avg_24h)
```

> Les colonnes `em_*` restent dans `features_dataset.csv` après fusion mais sont
> **exclues du modèle** via `EXTERNAL_SIGNAL_PREFIXES = ("em_",)` dans
> `src/utils/config.py` (décalage entraînement/service si utilisées comme features).

---

## 7. Source 6 — EskomSePush (délestage RSA, contexte app)

Cause **directe** des coupures en Afrique du Sud : le délestage programmé
(load-shedding) d'Eskom et des municipalités est publié à l'avance.

| Attribut | Valeur |
|----------|--------|
| **URL** | `https://developer.sepush.co.za/business/2.0` |
| **Token** | Variable d'env. `ESKOM_SEPUSH_TOKEN` (inscription gratuite, quota journalier) |
| **Module** | `src/loadshedding.py` (appelé par `src/app_data.load_loadshedding`) |
| **Couverture app** | `groote_schuur_sa` → bloc `capetown` (`ESKOM_SEPUSH_STATUS_BLOCK`) |

**Honnêteté méthodologique :** ce signal ne peut pas être validé sur Lacor
(Ouganda, réseau UMEME, pas de calendrier équivalent). Il est donc affiché
comme **contexte temps réel** dans l'onglet « Prochaine coupure », et **n'est
pas** une feature du modèle entraîné sur Lacor.

**Note sur `cape` :** la variable est fournie par Open-Meteo Archive mais
reste **nulle** pour Gulu en ERA5 2022 ; le panneau « contexte orage » de
l'app affiche le CAPE des **prévisions** Open-Meteo Forecast (temps réel).

---

## 8. Schéma de fusion des données

```
        Bases hospitalières horaires (Lacor + ERIC NHS + NYC LL84)
                                │
                                ▼
                    ┌───────────────────────────────┐
                    │ Enrichissement par hôpital    │
                    │ (météo + réseau EM)           │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            Open-Meteo (historique)        Electricity Maps (`em_*`)
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                  `data/processed/hospital_merged.csv`
                                    │
                                    ▼
                  `data/features/features_dataset.csv`
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
         `train_baseline` (nowcast)      `train_horizons` (1/3/6 h)
                    │                               │
                    ▼                               ▼
        `models/calibrated_model.joblib`   `models/nowcast_horizons/`
```

### Types de jointure

| Jointure | Type | Clé | Tolérance |
|----------|------|-----|-----------|
| Lacor (15 min) → horaire | Resample temporel | `datetime` | — |
| Hôpital + Météo / EM | `merge_asof` | `datetime` | ±1 h |

### Volumétrie indicative

| Source | Lignes | Hôpitaux concernés |
|--------|--------|---------------------|
| Lacor (horaire) | 8 760 | 1 |
| ERIC NHS (horaire) | 87 600 | 10 |
| NYC LL84 (horaire) | 43 800 | 5 |
| **Dataset final fusionné** | **~140 160** | **16 hôpitaux** (entraînement multi-sources) |

Le dataset de features compte **≈ 76 colonnes**
(météo + charge + `em_*` + historique coupures). Le nowcast utilise **54
features** numériques (`train_baseline`, scope `real`) ; les horizons 1/3/6 h
réutilisent **le même jeu** (`train_horizons.py`, étape 5 du pipeline).

---

## 9. Dictionnaire des variables

### Variables brutes (après preprocessing)

| Variable | Source | Type | Unité | Description |
|----------|--------|------|-------|-------------|
| `datetime` | toutes | datetime | — | Horodatage horaire |
| `hospital` | toutes | string | — | Clé hôpital (`lacor_uganda`, `st_thomas_nhs`, …) |
| `total_load_kw` | Lacor / ERIC / NYC | float | kW | Consommation totale |
| `solar_pv_kw` | Lacor / ERIC | float | kW | Production solaire |
| `generators_kw` | Lacor | float | kW | Production générateurs |
| `sterilization_kw` | Lacor / ERIC | float | kW | Consommation stérilisation |
| `base_load_kw` | Lacor / ERIC / NYC | float | kW | Charge de base |
| `grid_availability_ratio` | Lacor | float | [0,1] | Fraction de l'heure avec réseau |
| `is_outage` | Lacor (réelle) / ERIC (synthétique) | int | 0/1 | **Variable cible** |
| `temperature_2m` | Open-Meteo | float | °C | Température |
| `relative_humidity_2m` | Open-Meteo | float | % | Humidité relative |
| `dew_point_2m` | Open-Meteo | float | °C | Point de rosée |
| `wind_speed_10m` | Open-Meteo | float | km/h | Vent à 10 m |
| `wind_gusts_10m` | Open-Meteo | float | km/h | Rafales à 10 m |
| `precipitation` | Open-Meteo | float | mm | Précipitations |
| `surface_pressure` | Open-Meteo | float | hPa | Pression au sol |
| `shortwave_radiation` | Open-Meteo | float | W/m² | Rayonnement solaire |
| `cloud_cover` | Open-Meteo | float | % | Couverture nuageuse |
| `visibility` | Open-Meteo | float | m | Visibilité |
| `et0_fao_evapotranspiration` | Open-Meteo | float | mm | Évapotranspiration FAO |
| `cape` | Open-Meteo | float | J/kg | Énergie convective (souvent 0 en archive Gulu) |
| `weathercode` | Open-Meteo | int | — | Code météo WMO |
| `em_*` | Electricity Maps | mixte | — | Charge MW, mix, carbone (**contexte**, exclu du modèle) |

### Variables dérivées (features engineering)

→ Voir [`DOCUMENTATION_MODELE_ET_PREDICTIONS.md`](DOCUMENTATION_MODELE_ET_PREDICTIONS.md) pour le détail des
features réellement utilisées à l'entraînement (liste calculée
dynamiquement via `COLS_TO_DROP` dans `src/models/train_baseline.py`).

---

## 10. Modes train / live et fenêtres temporelles

`run_pipeline.py` expose deux modes :

### Mode `train` (défaut)

```bash
python run_pipeline.py
# ou explicite
python run_pipeline.py --mode train
```

- Météo Open-Meteo Archive : année 2022 entière
- Open-Meteo Forecast : prévisions 7 j par hôpital
- Electricity Maps : ingestion historique complète (`run`)

### Mode `live`

```bash
python run_pipeline.py --mode live --window-days 30
```

- Météo Open-Meteo : `[today − window_days, today]`
- Electricity Maps : appel `run_live(window_hours = window_days × 24)`
- L'ingestion de consommation Lacor reste sur le fichier 2022 (pas de
  flux temps réel public pour cet hôpital).

### Note importante

Le système n'est **pas** un flux streaming strict. Il fonctionne en :

- **train** : données historiques (principalement 2022)
- **live** : fenêtre glissante récente avec rafraîchissement par appels API

On parle donc de **quasi temps réel / near real-time** : données récentes
agrégées par pas horaire, pas de streaming continu seconde par seconde.

### Paramètres hors pipeline

La **fiabilité électrique OMS** par pays est codée dans `src/utils/hospitals.py`
(ajustement d'affichage par profil d'hôpital). Elle n'est pas ingérée comme
fichier CSV séparé.
