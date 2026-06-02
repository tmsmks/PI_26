# Prédiction de coupures d'électricité en hôpitaux

Projet de data science / machine learning pour **prédire les coupures d'électricité** dans les hôpitaux :
probabilité, moment estimé, durée et causes probables.

## Données d'entraînement

Le pipeline d'entraînement est **multi-hôpitaux** :
- **Lacor Hospital** (Ouganda, historique 15 min rééchantillonné à l'heure) — site de référence
- **ERIC NHS** (UK, profils horaires par hôpital, 10 sites)
- **NYC LL84** (USA, profils horaires bâtiment, 5 sites)
- **Enrichissement contextuel** : météo (Open-Meteo Archive + Forecast) en entrée du modèle ; réseau électrique (Electricity Maps) et délestage (EskomSePush) en **contexte temps réel** dans l’interface (hors features du modèle)

La variable cible est `is_outage` (1 = coupure, 0 = pas de coupure).

> ⚠️ **Seules les coupures de Lacor sont réellement observées.** Les coupures
> ERIC/NYC sont générées par une formule stochastique (profils de consommation
> réalistes mais cible synthétique). C'est pourquoi l'entraînement se fait
> **par défaut sur Lacor uniquement** (`--scope real`) : voir la section
> *Exécution*. Le multi-hôpitaux complet reste disponible via `--scope all`.

## Structure du projet

```
PI_26/
├── data/
│   ├── raw/                  ← données brutes (APIs, Excel, CSV)
│   │   ├── eric/             ← profils horaires ERIC NHS
│   │   └── nyc_ll84/         ← profils horaires NYC LL84
│   ├── processed/            ← hospital_merged.csv (multi-hôpitaux fusionné)
│   └── features/             ← features_dataset.csv (dataset d'entraînement)
├── models/
│   ├── baseline_model.joblib       ← meilleur modèle brut (RF / XGB / LGBM selon la run)
│   ├── calibrated_model.joblib     ← modèle calibré isotonique (utilisé par l'app)
│   ├── shap_explainer.joblib       ← TreeExplainer SHAP
│   ├── shap_values.npz             ← SHAP values du test set
│   ├── shap_feature_importance.csv ← importance SHAP globale
│   ├── feature_importance.csv      ← importance MDI
│   ├── model_comparison.csv        ← tableau comparatif RF / XGB / LGBM
│   ├── training_summary.json       ← hyperparamètres + métriques nowcast
│   └── nowcast_horizons/           ← modèles 1/3/6 h + horizons_summary.json
├── docs/
│   ├── DOCUMENTATION_DONNEES_ET_APIS.md
│   └── DOCUMENTATION_MODELE_ET_PREDICTIONS.md
├── src/
│   ├── data/
│   │   ├── ingest_consumption.py        ← Lacor Hospital (Excel → CSV)
│   │   ├── ingest_eric.py               ← ERIC NHS (10 sites UK, profils horaires)
│   │   ├── ingest_nyc_ll84.py           ← NYC LL84 (5 sites NYC, profils horaires)
│   │   ├── ingest_meteo.py              ← Open-Meteo Archive (météo historique)
│   │   ├── ingest_openmeteo_forecast.py ← Open-Meteo Forecast (prévisions 7 j)
│   │   ├── ingest_electricitymaps.py    ← Electricity Maps (charge & mix réseau)
│   │   └── preprocessing.py             ← rééchantillonnage + fusion multi-hôpitaux
│   ├── features/
│   │   └── build_features.py     ← feature engineering (temporel, charge, météo)
│   ├── models/
│   │   ├── train_baseline.py     ← modèle Lacor (RF/XGB/LGBM + calibration + SHAP)
│   │   ├── train_horizons.py     ← horizons 1/3/6 h (mêmes features, cible future)
│   │   └── backtest.py           ← backtest walk-forward mensuel (optionnel)
│   ├── nowcast_horizons.py       ← inférence « coupure dans les H h »
│   ├── realtime_forecast.py        ← fenêtre temps réel (Electricity Maps)
│   └── utils/
│       ├── config.py             ← configuration centralisée (APIs, hôpitaux)
│       └── io.py                 ← helpers I/O + logging
├── app.py                    ← interface Streamlit (27 hôpitaux, SHAP local)
├── run_pipeline.py           ← exécution complète du pipeline (CLI train|live)
├── requirements.txt          ← dépendances avec versions fixées
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Exécution

```bash
# Pipeline complet (ingestion historique 2022 → features → entraînement + SHAP)
# Par défaut : --scope real (entraînement sur coupures réellement observées)
python run_pipeline.py

# Entraînement multi-hôpitaux complet (inclut coupures synthétiques ERIC/NYC)
python run_pipeline.py --scope all

# Pipeline en mode "live" (fenêtre glissante récente, par défaut 30 jours)
python run_pipeline.py --mode live --window-days 30

# Pipeline rapide pour itération (CV réduit, grille compacte, SHAP échantillonné)
python run_pipeline.py --fast

# Tuning fin (override CV folds, taille SHAP, taille de grille…)
python run_pipeline.py --grid-scale compact --cv-folds 3 --shap-sample-size 2000

# Interface Streamlit
streamlit run app.py
```

Tous les flags CLI sont définis dans `run_pipeline.py` (`--mode {train,live}`, `--window-days`, `--fast`, `--grid-scale {compact,full}`, `--cv-folds`, `--shap-sample-size`, `--no-full-artifacts`, `--scope {real,all}`).

> **Portée d'entraînement (`--scope`)** — `real` (défaut) n'entraîne et n'évalue
> que sur les hôpitaux à coupures **réellement observées** (Lacor). Les sites
> ERIC/NYC ont des coupures **générées par une formule stochastique** : les
> inclure (`--scope all`) revient à entraîner sur ~94 % de bruit et gonfle
> artificiellement le F1 global. Le défaut `real` garantit des métriques honnêtes.

## Pipeline d'entraînement

`run_pipeline.py` orchestre **5 étapes** :

1. **Ingestion** — appelle séquentiellement les scripts `src/data/ingest_*.py`
2. **Preprocessing** ([src/data/preprocessing.py](src/data/preprocessing.py)) — rééchantillonnage Lacor 15 min → 1 h, fusion multi-hôpitaux (météo + contexte réseau Electricity Maps)
3. **Feature engineering** ([src/features/build_features.py](src/features/build_features.py)) — features temporelles cycliques, rolling de charge, interactions météo, historique des coupures
4. **Entraînement nowcast** ([src/models/train_baseline.py](src/models/train_baseline.py)) :
   1. Split temporel 80/20 **par hôpital** (train réordonné chronologiquement pour une vraie CV temporelle)
   2. **GridSearchCV** + **TimeSeriesSplit** (5 folds) pour RF / XGBoost / LightGBM
   3. Comparaison → sélection automatique du meilleur (F1 sur CV)
   4. **Calibration adaptative** (`--calibration auto`) : compare *aucune calibration* / isotonique / sigmoïde sur une validation interne et ne recalibre que si le Brier s'améliore d'une marge nette (un GBM est souvent déjà bien calibré)
   5. Évaluation hold-out (brut + calibré)
   6. **SHAP TreeExplainer** + sauvegarde des artefacts
5. **Horizons 1/3/6 h** ([src/models/train_horizons.py](src/models/train_horizons.py)) — même features que le nowcast, cible « coupure dans les H prochaines heures » → `models/nowcast_horizons/`

> **Contexte réseau exclu du modèle (`config.EXTERNAL_SIGNAL_PREFIXES`)** — la
> charge réseau Electricity Maps (`em_*`) est ingérée et affichée comme contexte
> (et sert la prévision temps réel via l'API live), mais reste **exclue du jeu de
> features du modèle** : elle n'est pas disponible à l'échelle du site et créerait
> un décalage entraînement/service. Le modèle s'appuie donc sur des features
> robustes (météo + charge + temporel + historique des coupures pour le nowcast),
> identiques en entraînement et en service pour tous les hôpitaux.

## Métriques et features importantes

Les métriques exactes du run courant sont écrites par `train_baseline.py` dans `models/training_summary.json` (modèle gagnant, hyperparamètres, F1 CV, accuracy, precision, recall, ROC AUC, Brier brut + calibré).

**Run courante** (`models/training_summary.json`, `--scope real`, **54 features**, signaux `em_*` exclus) :
- **Modèle gagnant : LightGBM** (`n_estimators=200`, `max_depth=15`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=15`)
- Hold-out test (brut = calibré, calibration `none`) : F1 ≈ 0.99 · ROC AUC ≈ 1.00 · Brier ≈ 0.002 · Precision ≈ 0.99 · Recall ≈ 0.99
- **Horizons** (`models/nowcast_horizons/horizons_summary.json`) : F1 walk-forward ≈ 0.72 / 0.70 / 0.70 (1 h / 3 h / 6 h)

> ⚠️ **Ce hold-out est trompeusement élevé** et ne doit PAS être communiqué tel
> quel : il est dominé par les features **autorégressives** de l'historique des
> coupures (`hours_since_last_outage`, `outage_frequency_7d`…), qui rendent la
> prédiction quasi-déterministe quand les coupures sont groupées dans le temps.
> Le **chiffre honnête est le backtest walk-forward ci-dessous**.
>
> Ces chiffres portent **uniquement sur des coupures réellement observées** (Lacor,
> ~9,4 % des heures). Le jeu de features exclut les colonnes à fuite directe
> (`grid_available`, `grid_availability_ratio`, etc.) et conserve la
> **consommation** et l’**historique des coupures** — modèle pilote **mono-site**.

### Validation temporelle (généralisation dans le temps)

Comme on ne dispose que d'**un site × une année**, la robustesse temporelle est
évaluée explicitement par [`src/models/backtest.py`](src/models/backtest.py)
(`python -m src.models.backtest`) — bien plus honnête qu'un hold-out unique :

- **Hold-out chronologique** (train mois 1–9 → test oct–déc) : F1 = 0.90 · ROC AUC = 0.995 · Recall = 0.92 · Brier = 0.015
- **Backtest walk-forward** (origine glissante, 6 folds mensuels) : F1 = **0.83 ± 0.14** [0.64–0.99] · Recall = 0.81 · ROC AUC = **0.96 ± 0.04** · Brier = 0.026

Lecture : la discrimination (ROC AUC) reste élevée toute l'année ; le F1 progresse
fortement avec l'historique disponible (≈0.64 aux premiers mois → ≈0.99 en fin
d'année), ce qui confirme que l'**autorégression des coupures** porte une grande
part du signal. La variance ±0.14 est le prix honnête d'un seul site × une année.
⚠️ Ceci mesure la stabilité **dans le temps sur Lacor** — pas la généralisation à
**d'autres sites**. Détail par mois : `models/backtest_by_month.csv`.

### Généralisation spatiale

Le modèle est **entraîné et validé sur Lacor** (coupures terrain 2022). Pour les
autres hôpitaux de l’interface, le **même modèle** est appliqué à un profil de
consommation et une météo locales : les scores sont **illustratifs**, non validés
inter-sites. Il n’existe pas, à ce jour, de jeu public équivalent « hôpital ×
coupure horaire » pour plusieurs pays ; le clonage de profil sert à comparer des
contextes réseau et climatiques, pas à prétendre à une validation multi-sites.

### Ce que sert l'app

| Site | Modèle servi | Explication |
|---|---|---|
| **Lacor** | nowcast calibré + horizons 1/3/6 h (mêmes features) | proba calibrée + SHAP |
| **Tous les autres** | **même** modèle Lacor, profil de consommation cloné | ⚠️ **score illustratif, non validé** pour ce site |

L'app sert **un seul** modèle (Lacor). Pour un site ≠ Lacor, c'est ce modèle
appliqué à un profil de consommation emprunté : un score **illustratif** et non
une capacité validée — la généralisation inter-sites n'est pas démontrée
(cf. ci-dessus). L'app l'annonce explicitement dans le profil de chaque site.

Les classements de features sont disponibles ici :
- `models/feature_importance.csv` — importance MDI du modèle gagnant
- `models/shap_feature_importance.csv` — importance SHAP globale (|mean|)

Le fichier `calibrated_model.joblib` est chargé par défaut dans l'app. Il contient le **gagnant** de la comparaison RF / XGBoost / LightGBM, avec la stratégie de calibration retenue (`calibration_method` dans `training_summary.json`).

## Sources de données

| Source | Type | Granularité | Utilisation |
|---|---|---|---|
| Lacor Hospital (Zenodo) | Excel | 15 min → horaire | Dataset principal + cible `is_outage` |
| NHS ERIC 2022-23 | Statistiques publiées + profils dérivés | horaire | 10 hôpitaux anglais |
| NYC LL84 | CSV mensuel + profils horaires dérivés | horaire | 5 hôpitaux NYC |
| Open-Meteo Archive | API publique | horaire | Météo historique 2022 |
| Open-Meteo Forecast | API publique | horaire (7 j) | Prévisions pour le mode live / app |
| Electricity Maps | API token | horaire | Charge & mix réseau de zone (contexte) |
| EskomSePush | API token | temps réel | Stade de délestage RSA (contexte, sites sud-africains) |

### Données ERIC NHS

Les données [ERIC (Estates Returns Information Collection)](https://digital.nhs.uk/data-and-information/publications/statistical/estates-returns-information-collection) sont une collecte annuelle obligatoire des NHS Trusts en Angleterre. Le script [src/data/ingest_eric.py](src/data/ingest_eric.py) génère des profils horaires réalistes (8 760 h/hôpital) calibrés sur les statistiques publiées ERIC 2022-23 pour 10 hôpitaux NHS.

## Interface Streamlit

L'application [app.py](app.py) propose :
- **27 hôpitaux** sélectionnables (Lacor, 10 NHS ERIC, 5 NYC LL84, 12 profils `africa_grid`)
- **Bandeau réseau temps réel** (Electricity Maps) : zone, charge MW, carbone, mix, consommation estimée
- **Prochaine coupure (24 h)** : probabilité de coupure dans les 1 / 3 / 6 h (modèles horizons + mode temps réel)
- **Analyse historique** : période au choix, probabilité horaire + SHAP waterfall
- **Prévisions J+7** : trajectoire de risque sur 7 jours (Open-Meteo Forecast)
- **Simulation manuelle** : 13 paramètres (temporel, énergie, météo) + jauge de risque + SHAP
- **Ajustement par profil** : adaptation au réseau électrique de chaque hôpital (fiabilité OMS estimée + stabilité du réseau, voir `adjust_for_hospital_profile`)
- **Garde-fou features** : détection automatique d'une désynchronisation entre le dataset (`features_dataset.csv`) et le modèle entraîné (`feature_names_in_`)
- **Gestion d'erreurs** : messages explicatifs si le modèle ou les données sont manquants

## Hôpitaux couverts

L'app Streamlit propose les hôpitaux **réels** : Lacor + 10 NHS ERIC + 5 NYC LL84, plus les profils `africa_grid` (charge estimée pilotée par le réseau live Electricity Maps).

| Catégorie | Nb | Hôpitaux |
|---|---|---|
| Référence (terrain) | 1 | Lacor (Ouganda) |
| ERIC NHS (UK) | 10 | St Thomas', Guy's, John Radcliffe, Addenbrooke's, Manchester Royal, Leeds General, Birmingham Heartlands, Royal Victoria Newcastle, Royal Devon, King's College |
| NYC LL84 (USA) | 5 | Bellevue, NYU Tisch, NYP Brooklyn Methodist, Elmhurst, Lincoln |
| Profils estimés `africa_grid` | 12 | Kenyatta, Tikur Anbessa, Groote Schuur, Dhaka, Fann, Parirenyatwa, Muhimbili, LUTH Lagos, Korle Bu, Ibn Sina, Kasr Al Ainy, CHUK Kigali |

Le fichier [src/utils/config.py](src/utils/config.py) référence les coordonnées des sites (`HOSPITAL_LOCATIONS`) utilisés par les ingestions géo-localisées (Open-Meteo, Electricity Maps). Les profils `africa_grid` clonent le profil temporel de Lacor mis à l'échelle (`avg_load_kw`) puis y injectent météo locale + Electricity Maps.

## Facteurs utilisés (features)

Le dataset de features compte environ 76 colonnes après engineering ; le modèle en utilise **54** numériques, sélectionnées via `COLS_TO_DROP` dans `src/utils/config.py`.

Familles de facteurs effectivement utilisées :

- **Charge/énergie** : `total_load_kw`, `solar_pv_kw`, `base_load_kw`, `load_rolling_*`, `load_diff_*`, `solar_ratio`, `peak_ratio`, `base_load_ratio`, etc.
- **Historique coupures** : `hours_since_last_outage`, `last_outage_duration_h`, `outage_frequency_7d`, `avg_outage_duration_7d`, `outage_trend_7d` (toutes calculées avec un `shift(1)` par hôpital pour éviter le leakage)
- **Temporels** : `hour`, `day_of_week`, `month`, `is_weekend`, `is_public_holiday`, encodages cycliques (`hour_sin/cos`, `month_sin/cos`)
- **Météo** : température, humidité, point de rosée, vent (vitesse + rafales), pluie, pression, rayonnement, CAPE, weathercode + interactions (`temp_humidity_interaction`, `wind_precipitation_interaction`, `heat_stress`, `solar_available`)
- **Météo avancée** : `cloud_cover_pct`, `visibility_m`, `evapotranspiration`, `rain_intensity`, `thermal_amplitude_24h`, `humidity_change_3h`, `pressure_change_3h`
- **Contexte réseau (hors modèle)** : variables Electricity Maps `em_*` (zone, charge MW, intensité carbone gCO₂/kWh, % renouvelable / fossile / bas carbone) — ingérées et affichées, mais exclues du jeu de features.

> **Horizons 1/3/6 h** ([src/models/train_horizons.py](src/models/train_horizons.py)) : même jeu de features que le nowcast, cible = coupure dans les H prochaines heures. Entraînés automatiquement à l’étape 5 de `run_pipeline.py`.

## Temps réel : périmètre exact

Le système n'est **pas** un flux streaming strict (seconde par seconde). Il fonctionne en :

- **Mode `train`** : données historiques (principalement 2022).
- **Mode `live`** : fenêtre glissante récente (`--window-days`) avec rafraîchissement par appels API.

On parle donc de **quasi temps réel** / **near real-time** : données récentes agrégées par pas horaire, pas de streaming continu.
