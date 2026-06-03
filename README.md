# Prédiction de coupures d'électricité en hôpitaux

Projet de data science / machine learning pour **prédire les coupures d'électricité** dans les hôpitaux :
probabilité, moment estimé, durée et causes probables.

## Données d'entraînement

Le pipeline d'entraînement est **multi-hôpitaux** :
- **Lacor Hospital** (Ouganda, historique 15 min rééchantillonné à l'heure) — site de référence
- **ERIC NHS** (UK, profils horaires par hôpital, 10 sites)
- **NYC LL84** (USA, profils horaires bâtiment, 5 sites)
- **Enrichissement** : météo Open-Meteo Archive ; EAGLE-I comté (NYC) ; **Electricity Maps** (zone au GPS + risque live dans l’app) si `ELECTRICITY_MAPS_TOKEN` ; Forecast / Eskom optionnels

La variable cible est `is_outage` (1 = coupure, 0 = pas de coupure).

> ⚠️ **Coupures terrain** : Lacor (hôpital). NYC : **EAGLE-I comté** si ingéré
> (réseau borough, pas compteur). ERIC : coupures **simulées** à l'ingestion.
> Entraînement **par défaut sur Lacor seul** (`--scope real`) — voir *Exécution*.
> Multi-hôpitaux complet : `--scope all`.

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
│   │   ├── ingest_eaglei.py             ← EAGLE-I comté (NYC + opt-in multisite)
│   │   ├── eaglei_outages.py            ← fusion EAGLE-I → profils NYC
│   │   ├── ingest_openmeteo_forecast.py ← optionnel (prévisions 7 j)
│   │   ├── ingest_electricitymaps.py    ← zone GPS + charge/mix réseau (pipeline + app)
│   │   └── preprocessing.py             ← rééchantillonnage + fusion multi-hôpitaux
│   ├── features/
│   │   └── build_features.py     ← feature engineering (temporel, charge, météo)
│   ├── models/
│   │   ├── train_baseline.py     ← modèle Lacor (RF/XGB/LGBM + calibration + SHAP)
│   │   ├── train_horizons.py     ← horizons 1/3/6 h (mêmes features, cible future)
│   │   └── backtest.py           ← backtest walk-forward mensuel (optionnel)
│   ├── nowcast_horizons.py       ← inférence « coupure dans les H h »
│   ├── realtime_forecast.py        ← variante API horizons (optionnel ; UI → grid_outage_risk)
│   └── utils/
│       ├── config.py             ← configuration centralisée (APIs, hôpitaux)
│       ├── hospitals.py          ← catalogue, `get_target_source`, EAGLE-I comté
│       ├── public_holidays.py    ← fériés 2022 UGA / GBR / USA par site
│       ├── local_signals.py      ← profil « données vraiment locales »
│       ├── em_zone.py            ← résolution zone EM fine (lat/lon)
│       └── io.py                 ← helpers I/O + logging
│   ├── grid_outage_risk.py       ← score risque réseau + météo (hors modèle ML)
│   └── ui_components.py          ← jauge risque, SHAP, panneau EM live
├── scripts/
│   └── prune_unused_data.py    ← nettoyage disque (bruts EAGLE-I, EM, forecast…)
├── app.py                    ← interface Streamlit (3 onglets, 27 hôpitaux)
├── src/app_data.py           ← chargements cachés (modèles, données, résumés)
├── src/ui_content.py         ← libellés features, catalogue des sources
├── src/ui_components.py      ← cartes risque, waterfall SHAP
├── run_pipeline.py           ← exécution complète du pipeline (CLI train|live)
├── requirements.txt          ← dépendances avec versions fixées
└── README.md
```

## Documentation

| Document | Contenu |
|----------|---------|
| [README.md](README.md) | Vue d'ensemble, installation, exécution, UI |
| [docs/DOCUMENTATION_DONNEES_ET_APIS.md](docs/DOCUMENTATION_DONNEES_ET_APIS.md) | Sources, ingestion, fusion, artefacts data |
| [docs/DOCUMENTATION_MODELE_ET_PREDICTIONS.md](docs/DOCUMENTATION_MODELE_ET_PREDICTIONS.md) | Features, entraînement, calibration, onglets Streamlit |
| [docs/CDC_UNIFIE_MSE.md](docs/CDC_UNIFIE_MSE.md) | Cahier des charges unifié (périmètre projet) |

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

# Validation généralisation inter-sites (EAGLE-I, leave-one-site-out)
python run_pipeline.py --multisite

# Interface Streamlit
streamlit run app.py
```

Tous les flags CLI sont définis dans `run_pipeline.py` (`--mode {train,live}`, `--window-days`, `--fast`, `--grid-scale {compact,full}`, `--cv-folds`, `--shap-sample-size`, `--no-full-artifacts`, `--scope {real,all}`, `--multisite`).

> **Portée d'entraînement (`--scope`)** — `real` (défaut) n'entraîne et n'évalue
> que sur les hôpitaux à coupures **réellement observées** (Lacor). Les sites
> ERIC/NYC ont des coupures **générées par une formule stochastique** : les
> inclure (`--scope all`) revient à entraîner sur ~94 % de bruit et gonfle
> artificiellement le F1 global. Le défaut `real` garantit des métriques honnêtes.

## Pipeline d'entraînement

`run_pipeline.py` orchestre **5 étapes** + **5 bis** (durée) + **6 opt-in** (EAGLE-I) :

1. **Ingestion** — Lacor, météo, ERIC, EAGLE-I, NYC, **Electricity Maps** (si token)
2. **Preprocessing** ([src/data/preprocessing.py](src/data/preprocessing.py)) — rééchantillonnage Lacor 15 min → 1 h, fusion multi-hôpitaux (météo uniquement ; pas de fusion Electricity Maps par défaut)
3. **Feature engineering** ([src/features/build_features.py](src/features/build_features.py)) — features temporelles cycliques, rolling de charge, interactions météo, historique des coupures
4. **Entraînement nowcast** ([src/models/train_baseline.py](src/models/train_baseline.py)) :
   1. Split temporel 80/20 **par hôpital** (train réordonné chronologiquement pour une vraie CV temporelle)
   2. **GridSearchCV** + **TimeSeriesSplit** (5 folds) pour RF / XGBoost / LightGBM
   3. Comparaison → sélection automatique du meilleur (F1 sur CV)
   4. **Calibration adaptative** (`--calibration auto`) : compare *aucune calibration* / isotonique / sigmoïde sur une validation interne et ne recalibre que si le Brier s'améliore d'une marge nette (un GBM est souvent déjà bien calibré)
   5. Évaluation hold-out (brut + calibré)
   6. **SHAP TreeExplainer** + sauvegarde des artefacts
5. **Horizons 1/3/6 h** ([src/models/train_horizons.py](src/models/train_horizons.py)) — même features que le nowcast, cible « coupure dans les H prochaines heures » → `models/nowcast_horizons/`
5 bis. **Modèle de durée** ([src/models/train_duration.py](src/models/train_duration.py)) — régression sur la durée réelle des épisodes Lacor → `models/duration_model.joblib` + `models/duration_summary.json`
6. **Multi-sites EAGLE-I** (opt-in `--multisite`) — [src/data/ingest_eaglei.py](src/data/ingest_eaglei.py) + [src/models/multisite_experiment.py](src/models/multisite_experiment.py) → `models/multisite_summary.json`

> **Electricity Maps** — ingéré à l’étape 1 si `ELECTRICITY_MAPS_TOKEN` est défini
> (`data/raw/electricitymaps_*.csv`, cache `em_zone_resolution.json`). Colonnes `em_*`
> **exclues du modèle ML** ; l’app affiche un **risque contextuel** live (sous-onglet
> dédié). Voir `src/grid_outage_risk.py`.

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

Deux niveaux à ne pas confondre :

**1. Modèle servi dans l’app (nowcast + horizons + durée)** — entraîné sur **Lacor
seul** (`--scope real`, coupures terrain 2022). Pour tout autre hôpital de
l’interface, c’est **le même** `calibrated_model.joblib` : scores **illustratifs**
pour ce site (l’app l’affiche via badges et notes).

**2. Validation reproductible inter-sites (EAGLE-I, opt-in)** — depuis tes derniers
ajouts, ce n’est plus un résultat orphelin : `ingest_eaglei.py` +
`multisite_experiment.py` + `python run_pipeline.py --multisite` régénèrent
`models/multisite_summary.json` et `models/multisite_loso_by_site.csv`.

- **Données** : Lacor (réel) + **8 comtés US** avec coupures **réelles** EAGLE-I
  (clients coupés, 15 min → horaire, seuil p90 ≈ 10 %/site).
- **Protocole** : leave-one-site-out — LightGBM **exogène** (29 features météo +
  calendrier, **sans** consommation ni auto-régression des coupures).
- **Résultat typique** : ROC-AUC LOSO moyen **≈ 0,63** (vs **≈ 0,99** intra-Lacor)
  → la météo seule **généralise partiellement** entre réseaux réels ; la
  sur-spécialisation au contexte ougandais est **mesurée**, pas supposée.

Ce modèle LOSO **ne remplace pas** le modèle hôpital de l’app (qui utilise charge +
historique des coupures). Il borne ce que l’on peut espérer hors Lacor sans
prétendre valider Bellevue ou Lagos comme Lacor. Panneau Streamlit : expander
« Validation & robustesse » → onglet généralisation.

### Ce que sert l'app

| Site | Modèle servi | Explication |
|---|---|---|
| **Lacor** | nowcast + horizons 1/3/6 h + durée | Coupures réelles — seul site calibré |
| **ERIC / NYC** | **même** modèle Lacor | Charge **réelle**, coupures **synthétiques** — illustratif |
| **`africa_grid`** | **même** modèle Lacor | Profil Lacor **cloné** + météo locale — illustratif |

L’app charge **un seul** modèle de probabilité (Lacor). La généralisation du
**modèle hôpital complet** vers un autre site n’est pas validée ; en revanche
l’expérience **EAGLE-I reproductible** quantifie une généralisation **partielle**
du signal météo (cf. ci-dessus).

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
| Open-Meteo Forecast | API publique | horaire (7 j) | Optionnel (`ingest_openmeteo_forecast.py`) — hors pipeline par défaut |
| Electricity Maps | API token | horaire | Pipeline + sous-onglet **Réseau & météo (live)** ; hors features ML |
| EskomSePush | API token | API | Délestage RSA — module `loadshedding.py`, **hors UI** |
| EAGLE-I (ORNL) | figshare (opt-in) | 15 min → horaire | Validation LOSO inter-sites (`--multisite`) ; panneau app |

### Données ERIC NHS

Les données [ERIC (Estates Returns Information Collection)](https://digital.nhs.uk/data-and-information/publications/statistical/estates-returns-information-collection) sont une collecte annuelle obligatoire des NHS Trusts en Angleterre. Le script [src/data/ingest_eric.py](src/data/ingest_eric.py) génère des profils horaires réalistes (8 760 h/hôpital) calibrés sur les statistiques publiées ERIC 2022-23 pour 10 hôpitaux NHS.

## Interface Streamlit

Lancer : `streamlit run app.py` (port par défaut 8501).

### Sélection des hôpitaux

**27 sites** dans le menu (Lacor + 10 ERIC NHS + 5 NYC LL84 + 11 profils `africa_grid` visibles ; **12** au catalogue dont Dhaka masqué).

- **Filtre radio** « Type de cible (coupures) » : Tous · Réel (Lacor) · Conso bâtiment, coupures comté (NYC/EAGLE-I) · Charge réelle — coupures simulées (NHS) · Cloné (`africa_grid`).
- **Liste déroulante** : préfixes `🎯 [Réel]`, `🧪 [Charge réelle, coupures simulées]`, `♻️ [Cloné]` + drapeau + nom. Seules ~10 lignes visibles : **tapez le nom** pour filtrer (`St Thomas`, `Bellevue`, `Lagos`, etc.).

Chaque fiche affiche un **badge de provenance de la cible** (`get_target_source` / `TARGET_SOURCE_META` dans [src/utils/hospitals.py](src/utils/hospitals.py)) :

| Badge (UI) | Signification |
|------------|----------------|
| Coupures réelles observées (terrain) | Lacor — modèle entraîné et évalué sur ce site |
| Conso bâtiment + coupures comté (EAGLE-I) | NYC LL84 — conso LL84 ; `is_outage` = réseau du borough/comté si EAGLE-I ingéré |
| Charge réelle — coupures simulées | ERIC — conso réelle, étiquettes `is_outage` simulées à l’ingestion |
| Aucune coupure étiquetée (profil cloné Lacor) | `africa_grid` — profil Lacor cloné/redimensionné, score illustratif |

### Onglets ([app.py](app.py))

**3 onglets principaux** :

| Onglet | Rôle |
|--------|------|
| **Prochaine coupure** | 2 **sous-onglets** (voir ci-dessous) |
| **Historique** | Période au choix, courbe de risque horaire, SHAP, stats conso/coupures |
| **Simulation** | 13 curseurs → proba + durée + jauge + SHAP |

**Sous-onglets « Prochaine coupure »** :

| Sous-onglet | Rôle |
|-------------|------|
| **Réseau & météo (live)** | Zone Electricity Maps au **point GPS** + météo récente → score de risque contextuel 0–100 % (`grid_outage_risk.py`) |
| **Modèle 1 / 3 / 6 h (replay)** | Date/heure sur données 2022 → P(coupure) à 1 / 3 / 6 h (`nowcast_horizons/`) + fenêtre 24 h conso/météo |

### Panneaux transverses
- **Expander « Sources & facteurs »** : SHAP global, catalogue des sources ([src/ui_content.py](src/ui_content.py)).
- **Expander « Données vraiment locales »** : profil honnête (`local_signals.py`) + courbe EAGLE-I (NYC).
- **Expander « Validation & robustesse »** : modèle de durée (`duration_summary.json`) et LOSO EAGLE-I (`multisite_summary.json`).
- **Sidebar** : modèle gagnant, scope `real`/`all`, statut durée, ROC-AUC LOSO si disponible.
- **Garde-fou features** : alerte si désynchronisation dataset ↔ `feature_names_in_`.

Couche de chargement : [src/app_data.py](src/app_data.py). Détail calculs : [docs/DOCUMENTATION_MODELE_ET_PREDICTIONS.md](docs/DOCUMENTATION_MODELE_ET_PREDICTIONS.md) §17.

## Hôpitaux couverts

Le sélecteur Streamlit propose **27 hôpitaux** : Lacor, 10 NHS ERIC, 5 NYC LL84 et les profils `africa_grid` visibles.

**Granularité « locale » (honnête)** — voir `src/utils/local_signals.py` :
| Signal | Lacor | ERIC / NYC (conso) | NYC (coupures) | `africa_grid` |
|--------|-------|---------------------|----------------|---------------|
| Consommation | Compteur terrain | Profil bâtiment horaire | Profil bâtiment (LL84) | Profil cloné Lacor |
| Coupures | Relevés terrain | Simulées à l'ingestion | **EAGLE-I comté** (borough) si ingéré | Aucune étiquette locale |
| Météo | lat/lon site | lat/lon site | lat/lon site | lat/lon site |
| Risque réseau (EM zone GPS) | Live UI | Live UI | Live UI | Live UI |

Les hôpitaux NYC sont rattachés à leur comté EAGLE-I (`eaglei_county_key` dans `hospitals.py`) : coupures réelles au niveau borough/comté si `data/raw/eaglei_<comté>.csv` est présent.

| Catégorie | Nb | Hôpitaux |
|---|---|---|
| Référence (terrain) | 1 | Lacor (Ouganda) |
| ERIC NHS (UK) | 10 | St Thomas', Guy's, John Radcliffe, Addenbrooke's, Manchester Royal, Leeds General, Birmingham Heartlands, Royal Victoria Newcastle, Royal Devon, King's College |
| NYC LL84 (USA) | 5 | Bellevue, NYU Tisch, NYP Brooklyn Methodist, Elmhurst, Lincoln |
| Profils estimés `africa_grid` | 11 visibles (+ Dhaka masqué) | Kenyatta, Tikur Anbessa, Groote Schuur, Fann, Parirenyatwa, Muhimbili, LUTH Lagos, Korle Bu, Ibn Sina, Kasr Al Ainy, CHUK Kigali |

Le fichier [src/utils/config.py](src/utils/config.py) dérive `HOSPITAL_LOCATIONS` du catalogue (`hospitals.py`). Les profils `africa_grid` clonent Lacor + météo locale. **EM** = contexte réseau au point GPS (pas la conso du bâtiment).

## Facteurs utilisés (features)

Le dataset de features compte environ **74 colonnes** après engineering (sans `em_*`) ; le modèle en utilise **54** numériques, sélectionnées via `COLS_TO_DROP` dans `src/utils/config.py`.

Familles de facteurs effectivement utilisées :

- **Charge/énergie** : `total_load_kw`, `solar_pv_kw`, `base_load_kw`, `load_rolling_*`, `load_diff_*`, `solar_ratio`, `peak_ratio`, `base_load_ratio`, etc.
- **Historique coupures** : `hours_since_last_outage`, `last_outage_duration_h`, `outage_frequency_7d`, `avg_outage_duration_7d`, `outage_trend_7d` (toutes calculées avec un `shift(1)` par hôpital pour éviter le leakage)
- **Temporels** : `hour`, `day_of_week`, `month`, `is_weekend`, `is_public_holiday` (calendrier 2022 du **pays du site** : UGA / GBR / USA via `public_holidays.py`), encodages cycliques (`hour_sin/cos`, `month_sin/cos`)
- **Météo** : température, humidité, point de rosée, vent (vitesse + rafales), pluie, pression, rayonnement, CAPE, weathercode + interactions (`temp_humidity_interaction`, `wind_precipitation_interaction`, `heat_stress`, `solar_available`)
- **Météo avancée** : `cloud_cover_pct`, `visibility_m`, `evapotranspiration`, `rain_intensity`, `thermal_amplitude_24h`, `humidity_change_3h`, `pressure_change_3h`
- **Electricity Maps (`em_*`)** : ingérées en CSV si token ; **exclues** du vecteur d’entraînement (`EXTERNAL_SIGNAL_PREFIXES`).

> **Horizons 1/3/6 h** ([src/models/train_horizons.py](src/models/train_horizons.py)) : même jeu de features que le nowcast, cible = coupure dans les H prochaines heures. Entraînés automatiquement à l’étape 5 de `run_pipeline.py`.

## Temps réel : périmètre exact

| Composant | Live ? | Détail |
|-----------|--------|--------|
| Sous-onglet **Réseau & météo** | Oui (API EM + météo) | Score **indicatif** ; cache 15 min ; token `ELECTRICITY_MAPS_TOKEN` |
| Sous-onglet **Modèle 1/3/6 h** | Non (replay 2022) | Même modèle Lacor ; date/heure sur jeu hospitalier chargé |
| **Historique / Simulation** | Replay | Données locales du site |
| **EskomSePush** | Hors UI | Module `loadshedding.py` (contexte RSA) |

Pipeline `--mode live` : rafraîchit météo + Electricity Maps (24 h glissantes).

```bash
export ELECTRICITY_MAPS_TOKEN="…"
streamlit run app.py
```

Nettoyage disque : `python scripts/prune_unused_data.py` (optionnel ; supprime aussi les CSV EM si vous ne les utilisez plus).

`src/realtime_forecast.py` : variante API pour inférence horizons live ; l’UI standard passe par `grid_outage_risk.py` + sous-onglet replay.
