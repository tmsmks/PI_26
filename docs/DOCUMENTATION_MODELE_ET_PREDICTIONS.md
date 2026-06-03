# Documentation du modèle et des calculs de prédiction

Le système repose sur un **modèle Lacor** unique servi dans l'application :

- **Nowcast** — `models/calibrated_model.joblib` : probabilité de coupure à l'heure courante (~54 features, dont historique des coupures).
- **Horizons 1/3/6 h** — `models/nowcast_horizons/` : probabilité de coupure dans les H prochaines heures (même jeu de features, cibles futures).
- **Durée** — `models/duration_model.joblib` : durée estimée d'un épisode (Lacor, régression).
- **Validation EAGLE-I** (opt-in) — `models/multisite_summary.json` : LOSO inter-sites, **hors** moteur de l'app.

Métriques et hyperparamètres : `models/training_summary.json`,
`models/nowcast_horizons/horizons_summary.json`, `models/duration_summary.json`,
`models/multisite_summary.json`.

## Table des matières

1. [Vue d'ensemble du pipeline de modélisation](#1-vue-densemble-du-pipeline-de-modélisation)
2. [Feature engineering — jeu multi-hôpitaux](#2-feature-engineering--jeu-multi-hôpitaux)
3. [Préparation des données d'entraînement](#3-préparation-des-données-dentraînement)
4. [Comparaison multi-modèles (RF / XGBoost / LightGBM)](#4-comparaison-multi-modèles-rf--xgboost--lightgbm)
5. [Validation croisée temporelle (TimeSeriesSplit)](#5-validation-croisée-temporelle-timeseriessplit)
6. [Calibration des probabilités](#6-calibration-des-probabilités)
7. [Évaluation et résultats](#7-évaluation-et-résultats)
8. [Interprétabilité — SHAP (explications locales)](#8-interprétabilité--shap-explications-locales)
9. [Calcul de la prédiction historique (onglet « Analyse historique »)](#9-calcul-de-la-prédiction-historique-onglet-analyse-historique)
10. [Calcul de la prédiction prévisionnelle J+7](#10-calcul-de-la-prédiction-prévisionnelle-j7-onglet-2)
10 bis. [Prédiction « Prochaine coupure » (horizons + temps réel)](#10-bis-prédiction-prochaine-coupure-horizons--temps-réel)
11. [Simulation manuelle — construction d'un scénario](#11-simulation-manuelle--construction-dun-scénario)
12. [Correction d'extrapolation (stress hors distribution)](#12-correction-dextrapolation-stress-hors-distribution)
13. [Honnêteté inter-sites — provenance de la cible](#13-honnêteté-inter-sites--provenance-de-la-cible)
14. [Calcul du temps estimé et de la durée](#14-calcul-du-temps-estimé-et-de-la-durée)
15. [Exclusion des fuites de cible](#15-exclusion-des-fuites-de-cible)
16. [Limites connues et pistes d'amélioration](#16-limites-connues-et-pistes-damélioration)
17. [Interface Streamlit (`app.py`)](#17-interface-streamlit-apppy)

---

## 1. Vue d'ensemble du pipeline de modélisation

Le pipeline transforme les données brutes en prédictions de coupure en
plusieurs étapes :

```
Données brutes     Features              Nowcast (4)        Horizons (5)      Durée (5 bis)
──────────────     ────────              ───────────        ────────────      ─────────────
multi-hôpitaux ──► features_dataset ──► RF/XGB/LGBM ──► calibrated_model   train_horizons
                       │                    + SHAP              │          → nowcast_horizons/
                       │                    (is_outage)           │
                       └──────────────────────────────────────────► train_duration
                                                                            → duration_model.joblib

(opt-in --multisite) EAGLE-I + Lacor → multisite_experiment → multisite_summary.json (LOSO)
```

### Fichiers impliqués

| Étape | Script | Entrée | Sortie |
|-------|--------|--------|--------|
| Feature engineering | `src/features/build_features.py` | `data/processed/hospital_merged.csv` | `data/features/features_dataset.csv` |
| Entraînement nowcast | `src/models/train_baseline.py` | `data/features/features_dataset.csv` | `models/calibrated_model.joblib` + `models/baseline_model.joblib` |
| Entraînement horizons 1/3/6 h | `src/models/train_horizons.py` | Lacor dans `features_dataset.csv` | `models/nowcast_horizons/horizon_{1,3,6}h/` |
| Modèle de durée | `src/models/train_duration.py` | Épisodes réels Lacor | `models/duration_model.joblib`, `duration_summary.json` |
| Validation multi-sites | `src/models/multisite_experiment.py` | Lacor + EAGLE-I (opt-in) | `multisite_summary.json`, `multisite_loso_by_site.csv` |
| Comparaison + métriques | `src/models/train_baseline.py` | tous les modèles | `models/model_comparison.csv` + `models/training_summary.json` |
| Importance MDI / SHAP | `src/models/train_baseline.py` | Modèle gagnant | `models/feature_importance.csv`, `models/shap_*` |
| Prédiction (app) | `app.py`, `src/app_data.py`, `src/nowcast_horizons.py` | Nowcast + horizons + durée (replay historique) | Probabilité, durée (h), waterfall SHAP |

Les fichiers `baseline_model.joblib` et `calibrated_model.joblib` contiennent
le **gagnant** de la comparaison RF / XGBoost / LightGBM (selon la dernière
exécution de `train_baseline.py`).

---

## 2. Feature engineering — jeu multi-hôpitaux

Le dataset final est **multi-hôpitaux** (≈ 140 000 lignes pour 16 sites
temps réel : Lacor + 10 NHS ERIC + 5 NYC LL84) et le nombre exact de
features dépend des sources disponibles au moment du run.

La sélection des features d'entraînement est pilotée par `COLS_TO_DROP`
dans `src/utils/config.py` (appliquée par `train_baseline.prepare_data()` et
`app.get_feature_columns()`), puis :

```python
X = df.drop(columns=drop).select_dtypes(include=[np.number])
```

### 2.1. Features temporelles

| Feature | Formule | Intervalle | Raison |
|---------|---------|-----------|--------|
| `hour` | Heure brute (0-23) | [0, 23] | Cycle jour/nuit |
| `day_of_week` | Jour de la semaine (0=lundi) | [0, 6] | Cycle hebdomadaire |
| `month` | Mois (1-12) | [1, 12] | Saisonnalité |
| `is_weekend` | `1 si day_of_week ≥ 5, sinon 0` | {0, 1} | Activité réduite |
| `is_public_holiday` | Fériés 2022 du **pays du site** (`public_holidays.py` : UGA, GBR, USA ; sinon 0) | {0, 1} | Par hôpital en pipeline ; par `hospital_key` dans l'app |
| `hour_sin / hour_cos` | `sin/cos(2π × hour / 24)` | [-1, 1] | Encodage cyclique heure |
| `month_sin / month_cos` | `sin/cos(2π × month / 12)` | [-1, 1] | Encodage cyclique mois |

### 2.2. Features de consommation (rolling)

Calculées **par hôpital** (`groupby("hospital").transform`) pour éviter le
mélange inter-sites :

| Feature | Formule | Unité |
|---------|---------|-------|
| `load_rolling_6h` | `mean(total_load_kw, 6h)` | kW |
| `load_rolling_24h` | `mean(total_load_kw, 24h)` | kW |
| `load_std_24h` | `std(total_load_kw, 24h)` | kW |
| `load_diff_1h` | `total_load_kw[t] − total_load_kw[t−1]` | kW |
| `load_diff_24h` | `total_load_kw[t] − total_load_kw[t−24]` | kW |
| `load_pct_change_1h` | `(load[t] − load[t−1]) / load[t−1]` | ratio |
| `peak_ratio` | `total_load_kw / load_rolling_24h` | ratio |

**kW absolus conservés dans X** (`total_load_kw`, `solar_pv_kw`, `base_load_kw`,
`sterilization_kw`, rolling/diff en kW) : choix **mono-site Lacor** documenté dans
`config.COLS_TO_DROP` — signal le plus prédictif sur le site pilote ; transfert
inter-sites non validé.

### 2.2 bis. Features de consommation sans dimension (transférables)

Dérivées dans `add_load_features()` — utiles pour comparer la *forme* du profil
hors échelle Lacor (~133 kW) :

| Feature | Rôle |
|---------|------|
| `load_zscore_24h` | Écart à la moyenne 24 h en σ |
| `load_cv_24h` | Coefficient de variation 24 h |
| `load_ratio_6h_24h` | Tendance court / long terme |
| `load_diff_1h_rel`, `load_diff_24h_rel` | Variations / baseline 24 h |

### 2.3. Features de sources d'énergie

| Feature | Formule | Inclus dans X ? |
|---------|---------|------------------|
| `solar_ratio` | `solar_pv_kw / total_load_kw` | ✅ |
| `base_load_ratio` | `base_load_kw / total_load_kw` | ✅ |
| `generator_active` | `1 si generators_kw > 1.0` | ❌ (leakage) |
| `generator_ratio` | `generators_kw / total_load_kw` | ❌ (leakage) |
| `grid_availability_rolling_6h` | `mean(grid_availability_ratio, 6h)` | ❌ (leakage) |
| `recent_outages_6h` | `sum(is_outage, 6h)` | ❌ (leakage) |
| `recent_outages_24h` | `sum(is_outage, 24h)` | ❌ (leakage) |

### 2.4. Features d'historique de coupures (sans leakage)

Ces features utilisent un `shift(1)` par hôpital sur `is_outage` pour ne
voir que le passé strict, et restent donc dans X :

| Feature | Description |
|---------|-------------|
| `hours_since_last_outage` | Heures depuis la dernière coupure |
| `last_outage_duration_h` | Durée de la dernière coupure terminée |
| `outage_frequency_7d` | Nombre d'heures en coupure sur 7 j glissants |
| `avg_outage_duration_7d` | Durée moyenne par épisode sur 7 j |
| `outage_trend_7d` | Ratio coupures 7 j récents / 7 j précédents |

### 2.5. Features météorologiques

13 variables brutes Open-Meteo (cf.
[`DOCUMENTATION_DONNEES_ET_APIS.md`](DOCUMENTATION_DONNEES_ET_APIS.md))
+ 8 variables dérivées :

| Feature | Formule | Raison |
|---------|---------|--------|
| `temp_humidity_interaction` | `temperature_2m × relative_humidity_2m / 100` | Index de chaleur simplifié |
| `wind_precipitation_interaction` | `wind_speed_10m × precipitation` | Intensité des intempéries |
| `solar_available` | `1 si shortwave_radiation > 50 W/m²` | Énergie solaire disponible |
| `heat_stress` | `1 si temperature_2m > 30°C` | Stress thermique |
| `cloud_cover_pct` | `cloud_cover` ou proxy via rayonnement | Couverture nuageuse |
| `dew_point_2m` | brute si dispo, sinon Magnus | Point de rosée |
| `visibility_m` | brute si dispo | Visibilité |
| `evapotranspiration` | `et0_fao_evapotranspiration` si dispo | Évapotranspiration |
| `rain_intensity` | `precipitation × wind_speed_10m` | Intensité pluie + vent |
| `thermal_amplitude_24h` | `max(T,24h) − min(T,24h)` | Amplitude thermique |
| `humidity_change_3h` | `Δhumidité sur 3 h` | Variation |
| `pressure_change_3h` | `Δpression sur 3 h` | Front orageux |

`cloud_cover` (brut), `visibility` (brut) et `et0_fao_evapotranspiration`
(brut) sont **exclus** par `COLS_TO_DROP` dans `src/utils/config.py`
(redondance avec leurs versions dérivées : `cloud_cover_pct`, `visibility_m`,
`evapotranspiration`).

### 2.6. Electricity Maps (`em_*`) — ingéré, hors modèle ML

Le preprocessing **ne fusionne pas** les CSV Electricity Maps dans
`features_dataset.csv` (pas de colonnes `em_*` à l'entraînement). L'ingestion
`ingest_electricitymaps.py` est appelée par `run_pipeline.py` si
`ELECTRICITY_MAPS_TOKEN` est défini. `EXTERNAL_SIGNAL_PREFIXES` dans `config.py`
exclut tout préfixe `em_` si un export local ancien en contient encore.

**Pourquoi hors ML :** signal **zone réseau entière**, pas local au bâtiment.
**Usage app :** sous-onglet **Réseau & météo (live)** via `grid_outage_risk.py`
(zone GPS `em_zone.py`). `realtime_forecast.py` = variante pipeline, non utilisée
par l'UI standard.

### 2.7. Modèles horizons 1/3/6 h (sous-onglet replay)

Script canonique : `src/models/train_horizons.py` →
`models/nowcast_horizons/horizons_summary.json`.

| Attribut | Valeur |
|----------|--------|
| **Cible** | Coupure dans les 1, 3 ou 6 prochaines heures (un modèle par horizon) |
| **Features** | **Même jeu que le nowcast** (~54 features, historique coupures inclus) |
| **Données** | **Lacor seul** — `scope` dans `horizons_summary.json` = traceabilité pipeline |
| **Algorithme** | Même famille que le gagnant `train_baseline` (hyperparamètres repris) |
| **Calibration** | IsotonicRegression sur holdout chronologique final (20 %) |
| **Évaluation** | Backtest walk-forward mensuel (train mois &lt; M, test mois M) |

**Métriques walk-forward indicatives (Lacor 2022, dernière exécution) :**

| Horizon | F1 | ROC AUC | Recall | Precision | Brier |
|---------|-----|---------|--------|-----------|-------|
| 1 h | 0.72 | 0.92 | 0.71 | 0.77 | 0.039 |
| 3 h | 0.70 | 0.90 | 0.74 | 0.68 | 0.073 |
| 6 h | 0.70 | 0.89 | 0.77 | 0.66 | 0.105 |

**Inférence (app) :** `src/nowcast_horizons.py` — prédiction à l’instant `ref_ts`
sur données chargées (sous-onglet **Modèle 1 / 3 / 6 h (replay)**) ; repli
nowcast horaire si bundles absents. Le score EM live est **indépendant** (cf. §10 bis.1).

---

## 3. Préparation des données d'entraînement

### 3.1. Sélection des features (`COLS_TO_DROP`)

Liste centralisée dans **`src/utils/config.py`** (`COLS_TO_DROP`) — utilisée
par `train_baseline.prepare_data()` et par `app.get_feature_columns()`.

Exclusions principales :
- identifiants et cible (`datetime`, `is_outage`, `hospital`) ;
- fuite directe (`grid_available`, `grid_availability_ratio`, `recent_outages_*`, …) ;
- colonnes météo brutes redondantes (`cloud_cover`, `visibility`, …) ;
- colonnes résiduelles non utilisées listées dans `COLS_TO_DROP` (ex. `storm_risk` si présente dans un export).

`prepare_data` applique aussi `drop_external_signal_columns()` → toutes les
colonnes `em_*` sont exclues du modèle. **Run courante : 54 features**
numériques (scope `real`, Lacor seul).

> 💡 La cohérence features ↔ modèle est protégée côté app : si
> `features_dataset.csv` est régénéré sans réentraîner (ou inversement),
> Streamlit affiche un avertissement « **Désynchronisation features ↔
> modèle détectée** ».

### 3.2. Split temporel par hôpital

Le fichier `train_baseline.py` (`temporal_split`) découpe **chaque
hôpital chronologiquement** (80/20 par défaut) puis concatène les indices.
Cela garantit qu'aucun hôpital n'est entièrement dans un seul split :

```
                par hôpital                 par hôpital
       ┌────────────────────────┐    ┌──────────────┐
       │    80 % train (early)  │    │  20 % test   │
       └────────────────────────┘    └──────────────┘
```

Si la colonne `hospital` n'est pas présente, on retombe sur un split
chronologique global. Les volumes par hôpital sont loggués dans
`log_hospital_split_stats`.

### 3.3. Validation croisée temporelle (TimeSeriesSplit)

Sur le train set, **TimeSeriesSplit à 5 folds** (3 en mode `--fast`) est
utilisé pour le grid search.

```
Fold 1: [████ train ████][test]
Fold 2: [████████ train ████████][test]
Fold 3: [████████████ train ████████████][test]
Fold 4: [████████████████ train ████████████████][test]
Fold 5: [████████████████████ train ████████████████████][test]
```

### 3.4. Gestion du déséquilibre de classes

Le taux de coupures global est de **9.7 %** (largement dominé par
Lacor — les hôpitaux NHS/NYC ont une cible synthétique à ~0.5 %).

- LightGBM / XGBoost : `scale_pos_weight ∈ {round(1/0.097), round(1.5/0.097)}`
  ≈ `{10, 15}`
- Random Forest : `class_weight ∈ [{0:1, 1:18}, {0:1, 1:22}]`

---

## 4. Comparaison multi-modèles (RF / XGBoost / LightGBM)

### 4.1. Modèles comparés

| Modèle | Bibliothèque | Principe |
|--------|-------------|----------|
| **Random Forest** | scikit-learn | Bagging : arbres indépendants, vote par moyenne |
| **XGBoost** | xgboost | Boosting : arbres séquentiels corrigeant les erreurs |
| **LightGBM** | lightgbm | Boosting histogramme, growth leaf-wise |

### 4.2. Grilles d'hyperparamètres (`build_model_configs`)

**Random Forest :**

| Paramètre | Valeurs (full) | Valeurs (compact) |
|-----------|----------------|-------------------|
| `n_estimators` | 200, 300 | 200 |
| `max_depth` | 12, 18, 25 | 12, 18 |
| `min_samples_leaf` | 4, 8 | 4 |
| `class_weight` | {0:1, 1:18}, {0:1, 1:22} | {0:1, 1:18} |

**XGBoost / LightGBM :**

| Paramètre | Valeurs (full) | Valeurs (compact) |
|-----------|----------------|-------------------|
| `n_estimators` | 200, 300 | 200 |
| `max_depth` | XGB: 5/8/12 — LGBM: 8/15/-1 | XGB: 5/8 — LGBM: 8/-1 |
| `learning_rate` | 0.05, 0.1 | 0.1 |
| `scale_pos_weight` | 10, 15 | 10 |
| `subsample` | 0.8 | 0.8 |
| `colsample_bytree` | 0.8 | 0.8 |

Les deux modes (`full` / `compact`) sont sélectionnables via
`run_pipeline.py --grid-scale {full,compact}` (ou implicitement avec
`--fast`).

### 4.3. Résultats comparatifs (run actuelle)

Source : `models/model_comparison.csv` + `models/training_summary.json`.

| Modèle | F1 (CV) | Accuracy | Precision | Recall | F1 (test) | ROC AUC | Brier |
|--------|---------|----------|-----------|--------|-----------|---------|-------|
| **LightGBM** | **0.7626** | **99.82 %** | 85.96 % | **84.48 %** | **85.22 %** | **99.90 %** | **0.0014** |
| XGBoost | 0.7503 | 99.63 % | 65.92 % | 84.48 % | 74.06 % | 99.86 % | 0.0031 |
| RandomForest | 0.1235 | 99.70 % | 79.47 % | 68.97 % | 73.85 % | 99.87 % | 0.0035 |

**LightGBM gagne** la run actuelle.

### 4.4. Hyperparamètres du gagnant

```python
LGBMClassifier(
    n_estimators=300,
    max_depth=-1,
    learning_rate=0.05,
    scale_pos_weight=15,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=1,
    verbose=-1,
)
```

Sauvegardé dans `models/baseline_model.joblib` (modèle brut) et
`models/calibrated_model.joblib` (modèle calibré utilisé par l'app).

---

## 5. Validation croisée temporelle (TimeSeriesSplit)

Le `TimeSeriesSplit(n_splits=5)` (ou 3 en `--fast`) garantit que chaque
fold respecte la chronologie. Le tuning utilise la métrique `f1` comme
score de référence (voir `run_model_comparison`).

Le meilleur score CV par modèle est exposé dans
`models/training_summary.json` (`comparison.<modèle>.best_f1_cv`). Pour la
run courante :

| Modèle | F1 (CV moyen sur 5 folds) |
|--------|---------------------------|
| LightGBM | **0.7626** |
| XGBoost | 0.7503 |
| Random Forest | 0.1235 |

L'écart entre LightGBM et RF traduit la difficulté du RF à capter les
interactions non linéaires entre signaux (météo, charge, temporel) sur
ce dataset multi-hôpitaux déséquilibré.

---

## 6. Calibration des probabilités

### Problème

Les probabilités brutes des modèles d'arbres peuvent être mal calibrées :
quand le modèle dit « 30 % de risque », la fréquence empirique observée
peut différer.

### Mécanisme (`--calibration auto`, défaut pipeline)

`train_baseline.calibrate_model()` compare sur une validation interne :

- **`none`** — modèle brut (référence) ;
- **`isotonic`** / **`sigmoid`** — `CalibratedClassifierCV` + `TimeSeriesSplit`.

La recalibration n'est retenue que si le **Brier** s'améliore d'une **marge nette**
(> 5 %) par rapport à `none` ; sinon `calibration_method: "none"` est écrit dans
`training_summary.json`.

### Run courante (source : `models/training_summary.json`)

| Champ | Valeur |
|-------|--------|
| `calibration_method` | `none` |
| Hold-out (brut = servi dans `calibrated_model.joblib`) | F1 ≈ 0.994 · ROC-AUC ≈ 1.00 · Brier ≈ 0.002 |

> ⚠️ Ce hold-out Lacor est **trompeusement élevé** (features autorégressives de
> l'historique des coupures) — voir [§7](#7-évaluation-et-résultats) et le README.
> Toujours régénérer la doc chiffrée après un nouveau `run_pipeline.py`.

---

## 7. Évaluation et résultats

> **Source de vérité** : `models/training_summary.json` (nowcast),
> `models/nowcast_horizons/horizons_summary.json`, `models/duration_summary.json`,
> `models/multisite_summary.json`. Les tableaux ci-dessous reflètent la run
> `--scope real` présente dans le dépôt ; ils deviennent obsolètes après réentraînement.

### 7.1. Nowcast — LightGBM, hold-out Lacor (`scope: real`)

| Métrique | Valeur (`test_metrics_raw`) |
|----------|----------------------------|
| F1 | 0.994 |
| ROC AUC | 1.000 |
| Brier | 0.002 |
| Precision / Recall | ≈ 0.994 |

F1 CV (sélection modèle) : **0.702** (LightGBM). Ne pas confondre avec le hold-out ci-dessus.

### 7.1 bis. Horizons walk-forward (Lacor)

| Horizon | F1 | ROC AUC |
|---------|-----|---------|
| 1 h | 0.724 | 0.923 |
| 3 h | 0.701 | 0.904 |
| 6 h | 0.703 | 0.888 |

### 7.1 ter. Durée d'épisode (hold-out 56 épisodes)

MAE modèle dédié **1.47 h** vs médiane constante **2.36 h** (`duration_summary.json`).

### 7.1 quater. Généralisation EAGLE-I (LOSO, hors app)

ROC-AUC LOSO moyen **≈ 0.63** (modèle exogène 29 features, sans conso) — cf. §16.

### 7.2. Top 15 features SHAP (run courante)

Source : `models/shap_feature_importance.csv` (LightGBM servi par l'app).

| Rang | Feature | \|SHAP\| moyen | Catégorie |
|------|---------|---------------|-----------|
| 1 | `hours_since_last_outage` | 1.36 | Historique coupures |
| 2 | `solar_ratio` | 0.76 | Énergie |
| 3 | `solar_pv_kw` | 0.46 | Énergie |
| 4 | `load_zscore_24h` | 0.45 | Énergie (sans dimension) |
| 5 | `load_ratio_6h_24h` | 0.38 | Énergie (sans dimension) |
| 6 | `outage_frequency_7d` | 0.37 | Historique coupures |
| 7 | `outage_trend_7d` | 0.36 | Historique coupures |
| 8 | `hour` | 0.34 | Temporel |
| 9 | `sterilization_kw` | 0.24 | Énergie |
| 10 | `pressure_change_3h` | 0.24 | Météo |
| 11 | `peak_ratio` | 0.23 | Énergie |
| 12 | `shortwave_radiation` | 0.23 | Météo |
| 13 | `load_rolling_6h` | 0.22 | Énergie |
| 14 | `surface_pressure` | 0.22 | Météo |
| 15 | `load_cv_24h` | 0.20 | Énergie (sans dimension) |

**Lecture rapide (nowcast, scope `real`, 54 features) :**
- Le **temps écoulé depuis la dernière coupure** et la **fréquence 7 j** dominent
  — la cible est très auto-corrélée à court terme.
- **Énergie** : ratios solaire, `load_zscore_24h`, `peak_ratio`, kW absolus
  (`solar_pv_kw`, `sterilization_kw`, `load_rolling_6h`).
- **Météo** : `pressure_change_3h`, `shortwave_radiation`, `surface_pressure`.

> Le panneau « Sources & facteurs du modèle » affiche l'importance SHAP du
> **nowcast** (`models/shap_feature_importance.csv`), incluant l'historique des coupures.

---

## 8. Interprétabilité — SHAP (explications locales)

### Principe

**SHAP** (SHapley Additive exPlanations) calcule la contribution de chaque
feature à une prédiction **individuelle**. Contrairement à l'importance
MDI (globale), SHAP explique **pourquoi cette prédiction précise** a cette
valeur.

### Implémentation

Le pipeline utilise `shap.TreeExplainer`, optimisé pour les modèles
d'arbres (LightGBM, XGBoost, RF) :

```python
explainer = shap.TreeExplainer(raw_model)
shap_values = explainer.shap_values(X_shap)
```

Pour limiter le temps de calcul, `train_baseline.py` échantillonne
`SHAP_SAMPLE_SIZE = 5000` lignes du test set par défaut (overridable via
`--shap-sample-size`).

### Fichiers générés

| Fichier | Contenu |
|---------|---------|
| `models/shap_explainer.joblib` | TreeExplainer sérialisé (omis si `--no-full-artifacts`) |
| `models/shap_values.npz` | Matrice SHAP (`shap_values`, `expected_value`, `feature_names`) |
| `models/shap_feature_importance.csv` | |SHAP| moyen par feature |

### Utilisation dans l'interface

Pour chaque prédiction (historique, prévisionnelle ou simulation), l'app
calcule les SHAP values en temps réel et affiche un **waterfall chart**
via `show_shap_waterfall` :

- Barres **rouges** : features qui poussent vers la coupure
- Barres **vertes** : features qui réduisent le risque
- Préfixe emoji par catégorie (énergie, météo, temporel, etc.)
- La **base** (`expected_value`) représente la prédiction moyenne

---

## 9. Calcul de la prédiction historique (onglet « Analyse historique »)

L'onglet **« Analyse historique (par période) »** de l'app Streamlit (code dans
`app.py`, section `with tab_predict`) propose une **analyse sur la période de
votre choix** avec le modèle **nowcast** (54 features, inclut l'historique des
coupures). L'onglet « Prochaine coupure » utilise les modèles **horizons**
(même features, cible future).

### Étape 1 : Choix de la période

L'utilisateur sélectionne une période parmi 7 presets ou en mode
« Personnalisé » via `st.date_input` :

```python
PRESETS = {
    "Personnalisé": None,
    "Dernières 72 h disponibles": (df_max_d − 2j, df_max_d),
    "Janvier 2022": (2022-01-01, 2022-01-31),
    "Saison sèche (déc-fév)": (2022-01-01, 2022-02-28),
    "Saison des pluies (mars-mai)": (2022-03-01, 2022-05-31),
    "Été (juin-août)": (2022-06-01, 2022-08-31),
    "Automne (sept-nov)": (2022-09-01, 2022-11-30),
    "Toute l'année 2022": (2022-01-01, 2022-12-31),
}
```

Pour les hôpitaux `africa_grid` la grille temporelle est rebasée à un
horaire glissant terminant maintenant — les presets « 2022 » prennent
alors le sens de fenêtres relatives.

### Étape 2 : Extraction et prédiction par heure

```python
selected_df = df[(df_dt >= start_dt_sel) & (df_dt <= end_dt_sel)].copy()
X = ensure_numeric_feature_frame(selected_df, feature_cols)
proba_series = model.predict_proba(X)[:, 1]
selected_df["outage_probability"] = proba_series
```

### Étape 3 : Identification du pic de risque

```python
high_risk = recent[recent["outage_probability"] > 0.5]
if high_risk.empty:
    max_idx = recent["outage_probability"].idxmax()
    max_proba = recent.loc[max_idx, "outage_probability"]
else:
    max_proba = high_risk.iloc[0]["outage_probability"]
```

### Étape 4 : Probabilités et durée au pic

```python
recent["outage_probability"] = site_predict_proba(recent)  # modèle Lacor, même X pour tous les sites
duration = estimate_outage_duration(max_proba, peak_row)   # duration_model.joblib ou repli 1+4p
h_notes = site_profile_notes(hospital_key, hospital)       # honnêteté (réel / synthétique / cloné)
```

Les probabilités **ne sont plus re-scalées** par la fiabilité OMS (ancienne fonction
`adjust_for_hospital_profile`, retirée). L’honnêteté inter-sites repose sur les notes
et badges de [section 13](#13-honnêteté-inter-sites--provenance-de-la-cible).

### Étape 5 : Affichage

| Probabilité | Niveau | Couleur | Icône |
|-------------|--------|---------|-------|
| > 70 % | ÉLEVÉ | Rouge (`#e74c3c`) | 🔴 |
| 40 % — 70 % | MOYEN | Orange (`#f39c12`) | 🟠 |
| < 40 % | FAIBLE | Vert (`#2ecc71`) | 🟢 |

Plus, dans la même page :
- waterfall SHAP local sur la **dernière heure** de la période,
- courbe `outage_probability` heure par heure (avec seuil 50 %),
- vue de la **consommation observée** (Auto / Horaire / Journalier /
  Hebdomadaire — bascule auto en fonction de la longueur de la période),
- résumé hebdomadaire des heures en coupure (si ≥ 48 h sélectionnées),
- 4 statistiques clés (nombre de coupures, taux %, charge moyenne, charge
  max).

### Persistance via `st.session_state`

Le résultat (`recent`, `max_proba`, `h_notes`, `factors`, SHAP…) est
sauvegardé sous la clé `predict_analysis_result` avec un identifiant
`f"{hospital_key}:{start}:{end}"`. Tant que la sélection ne change pas, la
page reste affichée même après un re-render Streamlit.

---

## 10. Calcul de la prédiction prévisionnelle J+7 (retiré de l'UI)

> **Note :** l'onglet Streamlit « Prévisions J+7 » a été supprimé (projection
> non validée). Les fichiers `meteo_forecast_*.csv` ne sont plus produits par
> `run_pipeline.py` (module optionnel + `prune_unused_data.py`). Algorithme
> conservé à titre de référence technique uniquement.

Les fichiers
`data/raw/meteo_forecast_<hospital_key>.csv` produits par
`ingest_openmeteo_forecast.py`.

### Algorithme (`build_forecast_predictions`)

Pour chaque heure future du CSV de prévisions :

1. Trouver dans l'historique `df` la **ligne historique la plus
   similaire** (même heure, même mois, même jour-type, température
   proche) via `_match_similar_historical_row`.
2. Cloner ses features (rolling, énergie, historique coupures) puis
   **remplacer la météo** par les variables prévisionnelles :
   `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`,
   `wind_speed_10m`, `wind_gusts_10m`, `precipitation`,
   `surface_pressure`, `shortwave_radiation`, `cloud_cover`,
   `visibility`, `et0_fao_evapotranspiration`, `cape`, `weathercode`.
3. Recalculer les features dérivées météo
   (`temp_humidity_interaction`, `solar_available`, `heat_stress`,
   `rain_intensity`, `cloud_cover_pct`, `dew_point_2m` Magnus si
   manquant…).
4. Prédire la probabilité avec `site_predict_proba(feat_batch)` (batch vectorisé).
5. Estimer la durée avec `_estimate_durations_batch(proba, feat_batch)` → colonne
   `duration_est_h`.

### Synthèse affichée

- Pic de risque (date + probabilité + délai en heures + **durée estimée**)
- Courbe de probabilité 7 j avec seuils 50 % / 70 %
- Contexte météo (température + précipitations)
- Synthèse par jour : `proba_max`, `proba_mean`, `heures_risque (>50%)`,
  `duration_max_h` (si modèle de durée chargé), `temp_max`, `pluie_mm`
- Top 5 heures : proba, durée est., météo

> Cet onglet utilise le modèle **nowcast** (`calibrated_model.joblib`).

---

## 10 bis. Prédiction « Prochaine coupure » (live + replay)

L'onglet **Prochaine coupure** comporte **deux sous-onglets** distincts :

| Sous-onglet | Nature | Module principal |
|-------------|--------|------------------|
| **Réseau & météo (live)** | Score contextuel 0–100 % (indicatif) | `src/grid_outage_risk.py` → `show_live_grid_weather_risk()` |
| **Modèle 1 / 3 / 6 h (replay)** | P(coupure) à 1 / 3 / 6 h sur données 2022 | `nowcast_horizons` + `forecast_next_outage` |

### 10 bis.1 — Réseau & météo (live)

- Appels API Electricity Maps (charge, carbone, mix) sur la **zone résolue au GPS**
  (`src/utils/em_zone.py`, cache `em_zone_resolution.json`).
- Météo récente : prévision Open-Meteo ou dernières lignes `meteo_<key>.csv`.
- Heuristique documentée dans `grid_outage_risk.py` (pas le modèle Lacor calibré).
- Cache session : `load_live_outage_risk` (@st.cache_data, TTL 15 min).
- Prérequis : `export ELECTRICITY_MAPS_TOKEN="…"`.

EskomSePush et `realtime_forecast.py` restent **hors** cet écran.

### 10 bis.2 — Modèle 1 / 3 / 6 h (replay)

Probabilité de coupure dans **1 h, 3 h et 6 h** à partir des **24 h qui précèdent**
un point date/heure choisi sur le dataset déjà chargé
(`load_hospital_data` → `forecast_next_outage` / `predict_horizons`).

### Sources des 24 dernières heures (par type d'hôpital)

| `data_source` | Consommation | Coupures (`is_outage` dans le jeu chargé) |
|---------------|--------------|---------------------------------------------|
| Lacor | Relevés terrain 2022 | Réelles |
| `eric` / `nyc_ll84` | Profil bâtiment horaire | ERIC : simulées ; NYC : **EAGLE-I comté** si ingéré |
| `africa_grid` | Profil Lacor cloné + météo locale | Aucune étiquette locale |

### Chaîne de calcul (app)

```
load_hospital_data(hospital_key)
        → fenêtre 24 h avant ref_ts
        → predict_horizons(..., horizon_models=nowcast_horizons/)
        → (repli) nowcast horaire si bundles absents
```

Modèle : bundles `models/nowcast_horizons/horizon_{1,3,6}h/` (mêmes 54 features
que le nowcast Lacor, entraînés sur Lacor seul).

### Contexte documenté hors sous-onglet replay

- **EAGLE-I** : courbe comté dans l'expander « Données vraiment locales » (NYC).
- **EskomSePush** : catalogue `DATA_SOURCES` uniquement (hors UI).
- **Electricity Maps** : sous-onglet live + ingestion CSV ; **exclu** des features ML.

---

## 11. Simulation manuelle — construction d'un scénario

L'onglet « Simulation manuelle » permet de définir un scénario hypothétique
(13 paramètres) et d'obtenir une prédiction. Le défi : construire un
vecteur de features cohérent à partir d'aussi peu d'entrées.

### 11.1. Paramètres utilisateur (13)

| Catégorie | Paramètre | Contrôle UI |
|-----------|-----------|-------------|
| Temporel | `hour` (0-23) | Slider |
| | `month` (1-12) | Slider |
| | `day_of_week` (0-6) | Selectbox |
| Énergie | `total_load_kw` | Slider (adapté à `avg_load_kw` / `max_load_kw` de l'hôpital) |
| | `solar_pv_kw` | Slider (désactivé si `has_solar=False`) |
| | `base_load_kw` | Slider |
| | `sterilization_kw` | Slider |
| Météo | `temperature_2m` (°C) | Slider |
| | `humidity` (%) | Slider |
| | `wind_speed` (km/h) | Slider |
| | `precipitation` (mm) | Slider |
| | `pressure` (hPa) | Slider |
| | `radiation` (W/m²) | Slider |

### 11.2. Stratégie : la ligne historique la plus similaire

Code source — `build_simulation_row` :

```python
candidates = df.copy()
candidates["_hour_dist"] = abs(candidates["hour"] - hour)
candidates["_month_dist"] = abs(candidates["month"] - month)
candidates["_load_dist"] = abs(candidates["total_load_kw"] - load)
candidates["_score"] = (
    candidates["_hour_dist"] * 3
    + candidates["_month_dist"]
    + candidates["_load_dist"] / 30
)
best_idx = candidates["_score"].idxmin()
ref = df.loc[best_idx, feature_cols].copy()
```

On part ensuite de `ref` (qui a des features rolling cohérentes) et on
remplace seulement les paramètres modifiés par l'utilisateur :

- variables temporelles (`hour`, `month`, `day_of_week`, `is_weekend`,
  `hour_sin/cos`, `month_sin/cos`, `is_public_holiday` via
  `is_public_holiday_for_hospital(hospital_key, …)`),
- variables énergétiques utilisateur (`total_load_kw`, `solar_pv_kw`,
  `base_load_kw`, `sterilization_kw`),
- variables météo brutes,
- features dérivées : `solar_ratio`, `base_load_ratio`, `peak_ratio`,
  `temp_humidity_interaction`, `wind_precipitation_interaction`,
  `solar_available`, `heat_stress`, `dew_point_2m` (Magnus),
  `rain_intensity`, `humidity_change_3h`, `pressure_change_3h`.

**Avantages :**
- Les rolling (`load_rolling_24h`, `load_std_24h`, etc.) sont réalistes
  car elles viennent d'un contexte historique réel.
- La cohérence inter-features est préservée.

### 11.3. Comparaison avec les conditions moyennes

L'app simule en parallèle un scénario « médian » (heure 12, mois 6, jour
mercredi, météo douce) et affiche `proba_scenario` vs
`proba_conditions_moyennes` + delta.

---

## 12. Correction d'extrapolation (stress hors distribution)

### Problème

Les modèles d'arbres **n'extrapolent pas**. Si l'utilisateur simule une
consommation de 500 kW alors que le max historique est 235 kW, le modèle
n'a jamais vu cette valeur et produit une prédiction conservative
(faussement basse).

### Solution : `apply_extrapolation_stress()`

Cette fonction détecte quand les paramètres dépassent les bornes
historiques (calculées sur le DataFrame de l'hôpital sélectionné) et
ajoute un bonus de risque proportionnel.

### Paramètres surveillés

| Paramètre | Borne (P95) | Borne (Max) |
|-----------|-------------|-------------|
| `total_load_kw` | `df["total_load_kw"].quantile(0.95)` | `df["total_load_kw"].max()` |
| `temperature_2m` | idem | idem |
| `wind_speed_10m` | idem | idem |
| `precipitation` | idem | idem |

### Calcul du bonus de stress

Pour chaque paramètre `x` avec borne max `M` et 95e percentile `P95` :

```
Si x > M :
    overshoot = (x − M) / max(M − P95, 1)
    bonus = min(0.25, overshoot × 0.10)

Si P95 < x ≤ M :
    overshoot = (x − P95) / max(M − P95, 1)
    bonus = min(0.10, overshoot × 0.05)
```

### Synergie multi-facteurs

Si **plusieurs** facteurs sont en stress simultanément, le risque est
amplifié :

```
Si nb_facteurs_stress ≥ 2 :
    stress_total *= 1.0 + 0.3 × (nb_facteurs − 1)
```

Exemple : 2 facteurs → ×1.3 ; 3 → ×1.6 ; 4 → ×1.9.

### Probabilité ajustée

```
P_ajustée = min(0.99, P_modèle + stress_total)
```

⚠️ Cette correction n'est appliquée **qu'en mode simulation manuelle**.
Les autres onglets utilisent les probabilités brutes du modèle Lacor via
`site_predict_proba`, sans re-scaling par la fiabilité OMS.

---

## 13. Honnêteté inter-sites — provenance de la cible

> **Historique** : une ancienne couche `adjust_for_hospital_profile` multipliait
> la probabilité selon la fiabilité OMS estimée. Elle a été **retirée** de l’app :
> elle donnait l’illusion d’une prédiction « adaptée » sans validation. La transparence
> repose désormais sur la **provenance de `is_outage`** et des messages explicites.

### Source de vérité : `get_target_source(hospital_key, hospital_info)`

| Valeur | Sites | Badge UI (`TARGET_SOURCE_META['label']`) |
|--------|-------|------------------------------------------|
| `real` | `lacor_uganda` | Coupures réelles observées (terrain) |
| `county_network` | `data_source` = `nyc_ll84` | Conso bâtiment + coupures comté EAGLE-I |
| `synthetic` | `data_source` = `eric` | Charge réelle — coupures simulées |
| `cloned` | `data_source` = `africa_grid` | Aucune coupure étiquetée (profil cloné Lacor) |

Métadonnées : `TARGET_SOURCE_META` dans `src/utils/hospitals.py`. Affichage :
- badge HTML sous la fiche hôpital (`st.markdown` dédié, hors gros bloc pour éviter
  la troncature Streamlit) ;
- première ligne de `detect_hospital_data_sources()` (liste des sources du site).

### Notes contextuelles (`site_profile_notes`)

Message principal selon `get_target_source` (réel / charge réelle–coupures simulées / cloné), puis
indicateurs optionnels :

| Condition | Message |
|-----------|---------|
| Fiabilité < 30 % | « Réseau [stabilité] — fiabilité OMS très basse (X %) » |
| Fiabilité < 55 % | « Réseau [stabilité] — fiabilité OMS basse (X %) » |
| Fiabilité > 90 % | « Réseau [stabilité] — fiabilité OMS élevée (X %) » |
| `has_solar=False` | « Pas de panneaux solaires — dépendance totale au réseau » |
| `has_generator=False` | « Pas de générateur de secours » |

---

## 14. Calcul du temps estimé et de la durée

Le modèle nowcast ne prédit que la **probabilité** ; le **délai** avant coupure
reste heuristique. La **durée** d’épisode est estimée par un modèle dédié (cf.
ci-dessous) avec repli sur `1 + 4p`.

### Temps avant la prochaine coupure

```python
# Analyse historique
hours_away = abs(
    (datetime_pic_risque − datetime_actuelle).total_seconds() / 3600
)

# Prévisions J+7
hours_away = max(0.0, (max_time − pd.Timestamp.now()).total_seconds() / 3600)

# Simulation manuelle
hours_away = max(1, round((1 − P) × 24))
```

| Probabilité (simulation) | Temps estimé |
|--------------------------|--------------|
| 90 % | ~2 h |
| 70 % | ~7 h |
| 50 % | ~12 h |
| 30 % | ~17 h |
| 10 % | ~22 h |

### Durée estimée de la coupure

**Modèle de durée dédié** (`src/models/train_duration.py` →
`models/duration_model.joblib`). Régression LightGBM (`objective=regression_l1`,
MAE) entraînée sur les **épisodes réels de Lacor seul** (`scope` dans
`duration_summary.json` = traceabilité pipeline), conditionnée aux **mêmes
features que le nowcast** prises à
l'heure de **déclenchement** de l'épisode. L'app passe donc le même vecteur `X`
qu'à l'inférence de probabilité.

```python
# app.py — estimate_outage_duration(proba, frame)
if proba <= 0.5:
    duration = 0.5                                  # risque négligeable
elif DURATION_MODEL is not None:
    duration = clip(DURATION_MODEL.predict(X), 0.5, 24)   # modèle dédié
else:
    duration = round(1.0 + proba * 4.0, 1)          # repli heuristique
```

**Pourquoi un modèle dédié.** Les coupures réelles de Lacor durent de 1 à 16 h
(médiane 2 h, moyenne 3 h) selon les conditions (charge, météo, heure) ;
l'ancienne heuristique `1 + 4p` était plafonnée à 5 h et **aveugle aux
conditions**. Sur un hold-out chronologique (56 épisodes), le modèle dédié
réduit nettement l'erreur :

| Estimateur | MAE (h) | RMSE (h) |
|------------|---------|----------|
| **Modèle de durée dédié** | **1.47** | **2.76** |
| Baseline médiane constante | 2.36 | 3.81 |
| Heuristique historique `1 + 4p` | 3.43 | 3.79 |

Features les plus prédictives de la durée : `load_diff_24h` (variation de charge
sur 24 h), `solar_pv_kw`, `hours_since_last_outage`. Métriques complètes :
`models/duration_summary.json`. Si le modèle est absent, l'app retombe
automatiquement sur l'heuristique (aucune régression fonctionnelle).

---

## 15. Exclusion des fuites de cible

Certaines colonnes du dataset fusionné portent une information **trop proche**
de `is_outage` pour être utilisées comme prédicteurs. Elles sont listées dans
`COLS_TO_DROP` (`src/utils/config.py`) et retirées avant l'entraînement (voir
[section 3.1](#31-sélection-des-features-cols_to_drop)).

| Feature exclue | Raison |
|----------------|--------|
| `grid_availability_ratio` | Corrélée quasi parfaitement à la disponibilité réseau / coupure |
| `grid_availability_rolling_6h` | Moyenne glissante de cette même information |
| `recent_outages_6h` / `recent_outages_24h` | Sommes glissantes de la cible sans décalage |
| `generators_kw`, `generator_active`, `generator_ratio` | Réaction **pendant** la coupure (conséquence, pas cause) |
| `grid_available` | Indicateur binaire inverse de la coupure sur Lacor |

### Historique des coupures (autorisé)

Les features `hours_since_last_outage`, `last_outage_duration_h`,
`outage_frequency_7d`, `avg_outage_duration_7d`, `outage_trend_7d` **entrent
dans le modèle** : elles sont calculées avec un `shift(1)` par hôpital sur
`is_outage` (`add_outage_history_features` dans `build_features.py`) et ne
utilisent que le passé strict. `hours_since_last_outage` domine en général
l'importance SHAP (cf. [section 7.2](#72-top-15-features-shap-run-courante)).

---

## 16. Limites connues et pistes d'amélioration

### Limites actuelles

| Limite | Impact | Sévérité |
|--------|--------|----------|
| Modèle servi = Lacor pour tous les sites UI (ERIC/NYC/africa_grid illustratifs) | Scores hors Lacor non validés comme terrain | Haute |
| Hétérogénéité multi-hôpitaux | Distributions et qualités de données différentes | Haute |
| Heuristique pour le **timing** (délai avant coupure) | La **durée** a désormais un modèle dédié (cf. §14) ; seul le *délai* reste heuristique | Faible |
| Calibration réduit le recall | Le modèle calibré détecte moins de coupures (81.6 % vs 84.5 %) | Moyenne |
| `apply_extrapolation_stress` uniquement en simulation manuelle | Pas de garde-fou en mode prédiction historique | Faible |

### Fonctionnalités du pipeline et de l'app

- Comparaison multi-modèles (RF / XGBoost / LightGBM) avec GridSearchCV et TimeSeriesSplit (5 folds).
- Calibration adaptative des probabilités (`--calibration auto`).
- SHAP global et waterfall local (TreeExplainer).
- Entraînement mono-site Lacor par défaut (`--scope real`) ; option multi-hôpitaux (`--scope all`).
- Modèles horizons 1/3/6 h (`train_horizons.py`, étape 5 du pipeline).
- Modes `train` (historique 2022) et `live` (fenêtre glissante API).
- Onglets Streamlit : Prochaine coupure (live EM + replay horizons), Historique, Simulation.
- Panneau validation : modèle de durée + généralisation EAGLE-I (LOSO).
- Fériés par pays (`public_holidays.py`) dans le feature engineering et la simulation.
- Nettoyage disque : `scripts/prune_unused_data.py`.
- Catalogue sources (`DATA_SOURCES`) : EM live + Forecast optionnel ; Eskom hors UI.
- Score risque réseau : `grid_outage_risk.py` (distinct du modèle Lacor).
- Garde-fou de cohérence entre `features_dataset.csv` et `feature_names_in_` du modèle.
- Honnêteté inter-sites : badges + `site_profile_notes` (sans re-scaling OMS des probas).

### Pistes d'évolution

| Piste | Complexité | Gain attendu |
|-------|------------|--------------|
| **Domain adaptation** inter-sites (pondération, calibrations locales) | Élevée | Meilleure robustesse hors site majoritaire |
| **Modèle de durée** : régression dédiée pour la durée de coupure — ✅ *fait* (`train_duration.py`, MAE 1.47 h vs 3.43 h heuristique) | Moyenne | Prédictions plus précises |
| **Modèle séquentiel** (LSTM / Transformer) : dépendances temporelles longues | Élevée | Meilleure détection des patterns multi-jours |
| **Cible réelle multi-sites** : coupures réelles EAGLE-I (comtés US) — ✅ *fait* (LOSO reproductible) | Moyenne | Mesure la sur-spécialisation Lacor |
| **Seuil de classification optimisé** : trouver le seuil F1-optimal du calibré | Faible | Meilleur recall sans perte de precision |
| **Streaming Lacor temps réel** : si une API ougandaise devient disponible | Moyenne | Prédiction live vraie (vs quasi temps réel) |

#### Validation multi-sites EAGLE-I (reproductible)

La piste « cible réelle multi-sites » est désormais **outillée et reproductible** :

- **Ingestion** — `src/data/ingest_eaglei.py` transforme les coupures réelles
  EAGLE-I (Oak Ridge NL, clients coupés par comté US, 15 min) en séries horaires
  binarisées au quantile p90. Bruts non versionnés ⇒ repli propre documenté
  (DOI figshare) si absents.
- **Expérience** — `src/models/multisite_experiment.py` empile Lacor (réel) + 8
  comtés (réels) et valide en **leave-one-site-out** avec un modèle **exogène**
  (météo + temporel, 29 features). Régénère `models/multisite_summary.json` et
  `models/multisite_loso_by_site.csv`.
- **Résultat** — ROC-AUC LOSO moyen **≈ 0,63** (vs 0,99 sur Lacor seul) :
  la sur-spécialisation au contexte ougandais est ainsi **mesurée, pas
  supposée**. Ce modèle exogène ne remplace pas le modèle hôpital complet ;
  il borne ce que la météo seule annonce hors du site source.
- **Lancer** — `python run_pipeline.py --multisite` (étape 6 opt-in) ou
  `python -m src.models.multisite_experiment`. Détails données :
  [`DOCUMENTATION_DONNEES_ET_APIS.md`](DOCUMENTATION_DONNEES_ET_APIS.md) §7 bis.

### Pipeline de prédiction complet (chaîne de calculs)

```
                        Période / paramètres utilisateur
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │ Analyse historique      │
                          │ Simulation (13 params)  │
                          │ Prochaine coupure       │
                          │  · live (grid_outage)   │
                          │  · replay (H 1/3/6)     │
                          └────────────┬────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │ ensure_numeric_feature   │
                          │ _frame() (X)             │
                          └────────────┬─────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │ model.predict_proba()    │
                          │ Modèle calibré (LightGBM)│
                          │ P_modèle ∈ [0, 1]        │
                          └────────────┬─────────────┘
                                       │
                            ┌──────────┼──────────┐
                            ▼                     ▼
              ┌─────────────────────┐   ┌─────────────────────────┐
              │ SHAP TreeExplainer  │   │ apply_extrapolation     │
              │ → waterfall chart   │   │ _stress()               │
              │                     │   │ (simulation uniquement) │
              └─────────────────────┘   └────────────┬────────────┘
                                                     │
                                                     ▼
                                       ┌─────────────────────────────┐
                                       │ estimate_outage_duration()    │
                                       │ (duration_model ou 1+4p)    │
                                       └────────────┬────────────────┘
                                                    │
                                                    ▼
                                       ┌─────────────────────────────┐
                                       │ Résultat final :              │
                                       │ • Probabilité (modèle Lacor)  │
                                       │ • Niveau : FAIBLE/MOYEN/ÉLEVÉ │
                                       │ • Délai (heuristique)         │
                                       │ • Durée (modèle dédié)        │
                                       │ • Waterfall SHAP local        │
                                       │ • Notes site_profile_notes    │
                                       └─────────────────────────────┘
```

---

## 17. Interface Streamlit (`app.py`)

### Architecture

| Module | Rôle |
|--------|------|
| `app.py` | Layout, onglets, inférence, graphiques Plotly |
| `src/app_data.py` | Chargements `@st.cache_*` (modèle, features, durée, multisite…) |
| `src/ui_content.py` | `FEATURE_LABELS`, `DATA_SOURCES`, catégories |
| `src/ui_components.py` | `show_risk_result`, `show_live_grid_weather_risk`, `show_shap_waterfall` |
| `src/grid_outage_risk.py` | Score risque EM + météo (sous-onglet live) |
| `src/utils/em_zone.py` | Résolution zone Electricity Maps au GPS |
| `src/utils/hospitals.py` | `HOSPITAL_DISPLAY`, `get_target_source`, `TARGET_SOURCE_META` |

### Sélecteur d’hôpital

```python
SELECTOR_HOSPITAL_KEYS = [
    k for k, v in HOSPITAL_DISPLAY.items()
    if k == "lacor_uganda"
    or v.get("data_source") in ("eric", "nyc_ll84", "africa_grid")
]
```

**27 clés** triées : réel → comté (NYC) → charge réelle (ERIC) → cloné.

1. **Radio** `hospital_source_filter` — `Tous` · Réel · Conso bâtiment, coupures comté ·
   Charge réelle, coupures simulées · Profil cloné.
2. **Selectbox** — `format_func=_hospital_select_label` : emoji + `[Réel]` /
   `[Conso bâtiment, coupures comté]` / `[Charge réelle, coupures simulées]` / `[Cloné]` + drapeau + nom.

**Fériés** : `is_public_holiday` utilise le calendrier 2022 du pays du site
(`src/utils/public_holidays.py` : UGA, GBR, USA ; sinon 0). Simulation et
feature engineering app passent par `hospital_key`.

Légende : « tapez dans la liste pour filtrer » (~10 lignes visibles sans recherche).

### Sidebar (état modèles)

- Modèle gagnant (`training_summary.json`) — message unique (plus dans `load_model`).
- Scope `real` / `all`.
- Modèle de durée actif + MAE hold-out si `duration_summary.json` présent.
- ROC-AUC LOSO moyen si `multisite_summary.json` présent.

### Expanders

1. **Sources & facteurs** — SHAP global Lacor, catalogue `DATA_SOURCES`.
2. **Données vraiment locales** — profil honnête (`local_signals`) + courbe EAGLE-I (NYC).
3. **Validation & robustesse** — durée + LOSO EAGLE-I.

### Inférence unifiée

```python
def site_predict_proba(frame) -> np.ndarray:
    return model.predict_proba(ensure_numeric_feature_frame(frame, feature_cols))[:, 1]

def estimate_outage_duration(proba, frame=None) -> float:
    # DURATION_MODEL.predict si chargé, sinon 1 + 4p pour proba > 0.5
```

Utilisé dans : analyse historique, simulation, sous-onglet **Modèle 1 / 3 / 6 h (replay)**.

### Structure des onglets

```
Prochaine coupure
├── Réseau & météo (live)     → show_live_grid_weather_risk
└── Modèle 1 / 3 / 6 h (replay) → date/heure, horizons, fenêtre 24 h
Historique                    → courbe + SHAP
Simulation                    → 13 curseurs
```

### Tests live (optionnel)

Scripts Playwright sous `scripts/test_streamlit_*.py` — parcourent
`http://localhost:8503` (ou URL passée en argument) pour plusieurs hôpitaux.
Vérifient les **3 onglets**. Sur Prochaine coupure, cliquer le sous-onglet
**Modèle 1 / 3 / 6 h (replay)** avant de chercher « Date de référence » / « Heure de référence »
(le sous-onglet **Réseau & météo (live)** s’affiche en premier). Pas d’onglet « Prévisions J+7 ».
Rapports JSON dans `reports/`.
