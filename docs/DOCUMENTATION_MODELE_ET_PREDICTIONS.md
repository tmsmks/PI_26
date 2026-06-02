# Documentation du modèle et des calculs de prédiction

Le système repose sur un **modèle Lacor** unique servi dans l'application :

- **Nowcast** — `models/calibrated_model.joblib` : probabilité de coupure à l'heure courante (~54 features, dont historique des coupures).
- **Horizons 1/3/6 h** — `models/nowcast_horizons/` : probabilité de coupure dans les H prochaines heures (même jeu de features, cibles futures).

Métriques et hyperparamètres : `models/training_summary.json` et
`models/nowcast_horizons/horizons_summary.json`.

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
13. [Ajustement par profil d'hôpital](#13-ajustement-par-profil-dhôpital)
14. [Calcul du temps estimé et de la durée](#14-calcul-du-temps-estimé-et-de-la-durée)
15. [Exclusion des fuites de cible](#15-exclusion-des-fuites-de-cible)
16. [Limites connues et pistes d'amélioration](#16-limites-connues-et-pistes-damélioration)

---

## 1. Vue d'ensemble du pipeline de modélisation

Le pipeline transforme les données brutes en prédictions de coupure en
plusieurs étapes :

```
Données brutes     Features              Nowcast (étape 4)              Horizons (étape 5)
──────────────     ────────              ─────────────────              ──────────────────
multi-hôpitaux ──► features_dataset ──► RF/XGB/LGBM + SHAP ──► calibrated_model.joblib
                       │                    (cible: is_outage)         │
                       └──────────────────────────────────────────────► train_horizons
                                                                            (cibles 1/3/6 h)
                                                                            → nowcast_horizons/
```

### Fichiers impliqués

| Étape | Script | Entrée | Sortie |
|-------|--------|--------|--------|
| Feature engineering | `src/features/build_features.py` | `data/processed/hospital_merged.csv` | `data/features/features_dataset.csv` |
| Entraînement nowcast | `src/models/train_baseline.py` | `data/features/features_dataset.csv` | `models/calibrated_model.joblib` + `models/baseline_model.joblib` |
| Entraînement horizons 1/3/6 h | `src/models/train_horizons.py` | Lacor dans `features_dataset.csv` | `models/nowcast_horizons/horizon_{1,3,6}h/` |
| Comparaison + métriques | `src/models/train_baseline.py` | tous les modèles | `models/model_comparison.csv` + `models/training_summary.json` |
| Importance MDI / SHAP | `src/models/train_baseline.py` | Modèle gagnant | `models/feature_importance.csv`, `models/shap_*` |
| Prédiction (app) | `app.py`, `src/nowcast_horizons.py`, `src/realtime_forecast.py` | Nowcast + horizons + APIs live | Probabilité + waterfall SHAP |

Les fichiers `baseline_model.joblib` et `calibrated_model.joblib` contiennent
le **gagnant** de la comparaison RF / XGBoost / LightGBM (selon la dernière
exécution de `train_baseline.py`).

---

## 2. Feature engineering — jeu multi-hôpitaux

Le dataset final est **multi-hôpitaux** (≈ 140 000 lignes pour 16 sites
temps réel : Lacor + 10 NHS ERIC + 5 NYC LL84) et le nombre exact de
features dépend des sources disponibles au moment du run.

La sélection des features d'entraînement est pilotée par `COLS_TO_DROP`
dans `src/models/train_baseline.py`, puis :

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
| `is_public_holiday` | Calendrier fériés Ouganda 2022 | {0, 1} | Pour Lacor uniquement |
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

### 2.6. Contexte réseau (`em_*`) — exclu du modèle

Les colonnes Electricity Maps (`em_total_load_mw`, `em_carbon_intensity_gco2_kwh`,
mix renouvelable/fossile, etc.) sont fusionnées dans `features_dataset.csv` mais
**exclues** du jeu d'entraînement via `EXTERNAL_SIGNAL_PREFIXES = ("em_",)` dans
`config.py` et `drop_external_signal_columns()` dans `train_baseline.py`.

**Pourquoi :** indisponibles à l'échelle d'un hôpital isolé et source de décalage
entraînement/service. Elles restent exploitées en **contexte** dans l'app (bandeau
réseau, estimation de charge pour `africa_grid`, pipeline
`src/realtime_forecast.py`).

### 2.7. Modèles horizons 1/3/6 h (onglet « Prochaine coupure »)

Script canonique : `src/models/train_horizons.py` →
`models/nowcast_horizons/horizons_summary.json`.

| Attribut | Valeur |
|----------|--------|
| **Cible** | Coupure dans les 1, 3 ou 6 prochaines heures (un modèle par horizon) |
| **Features** | **Même jeu que le nowcast** (~54 features, historique coupures inclus) |
| **Algorithme** | Même famille que le gagnant `train_baseline` (hyperparamètres repris) |
| **Calibration** | IsotonicRegression sur holdout chronologique final (20 %) |
| **Évaluation** | Backtest walk-forward mensuel (train mois &lt; M, test mois M) |

**Métriques walk-forward indicatives (Lacor 2022, dernière exécution) :**

| Horizon | F1 | ROC AUC | Recall | Precision | Brier |
|---------|-----|---------|--------|-----------|-------|
| 1 h | 0.72 | 0.92 | 0.71 | 0.77 | 0.039 |
| 3 h | 0.70 | 0.90 | 0.74 | 0.68 | 0.073 |
| 6 h | 0.70 | 0.89 | 0.77 | 0.66 | 0.105 |

**Inférence :** `src/nowcast_horizons.py` — prédiction à l’instant `ref_ts` ;
repli sur agrégation du nowcast horaire si bundles absents. Temps réel :
`src/realtime_forecast.py` (fenêtre EM + météo).

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

Les probabilités brutes des modèles d'arbres ne sont pas bien calibrées :
quand le modèle dit « 30 % de risque », la fréquence empirique observée
peut être très différente.

### Solution : `CalibratedClassifierCV` (isotonic)

```python
CalibratedClassifierCV(
    estimator=lightgbm_model,
    method="isotonic",
    cv=TimeSeriesSplit(n_splits=3),
)
```

### Résultats (run courante)

| Métrique | Brut | Calibré | Interprétation |
|----------|------|---------|----------------|
| Accuracy | 99.82 % | 99.86 % | Légère amélioration |
| Precision | 85.96 % | **94.67 %** | Calibré plus prudent |
| Recall | 84.48 % | 81.61 % | Léger compromis |
| F1 | 0.8522 | **0.8765** | Calibré meilleur |
| ROC AUC | 0.999 | **0.9996** | Calibré meilleur |
| Brier | **0.0014** | 0.0015 | Quasi équivalents |

Le modèle calibré est utilisé par défaut par l'application Streamlit.

---

## 7. Évaluation et résultats

### 7.1. Métriques du LightGBM sur le hold-out test

| Métrique | Brut | Calibré |
|----------|------|---------|
| Accuracy | 99.82 % | 99.86 % |
| Precision | 85.96 % | 94.67 % |
| Recall | 84.48 % | 81.61 % |
| F1 | 0.8522 | 0.8765 |
| ROC AUC | 0.999 | 0.9996 |
| Brier | 0.0014 | 0.0015 |

### 7.2. Top 15 features SHAP (run courante)

Source : `models/shap_feature_importance.csv` (LightGBM calibré).

| Rang | Feature | |SHAP| moyen | Catégorie |
|------|---------|--------------|-----------|
| 1 | `hours_since_last_outage` | 33.20 | Historique coupures |
| 2 | `base_load_kw` | 17.27 | Énergie |
| 3 | `total_load_kw` | 16.98 | Énergie |
| 4 | `dew_point_2m` | 14.13 | Météo |
| 5 | `thermal_amplitude_24h` | 12.50 | Météo |
| 6 | `month` | 12.19 | Temporel |
| 7 | `load_diff_1h` | 10.82 | Énergie |
| 8 | `peak_ratio` | 10.63 | Énergie |
| 9 | `evapotranspiration` | 9.43 | Météo |
| 10 | `pressure_change_3h` | 7.88 | Météo |
| 11 | `load_rolling_6h` | 7.83 | Énergie |
| 12 | `hour` | 6.77 | Temporel |
| 14 | `outage_frequency_7d` | 5.96 | Historique coupures |
| 15 | `temperature_2m` | 5.18 | Météo |

**Lecture rapide (nowcast, scope `real`, 54 features) :**
- Le **temps écoulé depuis la dernière coupure** est de loin le facteur
  dominant — la cible est très auto-corrélée à court terme.
- Les variables **énergétiques** (`base_load_kw`, `total_load_kw`,
  `peak_ratio`, `load_diff_1h`, `load_rolling_6h`) restent au cœur du
  signal.
- Les variables **météo** (`dew_point_2m`, `thermal_amplitude_24h`,
  `evapotranspiration`, `pressure_change_3h`, `temperature_2m`) capturent
  le contexte climatique.

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

### Étape 4 : Ajustement par profil d'hôpital

```python
max_proba, h_notes = adjust_for_hospital_profile(max_proba, hospital)
recent["outage_probability"] = recent["outage_probability"].apply(
    lambda p: adjust_for_hospital_profile(p, hospital)[0]
)
```

(Voir [section 13](#13-ajustement-par-profil-dhôpital) pour le détail.)

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

## 10. Calcul de la prédiction prévisionnelle J+7 (onglet « Prévisions J+7 »)

L'onglet « Prévisions J+7 » utilise les fichiers
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
4. Prédire la probabilité avec `model.predict_proba(X)[:, 1]`.

### Synthèse affichée

- Pic de risque (date + probabilité + délai en heures)
- Courbe de probabilité 7 j avec seuils 50 % / 70 %
- Contexte météo (température + précipitations)
- Synthèse par jour : `proba_max`, `proba_mean`, `heures_risque (>50%)`,
  `temp_max`, `pluie_mm`
- Top 5 heures les plus à risque

> Cet onglet utilise le modèle **nowcast** (`calibrated_model.joblib`).

---

## 10 bis. Prédiction « Prochaine coupure » (horizons + temps réel)

Onglet principal pour la **prédiction opérationnelle** : probabilité de coupure
dans 1 h, 3 h et 6 h à partir des **dernières 24 h** (conso estimée ou réelle +
météo + charge réseau).

### Sources des 24 dernières heures

| Mode | Hôpitaux | Données |
|------|----------|---------|
| Historique Lacor | `lacor_uganda` | `features_dataset` (replay 2022) |
| Temps réel | `africa_grid`, Lacor (option) | Electricity Maps + Open-Meteo Forecast via `src/realtime_forecast.py` |

Pour `africa_grid`, la consommation est **estimée** à partir de la charge réseau
zone puis **mise à l'échelle** sur la moyenne Lacor (`LACOR_REF_AVG ≈ 133 kW`)
pour éviter la saturation du modèle.

### Chaîne de calcul

```
API EM (charge 24 h) + API météo (archive + forecast)
        → build_realtime_window()
        → apply_feature_engineering_single()  # historique coupure si connu
        → bundle nowcast_horizons/horizon_{1,3,6}h
        → predict_proba → calibrator isotonique
```

### Contexte affiché (hors modèle)

- Bandeau Electricity Maps (charge, carbone, mix)
- EskomSePush pour Groote Schuur (délestage programmé RSA)
- Panneau orage (pluie, rafales, CAPE forecast) — informatif uniquement

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
  `hour_sin/cos`, `month_sin/cos`),
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
Les onglets « Prédiction historique » et « Prévisions J+7 » utilisent les
probabilités brutes du modèle calibré, puis appliquent uniquement
`adjust_for_hospital_profile`.

---

## 13. Ajustement par profil d'hôpital

### Pourquoi

Le pipeline est **multi-hôpitaux**, mais il reste des écarts de domaine
entre sites (qualité des données, fiabilité du réseau, structure
électrique). L'ajustement par profil reste utilisé comme couche de
robustesse métier dans l'application.

### Formule (`adjust_for_hospital_profile`)

```python
ref_reliability = 50.0
hospital_reliability = hospital_info.get("who_reliability", ref_reliability)

delta = (ref_reliability - hospital_reliability) / 100.0
factor = 1.0 + delta * 1.5
adjusted = min(0.99, max(0.01, proba * factor))
```

`delta > 0` → hôpital moins fiable que la référence → risque ↑.
`delta < 0` → hôpital plus fiable → risque ↓.

### Exemples (run courante)

| Hôpital | Fiabilité OMS | delta | facteur | P(20%) ajustée |
|---------|---------------|-------|---------|----------------|
| Lacor (Ouganda) | 50 % | 0.00 | 1.00 | 20.0 % |
| LUTH (Nigeria) | 30 % | +0.20 | 1.30 | 26.0 % |
| Parirenyatwa (Zimbabwe) | 35 % | +0.15 | 1.225 | 24.5 % |
| Tikur Anbessa (Éthiopie) | 45 % | +0.05 | 1.075 | 21.5 % |
| Muhimbili (Tanzanie) | 58 % | −0.08 | 0.88 | 17.6 % |
| Kenyatta (Kenya) | 65 % | −0.15 | 0.775 | 15.5 % |
| Korle Bu (Ghana) | 70 % | −0.20 | 0.70 | 14.0 % |
| CHUK (Rwanda) | 75 % | −0.25 | 0.625 | 12.5 % |
| Groote Schuur (Af. Sud) | 88 % | −0.38 | 0.43 | 8.6 % |
| Kasr Al Ainy (Égypte) | 88 % | −0.38 | 0.43 | 8.6 % |
| Ibn Sina (Maroc) | 92 % | −0.42 | 0.37 | 7.4 % |
| NHS (Angleterre) | 99.5 % | −0.495 | 0.2575 | 5.2 % |
| NYC LL84 (USA) | 99.96 % | −0.4996 | 0.2506 | 5.0 % |

### Notes contextuelles générées

| Condition | Message |
|-----------|---------|
| Fiabilité < 30 % | « Réseau [stabilité] — fiabilité OMS très basse (X %) » |
| Fiabilité < 55 % | « Réseau [stabilité] — fiabilité OMS basse (X %) » |
| Fiabilité > 90 % | « Réseau [stabilité] — fiabilité OMS élevée (X %) » |
| `has_solar=False` | « Pas de panneaux solaires — dépendance totale au réseau » |
| `has_generator=False` | « Pas de générateur de secours » |

---

## 14. Calcul du temps estimé et de la durée

Ces estimations sont des **heuristiques** (le modèle baseline ne prédit
que la probabilité, pas le timing).

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

```python
if P > 0.5:
    duration = round(1.0 + P × 4.0, 1)   # entre 1.5 h et 5 h
else:
    duration = 0.5                        # risque faible → courte coupure
```

| Probabilité | Durée estimée |
|-------------|---------------|
| 90 % | 4.6 h |
| 70 % | 3.8 h |
| 50 % | 3.0 h |
| 30 % | 0.5 h |

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
| Cible `is_outage` essentiellement portée par Lacor (les sites NHS/NYC ont une cible synthétique faible) | Risque de sur-spécialisation au profil ougandais | Haute |
| Hétérogénéité multi-hôpitaux | Distributions et qualités de données différentes | Haute |
| Heuristiques pour temps/durée | Pas de modèle dédié pour le timing et la durée | Moyenne |
| Calibration réduit le recall | Le modèle calibré détecte moins de coupures (81.6 % vs 84.5 %) | Moyenne |
| `apply_extrapolation_stress` uniquement en simulation manuelle | Pas de garde-fou en mode prédiction historique | Faible |

### Fonctionnalités du pipeline et de l'app

- Comparaison multi-modèles (RF / XGBoost / LightGBM) avec GridSearchCV et TimeSeriesSplit (5 folds).
- Calibration adaptative des probabilités (`--calibration auto`).
- SHAP global et waterfall local (TreeExplainer).
- Entraînement mono-site Lacor par défaut (`--scope real`) ; option multi-hôpitaux (`--scope all`).
- Modèles horizons 1/3/6 h (`train_horizons.py`, étape 5 du pipeline).
- Modes `train` (historique 2022) et `live` (fenêtre glissante API).
- Onglets Streamlit : Prochaine coupure, Analyse historique, Prévisions J+7, Simulation manuelle.
- Contexte temps réel : Electricity Maps (tous les sites) et EskomSePush (Groote Schuur).
- Garde-fou de cohérence entre `features_dataset.csv` et `feature_names_in_` du modèle.

### Pistes d'évolution

| Piste | Complexité | Gain attendu |
|-------|------------|--------------|
| **Domain adaptation** inter-sites (pondération, calibrations locales) | Élevée | Meilleure robustesse hors site majoritaire |
| **Modèle de durée** : régression dédiée pour la durée de coupure | Moyenne | Prédictions plus précises |
| **Modèle séquentiel** (LSTM / Transformer) : dépendances temporelles longues | Élevée | Meilleure détection des patterns multi-jours |
| **Cible réelle multi-sites** : ingérer des outage logs publics (NHS, ConEd…) | Moyenne | Réduit la sur-spécialisation Lacor |
| **Seuil de classification optimisé** : trouver le seuil F1-optimal du calibré | Faible | Meilleur recall sans perte de precision |
| **Streaming Lacor temps réel** : si une API ougandaise devient disponible | Moyenne | Prédiction live vraie (vs quasi temps réel) |

### Pipeline de prédiction complet (chaîne de calculs)

```
                        Période / paramètres utilisateur
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │ Analyse historique      │
                          │ Prévisions J+7          │
                          │ Simulation (13 params)  │
                          │ Prochaine coupure (H)   │
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
                                       ┌────────────────────────────┐
                                       │ adjust_for_hospital_profile│
                                       │ P × facteur_fiabilité_OMS  │
                                       └────────────┬───────────────┘
                                                    │
                                                    ▼
                                       ┌─────────────────────────────┐
                                       │ Résultat final :              │
                                       │ • Probabilité ∈ [0.01, 0.99]  │
                                       │ • Niveau : FAIBLE/MOYEN/ÉLEVÉ │
                                       │ • Temps estimé (heuristique)  │
                                       │ • Durée estimée (heuristique) │
                                       │ • Waterfall SHAP local        │
                                       │ • Notes profil hôpital        │
                                       └─────────────────────────────┘
```
