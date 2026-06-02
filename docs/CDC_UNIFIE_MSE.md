# Cahier des Charges Unifie - Systeme Predictif et Resilience Energetique Hospitaliere

**Projet Master (MSE) - HES-SO**  
**Periode :** Mi-mars - Mi-juin 2026 (13 semaines)  
**Equipe :** Rafael, Thomas, Ines, Muhammed Ali

---

## 1) Contexte et finalite

Ce projet vise a concevoir une solution **data/IA** capable de predire les coupures electriques dans un contexte hospitalier (cas d'etude : Lacor Hospital) et a **dimensionner un systeme de stockage hybride** (batteries, H2, EnR) pour garantir la continuite des soins critiques.

Le systeme s'appuie sur :
- des donnees reelles (Lacor Hospital),
- des profils comparables (NHS ERIC, NYC LL84, sites africa_grid clones),
- des donnees meteo (Open-Meteo),
- des donnees de contexte reseau/pays (OMS, Eskom).

---

## 2) Objectifs du projet

### 2.1 Objectif principal

Construire un **pipeline complet de prediction de coupures** electriques et une **interface applicative** permettant a la fois :
- une interpretation metier des resultats (cote Data/ML),
- un **dimensionnement** energetique oriente charges vitales (cote Energie/Electricite).

### 2.2 Objectifs fonctionnels

- Predire la probabilite de coupure (`is_outage` / `P_outage`).
- Estimer le temps avant prochaine coupure (indicateur derive).
- Estimer la duree probable de coupure (indicateur derive).
- Fournir des explications interpretables du resultat (SHAP / importance des variables).
- Permettre une utilisation via une interface Streamlit en mode **analyse** et **simulation** (scenario Data + scenario stockage).

### 2.3 Objectifs techniques

- Industrialiser un pipeline data reproductible (ingestion -> preprocessing -> features -> entrainement -> evaluation).
- Comparer plusieurs modeles ML (Random Forest, XGBoost, LightGBM, LSTM optionnel).
- Eviter les biais de fuite de donnees (data leakage) via split temporel et selection de features.
- Produire une base documentaire claire et exploitable (Data + Energie).
- Relier les sorties ML (profils de charge critique) a des **calculs de dimensionnement** (kW / kWh) sur les charges vitales.

### 2.4 Objectif specifique du binome Energie

Creer un **systeme de stockage energetique** adapte a des facteurs multi-criteres :
- duree et timing des coupures (resultats ML),
- conditions meteo locales,
- zone d'implantation et contraintes reseau,
- niveau de criticite des charges hospitalieres (vitales vs non vitales).

---

## 3) Perimetre

### 3.1 Inclus dans le perimetre

- Ingestion multi-sources:
  - Lacor (donnees de charge et cible `is_outage` terrain),
  - NHS ERIC et NYC LL84 (profils hospitaliers comparables),
  - Open-Meteo Archive + Forecast (meteo historique et previsionnelle),
  - Electricity Maps (contexte reseau zone, hors features modele),
  - EskomSePush (delestage programme RSA, contexte app),
  - OMS (fiabilite electrique par pays, parametres dans `hospitals.py`).
- Nettoyage et fusion des donnees temporelles.
- Feature engineering (variables temporelles, consommation, meteo, contexte).
- Entrainement nowcast (RF / XGB / LGBM) + horizons 1/3/6 h (memes features).
- Calibration des probabilites.
- Generation d'explications locales/globales (SHAP).
- Exposition des resultats dans l'application Streamlit.
- Analyse energetique:
  - segmentation des charges (vital vs confort),
  - bilan de puissance/energie (kW / kWh),
  - pre-dimensionnement de stockage (batteries/H2) a partir des profils critiques.

### 3.2 Hors perimetre (phase actuelle)

- Deploiement cloud haute disponibilite.
- MLOps complet (CI/CD modele, monitoring production automatise).
- Collecte temps reel connectee a des IoT/SCADA hospitaliers.
- Validation clinique/reglementaire formelle.
- Detail d'ingenierie de raccordement (etudes d'execution detaillees, plans definitifs HT/BT).

---

## 4) Parties prenantes

- **Porteur du projet**: equipe Master MSE (Rafael, Thomas, Ines, Muhammed Ali).
- **Volet Data & IA**: Rafael, Thomas.
- **Volet Energie & Electricite**: Ines, Muhammed Ali.
- **Utilisateur cible**: responsable technique hopital, energie/maintenance, decideur operationnel.
- **Encadrant/lecteur**: evaluateurs academiques / jury.

---

## 5) Exigences fonctionnelles

### 5.1 Pipeline de donnees (Data/ML)

- Le pipeline doit pouvoir etre execute de bout en bout via `run_pipeline.py`.
- Chaque etape doit produire des artefacts lisibles et versionnes (CSV, joblib, JSON).
- Les erreurs de chargement doivent etre explicites (messages clairs).

### 5.2 Modelisation

- Le systeme doit entrainer plusieurs modeles et selectionner le meilleur selon des metriques definies.
- Le split entrainement/test doit respecter la chronologie.
- Les probabilites doivent etre calibrables.

### 5.3 Application (Streamlit)

- L'utilisateur doit pouvoir:
  - choisir un profil d'hopital,
  - consulter une probabilite de coupure,
  - visualiser les facteurs explicatifs,
  - simuler un scenario manuel de charge/meteo,
  - visualiser l'effet d'un scenario de stockage (batterie/H2) sur la couverture des charges vitales.

### 5.4 Volet Energie & Electricite

- Identifier et quantifier les **charges vitales** vs **charges de confort**.
- Calculer un bilan de puissance/energie representatif (profil journalier / annuel).
- Proposer un **pre-dimensionnement** de stockage (capacite kWh, puissance kW) et d'EnR compatible avec les profils critiques identifies par le modele.
- Determiner le potentiel local de production (solaire, biomasse, eolien, generateur) et en deduire un mix technique realiste.

---

## 6) Exigences non fonctionnelles

- **Performance**: inferer une prediction en quelques secondes sur poste local.
- **Fiabilite**: execution reproductible avec dependances figees (`requirements.txt`).
- **Maintenabilite**: code modulaire par couche (`src/data`, `src/features`, `src/models`).
- **Lisibilite**: documentation fonctionnelle et technique a jour.
- **Traçabilite**: artefacts d'entrainement sauvegardes (`models/`, `data/features/`, `training_summary.json`).

---

## 7) Donnees et architecture cible

### 7.1 Flux de traitement

1. Ingestion des sources heterogenes (Excel/CSV/API).
2. Harmonisation temporelle et nettoyage.
3. Fusion des jeux de donnees.
4. Construction de features derivees.
5. Entrainement/validation/calibration.
6. Explicabilite des predictions.
7. Exploitation des profils critiques pour le dimensionnement (energie).
8. Consommation des resultats dans l'interface Streamlit.

### 7.2 Artefacts attendus

- Dataset fusionne preprocess (`data/processed/...`).
- Dataset features (`data/features/features_dataset.csv`).
- Modeles sauvegardes (`models/baseline_model.joblib`, `models/calibrated_model.joblib`).
- Explicabilite (`models/shap_explainer.joblib`, `models/shap_values.npz`).
- Fichiers de synthese (`models/model_comparison.csv`, `models/training_summary.json`).
- Dossier de dimensionnement (note de calcul, hypotheses, resultats).

---

## 8) Livrables

- **L1 - Pipeline data/ML fonctionnel** (scripts ingestion, preprocessing, features, training).
- **L2 - Modele baseline compare et evalue** (RF/XGB/LGBM + metriques).
- **L3 - Module de prediction** (`src/models/predict.py`).
- **L4 - Interface Streamlit operationnelle** (`app.py`).
- **L5 - Documentation complete** (donnees, APIs, modele, usage).
- **L6 - Dossier de dimensionnement energetique** (calculs, hypotheses, scenarios).
- **L7 - Cahier des charges unifie (ce document)**.

---

## 9) Planning previsionnel (mi-mars -> mi-juin)

Periode cible: **du 15 mars au 15 juin** (13 semaines environ).

### Phase 1 - Cadrage et collecte (S1 a S3: 15 mars -> 4 avril)

- Validation du besoin metier et des objectifs (Data + Energie).
- Inventaire et qualification des sources de donnees (Lacor, Meteo, OMS, Eskom, benchmarks).
- Mise en place de la structure projet et environnement Python.
- Premieres briques d'ingestion API/CSV/Excel.

**Jalon J1 (fin S3)**: sources principales accessibles, scripts d'ingestion initialises.

### Phase 2 - Preparation data et feature engineering (S4 a S6: 5 avril -> 25 avril)

- Nettoyage des donnees, traitement des valeurs manquantes/incoherentes.
- Synchronisation temporelle et fusion des sources.
- Construction des features (temporelles, charge, meteo, reseau).
- Verification de la qualite et de la stabilite des donnees.
- Premier bilan de charge (vital vs confort) pour le volet Energie.

**Jalon J2 (fin S6)**: dataset fusionne exploitable et jeu de features finalise + premier bilan energetique.

### Phase 3 - Modelisation et evaluation (S7 a S9: 26 avril -> 16 mai)

- Entrainement baseline et comparaison multi-modeles.
- Validation croisee temporelle et mesure des performances.
- Detection/correction des risques de fuite de donnees.
- Calibration des probabilites et analyse de robustesse.
- Utilisation des resultats ML pour identifier les **scenarios critiques** (jours/creux de reseau) qui alimentent le dimensionnement.

**Jalon J3 (fin S9)**: meilleur modele retenu, metriques consolidees, scenarios critiques identifies.

### Phase 4 - Explicabilite, application et dimensionnement (S10 a S11: 17 mai -> 30 mai)

- Integration SHAP (global/local).
- Mise en oeuvre des fonctions de prediction inferentielles.
- Integration dans l'interface Streamlit (analyse + simulation).
- Pre-dimensionnement detaille du stockage (batteries/H2) et du mix EnR en s'appuyant sur les scenarios critiques.

**Jalon J4 (fin S11)**: application demonstrable de bout en bout + pre-dimensionnement documente.

### Phase 5 - Stabilisation et livraison (S12 a S13: 31 mai -> 15 juin)

- Tests de non-regression et verification fonctionnelle.
- Finalisation de la documentation technique/fonctionnelle et du dossier de dimensionnement.
- Preparation de la soutenance/demo (scenarios, captures, narration, messages cles Data & Energie).

**Jalon J5 (15 juin)**: livraison complete du projet.

---

### 9.1 Macro-planning detaille du binome Energie (CdC specifique)

| Etape | Echeance cible | Resultat attendu |
| :--- | :--- | :--- |
| Determiner le cas a etudier | Mi-fin mars | Cas d'etude verrouille (site, hypotheses, perimetre) |
| 2A. Determiner le besoin en energie `Wh(t)` / `W(t)` par heure (electricite, chaud, froid) | Mi-avril | Profil horaire par usage, quantite/nature, repartition par salle, niveaux de priorite |
| 2B. Dimensionner (estimation) le stockage | Mi-avril | Premiere estimation capacite (kWh) et puissance (kW) du stockage |
| 2C. Determiner (estimation) le potentiel de production (solaire, biomasse, eolien, generateur) | Fin avril | Evaluation comparative des gisements et contraintes locales |
| Selectionner et dimensionner les moyens de production | Fin mai | Choix du mix final et dimensionnement consolidé |
| Rapport + presentation | Mi-juin | Dossier final + support de soutenance |

---

## 10) Criteres d'acceptation

Le projet est considere comme valide si:

- Le pipeline s'execute sans erreur sur l'environnement cible.
- Les donnees sont fusionnees correctement et les artefacts sont produits.
- Un modele de prediction est entraine, sauvegarde, et exploitable.
- L'application fournit une probabilite de coupure et des explications (SHAP).
- Les metriques minimales sont documentees (precision, recall, F1, ROC AUC, calibration).
- Le dossier de dimensionnement montre que les **charges vitales** sont couvertes sur la duree de coupure cible avec un coefficient de securite explicite.
- Le dimensionnement integre explicitement les facteurs: **timing des coupures, meteo, zone et priorites de charge**.
- Les limites et hypotheses sont explicitement decrites (cote Data et cote Energie).

---

## 11) Risques et plan de mitigation

- **Risque qualite donnees** (donnees manquantes/bruitees):
  - Mitigation: controles qualite, regles de nettoyage explicites, fallbacks.
- **Risque d'incoherence temporelle**:
  - Mitigation: normalisation timezone/frequence, verification des index datetime.
- **Risque de surapprentissage**:
  - Mitigation: split chronologique, TimeSeriesSplit, suivi des metriques train/test.
- **Risque de fuite de donnees**:
  - Mitigation: liste stricte des colonnes exclues, revue des features sensibles.
- **Risque de planning**:
  - Mitigation: jalons hebdomadaires, priorisation livrables critiques.
- **Risque de sous-dimensionnement energetique**:
  - Mitigation: scenarios de stress prudents + coefficient de securite (ex. `k = 1.2`) clairement indique.

---

## 12) Hypotheses et contraintes

- Travail realise en Python avec dependances definies.
- Execution majoritairement locale (pas d'obligation cloud).
- Le dataset principal de reference reste Lacor 2022.
- Les predicteurs de "temps avant coupure" et "duree" sont derives dans cette phase et peuvent etre raffines dans une phase 2.
- Le dimensionnement energetique reste au niveau **pre-etude / avant-projet**, non engageant pour une mise en oeuvre reelle sans etudes complementaires.

---

## 13) Gouvernance et suivi

- Point d'avancement hebdomadaire (etat, blocages, actions).
- Revue de jalon a chaque fin de phase (J1 a J5).
- Validation finale sur demonstration complete: pipeline + modele + app + dossier de dimensionnement + documentation.

---

## 14) Features prioritaires (Rafael & Thomas)

### P0 - Critiques (a implementer en premier)

| Feature | Source | Type | Definition / Formule | Usage |
| :--- | :--- | :--- | :--- | :--- |
| `total_load_kw` | Lacor | brute | Charge instantanee (kW) | Signal principal de stress |
| `is_outage` | Lacor | cible | `1 - grid_available` | Variable a predire |
| `load_rolling_6h` | derivee | numerique | Moyenne glissante sur 6h | Tendance court terme |
| `load_rolling_24h` | derivee | numerique | Moyenne glissante sur 24h | Tendance journaliere |
| `load_std_24h` | derivee | numerique | Ecart-type glissant 24h | Volatilite du reseau/charge |
| `load_diff_1h` | derivee | numerique | `load[t] - load[t-1]` | Changement brutal |
| `peak_ratio` | derivee | numerique | `total_load_kw / load_rolling_24h` | Detection de pics |
| `hour_sin`, `hour_cos` | temporel | numerique | Encodage cyclique heure | Saisonnalite intrajour |
| `day_of_week` | temporel | categoriel numerise | Jour 0..6 | Profil hebdomadaire |

### P1 - Importantes (meteo et mix energetique)

| Feature | Source | Type | Definition / Formule | Usage |
| :--- | :--- | :--- | :--- | :--- |
| `temperature_2m` | Open-Meteo | brute | Temperature horaire | Effet thermique sur charge |
| `relative_humidity_2m` | Open-Meteo | brute | Humidite horaire | Stress climatique |
| `precipitation` | Open-Meteo | brute | Pluie (mm) | Conditions defavorables |
| `wind_speed_10m` | Open-Meteo | brute | Vent (km/h) | Risque meteo |
| `shortwave_radiation` | Open-Meteo | brute | Rayonnement (W/m2) | Potentiel solaire |
| `solar_ratio` | Lacor derivee | numerique | `solar_pv_kw / total_load_kw` | Couverture EnR locale |
| `generator_ratio`* | Lacor derivee | numerique | `generators_kw / total_load_kw` | Dependance diesel |
| `temp_humidity_interaction` | derivee | numerique | `temperature * humidity / 100` | Proxy inconfort/charge |

`*` A exclure des modeles de prediction pure si fuite d'information confirmee.

### P2 - Contexte macro (a conserver pour extension multi-sites)

| Feature | Source | Type | Definition / Formule | Usage |
| :--- | :--- | :--- | :--- | :--- |
| `who_reliability_pct` | OMS | contexte | Fiabilite electrique pays (%) | Ajustement risque geographique |
| `reliability_risk` | OMS derivee | contexte | `1 - who_reliability_pct/100` | Score de fragilite |
| `loadshed_avg_stage` | Eskom | contexte | Moyenne niveau delestage | Indice d'instabilite reseau |
| `loadshed_pct_active` | Eskom | contexte | `% du temps en delestage` | Exposition au risque |

### Regles de gouvernance features

- Exclure toutes les variables introduisant de la fuite de cible.
- Garder les variables constantes pour le reporting, mais les retirer de l'entrainement.
- Versionner la liste finale des features actives dans la documentation modele.
- Valider les features via SHAP + importance globale + tests temporels.

---

**Version :** 1.2 (unifiee MSE)  
**Statut :** pret pour revue equipe / encadrant
