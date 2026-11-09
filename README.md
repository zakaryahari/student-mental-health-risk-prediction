# 🧠 Student Mental Health Risk Prediction

**Projet Croisé — Data Science & Machine Learning | Jury blanc 2026**

Prototype pédagogique de **classification multiclasse** visant à analyser des données simulées d'étudiants et à prédire la catégorie `Mental_Health_Risk` : **Low**, **Medium** ou **High**.

> **Avertissement important :** ce projet est strictement pédagogique. Les catégories prédites ne constituent **ni un diagnostic, ni un dépistage clinique** et ne doivent pas servir à des décisions disciplinaires, à sélectionner des étudiants ou à hiérarchiser des personnes réelles. Les données et résultats doivent être utilisés avec prudence et dans le respect de la confidentialité.

## 🎯 Objectifs

- Auditer et explorer le jeu de données (EDA) avec NumPy et Pandas.
- Nettoyer les données et construire un prétraitement reproductible sans *data leakage*.
- Entraîner et comparer au moins **3 modèles de classification**.
- Évaluer les modèles, optimiser leurs hyperparamètres et analyser leurs limites.
- Sauvegarder le pipeline complet de prétraitement et de prédiction (`joblib`).
- Orchestrer les tâches du projet avec **Apache Airflow**.
- Développer un tableau de bord et une démonstration interactive avec **Streamlit**.

## 📊 Jeu de données

**Source :** fichier `student_mental_health.csv` fourni par l'enseignant, issu d'une enquête étudiante **simulée**.

| Élément | Description |
|---|---|
| Observations | 2 000 étudiants simulés |
| Colonnes | 17 au total |
| Cible | `Mental_Health_Risk` |
| Classes | `Low` (660), `Medium` (660), `High` (680) |
| Problème | Classification supervisée multiclasse |
| Valeurs manquantes | Présentes dans plusieurs variables |

**Variables disponibles :**

| Groupe | Colonnes |
|---|---|
| Identifiant | `Student_ID` |
| Profil et études | `Age`, `Gender`, `Year_of_Study`, `Study_Hours_Per_Week`, `GPA`, `Attendance_Rate`, `Part_Time_Job` |
| Habitudes de vie | `Sleep_Hours`, `Social_Media_Hours`, `Screen_Time_Hours`, `Physical_Activity`, `Diet_Quality` |
| Facteurs psychosociaux | `Financial_Stress`, `Family_Support`, `Previous_Mental_Health_History` |
| Cible | `Mental_Health_Risk` |

`Student_ID` sert uniquement d'identifiant et sera exclu des variables d'entrée du modèle. Les corrélations étudiées dans ce jeu de données synthétique ne permettent pas de déduire une relation de causalité ni de valider un risque clinique réel.

## 🗂️ Structure du dépôt

```text
student-mental-health-risk-prediction/
├── data/
│   ├── raw/                 # Données originales, non versionnées
│   │   └── .gitkeep
│   └── processed/           # Données préparées, non versionnées
│       └── .gitkeep
├── notebooks/               # Exploration et expérimentations Jupyter
│   └── .gitkeep
├── src/                     # Modules Python réutilisables
│   └── .gitkeep
├── models/                  # Pipelines entraînés (.joblib / .pkl)
│   └── .gitkeep
├── dag/                     # DAG Apache Airflow
│   └── .gitkeep
├── app/                     # Application Streamlit
│   └── .gitkeep
├── reports/
│   └── figures/             # Visualisations et captures d'écran
│       └── .gitkeep
├── tests/                   # Tests automatisés
│   └── .gitkeep
├── .gitignore
├── README.md
└── requirements.txt
```

## 🛠️ Technologies prévues

- **Python**, **NumPy**, **Pandas** : manipulation et analyse des données.
- **Matplotlib**, **Seaborn** : visualisation.
- **scikit-learn** : prétraitement, pipelines, classification et évaluation.
- **imbalanced-learn** : rééchantillonnage *uniquement si nécessaire*.
- **joblib** : sérialisation du pipeline.
- **Apache Airflow** : planification et suivi du workflow.
- **Streamlit** : interface de démonstration.
- **Git / GitHub**, **Jira** : versionnement et suivi des tâches.

## ⚙️ Installation (environnement local)

**Prérequis :** Git et une version de Python compatible avec les dépendances du projet.

```bash
# 1. Cloner le dépôt (remplacer par l'URL réelle)
git clone <URL_DU_DEPOT>
cd student-mental-health-risk-prediction

# 2. Créer un environnement virtuel
python -m venv .venv

# 3. Activer l'environnement
# Windows PowerShell :
.venv\Scripts\Activate.ps1
# macOS / Linux : source .venv/bin/activate

# 4. Installer les dépendances, une fois requirements.txt renseigné
python -m pip install -r requirements.txt
```

Placer localement le fichier fourni par l'enseignant dans `data/raw/student_mental_health.csv` sans le publier dans Git. **Le fichier `requirements.txt` sera complété lors de l'implémentation.** Les commandes de démarrage de Streamlit et d'Airflow seront ajoutées lorsque leurs fichiers seront créés et testés.

## 🔬 Démarche de réalisation

### 1. Audit et analyse exploratoire (EDA)
- Charger les données et examiner dimensions, types et valeurs manquantes.
- Contrôler doublons, incohérences et valeurs atypiques.
- Étudier la cible et ses relations avec les variables explicatives.
- Générer des graphiques et résumer les observations sans interprétation clinique.

### 2. Préparation des données
- Séparer **80 % entraînement / 20 % test**, avec stratification si possible.
- Exclure `Student_ID` et ne jamais utiliser la cible dans les variables explicatives.
- Imputer les valeurs manquantes et encoder les catégories.
- Standardiser les variables numériques lorsque le modèle l'exige.
- Combiner les transformations avec `ColumnTransformer` et `Pipeline`.
- Traiter d'éventuels déséquilibres **uniquement si l'analyse le justifie** ; appliquer tout rééchantillonnage uniquement sur les données d'entraînement.

### 3. Entraînement et comparaison
- Comparer au minimum trois classifieurs (candidats : régression logistique, Random Forest, SVM).
- Utiliser les mêmes partitions et un protocole de validation cohérent.
- Examiner **accuracy, précision, rappel, F1-score** et matrice de confusion.

### 4. Optimisation et évaluation finale
- Rechercher des hyperparamètres avec `GridSearchCV` ou `RandomizedSearchCV`.
- Sélectionner un modèle en considérant les performances **et** les limites des données.
- Évaluer **une seule fois** le pipeline sélectionné sur le test indépendant.
- Analyser les erreurs, la robustesse, les risques de biais et les limites d'usage.
- Sauvegarder le pipeline complet dans `models/` au format `.joblib` ou `.pkl`.

### 5. Orchestration et application
- Définir un **DAG Airflow** : préparation → entraînement → évaluation, avec suivi des erreurs et planification.
- Construire une application **Streamlit** affichant les métriques, graphiques, matrice de confusion et un formulaire de démonstration sur de nouvelles entrées.
- Valider les saisies, éviter l'exposition de données personnelles et afficher l'avertissement pédagogique.

## 📈 Résultats

**À compléter après les expérimentations.** Aucun score de performance n'est annoncé à ce stade.

| Modèle | Accuracy test | F1 macro test | Observations |
|---|---|---|---|
| Modèle retenu | À compléter | À compléter | À compléter |

Les modèles intermédiaires seront comparés **en validation croisée sur le train** ; le tableau final sera complété après évaluation indépendante.

## ✅ Tests et reproductibilité

Prévoir des contrôles sur les colonnes attendues, les valeurs manquantes, les catégories inconnues, les saisies invalides, le chargement du pipeline et la séparation stricte entre entraînement et test. Les versions de dépendances, la graine aléatoire et les instructions d'exécution seront documentées au fur et à mesure.

## 📦 Livrables

- [ ] Dépôt Git organisé et documenté.
- [ ] Notebooks et scripts Python du workflow Data Science.
- [ ] Pipeline complet sauvegardé (`.joblib` / `.pkl`).
- [ ] DAG Apache Airflow fonctionnel.
- [ ] Application Streamlit avec graphiques et démonstration de prédiction.
- [ ] README final, métriques, captures d'écran et instructions d'exécution.
- [ ] Lien du projet Jira.

**Jira :** `À renseigner`  
**Dépôt GitHub :** `À renseigner`

## 📅 Planning

| Date | Travail prévu |
|---|---|
| 09/11/2026 | Initialisation, audit et EDA |
| 10/11/2026 | Prétraitement |
| 11/11/2026 | Comparaison des modèles |
| 12/11/2026 | Optimisation et évaluation |
| 13/11/2026 | Airflow et Streamlit |
| 14/11/2026 | Tests, documentation et livraison |
| 16/11/2026 | Soutenance — jury blanc |

**Date limite de remise : 14 novembre 2026 à 23 h 59.**

---

*Projet individuel à vocation pédagogique. Ce dépôt documente une démonstration technique de classification sur données simulées, et non un dispositif médical.*
