# Student Mental Health Risk Prediction

Projet pédagogique de Data Science visant à prédire trois catégories de risque liées à la santé mentale des étudiants : **Low**, **Medium** et **High**.

Le projet utilisera **Pandas**, **NumPy** et **scikit-learn** pour la préparation des données et la classification multiclasse, **Apache Airflow** pour l'orchestration et **Streamlit** pour l'interface interactive.

Ce projet est destiné à l'apprentissage et ne constitue pas un outil de diagnostic médical.

## Structure du projet

```text
.
├── data/
│   ├── raw/            # Données brutes (exclues de Git)
│   └── processed/      # Données préparées
├── notebooks/          # Exploration et expérimentations
├── src/                # Code Python du projet
├── models/             # Modèles entraînés et artefacts finaux
├── dag/                # Futurs workflows Apache Airflow
├── app/                # Future application Streamlit
├── reports/
│   └── figures/        # Visualisations et figures
├── tests/              # Tests du projet
├── README.md
├── requirements.txt    # Dépendances à renseigner ultérieurement
└── .gitignore
```

Les fichiers `.gitkeep` permettent de suivre les dossiers vides dans Git. Les données brutes des étudiants doivent rester dans `data/raw/` et ne doivent pas être versionnées. Les secrets et les fichiers de configuration sensibles sont également exclus de Git.

## État initial

Seule la structure du projet est créée. Les dépendances, la préparation des données, les modèles, les workflows Airflow et l'application Streamlit seront ajoutés progressivement.
