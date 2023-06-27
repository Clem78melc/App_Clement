# Description du projet

L’entreprise souhaite mettre en œuvre un outil de "scoring crédit" pour calculer la probabilité qu'un client rembourse son crédit, puis classer la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s'appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d'octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l'entreprise veut incarner.

Prêt à dépenser décide donc de développer un tableau de bord interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d'octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

## Fonction coût métier

J'ai créé une fonction coût métier qui prend en compte qu'un faux-négatif coûte 10 fois plus qu'un faux-positif. Nous allons donc créer une fonction de score qui augmentera de 10 en cas de faux-négatif et de 1 en cas de faux-positif. L'objectif sera de minimiser cette fonction de score. Nous prendrons également en compte le score AUC et le temps d'entraînement du modèle.

J'utiliserai la fonction `make_scorer` de scikit-learn afin de pouvoir utiliser ce score dans la fonction `RandomizedSearchCV` pour optimiser les hyperparamètres des modèles que je vais entraîner.

Pour optimiser le seuil de classification, j'ai également créé une fonction qui prendra en paramètres les probabilités prédites par le modèle et les classes réelles. Cette fonction retournera le seuil qui permet d'optimiser le coût métier créé. Pour cela, elle testera toutes les possibilités de 0 à 1 avec un pas personnalisable et retournera celui qui permet d'obtenir le coût minimum.

## Entraînement du modèle

Dans un premier temps, j'ai nettoyé les données grâce à un script récupéré sur Kaggle. Tout le crédit revient à Arezoo Dahesh.

Les algorithmes utilisés sont les suivants : XGBoost, LightGBM, LogisticRegression et AdaBoost.

J'ai également utilisé un DummyClassifier pour comparer les modèles utilisés à un modèle naïf.

Pour chaque modèle, à l'exception de la régression logistique, nous passerons par une recherche aléatoire des hyperparamètres (RandomizedSearchCV) afin de trouver ceux qui permettront d'optimiser la fonction de coût.

Je dispose d'un jeu de données d'entraînement et d'un jeu de données de test (qui ne contient pas les cibles). Je vais donc utiliser des méthodes de validation croisée (par exemple, grâce à la fonction `cross_val_predict` de scikit-learn) pour entraîner et tester les modèles.

LightGBM est le modèle qui permet d'obtenir les meilleurs scores, c'est donc avec lui que j'ai testé les différentes méthodes d'échantillonnage (sampling) pour optimiser encore le score.

![](scores_modèles.png "Performances des différents modèles")

## Traitement du déséquilibre des classes

Pour traiter le déséquilibre des classes, nous allons utiliser différentes méthodes d'échantillonnage (sampling).

- SMOTE
- Oversampling
- Undersampling


LightGBM est le modèle qui permet d'obtenir les meilleurs scores, c'est donc avec lui que j'ai testé les différentes méthodes d'échantillonnage pour optimiser encore le score.

SMOTE permet de créer de nouvelles lignes correspondant à la classe minoritaire en utilisant la méthode des plus proches voisins. L'oversampling duplique des lignes de la classe minoritaire de manière aléatoire pour rééquilibrer, tandis que l'undersampling supprime des lignes de la classe majoritaire.

Nous allons comparer ces différentes méthodes au simple fait de passer le paramètre `class_weight: balanced`, qui permet de donner un poids supérieur à la classe minoritaire par rapport à la classe majoritaire.

![](scores_sampling.png "Performances des différentes de sampling")

## L'interprétabilité globale et locale du modèle

Pour réaliser l'interprétabilité globale du modèle, nous allons utiliser la bibliothèque SHAP.

![](streamlit/Featur_exp.png "Explications des features globales")

Ce que nous constatons, c'est que les caractéristiques les plus importantes sont plutôt des métriques métier qui ne sont pas forcément interprétables pour nous.

Pour l'interprétabilité locale, nous allons plutôt utiliser la bibliothèque LIME qui est plus adaptée. 

## L'analyse du Data Drift

J'ai analysé grâce à la bibliothèque Evidency le data drift entre les données d'entraînement et les données de test pour vérifier que les données sur lesquelles nous allons entraîner notre modèle sont les plus similaires possibles aux données de test. Le rapport présent dans le dossier montre bien qu'il n'y a pas de data drift entre notre jeu de données d'entraînement et de test.



