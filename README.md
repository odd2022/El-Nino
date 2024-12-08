# El-Nino
Machine Learning for Climate and Energy Project with Nathan Toumi
Notes Prepresentation

1)ajouter plus de variables (pressure, plus de mois)
2) (transformation polynomiale des inputs)
3) base (ex LDA ou OLS) sur 1 mois de données seulement ou  + regularization pour classification (PCA) courbe de validation meilleur compromis biais variance 
4) tester un modèle non linéaire (MLP, random forest, hyper parameter important)

Semaine 1. Preprocessing pour avoir les données labellisées. DONE
Semaine 2. Faire les prédictions pour 1, 2 et 3 mois (chacun prend un modèle) (au début on split facile dans l'ordre du temps)
Semaine 3. Tester des modèles plus complexe + Faire le diapo

Ce qui a ete fait depuis le debut: 
- faire une baseline de LDA et trouver la PCA qui marche le mieux.
- tester différents modèles et faire la PCA dessus.

  A faire:
  - Faire Linear Regression et Regularization ?
  - Faire Transfo Polynomiales des Entrées ?
  - Changer les Inputs: Ajouter la Sea pressure et les données de mois
  - regarder les tutoriels pour avoir des idées de quoi changer 

Questions pour le prof lundi: 
1) ajouter plus de variables: ajouter 2 mois plutot qu'un pour augmenter la complexité du modèle ? ou juste ajouter un mois mais avant ? Dans ce cas comment procéder ? 
2) transformation polynomiale des inputs: comment procéder?
3) Faire une linear regression et courbe de validation ?
