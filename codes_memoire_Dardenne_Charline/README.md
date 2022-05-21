*******************************************************************************************************************
# README POUR LE MEMOIRE DE CHARLINE DARDENNE
*******************************************************************************************************************

Ce dossier comprend tous les codes nécessaires à la reproduction des résultats obtenus dans le Chapitre 4 et le 
Chapitre 5. Chaque sous-dossier correspond à une Section du mémoire excepté le sous-dossier nommé data_fMRI qui 
reprend les matrices comportant des signaux BOLD fournies par la groupe Coma Science Group de l’Université de 
Liège.

Les codes sont implémentés en langage Python (version 3.7.3).

## Autres_classificateurs
1. classification.py : Effectue une classification selon le modèle de classification sélectionné grâce à la 
variable « classif».
2. features_generation.py : Génère les features des données de type RGS selon les explications fournies lors de la 
troisième classification de la section 4.1 du mémoire. Les targets sont aussi générées par ce fichier.

## BoW
1. classification.py : Effectue une classification selon le modèle ExtraTreesClassifier avec le paramètre max_depth 
fixé à 4.
2. features_generation.py : Génère les features des données de type RGSF et RGS selon la méthode du sac de mots 
expliquée dans la Section 3.3 du mémoire. Les targets sont aussi générées par ce fichier.

## Data_augmentation 
1. classification1.py : Effectue une classification selon le modèle ExtraTreesClassifier avec le paramètre max_depth 
fixé à 4. Lors de cette classification, les sous-séries sont considérées comme indépendantes.
2. classification2.py : Effectue une classification selon le modèle ExtraTreesClassifier avec le paramètre max_depth 
fixé à 4. Lors de cette classification, les séries temporelles d’origine sont reformées en regroupant les 
sous-séries extraites.
3. features_generation.py : Applique une méthode d’augmentation des données de type RGS puis génère les features de 
ces données selon les explications fournies lors de la section 5.1 du mémoire. Les targets sont aussi générées par ce 
programme.

## Combinaison 
1. classification.py : Génère les features, les targets et effectue une combinaison de deux classifications selon les 
explications fournies dans la Section 5.4 du mémoire. 

## Gridsearch
1. classification.py : Effectue une classification selon le modèle ExtraTreesClassifier en choisissant, parmi une 
sélection, la valeur du paramètre max_depth qui optimise la performance du classificateur.
2. features_generation.py : Génère les features des données de type RGS selon les explications fournies lors de la 
troisième classification de la section 4.1 du mémoire. Les targets sont aussi générées par ce programme.
3. gs_param.py : Calcule les fréquences d’apparition des valeurs choisies par la méthode grid search pour le 
paramètre max_depth.

## Hankel 
1. classification.py : Effectue une classification selon le modèle ExtraTreesClassifier.
2. features_generation.py : Génère les features pour les données de type RGSF et RGS selon la méthode Hankel-DMD 
expliquée à la Section 5.3 du mémoire. Les targets sont aussi générées par ce programme.

## Premieres_classifications 
1. classification.py : Effectue une classification selon le modèle ExtraTreesClassifier.
2. features_generation.py : Génère les features des données de type RGSF et RGS en fonction de la variable « test ». 
Les explications concernant les sélections des features sont données dans la Section 4.1 du mémoire. Les targets sont 
aussi générées par ce programme.
