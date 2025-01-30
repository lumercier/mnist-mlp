# Classification de chiffres manuscrits (MNIST) avec un réseau de neurones (MLP)
## Phase 1 : Classification de fruits avec un arbre de décision
### 1. et 2.
De base, le script **_1_classification_fruits.py_** comporte une erreur.
> ValueError: could not convert string to float: 'Rouge'

Cela vient du fait que Scikit-learn ne peut pas traiter des chaînes de caractères comme "Rouge" ou "Ronde".\
Il faut convertir ces valeurs en nombres avant d'entraîner le modèle, grâce aux bibliothèques suivantes :
```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
```
On applique OneHotEncoder aux deux colonnes (couleur et forme).

```python
encoder = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0, 1])], # Appliquer OneHotEncoder aux colonnes 0 et 1 (couleur et forme)
    remainder='passthrough'  # Laisser les autres colonnes inchangées (s'il y en avait)
)
```

&nbsp;  

Voici donc le script fonctionnel :
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

attributs = [
    ["Rouge", "Ronde"],
    ["Jaune", "Allongée"],
    ["Orange", "Ronde"],
    ["Vert", "Ronde"],
    ["Jaune", "Ronde"]
]
etiquettes = ["Pomme", "Banane", "Orange", "Pomme", "Banane"]

encoder = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0, 1])],
    remainder='passthrough'
)

# Encoder les attributs
attributs_encoded = encoder.fit_transform(attributs)

# Créer un modèle d'arbre de décision (un modèle simple de classification)
modele = DecisionTreeClassifier()

# Apprendre au modèle à partir des données (l'entraîner)
modele.fit(attributs_encoded, etiquettes)

# Faire des prédictions pour de nouveaux fruits
nouveaux_fruits = [
    ["Rouge", "Ronde"],
    ["Jaune", "Allongée"],
    ["Vert", "Ronde"]
]

# Encoder les nouveaux fruits avant de faire des prédictions
nouveaux_fruits_encoded = encoder.transform(nouveaux_fruits)
predictions = modele.predict(nouveaux_fruits_encoded)

print("Prédictions pour les nouveaux fruits :")
for i in range(len(nouveaux_fruits)):
    print(f"Un fruit {nouveaux_fruits[i][0]} et {nouveaux_fruits[i][1]} est prédit comme étant un(e) : {predictions[i]}")

```
Grâce à ce code, on obtient :
> Prédictions pour les nouveaux fruits :  
> Un fruit Rouge et Ronde est prédit comme étant un(e) : Pomme  
> Un fruit Jaune et Allongée est prédit comme étant un(e) : Banane  
> Un fruit Vert et Ronde est prédit comme étant un(e) : Pomme

&nbsp;  
### 3. Modification des données
En modifiant les caractéristiques des nouveaux_fruits
```python
nouveaux_fruits = [
    ["Rouge", "Ronde"],
    ["Jaune", "Allongée"],
    ["Vert", "Ronde"],
    ["Orange", "Ronde"],
    ["Vert", "Allongée"],
    ["Orange", "Allongée"]
]
```
On obtient
> Prédictions pour les nouveaux fruits :  
> Un fruit Rouge et Ronde est prédit comme étant un(e) : Pomme  
> Un fruit Jaune et Allongée est prédit comme étant un(e) : Banane  
> Un fruit Vert et Ronde est prédit comme étant un(e) : Pomme  
> Un fruit Orange et Ronde est prédit comme étant un(e) : Orange  
> Un fruit Vert et Allongée est prédit comme étant un(e) : Pomme  
> Un fruit Orange et Allongée est prédit comme étant un(e) : Orange

&nbsp;   
En ajoutant des attributs et etiquettes
```python
attributs = [
    ...
    ["Bleu", "Ronde"],
    ["Marron", "Allongée"]
]
etiquettes = [..., "Myrtille", "Date"]
```
Puis en rajoutant de nouveaux fruits :
```python
nouveaux_fruits = [
    ...
    ["Bleu", "Ronde"],
    ["Marron", "Allongée"]
]
```
On obtient
>  Prédictions pour les nouveaux fruits :  
>  Un fruit Rouge et Ronde est prédit comme étant un(e) : Pomme  
>  Un fruit Jaune et Allongée est prédit comme étant un(e) : Banane  
>  Un fruit Vert et Ronde est prédit comme étant un(e) : Pomme  
>  Un fruit Orange et Ronde est prédit comme étant un(e) : Orange  
>  Un fruit Vert et Allongée est prédit comme étant un(e) : Date  
>  Un fruit Orange et Allongée est prédit comme étant un(e) : Date  
>  Un fruit Bleu et Ronde est prédit comme étant un(e) : Myrtille  
>  Un fruit Marron et Allongée est prédit comme étant un(e) : Date  

&nbsp; 
### 4. Résumé

Ce code entraîne un modèle d'arbre de décision pour classifier des fruits en fonction de leur couleur et forme.\
On lui donne des _attributs_ (par exemple *Jaune*, *Allongée*), auquels on associé une _étiquette_ (exemple : *Banane*).\
Comme les arbres de décision ne peuvent pas traiter du texte, le code va convertir ce texte en variables binaires (0 et 1).\
\
Une fois le modèle entraîné avec les données existantes, il est utilisé pour prédire le type de fruit en fonction de nouvelles entrées, ici _nouveaux_fruits_

* Un fruit Rouge et Ronde est classé comme une Pomme.
* Un fruit Jaune et Allongée est classé comme une Banane.
* Un fruit Orange et Ronde est classé comme une Orange.

Le modèle réussit ainsi à généraliser la classification des fruits sur la base des caractéristiques fournies.\
\
On peut imaginer combiner ce code avec une IA qui analyserait une photo, afin de détécter quels sont les fruits présents sur la photo.  
&nbsp;  
&nbsp;  
&nbsp;  

## Phase 2 : Construction d'un modèle MLP sur MNIST - étapes préliminaires


