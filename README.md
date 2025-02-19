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
### 1. Executing the script
Now, we run **_2_construire_mlp_mnist.py_** \
\
Once the script is executed, the output is :
>  Forme des données d'images (X) : (70000, 784)  
>  Forme des étiquettes (y) : (70000,)  
>  Modèle MLP simple créé : MLPClassifier(hidden_layer_sizes=(50,), max_iter=10)  
>  Modèle MLP à deux couches créé : MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10)  
>  MLP avec optimiseur Adam : MLPClassifier(hidden_layer_sizes=(50,), max_iter=10)  
>  MLP avec optimiseur SGD : MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01, max_iter=10, solver='sgd')  

&nbsp; 
### 2. Checking MNIST datas
The **MNIST database** is a large database of handwritten digits, which contains 60,000 training images and 10,000 testing images (each is a 28x28 pixel bounding box and anti-aliased, with grayscale levels). \
It is commonly used for training various image processing systems (and so in the field of machine learning). \
&nbsp;  
First, we import libraries then load the **MNIST database**.
```python
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import numpy as np

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target
```

We print data and label shapes
```python
print("Forme des données d'images (X) :", X.shape)
print("Forme des étiquettes (y) :", y.shape)
```
_X.shape → (70000, 784)_ : 70 000 images of 784 pixels (28x28). \
_y.shape → (70000,)_ : 70 000 labels, one for each image.

&nbsp; 

Then, data is being prepared
```python
X = X / 255.0
```
Scaling: Pixels have values between 0 and 255 (grayscale). \
We divide it by 255 to normalize between 0 and 1. \
Why? Better standardization accelerates learning and improves accuracy.

&nbsp; 

Then we build a first MLP (1 single hidden layer of 50 neurons)
```python
mlp_simple = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10)
print("Modèle MLP simple créé :", mlp_simple)
```
And a second MLP (2 hidden layers, the first with 100 neurons and the second one with 50 neurons).
```python
mlp_deux_couches = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10)
print("Modèle MLP à deux couches créé :", mlp_deux_couches)
```

&nbsp; 

Finally, we try 2 different optimization algorithms.  \
With Adam
```python
mlp_adam = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', max_iter=10)
print("MLP avec optimiseur Adam :", mlp_adam)
```

With SGD
```python
mlp_sgd = MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', learning_rate_init=0.01, max_iter=10)
print("MLP avec optimiseur SGD :", mlp_sgd)
```

&nbsp; 

#### The script does not perform a full training. It focuses on creating and configuring different MLP models. *_max_iter=10_* is too low to perform a good learning.

&nbsp; 

### 3. Steps required to build an MLP model with scikit-learn and hyperparameters
Building an MLP with Scikit-Learn requires :
- Load and prepare the data.
- Divide into train/test.
- Create a MLP model (MLPClassifier).
- Train (.fit())
- Evaluate (.score())

Key hyperparameters :
- *_hidden_layer_sizes_* → Number of neurons/layers.
- *_max_iter_* → Number of learning iterations.
- *_solver_* → Optimization algorithm.
- *_learning_rate_init_* → Learning rate.
- *_activation_* → Function of activation of neurons.

&nbsp; 

## Phase 3 : Training and evaluation of a MLP model on MNIST

















