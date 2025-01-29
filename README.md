# intro-ai
## Classification de chiffres manuscrits (MNIST) avec un réseau de neurones (MLP)
### Phase 1 : Classification de fruits avec un arbre de décision

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
etiquettes = ["Pomme", "Banane", "Orange", "Pomme", "Banane"] # Les noms des fruits correspondants

# Créer un OneHotEncoder pour convertir les caractéristiques catégorielles en numériques
encoder = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0, 1])], # Appliquer OneHotEncoder aux colonnes 0 et 1 (couleur et forme)
    remainder='passthrough'  # Laisser les autres colonnes inchangées (s'il y en avait)
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
Grâce à ce code, on obtiens :
> Prédictions pour les nouveaux fruits :
> 
> Un fruit Rouge et Ronde est prédit comme étant un(e) : Pomme
> 
> Un fruit Jaune et Allongée est prédit comme étant un(e) : Banane
> 
> Un fruit Vert et Ronde est prédit comme étant un(e) : Pomme
> 
Voilà !
