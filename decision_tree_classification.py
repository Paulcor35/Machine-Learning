import pandas as pd
import numpy as np
import math
from collections import Counter


# ------------------------- Chargement et préparation des données -------------------------

df_prepared = pd.read_csv("Data-20251001/Carseats_prepared.csv")

train_ratio = 0.8
n = len(df_prepared)
# Mélanger aléatoirement les indices pour créer un échantillon représentatif
shuffled_indices = np.random.permutation(n)
train_size = int(train_ratio * n)
train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:]

# Features (X_train) : toutes les colonnes sauf la cible 'High' pour les indices d'entraînement
# reset_index(drop=True) : réinitialise les indices du DataFrame pour qu'ils soient consécutifs
X_train = df_prepared.drop('High', axis=1).iloc[train_indices].reset_index(drop=True)

# Features (X_test) : toutes les colonnes sauf la cible 'High' pour les indices de test
X_test = df_prepared.drop('High', axis=1).iloc[test_indices].reset_index(drop=True)

y_train = df_prepared['High'].iloc[train_indices].reset_index(drop=True)
y_test = df_prepared['High'].iloc[test_indices].reset_index(drop=True)




# ------------------------- Arbre de classification from scratch -------------------------


def entropy(df, target_attr):
    """
    Calcule l'entropie d'une colonne cible.
    
    L'entropie est une mesure du désordre ou de l'incertitude dans un ensemble de données.
    Plus l'entropie est élevée, plus les classes sont mélangées et incertaines.
    
    Arguments :
    df : pandas.DataFrame -- le DataFrame contenant les données
    target_attr : str -- le nom de la colonne cible pour laquelle on calcule l'entropie
    
    Retour :
    float -- l'entropie de la colonne cible
    """

    values = df[target_attr]
    counter = Counter(values)
    total = len(df)
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count > 0)


def importance(attributes, df, target_attr):
    """
    Sélectionne le meilleur attribut pour diviser le dataset en maximisant le gain d'information.
    
    Pour chaque attribut :
    - Si l'attribut est catégoriel ou binaire, on calcule l'entropie moyenne pondérée
      des sous-ensembles correspondant à chaque valeur.
    - Si l'attribut est numérique non binaire, on teste tous les seuils possibles
      entre valeurs consécutives et on choisit le seuil qui maximise le gain d'information.
    
    Arguments :
    attributes : list -- liste des colonnes/features disponibles pour la division
    df : pandas.DataFrame -- le DataFrame contenant les données
    target_attr : str -- le nom de la colonne cible
    
    Retour :
    tuple (best_attr, best_threshold) :
        - best_attr : str, attribut qui maximise le gain d'information
        - best_threshold : float ou None, seuil optimal si numérique, None si catégoriel
    """

    base_entropy = entropy(df, target_attr)
    best_gain = -1
    best_attr = None
    best_threshold = None  # Pour stocker le seuil optimal si numérique
    
    for attr in attributes:
        if df[attr].dtype.kind in 'bifc' and len(df[attr].unique()) > 2:
            # Attribut numérique non binaire
            sorted_values = sorted(df[attr].unique())
            # Tester tous les seuils entre valeurs consécutives
            for i in range(len(sorted_values) - 1):
                threshold = (sorted_values[i] + sorted_values[i+1]) / 2
                subset1 = df[df[attr] <= threshold]
                subset2 = df[df[attr] > threshold]
                new_entropy = (len(subset1)/len(df)) * entropy(subset1, target_attr) + \
                              (len(subset2)/len(df)) * entropy(subset2, target_attr)
                gain = base_entropy - new_entropy
                if gain > best_gain:
                    best_gain = gain
                    best_attr = attr
                    best_threshold = threshold
        else:
            # Attribut catégoriel ou binaire
            values = df[attr].unique()
            new_entropy = 0
            for v in values:
                subset = df[df[attr] == v]
                new_entropy += (len(subset)/len(df)) * entropy(subset, target_attr)
            gain = base_entropy - new_entropy
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
                best_threshold = None  # pas de seuil pour les catégoriels
    
    return best_attr, best_threshold


def dt_learning(df, attributes, parent_df, target_attr):
    """
    Construit récursivement un arbre de décision pour la classification.
    
    À chaque nœud :
    1. Vérifie les cas de base :
       - DataFrame vide : retourne la classe majoritaire du parent
       - Toutes les instances appartiennent à la même classe : retourne cette classe
       - Plus d'attributs disponibles : retourne la classe majoritaire du noeud
    2. Sélectionne le meilleur attribut et seuil (si numérique) via la fonction importance
    3. Crée les branches :
       - Pour un attribut numérique, deux branches : <= seuil et > seuil
       - Pour un attribut catégoriel, une branche par valeur unique
    4. Applique la récursion sur chaque sous-ensemble
    
    Arguments :
    df : pandas.DataFrame -- DataFrame contenant les données courantes
    attributes : list -- liste des colonnes/features disponibles
    parent_df : pandas.DataFrame -- DataFrame parent, utilisé si df est vide
    target_attr : str -- le nom de la colonne cible
    
    Retour :
    dict ou valeur : arbre de décision sous forme de dictionnaire ou classe pour les feuilles
    """

    if df.empty:
        return plurality_val(parent_df, target_attr)
    if len(df[target_attr].unique()) == 1:
        return df[target_attr].iloc[0]
    if not attributes:
        return plurality_val(df, target_attr)
    
    A, threshold = importance(attributes, df, target_attr)
    tree = {A: {}}
    
    if threshold is not None:
        # Attribut numérique découpé en deux branches
        subset1 = df[df[A] <= threshold]
        subset2 = df[df[A] > threshold]
        tree[A][f"<= {threshold}"] = dt_learning(subset1, [attr for attr in attributes if attr != A], df, target_attr)
        tree[A][f"> {threshold}"] = dt_learning(subset2, [attr for attr in attributes if attr != A], df, target_attr)
    else:
        # Attribut catégoriel
        for v in df[A].unique():
            exs = df[df[A] == v]
            tree[A][v] = dt_learning(exs, [attr for attr in attributes if attr != A], df, target_attr)
    
    return tree


def plurality_val(df, target_attr):
    """
    Retourne la classe la plus fréquente dans la colonne cible.
    
    Utilisée pour :
    - Les feuilles lorsqu'aucune division supplémentaire n'est possible
    - Les cas où le DataFrame est vide
    
    Arguments :
    df : pandas.DataFrame -- DataFrame contenant les données
    target_attr : str -- le nom de la colonne cible
    
    Retour :
    valeur -- classe majoritaire
    """

    values = df[target_attr]
    return Counter(values).most_common(1)[0][0]





# ------------------------- Apprentissage et prédiction -------------------------

import time

def predict(tree, example):
    """
    Prédit la classe d'un exemple donné en utilisant un arbre de décision.
    
    Arguments :
    tree : dict ou valeur -- arbre de décision construit par dt_learning. 
           Si c'est une feuille, c'est directement la classe.
    example : pandas.Series -- un exemple contenant les valeurs des attributs
    
    Retour :
    str/int -- la classe prédite pour l'exemple
    """

    if not isinstance(tree, dict):
        return tree

    attr = next(iter(tree))
    node = tree[attr]

    # Vérifier si c'est un noeud avec seuil (numérique)
    if all(isinstance(k, str) and ("<=" in k or ">" in k) for k in node.keys()):
        # Extraire le seuil
        for k in node:
            if "<=" in k:
                threshold = float(k.split("<= ")[1])
                if example[attr] <= threshold:
                    return predict(node[k], example)
            elif ">" in k:
                threshold = float(k.split("> ")[1])
                if example[attr] > threshold:
                    return predict(node[k], example)
        # Cas improbable si aucune condition satisfaite
        return Counter([v if not isinstance(v, dict) else None for v in node.values()]).most_common(1)[0][0]
    
    # Sinon c'est un attribut catégoriel
    attr_value = example[attr]
    if attr_value not in node:
        # Descente pour récupérer toutes les feuilles
        values = []
        stack = [node]
        while stack:
            n = stack.pop()
            for v in n.values():
                if isinstance(v, dict):
                    stack.append(v)
                else:
                    values.append(v)
        return Counter(values).most_common(1)[0][0]

    return predict(node[attr_value], example)


# Définir les attributs
attributes = list(X_train.columns)

# Mesurer le temps d'apprentissage
start_train = time.time()
tree = dt_learning(pd.concat([X_train, y_train], axis=1), attributes, pd.DataFrame(), "High")
end_train = time.time()
print(f"Temps d'apprentissage (from scratch) : {end_train - start_train:.4f} secondes")

# Mesurer le temps de prédiction
start_pred = time.time()
y_pred = [predict(tree, X_test.iloc[i]) for i in range(len(X_test))]
end_pred = time.time()
print(f"Temps de prédiction (from scratch) : {end_pred - start_pred:.4f} secondes")

# Évaluation
y_pred = pd.Series(y_pred, name="Predicted")
y_true = y_test.reset_index(drop=True)

accuracy = (y_true == y_pred).mean()
print(f"\nAccuracy (test from scratch) : {accuracy*100:.2f}%")
print("Matrice de confusion :\n", pd.crosstab(y_true, y_pred, rownames=["Actual"], colnames=["Predicted"]))





# ------------------------- Avec sklearn -------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

clf = DecisionTreeClassifier(criterion="entropy", random_state=42)

# Temps d'apprentissage
start_train = time.time()
clf.fit(X_train, y_train)
end_train = time.time()
print(f"\nTemps d'apprentissage (scikit-learn) : {end_train - start_train:.4f} secondes")

# Temps de prédiction
start_pred = time.time()
y_pred_test = clf.predict(X_test)
end_pred = time.time()
print(f"Temps de prédiction (scikit-learn) : {end_pred - start_pred:.4f} secondes")

# Évaluation
accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nAccuracy (test scikit-learn) : {accuracy*100:.2f}%")
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred_test))
print("Rapport de classification :\n", classification_report(y_test, y_pred_test))
