import pandas as pd
import numpy as np


# ------------------------- Chargement et préparation des données -------------------------

df = pd.read_csv("Data-20251001/ozone_prepared.csv")

# Colonne cible
target_col = "maxO3"

# Vérifier qu’elle existe
assert target_col in df.columns, f"La colonne cible '{target_col}' est absente du fichier !"

# Séparer features et target (exclure l'id si présent)
X = df.drop(columns=[target_col, "id"], errors='ignore')
y = df[target_col]


# ------------------------- Séparer train/test -------------------------

train_ratio = 0.8 
n = len(df)
np.random.seed(42) # Pour reproductibilité
shuffled_indices = np.random.permutation(n) # Indices mélangés
train_size = int(train_ratio * n)

train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:]

X_train = X.iloc[train_indices].reset_index(drop=True)
X_test = X.iloc[test_indices].reset_index(drop=True)
y_train = y.iloc[train_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)


# ------------------------- Arbre de régression from scratch -------------------------

def variance(df, target_attr):
    """
    Calcule la variance d'une colonne cible dans un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        target_attr (str): Nom de la colonne cible.
    
    Returns:
        float: Variance de la colonne cible.
    """
    return np.var(df[target_attr])


def importance_regression(attributes, df, target_attr):
    """
    Choisit l'attribut et le seuil qui maximise la réduction de variance
    (critère utilisé pour la régression dans un arbre de décision).
    
    Pour chaque attribut :
        - On trie les valeurs uniques.
        - On teste tous les seuils possibles (milieu entre deux valeurs consécutives).
        - On calcule la variance pondérée des sous-ensembles gauche et droite.
        - On choisit l'attribut et le seuil qui maximisent le gain de variance.
    
    Args:
        attributes (list): Liste des colonnes candidates pour le split.
        df (pd.DataFrame): DataFrame actuel pour le calcul.
        target_attr (str): Nom de la colonne cible.
    
    Returns:
        best_attr (str): Attribut qui donne le meilleur split.
        best_threshold (float): Seuil optimal pour ce split.
    """

    base_var = variance(df, target_attr)
    best_gain = -1
    best_attr = None
    best_threshold = None

    for attr in attributes:
        values = np.sort(df[attr].unique())
        for i in range(1, len(values)):
            # Seuil = milieu entre deux valeurs consécutives
            threshold = (values[i - 1] + values[i]) / 2

            # Séparer le DataFrame en deux sous-ensembles
            left = df[df[attr] <= threshold]
            right = df[df[attr] > threshold]

            # Ignorer si un des sous-ensembles est vide
            if len(left) == 0 or len(right) == 0:
                continue

            # Variance pondérée des sous-ensembles
            new_var = (len(left)/len(df)) * variance(left, target_attr) + \
                      (len(right)/len(df)) * variance(right, target_attr)

            # Gain = réduction de variance
            gain = base_var - new_var

            # Mémoriser le meilleur split
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
                best_threshold = threshold

    return best_attr, best_threshold


def mean_val(df, target_attr):
    """
    Calcule la moyenne de la colonne cible.
    Utilisé pour créer une feuille lorsque le split n'est plus possible.
    
    Args:
        df (pd.DataFrame): DataFrame actuel.
        target_attr (str): Nom de la colonne cible.
    
    Returns:
        float: Moyenne de la colonne cible.
    """

    return df[target_attr].mean()


def dt_learning_regression(df, attributes, parent_df, target_attr, depth=0, max_depth=None):
    """
    Construit un arbre de régression récursivement.
    
    Conditions d'arrêt :
        - DataFrame vide : retourne la moyenne du parent.
        - Toutes les valeurs cibles identiques : retourne cette valeur.
        - Plus d'attributs disponibles ou profondeur max atteinte : retourne la moyenne.
    
    Pour chaque noeud :
        - Sélectionner l'attribut et le seuil qui maximisent la réduction de variance.
        - Séparer le DataFrame en sous-ensembles gauche et droite.
        - Appeler récursivement la fonction sur chaque sous-ensemble.
    
    Args:
        df (pd.DataFrame): DataFrame à diviser.
        attributes (list): Liste des attributs disponibles.
        parent_df (pd.DataFrame): DataFrame du noeud parent (pour gérer les DataFrame vides).
        target_attr (str): Nom de la colonne cible.
        depth (int, optional): Profondeur actuelle de l'arbre.
        max_depth (int, optional): Profondeur maximale autorisée.
    
    Returns:
        dict ou float: L'arbre construit (ou valeur si feuille).
    """

    # Cas où le DataFrame est vide : retourner la moyenne du parent
    if df.empty:
        return mean_val(parent_df, target_attr)

    # Cas où toutes les valeurs cibles sont identiques : feuille
    if len(df[target_attr].unique()) == 1:
        return df[target_attr].iloc[0]

    # Cas où plus d'attributs ou profondeur max atteinte : feuille
    if not attributes or (max_depth is not None and depth >= max_depth):
        return mean_val(df, target_attr)

    # Sélection du meilleur attribut et seuil pour ce noeud
    A, threshold = importance_regression(attributes, df, target_attr)
    if A is None:  # Aucun split possible
        return mean_val(df, target_attr)

    # Construction de l'arbre récursivement
    tree = {A: {}}
    left = df[df[A] <= threshold]
    right = df[df[A] > threshold]

    tree[A][f"<= {threshold:.3f}"] = dt_learning_regression(left, attributes, df, target_attr, depth + 1, max_depth)
    tree[A][f"> {threshold:.3f}"] = dt_learning_regression(right, attributes, df, target_attr, depth + 1, max_depth)

    return tree




# ------------------------- Apprentissage et prédiction -------------------------

import time


def predict_regression(tree, example):
    """
    Prédit la valeur cible pour un exemple donné à partir d'un arbre de régression.

    Fonction récursive qui descend dans l'arbre en fonction des conditions 
    (attribut <= seuil ou > seuil) jusqu'à atteindre une feuille.

    Args:
        tree (dict ou float): Arbre de régression construit par dt_learning_regression.
                              Si c'est une feuille, c'est une valeur float.
        example (pd.Series): Exemple pour lequel on veut prédire la cible.

    Returns:
        float: Valeur prédite pour la cible.
    """
    # Cas feuille : retourner directement la valeur
    if not isinstance(tree, dict):
        return tree

    # Récupérer l'attribut sur lequel porte le noeud actuel
    attr = next(iter(tree))

    # Parcourir toutes les conditions de ce noeud (gauche/droite)
    for condition, subtree in tree[attr].items():
        # Extraire l'opérateur et le seuil depuis la clé
        op, threshold = condition.split(" ")
        threshold = float(threshold)

        # Valeur de l'exemple pour cet attribut
        value = example[attr]

        # Vérifier quelle branche suivre
        if op == "<=" and value <= threshold:
            return predict_regression(subtree, example)
        elif op == ">" and value > threshold:
            return predict_regression(subtree, example)

    # Si aucune condition ne correspond, retourner NaN
    return np.nan


print("Évaluation model from scratch:\n")
attributes = [col for col in X_train.columns]

start_train = time.time()
tree = dt_learning_regression(pd.concat([X_train, y_train], axis=1), attributes, pd.DataFrame(), target_col, max_depth=5) # Limite de profondeur à 5
end_train = time.time()
print(f"Temps d'apprentissage : {end_train - start_train:.4f} s")

start_pred = time.time()
y_pred_scratch = [predict_regression(tree, row) for _, row in X_test.iterrows()]
end_pred = time.time()
print(f"Temps de prédiction : {end_pred - start_pred:.4f} s\n")

mse = mean_squared_error(y_test, y_pred_scratch)
r2 = r2_score(y_test, y_pred_scratch)
print(f"MSE: {mse:.4f}\nR²: {r2:.4f}\n")



# ------------------------- Avec sklearn -------------------------

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


print("Évaluation avec sklearn:\n")
regressor = DecisionTreeRegressor(criterion="squared_error", max_depth=5, random_state=42) # Limiter la profondeur pour comparaison équitable


start_train = time.time()
regressor.fit(X_train, y_train)
end_train = time.time()
print(f"Temps d'apprentissage : {end_train - start_train:.4f} s")

start_pred = time.time()
y_pred_sklearn = regressor.predict(X_test)
end_pred = time.time()
print(f"Temps de prédiction : {end_pred - start_pred:.4f} s")

mse = mean_squared_error(y_test, y_pred_sklearn)
r2 = r2_score(y_test, y_pred_sklearn)
print(f"\nMSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
