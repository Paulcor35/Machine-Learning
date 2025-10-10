import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------- Chargement et préparation des données -------------------------

# Lecture du fichier CSV préparé
df = pd.read_csv("Data-20251001/ozone_prepared.csv")

target_col = "maxO3"
assert target_col in df.columns, f"La colonne cible '{target_col}' est absente du fichier !"

# Séparer les features (X) et la target (y)
X = df.drop(columns=[target_col, "id"], errors='ignore').to_numpy(dtype=np.float64)
y = df[target_col].to_numpy(dtype=np.float64)

# Assurer la continuité en mémoire pour des calculs plus rapides
X = np.ascontiguousarray(X)
y = np.ascontiguousarray(y)

# Supprimer les colonnes constantes (inutile pour la régression)
non_constant_cols = np.any(X != X[0, :], axis=0)
X = X[:, non_constant_cols]

# Split train/test (80% / 20%)
np.random.seed(2)
n = len(y)
indices = np.random.permutation(n)
train_size = int(0.8 * n)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ------------------------- Fonctions utilitaires -------------------------

def variance(y):
    """
    Calcule la variance d'un vecteur y.
    Utile pour mesurer l'hétérogénéité d'un sous-ensemble lors de la régression.
    """
    return np.var(y)

def mean_val(y):
    """
    Calcule la moyenne d'un vecteur y.
    Sert pour prédire la valeur d'une feuille.
    """
    return np.mean(y)

def best_split(X, y, threshold_sample=50, min_gain=1e-6):
    """
    Trouve le meilleur attribut et seuil pour diviser le dataset en régression.
    
    - X : tableau 2D des features
    - y : tableau 1D des labels
    - threshold_sample : nombre de seuils à tester pour limiter les calculs
    - min_gain : gain minimal pour effectuer une division
    
    Retour :
        best_attr : indice de l'attribut
        best_threshold : valeur du seuil
    """
    n_samples, n_features = X.shape
    base_var = variance(y)
    best_gain, best_attr, best_threshold = -1.0, -1, 0.0

    for j in range(n_features):
        X_j = X[:, j]

        # Tri des valeurs pour accélérer le calcul des variances cumulées
        sort_idx = np.argsort(X_j)
        X_j_sorted = X_j[sort_idx]
        y_sorted = y[sort_idx]

        unique_vals = np.unique(X_j_sorted)
        if len(unique_vals) > threshold_sample:
            # Sélection de seuils selon des quantiles pour limiter le calcul
            thresholds = np.quantile(unique_vals, np.linspace(0, 1, threshold_sample))
        else:
            thresholds = unique_vals

        # Sommes cumulées pour calculer variance plus efficacement
        sum_y = np.cumsum(y_sorted)
        sum_y2 = np.cumsum(y_sorted ** 2)
        total_y = sum_y[-1]
        total_y2 = sum_y2[-1]

        for t in thresholds:
            left_n = np.searchsorted(X_j_sorted, t, side='right')
            right_n = n_samples - left_n
            if left_n == 0 or right_n == 0:
                continue  # éviter division par zéro

            # Variance du sous-ensemble gauche
            sum_left_y = sum_y[left_n - 1]
            sum_left_y2 = sum_y2[left_n - 1]
            left_mean = sum_left_y / left_n
            left_var = sum_left_y2 / left_n - left_mean ** 2

            # Variance du sous-ensemble droit
            sum_right_y = total_y - sum_left_y
            sum_right_y2 = total_y2 - sum_left_y2
            right_mean = sum_right_y / right_n
            right_var = sum_right_y2 / right_n - right_mean ** 2

            # Variance moyenne pondérée
            new_var = (left_n / n_samples) * left_var + (right_n / n_samples) * right_var
            gain = base_var - new_var

            if gain > best_gain:
                best_gain = gain
                best_attr = j
                best_threshold = t

    if best_gain < min_gain:
        return -1, 0.0  # pas de division
    return best_attr, best_threshold

# ------------------------- Construction récursive de l'arbre -------------------------

def dt_learning_regression(X, y, idx, depth=0, max_depth=None, min_samples_split=5, parent_mean=None):
    """
    Construction récursive de l'arbre de régression.
    
    - X, y : données
    - idx : indices des échantillons à considérer
    - depth : profondeur actuelle
    - max_depth : profondeur maximale (early stopping)
    - min_samples_split : nombre minimal d'échantillons pour scinder
    - parent_mean : moyenne du parent pour feuilles vides
    
    Retour :
        Arbre sous forme de tuple (attr, seuil, left_tree, right_tree) ou valeur pour feuille
    """
    if len(idx) == 0:
        return parent_mean if parent_mean is not None else 0.0

    y_sub = y[idx]
    n_samples = len(y_sub)

    # Conditions d'arrêt
    if len(np.unique(y_sub)) == 1:
        return y_sub[0]
    if max_depth is not None and depth >= max_depth:
        return mean_val(y_sub)
    if n_samples < min_samples_split:
        return mean_val(y_sub)

    # Trouver le meilleur split
    attr, threshold = best_split(X[idx], y_sub)
    if attr == -1:
        return mean_val(y_sub)

    # Indices pour les sous-ensembles gauche et droit
    left_idx = idx[X[idx, attr] <= threshold]
    right_idx = idx[X[idx, attr] > threshold]

    # Construction récursive
    tree = (attr, threshold,
            dt_learning_regression(X, y, left_idx, depth + 1, max_depth, min_samples_split, mean_val(y_sub)),
            dt_learning_regression(X, y, right_idx, depth + 1, max_depth, min_samples_split, mean_val(y_sub))
           )
    return tree

# ------------------------- Prédiction avec l'arbre -------------------------

def predict_regression(tree, x):
    """
    Prédit la valeur d'une observation x donnée à partir de l'arbre de régression.
    
    - tree : arbre construit par dt_learning_regression
    - x : vecteur 1D d'une observation
    """
    if not isinstance(tree, tuple):
        return tree
    attr, threshold, left, right = tree
    if x[attr] <= threshold:
        return predict_regression(left, x)
    else:
        return predict_regression(right, x)

# ------------------------- Entraînement et évaluation -------------------------

print("Évaluation du modèle from scratch\n")

start_train = time.time()
tree = dt_learning_regression(X, y, np.arange(len(y_train)), max_depth=5)
end_train = time.time()
print(f"Temps d'apprentissage : {end_train - start_train:.4f} s")

# Prédiction sur test
start_pred = time.time()
y_pred_scratch = np.array([predict_regression(tree, x) for x in X_test])
end_pred = time.time()
print(f"Temps de prédiction : {end_pred - start_pred:.4f} s\n")

mse = np.mean((y_test - y_pred_scratch) ** 2)
r2 = 1 - np.sum((y_test - y_pred_scratch) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}\n")

# ------------------------- Comparaison avec scikit-learn -------------------------

print("Évaluation avec sklearn:\n")
regressor = DecisionTreeRegressor(criterion="squared_error", max_depth=5, random_state=42)

start_train = time.time()
regressor.fit(X_train, y_train)
end_train = time.time()
print(f"Temps d'apprentissage : {end_train - start_train:.4f} s")

start_pred = time.time()
y_pred_sklearn = regressor.predict(X_test)
end_pred = time.time()
print(f"Temps de prédiction : {end_pred - start_pred:.4f} s\n")

mse = mean_squared_error(y_test, y_pred_sklearn)
r2 = r2_score(y_test, y_pred_sklearn)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
