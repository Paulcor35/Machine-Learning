#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #

import numpy as np
import pandas as pd
from dataclasses import dataclass
import time

@dataclass
class _Node:
    """
    Classe générique pour représenter un noeud de l'arbre.
    """
    is_leaf: bool
    value: float = None           # pour feuille
    attr: int = None              # index de feature
    threshold: float = None       # seuil pour split
    left: object = None           # sous-arbre gauche
    right: object = None          # sous-arbre droit

# ------------------------- Fonctions utilitaires -------------------------

def variance(y):
    return np.var(y)

def mean_val(y):
    return np.mean(y)

def best_split(X, y, threshold_sample=50, min_gain=1e-6):
    n_samples, n_features = X.shape
    base_var = variance(y)
    best_gain, best_attr, best_threshold = -1.0, -1, 0.0

    for j in range(n_features):
        X_j = X[:, j]
        sort_idx = np.argsort(X_j)
        X_j_sorted = X_j[sort_idx]
        y_sorted = y[sort_idx]

        unique_vals = np.unique(X_j_sorted)
        if len(unique_vals) > threshold_sample:
            thresholds = np.quantile(unique_vals, np.linspace(0, 1, threshold_sample))
        else:
            thresholds = unique_vals

        sum_y = np.cumsum(y_sorted)
        sum_y2 = np.cumsum(y_sorted ** 2)
        total_y = sum_y[-1]
        total_y2 = sum_y2[-1]

        for t in thresholds:
            left_n = np.searchsorted(X_j_sorted, t, side='right')
            right_n = n_samples - left_n
            if left_n == 0 or right_n == 0:
                continue

            sum_left_y = sum_y[left_n - 1]
            sum_left_y2 = sum_y2[left_n - 1]
            left_mean = sum_left_y / left_n
            left_var = sum_left_y2 / left_n - left_mean ** 2

            sum_right_y = total_y - sum_left_y
            sum_right_y2 = total_y2 - sum_left_y2
            right_mean = sum_right_y / right_n
            right_var = sum_right_y2 / right_n - right_mean ** 2

            new_var = (left_n / n_samples) * left_var + (right_n / n_samples) * right_var
            gain = base_var - new_var

            if gain > best_gain:
                best_gain = gain
                best_attr = j
                best_threshold = t

    if best_gain < min_gain:
        return -1, 0.0
    return best_attr, best_threshold

# ------------------------- Arbre de régression -------------------------

class DecisionTreeRegressor:
    """
    Arbre de régression simplifié "from scratch" avec splits numériques uniquement.
    """

    typ = "r"

    def __init__(self, max_depth=5, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _build_tree(self, X, y, idx, depth=0, parent_mean=None):
        if len(idx) == 0:
            return _Node(is_leaf=True, value=parent_mean if parent_mean is not None else 0.0)

        y_sub = y[idx]
        n_samples = len(y_sub)

        if len(np.unique(y_sub)) == 1 or (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split:
            return _Node(is_leaf=True, value=mean_val(y_sub))

        attr, threshold = best_split(X[idx], y_sub)
        if attr == -1:
            return _Node(is_leaf=True, value=mean_val(y_sub))

        left_idx = idx[X[idx, attr] <= threshold]
        right_idx = idx[X[idx, attr] > threshold]

        left_node = self._build_tree(X, y, left_idx, depth + 1, mean_val(y_sub))
        right_node = self._build_tree(X, y, right_idx, depth + 1, mean_val(y_sub))

        return _Node(is_leaf=False, attr=attr, threshold=threshold, left=left_node, right=right_node)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        idx = np.arange(len(y))
        self.tree = self._build_tree(X, y, idx)
        return self

    def _predict_one(self, x, node):
        while not node.is_leaf:
            if x[node.attr] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(X[i], self.tree) for i in range(X.shape[0])])

# ------------------------- Exemple d'utilisation -------------------------

if __name__ == "__main__":
    # Exemple synthétique
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=200, n_features=5, noise=10.0, random_state=42)
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    print("Entraînement de l'arbre from scratch...")
    start_train = time.time()
    dtr = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
    dtr.fit(X_train, y_train)
    end_train = time.time()
    print(f"Temps d'apprentissage : {end_train - start_train:.4f} s")

    print("Prédiction...")
    start_pred = time.time()
    y_pred = dtr.predict(X_test)
    end_pred = time.time()
    print(f"Temps de prédiction : {end_pred - start_pred:.4f} s")

    mse = np.mean((y_test - y_pred) ** 2)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")


    # ---------------- Compat scikit-like ----------------

    def get_params(self, deep=True):
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self
