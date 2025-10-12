# classes/DecisionTreeRegressor.py
#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class _Node:
    is_leaf: bool
    value: float | None = None
    attr: int | None = None
    threshold: float | None = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

def _variance(y: np.ndarray) -> float:
    m = y.mean()
    return float((y * y).mean() - m * m)

def _best_split_vectorized(X: np.ndarray,
                           y: np.ndarray,
                           min_samples_leaf: int = 1,
                           max_splits_per_feature: int | None = None,
                           min_gain: float = 1e-6) -> tuple[int, float]:
    """
    Trouve le meilleur split sur TOUTES les features en un minimum de Python.
    - Tri par feature (au nœud).
    - Un scan vectorisé pour tous les splits valides.
    - Option: sous-échantillonne les positions si très nombreuses (max_splits_per_feature).
    Retourne (best_attr, best_threshold) ou (-1, 0.0) si pas de gain.
    """
    n, p = X.shape
    base_var = _variance(y)
    best_gain = -1.0
    best_attr = -1
    best_thr = 0.0
    min_gain = float(min_gain)

    for j in range(p):
        xj = X[:, j]
        order = np.argsort(xj, kind="mergesort")
        xs = xj[order]
        ys = y[order]

        # positions où la valeur change → seuls splits entre valeurs distinctes
        diff = np.diff(xs)
        valid_pos = np.nonzero(diff != 0)[0]  # positions i = 0..n-2, split entre i et i+1

        if valid_pos.size == 0:
            continue

        # min_samples_leaf : on restreint les positions valides
        mask_leaf = (valid_pos + 1 >= min_samples_leaf) & (valid_pos + 1 <= n - min_samples_leaf)
        valid_pos = valid_pos[mask_leaf]
        if valid_pos.size == 0:
            continue

        # si trop de positions, on en échantillonne
        if max_splits_per_feature is not None and valid_pos.size > max_splits_per_feature:
            step = int(np.ceil(valid_pos.size / max_splits_per_feature))
            valid_pos = valid_pos[::step]

        # préfixes pour calculer var gauche/droite rapidement
        csum = np.cumsum(ys, dtype=np.float64)
        csum2 = np.cumsum(ys * ys, dtype=np.float64)

        left_n = valid_pos + 1
        right_n = n - left_n

        left_sum = csum[valid_pos]
        left_sum2 = csum2[valid_pos]
        left_mean = left_sum / left_n
        left_var = left_sum2 / left_n - left_mean * left_mean

        right_sum = csum[-1] - left_sum
        right_sum2 = csum2[-1] - left_sum2
        right_mean = right_sum / right_n
        right_var = right_sum2 / right_n - right_mean * right_mean

        new_var = (left_n / n) * left_var + (right_n / n) * right_var
        gains = base_var - new_var

        k = int(np.argmax(gains))
        gain = float(gains[k])
        if gain > best_gain and gain > min_gain:
            best_gain = gain
            best_attr = j
            # seuil = milieu entre xs[i] et xs[i+1]
            i = int(valid_pos[k])
            best_thr = float((xs[i] + xs[i + 1]) * 0.5)

    if best_attr < 0:
        return -1, 0.0
    return best_attr, best_thr


class DecisionTreeRegressor:
    """
    DecisionTreeRegressor (arbre de décision pour la régression) implémenté en NumPy,
    avec recherche de split vectorisée et plusieurs garde-fous de régularisation.

    Principe
    --------
    On construit récursivement un arbre binaire qui partitionne l’espace des features.
    À chaque nœud, on choisit le split (feature j, seuil t) qui réduit le plus
    la variance de la cible (critère MSE). Une feuille prédit la moyenne des y
    présents dans cette feuille.

    Paramètres
    ----------
    max_depth : int | None
        Profondeur maximale de l’arbre. None → croissance jusqu’aux critères d’arrêt.
    min_samples_split : int
        Nombre minimal d’échantillons requis dans un nœud pour tenter un split.
    min_samples_leaf : int
        Nombre minimal d’échantillons requis dans chaque feuille après le split.
    max_splits_per_feature : int | None
        Limite le nombre de positions de coupure évaluées par feature au nœud (sous-échantillonnage
        des positions valides) afin d’accélérer l’apprentissage. None → toutes les positions valides.
    min_gain : float
        Gain de variance minimal requis pour accepter un split (sinon on crée une feuille).

    Notes
    -----
    - Critère : réduction de la variance (équivalent MSE). Le meilleur split minimise
      la variance pondérée des sous-nœuds gauche/droite.
    - Implémentation du split :
        • Tri des valeurs d’une feature au nœud (stable) ;
        • Calculs vectorisés via préfixes/cumsum pour évaluer toutes les coupures valides
          en O(n) par feature ;
        • Option `max_splits_per_feature` pour limiter le nombre de coupures testées
          quand il y en a beaucoup (gros n, valeurs continues).
    - Arrêts :
        • profondeur atteinte (max_depth),
        • effectif insuffisant (min_samples_split/min_samples_leaf),
        • gain < min_gain,
        • y constant dans le nœud.
    - Prédiction : parcours de l’arbre jusqu’à une feuille, renvoie la moyenne locale.

    Avantages
    ---------
    - Modèle interprétable et rapide sur données tabulaires.
    - Supporte les relations non linéaires et les interactions entre variables.
    - Version vectorisée significativement plus rapide que des boucles Python naïves.

    Attributs principaux
    --------------------
    tree : _Node | None
        Racine de l’arbre entraîné (structure récursive de nœuds).
    typ : str
        'r' pour indiquer une tâche de régression.
    """

    typ = "r"

    def __init__(self,
                 max_depth: int | None = 5,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 1,
                 max_splits_per_feature: int | None = 256,
                 min_gain: float = 1e-6):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_splits_per_feature = (None if max_splits_per_feature is None
                                       else int(max_splits_per_feature))
        self.min_gain = float(min_gain)
        self.tree: _Node | None = None

    def _build(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, depth: int) -> _Node:
        y_sub = y[idx]
        n = y_sub.size

        if n == 0:
            return _Node(is_leaf=True, value=0.0)
        if (self.max_depth is not None and depth >= self.max_depth) or n < self.min_samples_split:
            return _Node(is_leaf=True, value=float(y_sub.mean()))
        if np.all(y_sub == y_sub[0]):
            return _Node(is_leaf=True, value=float(y_sub[0]))

        X_sub = X[idx]
        attr, thr = _best_split_vectorized(
            X_sub, y_sub,
            min_samples_leaf=self.min_samples_leaf,
            max_splits_per_feature=self.max_splits_per_feature,
            min_gain=self.min_gain
        )
        if attr == -1:
            return _Node(is_leaf=True, value=float(y_sub.mean()))

        left_mask = X_sub[:, attr] <= thr
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]

        # sécurités min_samples_leaf
        if left_idx.size < self.min_samples_leaf or right_idx.size < self.min_samples_leaf:
            return _Node(is_leaf=True, value=float(y_sub.mean()))

        left = self._build(X, y, left_idx, depth + 1)
        right = self._build(X, y, right_idx, depth + 1)
        return _Node(is_leaf=False, attr=attr, threshold=thr, left=left, right=right)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32, order="C")
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        idx = np.arange(y.size, dtype=np.int32)
        self.tree = self._build(X, y, idx, depth=0)
        return self

    def _predict_one(self, x: np.ndarray, node: _Node) -> float:
        while not node.is_leaf:
            node = node.left if x[node.attr] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        if self.tree is None:
            raise RuntimeError("DecisionTreeRegressor non entraîné.")
        X = np.asarray(X, dtype=np.float32, order="C")
        return np.array([self._predict_one(X[i], self.tree) for i in range(X.shape[0])],
                        dtype=np.float32)
