#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Définition de la structure d’un nœud de l’arbre
# ---------------------------------------------------------------------------
@dataclass
class _Node:
    is_leaf: bool                         # Indique si le nœud est une feuille
    value: float | None = None            # Valeur prédite si c’est une feuille
    attr: int | None = None               # Index de la feature utilisée pour le split
    threshold: float | None = None        # Seuil de coupure pour cette feature
    left: Optional["_Node"] = None        # Sous-arbre gauche
    right: Optional["_Node"] = None       # Sous-arbre droit

# ---------------------------------------------------------------------------
# Fonction pour calculer la variance d’un vecteur cible y
# ---------------------------------------------------------------------------
def _variance(y: np.ndarray) -> float:
    """
    Calcule la variance de y selon Var(y) = E[y^2] - (E[y])^2
    """
    m = y.mean()
    return float((y * y).mean() - m * m)

# ---------------------------------------------------------------------------
# Recherche vectorisée du meilleur split pour la régression
# ---------------------------------------------------------------------------
def _best_split_vectorized(X: np.ndarray,
                           y: np.ndarray,
                           min_samples_leaf: int = 1,
                           max_splits_per_feature: int | None = None,
                           min_gain: float = 1e-6) -> tuple[int, float]:
    """
    Trouve le meilleur split (feature, seuil) pour minimiser la variance intra-nœud.
    
    Utilisation :
    - Tri stable des valeurs de chaque feature
    - Calcul vectorisé des gains de variance pour tous les splits valides
    - Possibilité de sous-échantillonner les seuils si trop nombreux

    Retour :
    - (best_feature, best_threshold)
    ou (-1, 0.0) si aucun split pertinent n'est trouvé
    """
    n, p = X.shape
    base_var = _variance(y)  # variance initiale du nœud
    best_gain = -1.0
    best_attr = -1
    best_thr = 0.0
    min_gain = float(min_gain)

    for j in range(p):
        xj = X[:, j]
        order = np.argsort(xj, kind="mergesort")  # tri stable pour conserver l'ordre relatif
        xs = xj[order]
        ys = y[order]

        # Positions où la valeur change (split possible uniquement entre valeurs distinctes)
        diff = np.diff(xs)
        valid_pos = np.nonzero(diff != 0)[0]  # positions 0..n-2

        if valid_pos.size == 0:
            continue

        # Restriction min_samples_leaf
        mask_leaf = (valid_pos + 1 >= min_samples_leaf) & (valid_pos + 1 <= n - min_samples_leaf)
        valid_pos = valid_pos[mask_leaf]
        if valid_pos.size == 0:
            continue

        # Si trop de splits, on prend un échantillon
        if max_splits_per_feature is not None and valid_pos.size > max_splits_per_feature:
            step = int(np.ceil(valid_pos.size / max_splits_per_feature))
            valid_pos = valid_pos[::step]

        # Sommes cumulées pour calcul vectorisé des variances gauche/droite
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
            i = int(valid_pos[k])
            best_thr = float((xs[i] + xs[i + 1]) * 0.5)  # seuil = milieu entre deux valeurs

    if best_attr < 0:
        return -1, 0.0
    return best_attr, best_thr

# ---------------------------------------------------------------------------
# Classe principale : Arbre de décision pour la régression
# ---------------------------------------------------------------------------
class DecisionTreeRegressor:
    """
    Arbre de décision pour régression implémenté avec NumPy et recherche vectorisée.
    
    Chaque feuille prédit la moyenne des y présents.
    """

    typ = 'r'  # type = regression

    def __init__(self,
                 max_depth: int | None = 5,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 1,
                 max_splits_per_feature: int | None = 256,
                 min_gain: float = 1e-6):
        """
        Initialisation avec paramètres de régularisation.
        """
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_splits_per_feature = (None if max_splits_per_feature is None
                                       else int(max_splits_per_feature))
        self.min_gain = float(min_gain)
        self.tree: _Node | None = None  # racine de l’arbre

    def _build(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, depth: int) -> _Node:
        """
        Construction récursive de l’arbre.
        """
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

        # Sécurité : min_samples_leaf
        if left_idx.size < self.min_samples_leaf or right_idx.size < self.min_samples_leaf:
            return _Node(is_leaf=True, value=float(y_sub.mean()))

        left = self._build(X, y, left_idx, depth + 1)
        right = self._build(X, y, right_idx, depth + 1)
        return _Node(is_leaf=False, attr=attr, threshold=thr, left=left, right=right)

    def fit(self, X, y):
        """
        Entraîne l’arbre sur X et y.
        """
        X = np.asarray(X, dtype=np.float32, order="C")
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        idx = np.arange(y.size, dtype=np.int32)
        self.tree = self._build(X, y, idx, depth=0)
        return self

    def _predict_one(self, x: np.ndarray, node: _Node) -> float:
        """
        Prédiction pour un échantillon unique.
        """
        while not node.is_leaf:
            node = node.left if x[node.attr] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        """
        Prédiction pour plusieurs échantillons.
        """
        if self.tree is None:
            raise RuntimeError("The model isn't trained, call `fit` first")
        X = np.asarray(X, dtype=np.float32, order="C")
        return np.array([self._predict_one(X[i], self.tree) for i in range(X.shape[0])],
                        dtype=np.float32)
