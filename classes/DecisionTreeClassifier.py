#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Définition d’un nœud pour l’arbre de classification
# ---------------------------------------------------------------------------
@dataclass
class _Node:
    is_leaf: bool
    prediction: object | None = None      # Classe prédite si c’est une feuille
    attr: int | None = None               # Index de la feature utilisée pour le split
    threshold: float | None = None        # Seuil si split numérique
    branches: dict | None = None          # Dictionnaire de branches :
                                          # {"le": _Node, "gt": _Node} pour num, ou {val: _Node} pour catégoriel

# ---------------------------------------------------------------------------
# Fonction pour calculer l'entropie d’un vecteur y
# ---------------------------------------------------------------------------
def _entropy(y: np.ndarray) -> float:
    """
    Entropie = - sum(p_i * log2(p_i)), avec p_i = proportion de chaque classe
    """
    if y.size == 0:
        return 0.0
    vals, cnt = np.unique(y, return_counts=True)
    p = cnt.astype(np.float64) / y.size
    return -float(np.sum(p[p > 0.0] * np.log2(p[p > 0.0])))  # ignore p==0

# ---------------------------------------------------------------------------
# Recherche vectorisée du meilleur split pour classification
# ---------------------------------------------------------------------------
def _best_split_vectorized(
    X: np.ndarray,
    y: np.ndarray,
    min_samples_leaf: int = 1,
    max_thresholds: int | None = 256,
    min_gain: float = 1e-6,
) -> tuple[int | None, float | None, float]:
    """
    Cherche le meilleur split basé sur l'entropie sur toutes les colonnes.

    - Pour les colonnes numériques : seuil = milieu entre valeurs distinctes consécutives,
      sous-échantillonné si > max_thresholds
    - Pour les colonnes catégorielles : gain calculé par partition {valeur}
    
    Retour : (attr, threshold, gain)
      threshold=None pour split catégoriel
    """
    n, p = X.shape
    base_h = _entropy(y)
    best_gain, best_attr, best_thr = -1.0, None, None

    for j in range(p):
        col = X[:, j]

        # Split numérique (float/int et >2 valeurs uniques)
        if np.issubdtype(col.dtype, np.number) and np.unique(col).size > 2:
            order = np.argsort(col, kind="mergesort")
            xs = col[order].astype(np.float64, copy=False)
            ys = y[order]

            diff = np.diff(xs)
            pos = np.nonzero(diff != 0)[0]  # positions valides pour split
            if pos.size == 0:
                continue

            # min_samples_leaf
            mask_leaf = (pos + 1 >= min_samples_leaf) & (pos + 1 <= n - min_samples_leaf)
            pos = pos[mask_leaf]
            if pos.size == 0:
                continue

            # sous-échantillonnage si trop de seuils
            if max_thresholds is not None and pos.size > max_thresholds:
                step = int(np.ceil(pos.size / max_thresholds))
                pos = pos[::step]

            # cumulatif par classe pour calcul vectorisé des entropies
            classes, y_idx = np.unique(ys, return_inverse=True)
            C = classes.size
            one_hot = np.zeros((ys.size, C), dtype=np.int32)
            one_hot[np.arange(ys.size), y_idx] = 1
            csum = np.cumsum(one_hot, axis=0)           # cumule les comptes par classe

            left_cnt = csum[pos]                        # (k, C) pour chaque split
            left_n = left_cnt.sum(axis=1).astype(np.float64)
            right_cnt = csum[-1] - left_cnt
            right_n = right_cnt.sum(axis=1).astype(np.float64)

            # fonction pour calculer entropie à partir des comptes
            def ent_from_counts(cnt):
                tot = cnt.sum(axis=1, keepdims=True)
                p = np.divide(cnt, tot, out=np.zeros_like(cnt, dtype=np.float64), where=tot > 0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    e = -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
                return e

            H_left = ent_from_counts(left_cnt)
            H_right = ent_from_counts(right_cnt)
            new_H = (left_n / n) * H_left + (right_n / n) * H_right
            gains = base_h - new_H

            k = int(np.argmax(gains))
            gain = float(gains[k])
            if gain > best_gain and gain > float(min_gain):
                best_gain = gain
                best_attr = j
                i = int(pos[k])
                best_thr = float((xs[i] + xs[i + 1]) * 0.5)

        else:
            # Split catégoriel / binaire
            vals, counts = np.unique(col, return_counts=True)
            if vals.size <= 1:
                continue
            new_H = 0.0
            for v, c in zip(vals, counts):
                if c < min_samples_leaf or (n - c) < min_samples_leaf:
                    new_H = base_h
                    break
                y_sub = y[col == v]
                new_H += (c / n) * _entropy(y_sub)
            gain = base_h - new_H
            if gain > best_gain and gain > float(min_gain):
                best_gain = gain
                best_attr = j
                best_thr = None

    return best_attr, best_thr, best_gain

# ---------------------------------------------------------------------------
# Classe principale : Arbre de décision pour classification
# ---------------------------------------------------------------------------
class DecisionTreeClassifier:
    """
    Arbre de décision pour classification implémenté en NumPy avec recherche vectorisée.

    Chaque feuille prédit la classe majoritaire des échantillons.
    """
    typ = 'c'  # type classification

    def __init__(self,
                 max_depth: int | None = 10,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_thresholds: int | None = 50,
                 min_gain: float = 1e-6,
                 random_state: int | None = None):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_thresholds = None if max_thresholds is None else int(max_thresholds)
        self.min_gain = float(min_gain)
        self.random_state = random_state
        self._tree: _Node | None = None  # racine

    # -----------------------------------------------------------------------
    # Classe majoritaire pour les feuilles
    # -----------------------------------------------------------------------
    def _plurality(self, y: np.ndarray):
        return Counter(y).most_common(1)[0][0]

    # -----------------------------------------------------------------------
    # Construction récursive de l’arbre
    # -----------------------------------------------------------------------
    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n = y.size
        if n == 0:
            return _Node(is_leaf=True, prediction=None)
        if np.unique(y).size == 1:
            return _Node(is_leaf=True, prediction=y[0])
        if (self.max_depth is not None and depth >= self.max_depth) or n < self.min_samples_split:
            return _Node(is_leaf=True, prediction=self._plurality(y))

        attr, thr, gain = _best_split_vectorized(
            X, y,
            min_samples_leaf=self.min_samples_leaf,
            max_thresholds=self.max_thresholds,
            min_gain=self.min_gain
        )
        if attr is None or gain <= 0.0:
            return _Node(is_leaf=True, prediction=self._plurality(y))

        if thr is None:
            # split catégoriel
            branches = {}
            vals = np.unique(X[:, attr])
            for v in vals:
                mask = (X[:, attr] == v)
                if mask.sum() < self.min_samples_leaf or (n - mask.sum()) < self.min_samples_leaf:
                    branches[v] = _Node(is_leaf=True, prediction=self._plurality(y))
                else:
                    branches[v] = self._build(X[mask], y[mask], depth + 1)
            return _Node(is_leaf=False, attr=int(attr), threshold=None, branches=branches)
        else:
            # split numérique
            mask_le = X[:, attr] <= thr
            if mask_le.sum() < self.min_samples_leaf or (n - mask_le.sum()) < self.min_samples_leaf:
                return _Node(is_leaf=True, prediction=self._plurality(y))
            left = self._build(X[mask_le], y[mask_le], depth + 1)
            right = self._build(X[~mask_le], y[~mask_le], depth + 1)
            return _Node(is_leaf=False, attr=int(attr), threshold=float(thr), branches={"le": left, "gt": right})

    # -----------------------------------------------------------------------
    # API publique : entraînement
    # -----------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        self._tree = self._build(X, y, depth=0)
        return self

    # -----------------------------------------------------------------------
    # Si une valeur catégorielle inconnue apparaît, retourner la classe majoritaire
    # -----------------------------------------------------------------------
    def _majority_subtree(self, node: _Node):
        counts = Counter()
        stack = [node]
        while stack:
            n = stack.pop()
            if n.is_leaf:
                counts[n.prediction] += 1
            else:
                for child in n.branches.values():
                    stack.append(child)
        pred = counts.most_common(1)[0][0] if counts else None
        return _Node(is_leaf=True, prediction=pred)

    # -----------------------------------------------------------------------
    # Prédiction pour un seul échantillon
    # -----------------------------------------------------------------------
    def _predict_one(self, node: _Node, x: np.ndarray):
        while not node.is_leaf:
            if node.threshold is None:
                # split catégoriel
                v = x[node.attr]
                node = node.branches.get(v, self._majority_subtree(node))
                if node.is_leaf:
                    break
            else:
                # split numérique
                node = node.branches["le"] if x[node.attr] <= node.threshold else node.branches["gt"]
        return node.prediction

    # -----------------------------------------------------------------------
    # Prédiction pour plusieurs échantillons
    # -----------------------------------------------------------------------
    def predict(self, X):
        if self._tree is None:
            raise RuntimeError("DecisionTreeClassifier non entraîné : appelez fit d'abord.")
        X = np.asarray(X)
        return np.array([self._predict_one(self._tree, X[i]) for i in range(X.shape[0])])
