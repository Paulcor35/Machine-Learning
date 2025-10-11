#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #

import numpy as np
import pandas as pd
import math
from collections import Counter
from dataclasses import dataclass

@dataclass
class _Node:
    is_leaf: bool
    prediction: object = None
    attr: int = None  # index de la feature (entier)
    threshold: float = None
    branches: dict = None  # {"le": _Node, "gt": _Node} ou {val: _Node} si catégoriel


class DecisionTreeClassifier:
    """
    Arbre de décision from scratch optimisé (classification) basé sur l'entropie.
    Version numpy pour efficacité (semblable à sklearn.DecisionTreeClassifier minimal).

    - Gère les attributs numériques et catégoriels (encodés en int)
    - API : fit(X, y), predict(X)
    - Hyperparamètres :
        max_depth, min_samples_split, min_samples_leaf, max_thresholds
    """

    typ = ['c']

    def __init__(self,
                 max_depth: int = 10,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_thresholds: int = 50,
                 random_state: int = None):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_thresholds = max_thresholds
        self.random_state = random_state
        self._tree = None
        self._n_features = None

    # ----------- Mesures et critères -----------

    def _entropy(self, y):
        """Calcule l'entropie d'un vecteur 1D numpy."""
        if len(y) == 0:
            return 0.0
        values, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def _plurality(self, y):
        """Retourne la classe majoritaire."""
        return Counter(y).most_common(1)[0][0]

    # ----------- Sélection du meilleur split -----------

    def _best_split(self, X, y, feature_indices):
        base_entropy = self._entropy(y)
        best_gain = -1.0
        best_attr = None
        best_thresh = None

        for attr in feature_indices:
            col = X[:, attr]
            unique_vals = np.unique(col)

            # Attribut numérique
            if len(unique_vals) > 2 and np.issubdtype(col.dtype, np.number):
                if len(unique_vals) > self.max_thresholds:
                    thresholds = np.percentile(unique_vals, np.linspace(0, 100, self.max_thresholds))
                else:
                    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

                for thr in thresholds:
                    mask_le = col <= thr
                    mask_gt = ~mask_le
                    if mask_le.sum() < self.min_samples_leaf or mask_gt.sum() < self.min_samples_leaf:
                        continue

                    new_entropy = (mask_le.sum()/len(y)) * self._entropy(y[mask_le]) + \
                                  (mask_gt.sum()/len(y)) * self._entropy(y[mask_gt])
                    gain = base_entropy - new_entropy
                    if gain > best_gain:
                        best_gain = gain
                        best_attr = attr
                        best_thresh = thr
            else:
                # Attribut catégoriel
                new_entropy = 0.0
                for val in unique_vals:
                    mask = col == val
                    if mask.sum() == 0:
                        continue
                    new_entropy += (mask.sum()/len(y)) * self._entropy(y[mask])
                gain = base_entropy - new_entropy
                if gain > best_gain:
                    best_gain = gain
                    best_attr = attr
                    best_thresh = None

        return best_attr, best_thresh, best_gain

    # ----------- Construction récursive -----------

    def _build(self, X, y, feature_indices, depth=0):
        # Cas d'arrêt
        if len(y) == 0:
            return _Node(is_leaf=True, prediction=None)
        if len(np.unique(y)) == 1:
            return _Node(is_leaf=True, prediction=y[0])
        if self.max_depth is not None and depth >= self.max_depth:
            return _Node(is_leaf=True, prediction=self._plurality(y))
        if len(y) < self.min_samples_split:
            return _Node(is_leaf=True, prediction=self._plurality(y))

        # Choisir le meilleur split
        attr, thresh, gain = self._best_split(X, y, feature_indices)
        if attr is None or gain <= 0:
            return _Node(is_leaf=True, prediction=self._plurality(y))

        col = X[:, attr]

        if thresh is not None:
            # Split numérique
            mask_le = col <= thresh
            mask_gt = ~mask_le

            if mask_le.sum() < self.min_samples_leaf or mask_gt.sum() < self.min_samples_leaf:
                return _Node(is_leaf=True, prediction=self._plurality(y))

            left = self._build(X[mask_le], y[mask_le], feature_indices, depth + 1)
            right = self._build(X[mask_gt], y[mask_gt], feature_indices, depth + 1)
            return _Node(is_leaf=False, attr=attr, threshold=float(thresh), branches={"le": left, "gt": right})
        else:
            # Split catégoriel
            branches = {}
            for val in np.unique(col):
                mask = col == val
                if mask.sum() < self.min_samples_leaf:
                    branches[val] = _Node(is_leaf=True, prediction=self._plurality(y))
                else:
                    branches[val] = self._build(X[mask], y[mask],
                                                [f for f in feature_indices if f != attr],
                                                depth + 1)
            return _Node(is_leaf=False, attr=attr, branches=branches)

    # ----------- API publique -----------

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        self._n_features = X.shape[1]
        feature_indices = list(range(self._n_features))
        self._tree = self._build(X, y, feature_indices, depth=0)
        return self

    def _predict_one(self, node, x):
        while not node.is_leaf:
            if node.threshold is not None:
                # Numérique
                val = x[node.attr]
                node = node.branches["le"] if val <= node.threshold else node.branches["gt"]
            else:
                # Catégoriel
                val = x[node.attr]
                if val in node.branches:
                    node = node.branches[val]
                else:
                    node = self._majority_subtree(node)
                    break
        return node.prediction

    def _majority_subtree(self, node):
        """Vote majoritaire des feuilles sous un noeud."""
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

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(self._tree, X[i]) for i in range(len(X))])

