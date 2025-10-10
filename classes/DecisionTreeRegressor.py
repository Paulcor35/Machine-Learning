#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #

import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass

@dataclass
class _Node:
    is_leaf: bool
    prediction: float = None
    attr: str = None
    threshold: float = None
    branches: dict = None  # {"le": _Node, "gt": _Node} pour numériques, {val: _Node} pour catégoriels

class DecisionTreeRegressor:
    """
    Arbre de régression 'from scratch' basé sur la réduction de variance (MSE).

    - Gère attributs numériques (seuil optimal) et catégoriels (branche par valeur).
    - Aucune préparation interne : X/y doivent être prêts.
    - API:
        typ = ['r']
        fit(X, y) -> self
        predict(X) -> np.ndarray
        get_params / set_params

    Hyperparamètres:
        max_depth: profondeur max (None = illimitée)
        min_samples_split: nb min d'échantillons pour splitter
        min_samples_leaf: nb min par feuille
        random_state: non utilisé ici (placeholder compat)
    """

    typ = ['r']

    def __init__(self,
                 max_depth: int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = None):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state

        self._tree: _Node = None
        self._feature_names = None
        self._is_dataframe = True

    # ---------------- Utils conversion ----------------

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            self._is_dataframe = True
            return X
        self._is_dataframe = False
        X = np.asarray(X)
        cols = [f"f{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)

    def _to_series(self, y, name="target"):
        if isinstance(y, pd.Series):
            return y
        y = np.asarray(y).reshape(-1)
        return pd.Series(y, name=name)

    # ---------------- Critères ----------------

    def _variance(self, y: pd.Series) -> float:
        if len(y) == 0:
            return 0.0
        # population variance (pas biaisée) cohérente pour comparaison
        return float(np.var(y.values))

    def _reduction_var_cat(self, X_col: pd.Series, y: pd.Series, base_var: float) -> float:
        total = len(y)
        new_var = 0.0
        for v, idx in X_col.groupby(X_col).groups.items():
            y_sub = y.loc[idx]
            new_var += (len(y_sub) / total) * self._variance(y_sub)
        return base_var - new_var

    def _reduction_var_num(self, X_col: pd.Series, y: pd.Series, base_var: float):
        vals = np.unique(X_col.values)
        if vals.size < 2:
            return -1.0, None
        vals.sort()
        best_gain = -1.0
        best_thr = None
        total = len(y)

        for i in range(vals.size - 1):
            thr = (vals[i] + vals[i+1]) / 2.0
            mask_le = X_col <= thr
            y_le = y[mask_le]
            y_gt = y[~mask_le]

            # contraintes feuilles
            if len(y_le) < self.min_samples_leaf or len(y_gt) < self.min_samples_leaf:
                continue

            new_var = (len(y_le)/total) * self._variance(y_le) + (len(y_gt)/total) * self._variance(y_gt)
            gain = base_var - new_var
            if gain > best_gain:
                best_gain = gain
                best_thr = thr
        return best_gain, best_thr

    # ---------------- Construction arbre ----------------

    def _mean(self, y: pd.Series) -> float:
        return float(np.mean(y.values)) if len(y) else 0.0

    def _choose_split(self, X: pd.DataFrame, y: pd.Series):
        base_var = self._variance(y)
        best_gain = -1.0
        best_attr = None
        best_thr = None
        is_numeric = False

        for col in X.columns:
            s = X[col]
            if s.dtype.kind in 'bifc' and s.nunique() > 2:
                gain, thr = self._reduction_var_num(s, y, base_var)
                if gain > best_gain:
                    best_gain, best_attr, best_thr, is_numeric = gain, col, thr, True
            else:
                gain = self._reduction_var_cat(s, y, base_var)
                if gain > best_gain:
                    best_gain, best_attr, best_thr, is_numeric = gain, col, None, False

        return best_attr, best_thr, best_gain, is_numeric

    def _build(self, X: pd.DataFrame, y: pd.Series, depth: int) -> _Node:
        # arrêts
        if len(y) == 0:
            return _Node(is_leaf=True, prediction=0.0)
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split:
            return _Node(is_leaf=True, prediction=self._mean(y))
        if y.nunique() == 1:
            return _Node(is_leaf=True, prediction=float(y.iloc[0]))

        attr, thr, gain, is_num = self._choose_split(X, y)
        if attr is None or gain <= 0:
            return _Node(is_leaf=True, prediction=self._mean(y))

        if is_num and thr is not None:
            mask_le = X[attr] <= thr
            X_le, y_le = X[mask_le], y[mask_le]
            X_gt, y_gt = X[~mask_le], y[~mask_le]

            # re-check min_samples_leaf
            if len(y_le) < self.min_samples_leaf or len(y_gt) < self.min_samples_leaf:
                return _Node(is_leaf=True, prediction=self._mean(y))

            left = self._build(X_le.drop(columns=[attr], errors="ignore"), y_le, depth+1)
            right = self._build(X_gt.drop(columns=[attr], errors="ignore"), y_gt, depth+1)
            return _Node(is_leaf=False, attr=attr, threshold=float(thr), branches={"le": left, "gt": right})
        else:
            branches = {}
            groups = X[attr].groupby(X[attr]).groups
            for v, idx in groups.items():
                X_sub = X.loc[idx]
                y_sub = y.loc[idx]
                if len(y_sub) < self.min_samples_leaf:
                    branches[v] = _Node(is_leaf=True, prediction=self._mean(y))
                else:
                    branches[v] = self._build(X_sub.drop(columns=[attr], errors="ignore"), y_sub, depth+1)

            if len(branches) <= 1:
                return _Node(is_leaf=True, prediction=self._mean(y))
            return _Node(is_leaf=False, attr=attr, branches=branches)

    # ---------------- Prédiction ----------------

    def _mean_subtree(self, node: _Node) -> float:
        # moyenne des prédictions feuilles sous le noeud
        vals = []
        stack = [node]
        while stack:
            n = stack.pop()
            if n.is_leaf:
                vals.append(n.prediction)
            else:
                for ch in n.branches.values():
                    stack.append(ch)
        return float(np.mean(vals)) if vals else 0.0

    def _predict_one(self, node: _Node, x: pd.Series) -> float:
        while not node.is_leaf:
            attr = node.attr
            if node.threshold is not None:
                thr = node.threshold
                val = x.get(attr)
                if pd.isna(val):
                    # missing value -> envoie côté moyen (ici: "le")
                    node = node.branches["le"]
                    continue
                node = node.branches["le"] if val <= thr else node.branches["gt"]
            else:
                val = x.get(attr)
                if val in node.branches:
                    node = node.branches[val]
                else:
                    # valeur jamais vue -> moyenne des feuilles du sous-arbre
                    return self._mean_subtree(node)
        return float(node.prediction)

    # ---------------- API publique ----------------

    def fit(self, X, y):
        X_df = self._to_dataframe(X).reset_index(drop=True)
        y_sr = self._to_series(y, name="target").reset_index(drop=True)
        self._feature_names = list(X_df.columns)
        self._tree = self._build(X_df, y_sr, depth=0)
        return self

    def predict(self, X):
        if self._tree is None:
            raise RuntimeError("DecisionTreeRegressor non entraîné : appelez fit(X, y) d'abord.")
        X_df = self._to_dataframe(X)
        # Recompose colonnes comme à l'entraînement
        for c in self._feature_names:
            if c not in X_df.columns:
                X_df[c] = np.nan
        X_df = X_df[self._feature_names]
        preds = [self._predict_one(self._tree, X_df.iloc[i]) for i in range(len(X_df))]
        return np.asarray(preds)

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