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
    attr: str = None
    threshold: float = None
    branches: dict = None  # {value: _Node} pour catégoriels, {"le": _Node, "gt": _Node} pour numériques

class DecisionTreeClassifier:
    """
    Arbre de décision 'from scratch' (classification) basé sur l'entropie / gain d'information.

    - Gère à la fois attributs numériques (seuil optimal) et catégoriels (branche par valeur).
    - Aucun préprocessing interne : X/y doivent être prêts (encodage au besoin fait hors de la classe).
    - API minimale:
        typ = ['c']
        fit(X, y) -> self
        predict(X) -> np.ndarray
        get_params / set_params
    - Hyperparamètres basiques:
        max_depth: profondeur max (None = illimitée)
        min_samples_split: nb min d'échantillons pour splitter
        min_samples_leaf: nb min par feuille
        random_state: réservé (non utilisé ici mais gardé pour compat)
    """

    typ = ['c']

    def __init__(self,
                 max_depth: int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = None):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state
        self._tree = None
        self._feature_names = None
        self._is_dataframe = True  # pour reconstruire DataFrame si X ndarray

    # ----------- Utils de conversions -----------

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

    # ----------- Mesures et critères -----------

    def _entropy(self, y: pd.Series) -> float:
        cnt = Counter(y)
        total = len(y)
        if total == 0:
            return 0.0
        return -sum((c/total) * math.log2(c/total) for c in cnt.values() if c > 0)

    def _information_gain_cat(self, X_col: pd.Series, y: pd.Series, base_entropy: float) -> float:
        # Entropie pondérée des sous-ensembles (une branche par valeur)
        total = len(y)
        new_entropy = 0.0
        for v, idx in X_col.groupby(X_col).groups.items():
            y_sub = y.loc[idx]
            new_entropy += (len(y_sub)/total) * self._entropy(y_sub)
        return base_entropy - new_entropy

    def _information_gain_num(self, X_col: pd.Series, y: pd.Series, base_entropy: float):
        # Teste tous les seuils entre valeurs uniques triées
        values = np.unique(X_col.values)
        if values.size < 2:
            return -1.0, None  # pas de split utile
        values.sort()
        best_gain = -1.0
        best_threshold = None
        total = len(y)

        # seuil = moyenne de deux valeurs consécutives
        for i in range(values.size - 1):
            thr = (values[i] + values[i+1]) / 2.0
            mask_le = X_col <= thr
            y_le = y[mask_le]
            y_gt = y[~mask_le]

            # respect des min_samples_leaf
            if len(y_le) < self.min_samples_leaf or len(y_gt) < self.min_samples_leaf:
                continue

            new_entropy = (len(y_le)/total) * self._entropy(y_le) + (len(y_gt)/total) * self._entropy(y_gt)
            gain = base_entropy - new_entropy
            if gain > best_gain:
                best_gain = gain
                best_threshold = thr
        return best_gain, best_threshold

    # ----------- Construction de l'arbre -----------

    def _plurality(self, y: pd.Series):
        return Counter(y).most_common(1)[0][0]

    def _all_same_class(self, y: pd.Series) -> bool:
        return y.nunique() <= 1

    def _choose_split(self, X: pd.DataFrame, y: pd.Series):
        base_entropy = self._entropy(y)
        best_gain = -1.0
        best_attr = None
        best_threshold = None
        is_numeric_split = False

        for col in X.columns:
            s = X[col]
            # numériques non binaires => seuil
            if s.dtype.kind in 'bifc' and s.nunique() > 2:
                gain, thr = self._information_gain_num(s, y, base_entropy)
                if gain > best_gain:
                    best_gain = gain
                    best_attr = col
                    best_threshold = thr
                    is_numeric_split = True
            else:
                # catégoriel ou binaire
                gain = self._information_gain_cat(s, y, base_entropy)
                if gain > best_gain:
                    best_gain = gain
                    best_attr = col
                    best_threshold = None
                    is_numeric_split = False

        return best_attr, best_threshold, best_gain, is_numeric_split

    def _build(self, X: pd.DataFrame, y: pd.Series, depth: int) -> _Node:
        # Cas de base
        if len(y) == 0:
            # sécurité (ne devrait pas arriver si contrôles corrects)
            return _Node(is_leaf=True, prediction=None)

        if self._all_same_class(y):
            return _Node(is_leaf=True, prediction=y.iloc[0])

        if self.max_depth is not None and depth >= self.max_depth:
            return _Node(is_leaf=True, prediction=self._plurality(y))

        if len(y) < self.min_samples_split:
            return _Node(is_leaf=True, prediction=self._plurality(y))

        # Choisir meilleur split
        attr, thr, gain, is_num = self._choose_split(X, y)

        # si aucun gain ou impossible de splitter proprement
        if attr is None or gain <= 0:
            return _Node(is_leaf=True, prediction=self._plurality(y))

        # Construction des branches
        if is_num and thr is not None:
            mask_le = X[attr] <= thr
            X_le, y_le = X[mask_le], y[mask_le]
            X_gt, y_gt = X[~mask_le], y[~mask_le]

            # Si une branche viole min_samples_leaf, on stoppe
            if len(y_le) < self.min_samples_leaf or len(y_gt) < self.min_samples_leaf:
                return _Node(is_leaf=True, prediction=self._plurality(y))

            left = self._build(X_le.drop(columns=[attr], errors="ignore"), y_le, depth+1)
            right = self._build(X_gt.drop(columns=[attr], errors="ignore"), y_gt, depth+1)
            return _Node(is_leaf=False, attr=attr, threshold=float(thr), branches={"le": left, "gt": right})
        else:
            # catégoriel / binaire
            branches = {}
            for v, idx in X[attr].groupby(X[attr]).groups.items():
                X_sub = X.loc[idx]
                y_sub = y.loc[idx]
                if len(y_sub) < self.min_samples_leaf:
                    branches[v] = _Node(is_leaf=True, prediction=self._plurality(y))
                else:
                    branches[v] = self._build(X_sub.drop(columns=[attr], errors="ignore"), y_sub, depth+1)

            # S'il ne reste qu'une branche (données dégénérées), on fait une feuille
            if len(branches) <= 1:
                return _Node(is_leaf=True, prediction=self._plurality(y))
            return _Node(is_leaf=False, attr=attr, branches=branches)

    # ----------- API publique -----------

    def fit(self, X, y):
        X_df = self._to_dataframe(X)
        y_sr = self._to_series(y, name="target")

        # Assurer alignement des index
        X_df = X_df.reset_index(drop=True)
        y_sr = y_sr.reset_index(drop=True)

        self._feature_names = list(X_df.columns)
        self._tree = self._build(X_df, y_sr, depth=0)
        return self

    def _predict_one(self, node: _Node, x: pd.Series):
        while not node.is_leaf:
            attr = node.attr
            if node.threshold is not None:
                # split numérique
                thr = node.threshold
                val = x.get(attr)
                # Valeur manquante -> stratégie : aller côté majoritaire (ici "le")
                if pd.isna(val):
                    node = node.branches["le"]
                    continue
                node = node.branches["le"] if val <= thr else node.branches["gt"]
            else:
                # split catégoriel
                val = x.get(attr)
                if val in node.branches:
                    node = node.branches[val]
                else:
                    # valeur inconnue en prédiction -> vote majoritaire des feuilles sous ce noeud
                    node = self._majority_subtree(node)
                    break
        return node.prediction

    def _majority_subtree(self, node: _Node):
        # Descend et compte les feuilles pour renvoyer une pseudo-feuille majoritaire
        counts = Counter()
        stack = [node]
        while stack:
            n = stack.pop()
            if n.is_leaf:
                counts[n.prediction] += 1
            else:
                for child in n.branches.values():
                    stack.append(child)
        # créer une feuille synthétique
        pred = counts.most_common(1)[0][0] if counts else None
        return _Node(is_leaf=True, prediction=pred)

    def predict(self, X):
        if self._tree is None:
            raise RuntimeError("DecisionTreeClassifier non entraîné : appelez fit(X, y) d'abord.")
        X_df = self._to_dataframe(X)
        # Si X ndarray, il peut ne pas avoir exactement les colonnes de fit (on coupe au commun)
        # On recompose un DataFrame avec les colonnes vues à l'entraînement si possible
        if not all(c in X_df.columns for c in self._feature_names):
            # Crée les colonnes manquantes avec NaN
            for c in self._feature_names:
                if c not in X_df.columns:
                    X_df[c] = np.nan
            X_df = X_df[self._feature_names]
        else:
            X_df = X_df[self._feature_names]

        preds = [self._predict_one(self._tree, X_df.iloc[i]) for i in range(len(X_df))]
        return np.asarray(preds)

    # ----------- Compat scikit-like -----------

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