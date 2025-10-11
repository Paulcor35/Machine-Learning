#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union

# Arbre de base
from classes.DecisionTreeClassifier import DecisionTreeClassifier as BaseTree

def _to_df(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    X = np.asarray(X)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

def _to_sr(y):
    if isinstance(y, pd.Series):
        return y.copy()
    y = np.asarray(y).reshape(-1)
    return pd.Series(y, name="target")

class RandomForestClassifier:
    """
    RandomForestClassifier (from scratch)
    - Base learner: DecisionTreeClassifier (maison)
    - Bootstrap des échantillons + sous-échantillonnage des features
    - Vote majoritaire pour predict, moyenne des votes pour predict_proba

    Hyperparamètres principaux
    --------------------------
    n_estimators: int                nombre d'arbres
    max_depth: int|None             profondeur max des arbres
    min_samples_split: int          min d'échantillons pour splitter
    min_samples_leaf: int           min d'échantillons par feuille
    max_features: {"sqrt","log2",None} | int | float
        - "sqrt": sqrt(p)
               "log2": log2(p)
               None: p (toutes les features)
               int: ce nombre de features
               float in (0,1]: ce ratio * p
    bootstrap: bool                 activer le tirage avec remise des lignes
    random_state: int|None          graine RNG

    API
    ---
    typ = ['c']
    fit(X, y) -> self
    predict(X) -> np.ndarray (labels)
    predict_proba(X) -> np.ndarray (n_samples, n_classes)
    get_params / set_params
    """

    typ = 'c'

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, int, float, None] = "sqrt",
                 bootstrap: bool = True,
                 random_state: Optional[int] = None):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state

        # appris
        self._estimators: List[Tuple[BaseTree, List[str]]] = []
        self._feature_names: List[str] = []
        self.classes_: np.ndarray = None
        self.n_classes_: int = 0
        self._rng = np.random.default_rng(random_state)

    # -------------------- helpers --------------------

    def _resolve_max_features(self, p: int) -> int:
        mf = self.max_features
        if mf is None:
            k = p
        elif isinstance(mf, str):
            if mf == "sqrt":
                k = int(np.sqrt(p))
            elif mf == "log2":
                k = int(np.log2(p)) if p > 1 else 1
            else:
                raise ValueError(f"max_features string unsupported: {mf}")
        elif isinstance(mf, (int, np.integer)):
            k = int(mf)
        elif isinstance(mf, float):
            if not (0 < mf <= 1):
                raise ValueError("max_features float doit être dans (0,1].")
            k = int(np.ceil(mf * p))
        else:
            raise ValueError(f"max_features type unsupported: {type(mf)}")

        k = max(1, min(k, p))
        return k

    def _bootstrap_indices(self, n: int) -> np.ndarray:
        if self.bootstrap:
            return self._rng.integers(0, n, size=n)
        else:
            # sans remise, on peut juste permuter
            return self._rng.permutation(n)

    # -------------------- API --------------------

    def fit(self, X, y):
        X_df = _to_df(X).reset_index(drop=True)
        y_sr = _to_sr(y).reset_index(drop=True)

        self._feature_names = list(X_df.columns)
        self.classes_ = np.unique(y_sr.values)
        self.n_classes_ = self.classes_.size

        n, p = X_df.shape
        k = self._resolve_max_features(p)

        self._estimators = []

        for _ in range(self.n_estimators):
            # 1) bootstrap lignes
            idx = self._bootstrap_indices(n)
            X_boot = X_df.iloc[idx]

            # 2) sous-ensemble de features
            feat_idx = self._rng.choice(p, size=k, replace=False)
            feat_names = [self._feature_names[j] for j in feat_idx]
            X_sub = X_boot[feat_names]
            y_sub = y_sr.iloc[idx]

            # 3) fit arbre base
            tree = BaseTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state  # pour homogénéité, même si l'arbre ne l'utilise pas
            ).fit(X_sub, y_sub)

            self._estimators.append((tree, feat_names))

        return self

    def _ensure_columns(self, X_df: pd.DataFrame, feat_names: List[str]) -> pd.DataFrame:
        # ajoute colonnes manquantes si besoin
        for c in feat_names:
            if c not in X_df.columns:
                X_df[c] = np.nan
        return X_df[feat_names]

    def predict(self, X):
        X_df = _to_df(X)

        # votes (n_estimators, n_samples)
        votes = []
        for tree, feats in self._estimators:
            X_use = self._ensure_columns(X_df.copy(), feats)
            preds = tree.predict(X_use)
            votes.append(preds)

        votes = np.vstack(votes)  # shape: (T, n)
        # vote majoritaire par colonne
        n = votes.shape[1]
        out = np.empty(n, dtype=object)
        # classes_ pour tie-break déterministe (ordre)
        class_order = list(self.classes_)

        for i in range(n):
            col = votes[:, i]
            # compter
            uniq, counts = np.unique(col, return_counts=True)
            # en cas d'égalité, on choisit la classe apparaissant la première dans class_order
            max_count = counts.max()
            candidates = set(uniq[counts == max_count])
            chosen = next(c for c in class_order if c in candidates)
            out[i] = chosen

        return out

    def predict_proba(self, X):
        if self.classes_ is None:
            raise RuntimeError("Modèle non entraîné : appelez fit(X, y) d'abord.")
        X_df = _to_df(X)
        n = X_df.shape[0]
        probs = np.zeros((n, self.n_classes_), dtype=float)

        for tree, feats in self._estimators:
            X_use = self._ensure_columns(X_df.copy(), feats)
            preds = tree.predict(X_use)
            # hard-vote -> proba empirique
            for j, c in enumerate(self.classes_):
                probs[:, j] += (preds == c).astype(float)

        probs /= float(len(self._estimators))
        return probs

    # -------------------- compat scikit-like --------------------

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self
