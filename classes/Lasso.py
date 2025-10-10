#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #

import numpy as np
import pandas as pd

def _as_df(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    X = np.asarray(X, dtype=float)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

def _as_sr(y):
    if isinstance(y, pd.Series):
        return y.copy()
    y = np.asarray(y, dtype=float).reshape(-1)
    return pd.Series(y, name="target")

def _soft_threshold(u, lam):
    # Prox L1 : S(u, λ) = sign(u) * max(|u| - λ, 0)
    if u > lam:  return u - lam
    if u < -lam: return u + lam
    return 0.0

class Lasso:
    """
    Régression Lasso (L1) par gradient + étape proximale (soft-thresholding).

    - typ = ['r'] : pipeline régression
    - API : fit(X, y) -> self ; predict(X) ; get_params / set_params
    - Gère DataFrame/ndarray ; normalisation interne optionnelle.

    Paramètres
    ----------
    learning_rate : float          (pas de 'alpha' ici ; utiliser l1_penalty)
    max_iter      : int
    l1_penalty    : float          (λ)
    tol           : float          (tolérance d'arrêt sur la baisse du coût)
    fit_intercept : bool
    normalize     : bool           (centre X, et met à l'échelle par std si True)
    random_state  : int|None       (non utilisé ici, placeholder)
    """

    typ = ['r']

    def __init__(self,
                 learning_rate: float = 0.01,
                 max_iter: int = 1000,
                 l1_penalty: float = 0.1,
                 tol: float = 1e-6,
                 fit_intercept: bool = True,
                 normalize: bool = True,
                 random_state: int | None = None):
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.l1_penalty = float(l1_penalty)
        self.tol = float(tol)
        self.fit_intercept = bool(fit_intercept)
        self.normalize = bool(normalize)
        self.random_state = random_state

        # Attributs appris
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.cost_history_: list[float] = []

        # Stats de prétraitement
        self._x_mean = None
        self._x_scale = None
        self._y_mean = 0.0
        self._feature_names = None

    # -------------------- Prétraitements --------------------

    def _prep_fit(self, X, y):
        X = _as_df(X).reset_index(drop=True)
        y = _as_sr(y).reset_index(drop=True)

        self._feature_names = list(X.columns)

        # Centrage X si intercept ou normalize
        self._x_mean = X.mean(axis=0) if (self.fit_intercept or self.normalize) else pd.Series(0.0, index=X.columns)
        Xc = X - self._x_mean

        # Mise à l'échelle si normalize
        if self.normalize:
            self._x_scale = Xc.std(axis=0).replace(0.0, 1.0)
            Xc = Xc / self._x_scale
        else:
            self._x_scale = pd.Series(1.0, index=X.columns)

        # Centrage de y si intercept
        if self.fit_intercept:
            self._y_mean = float(y.mean())
            yc = y - self._y_mean
        else:
            self._y_mean = 0.0
            yc = y

        return Xc.to_numpy(dtype=float), yc.to_numpy(dtype=float)

    def _prep_predict(self, X):
        X = _as_df(X)
        # Alignement des colonnes
        for c in self._feature_names:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self._feature_names]

        Xc = X - self._x_mean
        if self.normalize:
            Xc = Xc / self._x_scale
        return Xc.to_numpy(dtype=float)

    # -------------------- Entraînement (GD + prox L1) --------------------

    def fit(self, X, y):
        Xc, yc = self._prep_fit(X, y)  # (n, p), (n,)
        n, p = Xc.shape

        w = np.zeros(p, dtype=float)
        b = 0.0
        self.cost_history_.clear()

        prev_cost = float('inf')
        lr = self.learning_rate
        lam = self.l1_penalty

        for _ in range(self.max_iter):
            # Prédiction et résidus
            y_pred = Xc @ w + b
            r = yc - y_pred

            # Gradients MSE (facteur 2/n)
            grad_w = -(2.0 / n) * (Xc.T @ r)
            grad_b = -(2.0 / n) * np.sum(r)

            # Pas de gradient
            w = w - lr * grad_w
            b = b - lr * grad_b

            # Prox L1 sur w (b n'est pas régularisé)
            # w := S(w, lr * lam)
            if lam != 0.0:
                w = np.vectorize(_soft_threshold)(w, lr * lam)

            # Coût = MSE + λ ||w||_1
            mse = float(np.mean((r) ** 2))
            cost = mse + lam * float(np.sum(np.abs(w)))
            self.cost_history_.append(cost)

            # Critère d'arrêt
            if abs(prev_cost - cost) < self.tol:
                prev_cost = cost
                break
            prev_cost = cost

        self.coef_ = w
        # Intercept en espace original : y = (Xc @ w + b) + y_mean
        self.intercept_ = (b + self._y_mean) if self.fit_intercept else b
        return self

    # -------------------- Prédiction --------------------

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Lasso non entraîné : appelez fit(X, y) d'abord.")
        Xc = self._prep_predict(X)
        return Xc @ self.coef_ + self.intercept_

    # -------------------- Compat scikit-like --------------------

    def get_params(self, deep=True):
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "l1_penalty": self.l1_penalty,
            "tol": self.tol,
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self