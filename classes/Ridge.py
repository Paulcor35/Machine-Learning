#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #

import numpy as np

class Ridge:
    """
    Ridge regression (solution fermée) avec biais non régularisé.
    API:
      - typ = ['r']
      - fit(X, y) -> self
      - predict(X) -> y_pred
      - coef_ (poids), intercept_
      - get_params / set_params
      - w (alias de coef_) pour compat avec tes plots
    """
    typ = 'r'

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        # Ajout de la colonne de biais
        Xb = np.column_stack([np.ones(X.shape[0]), X])

        n_features_plus_bias = Xb.shape[1]
        I = np.eye(n_features_plus_bias)
        I[0, 0] = 0.0  # ne pas régulariser l’intercept

        # (X^T X + alpha I)^{-1} X^T y (solve pour stabilité)
        A = Xb.T @ Xb + self.alpha * I
        b = Xb.T @ y
        beta = np.linalg.solve(A, b)

        # Stockage façon scikit
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:].astype(float)
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Ridge non entraîné : appelez fit(X, y) d'abord.")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    # Alias pour compat avec tes autres modèles (SVC/SVR) et tes plots
    @property
    def w(self):
        return self.coef_

    # utilitaires scikit-like
    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **params):
        if "alpha" in params:
            self.alpha = float(params["alpha"])
        return self
