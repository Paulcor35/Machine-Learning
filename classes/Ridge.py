#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #

import numpy as np

class Ridge:
    """
    Ridge regression (fermée) avec biais non régularisé.
    API minimale attendue par votre pipeline:
      - attribut 'typ' indiquant les types supportés (ici régression -> 'r')
      - fit(X, y) -> self
      - predict(X) -> y_pred
      - get_params/set_params (compat utilitaires)
    """
    typ = ['r']

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.w = None  # vecteur des coefficients [intercept, w1, ..., wp]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        # Ajout du biais
        Xb = np.column_stack([np.ones(X.shape[0]), X])

        n_features = Xb.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0.0  # ne pas régulariser l’intercept

        # Solution fermée : w = (X^T X + alpha I)^(-1) X^T y
        A = Xb.T @ Xb + self.alpha * I
        b = Xb.T @ y
        self.w = np.linalg.solve(A, b)
        return self

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("Ridge non entraîné : appelez fit(X, y) d'abord.")
        X = np.asarray(X, dtype=float)
        Xb = np.column_stack([np.ones(X.shape[0]), X])
        return Xb @ self.w

    # Petites méthodes utilitaires pour ressembler à scikit
    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **params):
        if "alpha" in params:
            self.alpha = float(params["alpha"])
        return self