# classes/Lasso.py (version numpy-only optimisée)
#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-

import numpy as np

class Lasso:
    """
    Lasso par gradient proximal (ISTA), numpy-only.
    """

    typ = 'r'

    def __init__(self,
                 learning_rate: float = 0.1,
                 max_iter: int = 1000,
                 l1_penalty: float = 0.01,
                 tol: float = 1e-6,
                 fit_intercept: bool = True,
                 normalize: bool = True,
                 random_state: int | None = None,
                 record_history: bool = False,
                 dtype: str = "float32"):
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.l1_penalty = float(l1_penalty)
        self.tol = float(tol)
        self.fit_intercept = bool(fit_intercept)
        self.normalize = bool(normalize)
        self.random_state = random_state
        self.record_history = bool(record_history)
        self.dtype = dtype

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.cost_history_: list[float] = []

        # stats de normalisation
        self._x_mean: np.ndarray | None = None
        self._x_scale: np.ndarray | None = None
        self._y_mean: float = 0.0

    # ---------- utils ----------

    @staticmethod
    def _soft_threshold_vec(w: np.ndarray, lam: float) -> np.ndarray:
        # proximal L1 vectoriel: sign(w) * max(|w| - lam, 0)
        absw = np.abs(w)
        return np.sign(w) * np.maximum(absw - lam, 0.0, dtype=w.dtype)

    def _prep_fit(self, X, y):
        X = np.asarray(X, dtype=self.dtype, order="C")
        y = np.asarray(y, dtype=self.dtype).reshape(-1)

        if self.fit_intercept or self.normalize:
            self._x_mean = X.mean(axis=0)
        else:
            self._x_mean = np.zeros(X.shape[1], dtype=X.dtype)

        Xc = X - self._x_mean

        if self.normalize:
            self._x_scale = Xc.std(axis=0, dtype=self.dtype)
            self._x_scale[self._x_scale == 0] = 1.0
            Xc = Xc / self._x_scale
        else:
            self._x_scale = np.ones(X.shape[1], dtype=X.dtype)

        if self.fit_intercept:
            self._y_mean = float(y.mean(dtype=self.dtype))
            yc = y - self._y_mean
        else:
            self._y_mean = 0.0
            yc = y

        return Xc, yc

    def _prep_predict(self, X):
        X = np.asarray(X, dtype=self.dtype, order="C")
        Xc = (X - self._x_mean)
        if self.normalize:
            Xc = Xc / self._x_scale
        return Xc

    # ---------- entraînement ----------

    def fit(self, X, y):
        Xc, yc = self._prep_fit(X, y)  # (n, p)
        n, p = Xc.shape

        w = np.zeros(p, dtype=self.dtype)
        b = np.array(0.0, dtype=self.dtype)
        self.cost_history_.clear()

        lr = self.learning_rate
        lam = self.l1_penalty
        prev_cost = np.inf

        # on calcule r = Xc @ w + b - yc pour réutiliser
        r = -yc.copy()
        # boucle ISTA
        for _ in range(self.max_iter):
            # y_pred = Xc @ w + b  => r = y_pred - yc, mais on met à jour r plus vite:
            # grad_w = (2/n) Xc^T r ; grad_b = (2/n) sum(r)
            grad_w = (2.0 / n) * (Xc.T @ r)
            grad_b = (2.0 / n) * np.sum(r, dtype=self.dtype)

            # tentative de descente
            w_new = w - lr * grad_w
            b_new = b - lr * grad_b

            # prox L1 (sur w seulement)
            if lam != 0.0:
                w_new = self._soft_threshold_vec(w_new, lr * lam)

            # mettre à jour r efficacement:
            # r_new = (Xc @ w_new + b_new) - yc
            #       = r + Xc@(w_new - w) + (b_new - b)
            dw = w_new - w
            db = b_new - b
            r = r + Xc @ dw + db

            w, b = w_new, b_new

            if self.record_history or self.tol > 0:
                mse = float(np.mean(r * r))
                cost = mse + lam * float(np.sum(np.abs(w)))
                if self.record_history:
                    self.cost_history_.append(cost)
                if abs(prev_cost - cost) < self.tol:
                    prev_cost = cost
                    break
                prev_cost = cost

        self.coef_ = w.astype(np.float64, copy=False)
        self.intercept_ = float(b + self._y_mean) if self.fit_intercept else float(b)
        return self

    # ---------- prédiction ----------

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Lasso non entraîné : appelez fit(X, y) d'abord.")
        Xc = self._prep_predict(X)
        return (Xc @ self.coef_) + self.intercept_

    # ---------- compat scikit-like ----------

    def get_params(self, deep=True):
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "l1_penalty": self.l1_penalty,
            "tol": self.tol,
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
            "random_state": self.random_state,
            "record_history": self.record_history,
            "dtype": self.dtype,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self