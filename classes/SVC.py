# classes/SVC.py (version optimisée sans historique)
#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-

import numpy as np

class SVC:
    """
    SVM linéaire (hinge + L2) vectorisée full-batch.
    Perte: 0.5||w||^2 + C * sum max(0, 1 - y*(w·x+b))
    Gradients:
      dL/dw = w - C * sum_{i: margin_i<1} (y_i x_i)
      dL/db = -C * sum_{i: margin_i<1} y_i
    """

    typ = "c"

    def __init__(self,
                 learning_rate: float = 1e-3,
                 C: float = 1.0,
                 n_iters: int = 300,
                 shuffle: bool = True,
                 random_state: int | None = 0,
                 dtype: str = "float32"):
        self.lr = float(learning_rate)
        self.C = float(C)
        self.n_iters = int(n_iters)
        self.shuffle = bool(shuffle)
        self.random_state = None if random_state is None else int(random_state)
        self.dtype = dtype

        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self._rng = np.random.default_rng(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVC":
        X = np.asarray(X, dtype=self.dtype, order="C")
        y = np.asarray(y, dtype=self.dtype).reshape(-1)
        y = np.where(y <= 0, -1.0, 1.0).astype(X.dtype, copy=False)

        n, d = X.shape
        self.w = np.zeros(d, dtype=X.dtype)
        self.b = np.array(0.0, dtype=X.dtype)

        idx = np.arange(n)

        for _ in range(self.n_iters):
            if self.shuffle:
                self._rng.shuffle(idx)
                X_ep = X[idx]
                y_ep = y[idx]
            else:
                X_ep, y_ep = X, y

            s = X_ep @ self.w + self.b          # (n,)
            m = y_ep * s                         # (n,)
            active = m < 1.0                     # hinge active

            if np.any(active):
                ya = y_ep[active]
                Xa = X_ep[active]
                grad_w = self.w - self.C * (Xa.T @ ya)
                grad_b = -self.C * np.sum(ya, dtype=X.dtype)
            else:
                grad_w = self.w
                grad_b = np.array(0.0, dtype=X.dtype)

            self.w = self.w - self.lr * grad_w
            self.b = self.b - self.lr * grad_b

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Le modèle n'est pas entraîné.")
        X = np.asarray(X, dtype=self.dtype, order="C")
        return (X @ self.w + self.b).astype(np.float64, copy=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        s = self.decision_function(X)
        return (s >= 0.0).astype(int)

    # compat scikit-like
    def get_params(self, deep=True):
        return {
            "learning_rate": self.lr,
            "C": self.C,
            "n_iters": self.n_iters,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "dtype": self.dtype,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if "learning_rate" in params:
            self.lr = float(params["learning_rate"])
        return self
