# classes/SVC.py
#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-

import numpy as np

class SVC:
    """
    SVM linéaire (classification binaire) optimisée full-batch,
    entraînée sur la perte hinge avec régularisation L2.

    Principe
    --------
    On cherche l’hyperplan f(x)=w·x+b qui maximise la marge entre les classes
    tout en pénalisant les exemples mal classés ou à l’intérieur de la marge.
    L’objectif à minimiser est :
        L(w, b) = ½‖w‖² + C · Σ_i max(0, 1 - y_i (w·x_i + b))
    où y_i ∈ {-1, +1}. La première partie (½‖w‖²) élargit la marge (régularisation L2),
    la seconde (perte hinge) pénalise les erreurs/marges violées.

    Paramètres
    ----------
    learning_rate : float
        Pas d’apprentissage pour la descente de gradient.
    C : float
        Poids de la pénalité des erreurs de marge (termes slack).
        Plus C est grand, plus on corrige agressivement les erreurs (risque d’overfit).
    n_iters : int
        Nombre d’itérations (époques) de descente de gradient.
    shuffle : bool
        Mélange des échantillons à chaque époque (utile pour la stabilité).
    random_state : int | None
        Graine aléatoire pour la reproductibilité du mélange.
    dtype : str
        Type numérique interne ("float32" par défaut pour vitesse/mémoire).

    Notes
    -----
    - Sous-gradient (vectorisé, full-batch) :
        Soit m_i = y_i (w·x_i + b). Pour les points « actifs » avec m_i < 1 :
            dL/dw = w - C · Σ_i y_i x_i
            dL/db = - C · Σ_i y_i
      Si tous les points satisfont m_i ≥ 1, seule la régularisation L2 agit (dL/dw = w, dL/db = 0).
    - Le schéma full-batch (sur tout X par itération) est bien plus rapide et stable
      que du SGD échantillon-par-échantillon en Python pur.

    Avantages
    ---------
    - Classifieur linéaire robuste, avec contrôle fin biais/variance via C.
    - Implémentation NumPy vectorisée : rapide, simple et déterministe.
    - Facilement comparable à `sklearn.svm.LinearSVC` (même objectif hinge + L2).

    Attributs principaux
    --------------------
    w : np.ndarray
        Vecteur des poids appris.
    b : float
        Biais (intercept).
    typ : str
        'c' pour indiquer une tâche de classification.
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