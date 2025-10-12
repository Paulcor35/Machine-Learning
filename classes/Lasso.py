# classes/Lasso.py (version numpy-only optimisée)
#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-

import numpy as np

class Lasso:
    """
    Lasso (régression linéaire à pénalisation L1) 
    implémenté en NumPy pur via l’algorithme du gradient proximal (ISTA).

    Principe
    --------
    Le Lasso (Least Absolute Shrinkage and Selection Operator) ajoute une 
    régularisation L1 à la régression linéaire classique afin de favoriser 
    la parcimonie (sparsité) des poids.  
    On résout :
        min_w,b  (1/n) * ||y - (Xw + b)||² + λ * ||w||₁

    Cette implémentation utilise l’algorithme **ISTA** (Iterative Shrinkage-Thresholding Algorithm),
    qui alterne entre une descente de gradient sur le terme quadratique et une opération 
    de seuillage doux (proximal L1) sur les poids.

    Paramètres
    ----------
    learning_rate : float
        Pas d’apprentissage du gradient (souvent noté η). 
        Doit être suffisamment petit pour garantir la convergence.
    max_iter : int
        Nombre maximal d’itérations de mise à jour.
    l1_penalty : float
        Coefficient λ du terme de pénalisation L1.
    tol : float
        Tolérance d’arrêt. Si la variation du coût entre deux itérations 
        est inférieure à ce seuil, l’entraînement s’arrête.
    fit_intercept : bool
        Si True, un biais (intercept) est appris et non régularisé.
    normalize : bool
        Si True, les colonnes de X sont centrées-réduites avant l’entraînement.
    random_state : int | None
        Graine du générateur aléatoire (utile si des variantes stochastiques sont ajoutées).
    record_history : bool
        Si True, enregistre l’évolution du coût (loss) à chaque itération dans `cost_history_`.
    dtype : str
        Type numérique interne utilisé (par défaut "float32" pour rapidité et légèreté mémoire).

    Notes
    -----
    - L’algorithme ISTA se décompose ainsi :
        1. Calcul du gradient de la perte MSE :  
           ∇w = (2/n) * Xᵀ(Xw + b - y)
        2. Mise à jour par descente de gradient :
           w ← w - η * ∇w
        3. Application du **proximal L1** (seuillage doux) :
           w ← sign(w) * max(|w| - ηλ, 0)
        4. Répétition jusqu’à convergence ou atteinte de `max_iter`.

    - Le biais `b` est mis à jour indépendamment du terme de pénalisation L1.
    - Les poids `w` sont mis à jour de manière entièrement vectorisée (aucune boucle Python).
    - Si `normalize=True`, la normalisation est inversée automatiquement à la prédiction.

    Attributs principaux
    --------------------
    coef_ : np.ndarray
        Coefficients appris (poids du modèle).
    intercept_ : float
        Biais appris (non régularisé).
    cost_history_ : list[float]
        Historique du coût total (MSE + régularisation L1) si `record_history=True`.
    _x_mean, _x_scale, _y_mean :
        Statistiques internes utilisées pour la normalisation.
    typ : str
        'r' → indique qu’il s’agit d’un modèle de régression.

    Avantages
    ----------
    - Sélection automatique des variables pertinentes (poids nuls possibles).
    - Implémentation pure NumPy, simple et rapide pour des datasets moyens.
    - Interprétable et compatible avec des pipelines standard (fit/predict).
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