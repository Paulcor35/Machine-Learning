# classes/Ridge.py
#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #

import numpy as np

class Ridge:
    """
    Régression Ridge (aussi appelée régression à crête),
    basée sur une solution analytique fermée avec régularisation L2.

    Principe
    --------
    Ridge ajoute un terme de pénalisation L2 aux moindres carrés pour
    contraindre la norme des coefficients et réduire la variance du modèle.
    On résout (en excluant le biais de la pénalisation) :
        min_{w,b}  ||y - (Xw + b)||² + α ||w||²
    La solution fermée s’obtient en augmentant X d’une colonne de 1 (biais)
    puis en résolvant un système linéaire régularisé où seule la partie
    “poids” est pénalisée.

    Paramètres
    ----------
    alpha : float
        Coefficient de régularisation (λ).
        Plus alpha est grand, plus les poids sont contraints à être petits,
        ce qui réduit le sur-apprentissage mais augmente le biais.

    Notes
    -----
    - Fonction de coût :
        L(w, b) = ||y - (X·w + b)||² + α * ||w||²
      où seule la partie w (les poids) est régularisée, pas le biais b.
    - Solution fermée :
        β = (XᵀX + αI)⁻¹ Xᵀy
      avec β = [b, w₁, …, w_p] et la première diagonale (b) non régularisée.
    - Implémentation stable : on utilise `np.linalg.solve` plutôt que
      l’inversion explicite pour de meilleures propriétés numériques.

    Avantages
    ---------
    - Fermée et rapide : pas d’itérations (utile pour des datasets moyens/grands).
    - Robuste à la multicolinéarité et aux matrices mal conditionnées.
    - Réduit la variance des estimations (meilleure généralisation).
    - Simple, interprétable et compatible avec un pipeline linéaire standard.

    Attributs principaux
    --------------------
    coef_      : np.ndarray
        Vecteur des coefficients (w).
    intercept_ : float
        Biais non régularisé.
    w          : np.ndarray
        Alias de coef_ (compatibilité avec d’autres modèles).
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