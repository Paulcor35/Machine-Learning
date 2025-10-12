# classes/SVR.py
#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #

import numpy as np

class SVR:
	"""
    SVR linéaire (régression ε-insensible) optimisé en NumPy, entraîné
    par descente de gradient full-batch sur une perte hinge-ε avec régularisation L2.

    Principe
    --------
    On cherche une fonction linéaire f(x)=w·x+b qui :
      1) reste dans un « tube » d’insensibilité de largeur ε autour des cibles (pas de pénalité
         tant que |y-f(x)|≤ε),
      2) garde des poids petits via une pénalisation L2 pour améliorer la généralisation.
    L’objectif à minimiser est :
        L(w,b) = ½‖w‖² + C · Σ_i max(0, |y_i - (w·x_i + b)| - ε)
    L’implémentation effectue des mises à jour full-batch : on agrège les sous-gradients
    des seuls points en dehors du tube ε, ce qui accélère et stabilise l’optimisation.

    Paramètres
    ----------
    learning_rate : float
        Pas d’apprentissage des mises à jour (taux de descente).
    C : float
        Poids de la pénalité au-delà du tube ε (compromis biais/variance).
        Plus C est grand, plus on pénalise fort les écarts, au risque d’overfit.
    epsilon : float
        Largeur du tube d’insensibilité (les erreurs < ε ne coûtent rien).
    n_iters : int
        Nombre d’itérations (époques) d’entraînement.
    shuffle : bool
        Mélanger les exemples à chaque époque (utile surtout en mini-batch/SGD).
    random_state : int
        Graine pseudo-aléatoire pour la reproductibilité.

    Notes
    -----
    - Sous-gradient full-batch :
        Soit err_i = (w·x_i + b) - y_i,  a_i = 1{|err_i| > ε}.
        Alors
            dL/dw = w/n + (C/n) · Σ_i a_i · sign(err_i) · x_i
            dL/db = (C/n) · Σ_i a_i · sign(err_i)
    - Les points « inside-tube » (|err_i| ≤ ε) n’influencent pas la mise à jour.
    - Convergence sensible au scaling : standardiser X (et éventuellement y).

    Avantages
    ---------
    - Robuste aux petites fluctuations des cibles grâce au tube ε.
    - Linéaire et rapide (NumPy vectorisé), donc adapté aux jeux de données larges.
    - Contrôle fin du compromis biais/variance via (C, ε).
    - Implémentation simple et déterministe, facile à comparer à scikit-learn (LinearSVR).

    Attributs principaux
    --------------------
    w : np.ndarray
        Vecteur des poids appris.
    b : float
        Biais (intercept) appris.
    typ : str
        'r' pour indiquer une tâche de régression.
    """

	typ = "r"

	def __init__(self, learning_rate: float = 1e-3, C: float = 1.0,
				 epsilon: float = 0.1, n_iters: int = 100,
				 shuffle: bool = True, random_state: int = 0):
		self.lr = float(learning_rate)
		self.C = float(C)
		self.epsilon = float(epsilon)
		self.n_iters = int(n_iters)
		self.shuffle = bool(shuffle)
		self.random_state = int(random_state)
		self.w: np.ndarray | None = None
		self.b: float = 0.0

	def fit(self, X: np.ndarray, y: np.ndarray) -> "SVR":
		X = np.asarray(X)
		y = np.asarray(y)

		n, d = X.shape
		rng = np.random.default_rng(self.random_state)

		self.w = np.zeros(d, dtype=float)
		self.b = 0.0

		for epoch in range(self.n_iters):
			if self.shuffle:
				idx = rng.permutation(n)
				X_ep, y_ep = X[idx], y[idx]
			else:
				X_ep, y_ep = X, y

			# Full batch vectorization - process all samples at once
			y_pred = X_ep @ self.w + self.b
			errors = y_pred - y_ep
			abs_errors = np.abs(errors)

			# Regularization gradient
			grad_w_reg = self.w / n

			# Loss gradient - vectorized
			outside_mask = abs_errors > self.epsilon
			n_outside = np.sum(outside_mask)

			if n_outside > 0:
				signs = np.sign(errors[outside_mask])
				X_outside = X_ep[outside_mask]

				# Average over all outside points
				grad_w_loss = self.C * (signs[:, None] * X_outside).sum(axis=0) / n
				grad_b_loss = self.C * signs.sum() / n
			else:
				grad_w_loss = np.zeros(d)
				grad_b_loss = 0.0

			# Combined gradients
			grad_w = grad_w_reg + grad_w_loss
			grad_b = grad_b_loss

			# Update parameters
			self.w -= self.lr * grad_w
			self.b -= self.lr * grad_b

		return self

	def decision_function(self, X: np.ndarray) -> np.ndarray:
		if self.w is None:
			raise RuntimeError("Le modèle n'est pas entraîné.")
		X = np.asarray(X)  # Handle pandas DataFrames
		return X @ self.w + self.b

	def predict(self, X: np.ndarray) -> np.ndarray:
		return self.decision_function(X)
