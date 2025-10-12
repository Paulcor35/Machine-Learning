#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #

import numpy as np

class SVR:
	"""
	SVR linéaire (régression ε-insensible) entraînée par SGD
	sur la perte epsilon + L2.

	Paramètres
	----------
	learning_rate : float
		Pas d'apprentissage pour l'update SGD.
	C : float
		Poids de la pénalité (termes slack).
	epsilon : float
		Tube ε de la perte SVR.
	n_iters : int
		Nombre d'époques.
	shuffle : bool
		Mélange des exemples à chaque époque.
	random_state : int
		Graine pour la permutation.

	Notes
	-----
	- Ici, on minimise : 1/2 ||w||^2 + C * sum_i max(0, |y - (w·x+b)| - ε)
	- Update SGD (sous-gradient) :
		si err >  ε :  grad_w = w + C * (+x), grad_b = +C
		si err < -ε :  grad_w = w + C * (-x), grad_b = -C
		sinon        :  grad_w = w,            grad_b = 0
	"""

	typ = "r"

	def __init__(self, learning_rate: float = 1e-3, C: float = 1.0,
				 epsilon: float = 0.1, n_iters: int = 100,
				 shuffle: bool = False, random_state: int = 0):
		self.lr = float(learning_rate)
		self.C = float(C)
		self.epsilon = float(epsilon)
		self.n_iters = int(n_iters)
		self.shuffle = bool(shuffle)
		self.random_state = int(random_state)
		self.w: np.ndarray | None = None
		self.b: float = 0.0

	# --- fit / predict ---
	def fit(self, X: np.ndarray, y: np.ndarray) -> "SVR":
		n, d = X.shape
		rng = np.random.default_rng(self.random_state)
		self.w = np.zeros(d, dtype=float)
		self.b = 0.0

		for _ in range(self.n_iters):
			if self.shuffle:
				idx = rng.permutation(n)
				X_ep, y_ep = X[idx], y[idx]
			else:
				X_ep, y_ep = X, y

			for x_i, y_i in zip(X_ep, y_ep):
				y_pred = float(np.dot(self.w, x_i) + self.b)
				err = y_pred - float(y_i)

				# Sous-gradient de la perte ε-insensible + régularisation L2
				if err > self.epsilon:
					grad_w = self.w + self.C * x_i
					grad_b = self.C
				elif err < -self.epsilon:
					grad_w = self.w - self.C * x_i
					grad_b = -self.C
				else:
					grad_w = self.w
					grad_b = 0.0

				# Mise à jour
				self.w -= self.lr * grad_w
				self.b -= self.lr * grad_b

		return self

	def decision_function(self, X: np.ndarray) -> np.ndarray:
		"""Ici équivaut aux prédictions continues (régression)."""
		if self.w is None:
			raise RuntimeError("Le modèle n'est pas entraîné.")
		return X @ self.w + self.b

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""Par convention régression: renvoie directement les valeurs continues."""
		return self.decision_function(X)
