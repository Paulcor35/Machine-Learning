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
	- Update SGD (sous-gradient):
		si |err| > ε: grad_w = w/n + C * sign(err) * x
					  grad_b = C * sign(err)
		sinon		: grad_w = w/n
					  grad_b = 0
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
