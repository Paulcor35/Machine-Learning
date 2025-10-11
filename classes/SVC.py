#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #

import numpy as np
import pandas as pd
import utils


class SVC:
	"""
	SVM linéaire entraînée par SGD sur la perte hinge + L2.

	Paramètres
	----------
	learning_rate : float
		Pas d'apprentissage pour l'update SGD.
	C : float
		Poids de la partie hinge (équivaut à 1/lambda en formulation primal).
	n_iters : int
		Nombre d'époques.
	shuffle : bool
		Mélange des exemples à chaque époque.
	random_state : int
		Graine pour la permutation.
	"""
	typ = "c"

	def __init__(self, learning_rate: float = 1e-4, C: float = 9.0,
				 n_iters: int = 100, shuffle: bool = True, random_state: int = 0):
		self.lr = learning_rate
		self.C = C
		self.n_iters = n_iters
		self.shuffle = shuffle
		self.random_state = random_state
		self.w: np.ndarray | None = None
		self.b: float = 0.0

	# --- fit / predict ---
	def fit(self, X: np.ndarray, y: np.ndarray) -> "SVC":
		y = np.where(y <= 0, -1, 1).astype(float)   # {0,1} -> {-1,+1}
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
				margin = y_i * (np.dot(self.w, x_i) + self.b)
				if margin >= 1:
					# gradient de la régularisation L2 uniquement
					self.w -= self.lr * self.w
				else:
					# hinge active : régul + terme de classification
					self.w -= self.lr * (self.w - self.C * y_i * x_i)
					self.b  += self.lr * (self.C * y_i)

		return self

	def decision_function(self, X: np.ndarray) -> np.ndarray:
		if self.w is None:
			raise RuntimeError("Le modèle n'est pas entraîné.")
		return X @ self.w + self.b

	def predict(self, X: np.ndarray) -> np.ndarray:
		s = self.decision_function(X)
		return (s >= 0.0).astype(int)
