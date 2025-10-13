#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np

class Ridge:
	"""
	Ridge Regression (also known as Tikhonov regularization or L2-regularized least squares).

	This model solves a linear regression problem with an added L2 penalty term
	to constrain the magnitude of the coefficients, thereby reducing model variance
	and mitigating overfitting. It uses a closed-form analytical solution with
	regularization applied only to the weight coefficients, not the bias.

	Principle
	---------
	Ridge regression minimizes the following objective:
		min_{w, b}  ||y - (Xw + b)||² + α ||w||²

	The solution is obtained by augmenting X with a column of ones (for the bias term)
	and solving a regularized linear system where only the weight vector w is penalized.

	Parameters
	----------
	alpha : float
		Regularization strength (λ).
		Larger values of alpha shrink the weights more strongly,
		reducing overfitting at the cost of increased bias.

	Notes
	-----
	Cost function:
		L(w, b) = ||y - (X·w + b)||² + α · ||w||²
	Only the weight vector w is regularized; the bias term b is excluded.

	Closed-form solution:
		β = (XᵀX + αI)⁻¹ Xᵀy
	where β = [b, w₁, …, w_p], and the first diagonal element (corresponding to b)
	is not regularized.

	Numerical stability:
	The implementation uses `np.linalg.solve` instead of explicit matrix inversion
	to ensure better numerical stability and performance.

	Attributes
	----------
	w : np.ndarray
		Learned coefficient vector.
	intercept_ : float
		Unregularized bias term.
	"""


	typ = 'r'

	def __init__(self, alpha: float = 1.0):
		self.alpha = float(alpha)
		self.w: np.ndarray | None = None
		self.intercept_: float = 0.0

	def fit(self, X: np.ndarray, y: np.ndarray):
		X = np.asarray(X, dtype=float)
		y = np.asarray(y, dtype=float).reshape(-1)

		# Add the bias column
		Xb = np.column_stack([np.ones(X.shape[0]), X])

		n_features_plus_bias = Xb.shape[1]
		I = np.eye(n_features_plus_bias)
		I[0, 0] = 0.0  # don't regularize the intercept

		A = Xb.T @ Xb + self.alpha * I
		b = Xb.T @ y
		beta = np.linalg.solve(A, b)

		# Stockage façon scikit
		self.intercept_ = float(beta[0])
		self.w = beta[1:].astype(float)
		return self

	def predict(self, X: np.ndarray):
		if self.w is None:
			raise RuntimeError("The model isn't trained, call `fit` first")
		X = np.asarray(X, dtype=float)
		return X @ self.w + self.intercept_
