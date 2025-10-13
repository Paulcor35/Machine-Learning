#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np

class SVR:
	"""
	Linear Support Vector Regression (ε-insensitive) implemented with NumPy.

	This model is trained using full-batch gradient descent on an ε-hinge loss
	with L2 regularization. It finds a linear function f(x) = w·x + b that stays
	within an ε-insensitive tube around the targets and keeps weights small to
	promote generalization.

	Objective
	---------
	Minimize the following loss:
		L(w, b) = ½‖w‖² + C · Σ_i max(0, |y_i - (w·x_i + b)| - ε)

	The implementation performs full-batch updates by aggregating subgradients
	only from points lying outside the ε-tube, improving speed and stability.

	Parameters
	----------
	learning_rate : float
		Step size for gradient updates (learning rate).
	C : float
		Regularization strength. Controls the trade-off between bias and variance.
		Larger values of C penalize deviations more strongly, increasing overfitting risk.
	epsilon : float
		Width of the ε-insensitive tube (errors smaller than ε incur no cost).
	n_iters : int
		Number of training iterations (epochs).
	shuffle : bool
		Whether to shuffle the examples at each epoch (mainly useful for mini-batch/SGD).
	random_state : int
		Seed for the random number generator to ensure reproducibility.

	Notes
	-----
	Full-batch subgradient computation:
		Let err_i = (w·x_i + b) - y_i and a_i = 1{|err_i| > ε}.
		Then:
		    dL/dw = w/n + (C/n) · Σ_i a_i · sign(err_i) · x_i
		    dL/db = (C/n) · Σ_i a_i · sign(err_i)

	Points inside the tube (|err_i| ≤ ε) do not affect the update.
	Convergence is sensitive to feature scaling—standardize X (and possibly y).

	Attributes
	----------
	w : np.ndarray
		Learned weight vector.
	b : float
		Learned bias (intercept).
	typ : str
		'r' indicating a regression task.
	"""


	typ = 'r'

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
			raise RuntimeError("The model isn't trained, call `fit` first")
		X = np.asarray(X)  # Handle pandas DataFrames
		return X @ self.w + self.b

	def predict(self, X: np.ndarray) -> np.ndarray:
		return self.decision_function(X)
