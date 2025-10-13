#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np

class Lasso:
	"""
	Lasso (L1-regularized linear regression) implemented in pure NumPy
	using the proximal gradient algorithm (ISTA).

	The Lasso (Least Absolute Shrinkage and Selection Operator) adds an L1 penalty
	to ordinary least squares regression to encourage sparsity in the weight vector.

	The optimization problem solved is:

		minimize over w, b:
			(1 / n) * ||y - (Xw + b)||² + λ * ||w||₁

	This implementation uses the Iterative Shrinkage-Thresholding Algorithm (ISTA),
	which alternates between a gradient descent step on the quadratic loss and a
	soft-thresholding (L1 proximal) update on the weights.

	Parameters
	----------
	learning_rate : float
		Gradient step size (often denoted η). Must be small enough to ensure convergence.
	max_iter : int
		Maximum number of update iterations.
	l1_penalty : float
		Regularization strength λ for the L1 penalty term.
	tol : float
		Stopping tolerance. Training stops if the change in cost between two iterations
		falls below this threshold.
	fit_intercept : bool
		If True, a bias term (intercept) is learned and not regularized.
	normalize : bool
		If True, columns of X are standardized (zero mean and unit variance) before training.
	random_state : int or None
		Random seed (useful if stochastic variants are added later).
	record_history : bool
		If True, stores the evolution of the total cost (loss) at each iteration
		in `cost_history_`.
	dtype : str
		Internal numeric type ("float32" by default for speed and memory efficiency).

	Notes
	-----
	ISTA algorithm steps:
		1. Compute the gradient of the MSE loss:
		   grad_w = (2 / n) * Xᵀ(Xw + b - y)
		2. Gradient descent update:
		   w <- w - η * grad_w
		3. Apply the L1 proximal operator (soft thresholding):
		   w <- sign(w) * max(|w| - η * λ, 0)
		4. Repeat until convergence or until `max_iter` iterations are reached.

	The bias term b is updated separately and is not regularized.
	Weight updates are fully vectorized (no Python loops).
	If `normalize=True`, normalization is automatically reversed at prediction time.

	Attributes
	----------
	coef_ : np.ndarray
		Learned coefficients (model weights).
	intercept_ : float
		Learned bias term (not regularized).
	cost_history_ : list
		Sequence of total cost values (MSE + L1 regularization) if `record_history=True`.
	_x_mean, _x_scale, _y_mean :
		Internal statistics used for normalization.
	typ : str
		'r' indicating a regression task.
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

		# normalization stats
		self._x_mean: np.ndarray | None = None
		self._x_scale: np.ndarray | None = None
		self._y_mean: float = 0.0

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

	def fit(self, X, y):
		Xc, yc = self._prep_fit(X, y)  # (n, p)
		n, p = Xc.shape

		w = np.zeros(p, dtype=self.dtype)
		b = np.array(0.0, dtype=self.dtype)
		self.cost_history_.clear()

		lr = self.learning_rate
		lam = self.l1_penalty
		prev_cost = np.inf

		# we compute r = Xc @ w + b - yc to reuse
		r = -yc.copy()
		# ISTA loop
		for _ in range(self.max_iter):
			grad_w = (2.0 / n) * (Xc.T @ r)
			grad_b = (2.0 / n) * np.sum(r, dtype=self.dtype)

			w_new = w - lr * grad_w
			b_new = b - lr * grad_b

			# prox L1 (only on w)
			if lam != 0.0:
				w_new = self._soft_threshold_vec(w_new, lr * lam)

			# mettre à jour r efficacement:
			# r_new = (Xc @ w_new + b_new) - yc
			#	   = r + Xc@(w_new - w) + (b_new - b)
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

	def predict(self, X):
		if self.coef_ is None:
			raise RuntimeError("The model isn't trained, call `fit` first")
		Xc = self._prep_predict(X)
		return (Xc @ self.coef_) + self.intercept_
