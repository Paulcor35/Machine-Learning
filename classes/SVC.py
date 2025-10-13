#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-
import numpy as np

class SVC:
	"""
	Linear Support Vector Machine (binary classification) implemented with NumPy.

	This model is trained using full-batch gradient descent on the hinge loss
	with L2 regularization. It finds the hyperplane f(x) = w·x + b that maximizes
	the margin between classes while penalizing misclassified points or those
	within the margin.

	Objective
	---------
	Minimize the following loss:
		L(w, b) = ½‖w‖² + C · Σ_i max(0, 1 - y_i (w·x_i + b))
	where y_i ∈ {-1, +1}.
	The first term (½‖w‖²) enforces a large margin via L2 regularization,
	while the second term (hinge loss) penalizes margin violations or misclassifications.

	Parameters
	----------
	learning_rate : float
		Step size for gradient descent updates.
	C : float
		Penalty strength for margin violations (slack terms).
		Larger C values correct errors more aggressively but may increase overfitting risk.
	n_iters : int
		Number of training iterations (epochs).
	shuffle : bool
		Whether to shuffle the training samples at each epoch (improves stability).
	random_state : int or None
		Random seed for reproducibility when shuffling data.
	dtype : str
		Internal numeric type (default "float32" for performance and memory efficiency).

	Notes
	-----
	Full-batch subgradient computation:
		Let m_i = y_i (w·x_i + b).
		For "active" points with m_i < 1:
		    dL/dw = w - C · Σ_i y_i x_i
		    dL/db = - C · Σ_i y_i
		If all points satisfy m_i ≥ 1, only the L2 regularization term applies:
		    dL/dw = w,  dL/db = 0.

	The full-batch update scheme (over all X per iteration) is significantly faster
	and more stable than sample-by-sample SGD in pure Python.

	Attributes
	----------
	w : np.ndarray
		Learned weight vector.
	b : float
		Learned bias (intercept).
	typ : str
		'c' indicating a classification task.
	"""


	typ = 'c'

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

			s = X_ep @ self.w + self.b		  # (n,)
			m = y_ep * s					  # (n,)
			active = m < 1.0				  # hinge active

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
			raise RuntimeError("The model isn't trained, call `fit` first")
		X = np.asarray(X, dtype=self.dtype, order="C")
		return (X @ self.w + self.b).astype(np.float64, copy=False)

	def predict(self, X: np.ndarray) -> np.ndarray:
		s = self.decision_function(X)
		return (s >= 0.0).astype(int)
