#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class _Node:
	is_leaf: bool
	value: float | None = None
	attr: int | None = None
	threshold: float | None = None
	left: Optional["_Node"] = None
	right: Optional["_Node"] = None

def _variance(y: np.ndarray) -> float:
	m = y.mean()
	return float((y * y).mean() - m * m)

def _best_split_vectorized(X: np.ndarray,
						   y: np.ndarray,
						   min_samples_leaf: int = 1,
						   max_splits_per_feature: int | None = None,
						   min_gain: float = 1e-6) -> tuple[int, float]:
	"""
	Finds the best split across all features using minimal Python loops.

	- Sort values by feature at the node.
	- Perform a fully vectorized scan over all valid splits.
	- Optional: subsample split positions if very numerous (`max_splits_per_feature`).

	Returns a tuple `(best_feature, best_threshold)` or `(-1, 0.0)` if no gain
	is found.
	"""
	n, p = X.shape
	base_var = _variance(y)
	best_gain = -1.0
	best_attr = -1
	best_thr = 0.0
	min_gain = float(min_gain)

	for j in range(p):
		xj = X[:, j]
		order = np.argsort(xj, kind="mergesort")
		xs = xj[order]
		ys = y[order]

		# Positions where the value changes → only splits between distinct values
		diff = np.diff(xs)
		valid_pos = np.nonzero(diff != 0)[0]  # positions i = 0..n-2, split between i and i+1

		if valid_pos.size == 0:
			continue

		# min_samples_leaf: restrict valid split positions
		mask_leaf = (valid_pos + 1 >= min_samples_leaf) & (valid_pos + 1 <= n - min_samples_leaf)
		valid_pos = valid_pos[mask_leaf]
		if valid_pos.size == 0:
			continue

		# If too many positions, sample a subset
		if max_splits_per_feature is not None and valid_pos.size > max_splits_per_feature:
			step = int(np.ceil(valid_pos.size / max_splits_per_feature))
			valid_pos = valid_pos[::step]

		# Prefix sums to quickly compute left/right variance
		csum = np.cumsum(ys, dtype=np.float64)
		csum2 = np.cumsum(ys * ys, dtype=np.float64)

		left_n = valid_pos + 1
		right_n = n - left_n

		left_sum = csum[valid_pos]
		left_sum2 = csum2[valid_pos]
		left_mean = left_sum / left_n
		left_var = left_sum2 / left_n - left_mean * left_mean

		right_sum = csum[-1] - left_sum
		right_sum2 = csum2[-1] - left_sum2
		right_mean = right_sum / right_n
		right_var = right_sum2 / right_n - right_mean * right_mean

		new_var = (left_n / n) * left_var + (right_n / n) * right_var
		gains = base_var - new_var

		k = int(np.argmax(gains))
		gain = float(gains[k])
		if gain > best_gain and gain > min_gain:
			best_gain = gain
			best_attr = j
			# threshold = middle of xs[i] and xs[i+1]
			i = int(valid_pos[k])
			best_thr = float((xs[i] + xs[i + 1]) * 0.5)

	if best_attr < 0:
		return -1, 0.0
	return best_attr, best_thr


class DecisionTreeRegressor:
	"""
	DecisionTreeRegressor implemented in NumPy with vectorized split search
	and several regularization safeguards.

	This model recursively builds a binary tree that partitions the feature space.
	At each node, it selects the split (feature j, threshold t) that most reduces
	the target variance (MSE criterion). Each leaf predicts the mean of y values
	contained in that leaf.

	Parameters
	----------
	max_depth : int or None
		Maximum depth of the tree. If None, the tree grows until stopping criteria
		are met.
	min_samples_split : int
		Minimum number of samples required at a node to attempt a split.
	min_samples_leaf : int
		Minimum number of samples required in each leaf after a split.
	max_splits_per_feature : int or None
		Limits the number of split positions evaluated per feature at a node
		(subsampling valid thresholds) to speed up training. If None, all valid
		split positions are tested.
	min_gain : float
		Minimum required variance reduction to accept a split; otherwise, a leaf
		is created.

	Notes
	-----
	Criterion:
		Variance reduction (equivalent to minimizing mean squared error).
		The best split minimizes the weighted variance of left and right subsets.

	Split implementation:
		- Sort the feature values at the node (stable sort).
		- Use vectorized prefix-sum computations to evaluate all valid split
		  positions in O(n) per feature.
		- Option `max_splits_per_feature` limits the number of tested thresholds
		  when many are available (large n, continuous values).

	Stopping conditions:
		- maximum depth reached (`max_depth`),
		- insufficient samples (`min_samples_split` / `min_samples_leaf`),
		- variance gain below `min_gain`,
		- constant target values in the node.

	Prediction:
		Traverse the tree down to a leaf; return the local mean of y.

	Attributes
	----------
	tree : _Node or None
		Root of the trained tree (recursive node structure).
	typ : str
		'r' indicating a regression task.
	"""


	typ = 'r'

	def __init__(self,
				 max_depth: int | None = 5,
				 min_samples_split: int = 5,
				 min_samples_leaf: int = 1,
				 max_splits_per_feature: int | None = 256,
				 min_gain: float = 1e-6):
		self.max_depth = max_depth
		self.min_samples_split = int(min_samples_split)
		self.min_samples_leaf = int(min_samples_leaf)
		self.max_splits_per_feature = (None if max_splits_per_feature is None
									   else int(max_splits_per_feature))
		self.min_gain = float(min_gain)
		self.tree: _Node | None = None

	def _build(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, depth: int) -> _Node:
		y_sub = y[idx]
		n = y_sub.size

		if n == 0:
			return _Node(is_leaf=True, value=0.0)
		if (self.max_depth is not None and depth >= self.max_depth) or n < self.min_samples_split:
			return _Node(is_leaf=True, value=float(y_sub.mean()))
		if np.all(y_sub == y_sub[0]):
			return _Node(is_leaf=True, value=float(y_sub[0]))

		X_sub = X[idx]
		attr, thr = _best_split_vectorized(
			X_sub, y_sub,
			min_samples_leaf=self.min_samples_leaf,
			max_splits_per_feature=self.max_splits_per_feature,
			min_gain=self.min_gain
		)
		if attr == -1:
			return _Node(is_leaf=True, value=float(y_sub.mean()))

		left_mask = X_sub[:, attr] <= thr
		left_idx = idx[left_mask]
		right_idx = idx[~left_mask]

		# sécurités min_samples_leaf
		if left_idx.size < self.min_samples_leaf or right_idx.size < self.min_samples_leaf:
			return _Node(is_leaf=True, value=float(y_sub.mean()))

		left = self._build(X, y, left_idx, depth + 1)
		right = self._build(X, y, right_idx, depth + 1)
		return _Node(is_leaf=False, attr=attr, threshold=thr, left=left, right=right)

	def fit(self, X, y):
		X = np.asarray(X, dtype=np.float32, order="C")
		y = np.asarray(y, dtype=np.float32).reshape(-1)
		idx = np.arange(y.size, dtype=np.int32)
		self.tree = self._build(X, y, idx, depth=0)
		return self

	def _predict_one(self, x: np.ndarray, node: _Node) -> float:
		while not node.is_leaf:
			node = node.left if x[node.attr] <= node.threshold else node.right
		return node.value

	def predict(self, X):
		if self.tree is None:
			raise RuntimeError("The model isn't trained, call `fit` first")
		X = np.asarray(X, dtype=np.float32, order="C")
		return np.array([self._predict_one(X[i], self.tree) for i in range(X.shape[0])],
						dtype=np.float32)
