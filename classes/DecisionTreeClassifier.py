#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Optional

@dataclass
class _Node:
	is_leaf: bool
	prediction: object | None = None
	attr: int | None = None			      # index of the feature
	threshold: float | None = None		  # threshold if numeric split
	branches: dict | None = None		  # {"le": _Node, "gt": _Node} or {val: _Node}

def _entropy(y: np.ndarray) -> float:
	if y.size == 0:
		return 0.0
	vals, cnt = np.unique(y, return_counts=True)
	p = cnt.astype(np.float64) / y.size
	# ignore p==0
	return -float(np.sum(p[p > 0.0] * np.log2(p[p > 0.0])))

def _best_split_vectorized(
	X: np.ndarray,
	y: np.ndarray,
	min_samples_leaf: int = 1,
	max_thresholds: int | None = 256,
	min_gain: float = 1e-6,
) -> tuple[int | None, float | None, float]:
	"""
	Searches for the best split ENTROPY on all the columns.
	- For numeric columns: midpoints between each distinct consecutive value,
	  sub-sampled at max_thresholds if necessary.
	- For the category columns (discretes non-numeric or few values):
	  gain with partition {value}.
	Returns (attr, threshold, gain). If category split: threshold=None.
	"""
	n, p = X.shape
	base_h = _entropy(y)
	best_gain, best_attr, best_thr = -1.0, None, None

	for j in range(p):
		col = X[:, j]
		# numeric (float/int if >2 values)
		if np.issubdtype(col.dtype, np.number) and np.unique(col).size > 2:
			order = np.argsort(col, kind="mergesort")
			xs = col[order].astype(np.float64, copy=False)
			ys = y[order]

			# positions where the value changes (split between i and i+1)
			diff = np.diff(xs)
			pos = np.nonzero(diff != 0)[0]
			if pos.size == 0:
				continue

			# contstraint min_samples_leaf
			mask_leaf = (pos + 1 >= min_samples_leaf) & (pos + 1 <= n - min_samples_leaf)
			pos = pos[mask_leaf]
			if pos.size == 0:
				continue

			# sub-sampling splits if necessary
			if max_thresholds is not None and pos.size > max_thresholds:
				step = int(np.ceil(pos.size / max_thresholds))
				pos = pos[::step]

			# prefixes to accelerate the entropies
			# we compute entropies left/right with cumulative histograms over y
			# (works very well if y is binary ; for multi-classes we aggregate per class)
			classes, y_idx = np.unique(ys, return_inverse=True)
			C = classes.size

			# cumulativ per class
			one_hot = np.zeros((ys.size, C), dtype=np.int32)
			one_hot[np.arange(ys.size), y_idx] = 1
			csum = np.cumsum(one_hot, axis=0)			# (n, C)
			left_cnt = csum[pos]						  # (k, C)
			left_n = left_cnt.sum(axis=1).astype(np.float64)  # (k,)
			right_cnt = csum[-1] - left_cnt			   # (k, C)
			right_n = right_cnt.sum(axis=1).astype(np.float64)

			# weighted entropies
			def ent_from_counts(cnt):
				tot = cnt.sum(axis=1, keepdims=True)
				p = np.divide(cnt, tot, out=np.zeros_like(cnt, dtype=np.float64), where=tot > 0)
				with np.errstate(divide='ignore', invalid='ignore'):
					e = -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
				return e

			H_left = ent_from_counts(left_cnt)   # (k,)
			H_right = ent_from_counts(right_cnt) # (k,)
			new_H = (left_n / n) * H_left + (right_n / n) * H_right
			gains = base_h - new_H

			k = int(np.argmax(gains))
			gain = float(gains[k])
			if gain > best_gain and gain > float(min_gain):
				best_gain = gain
				best_attr = j
				i = int(pos[k])
				best_thr = float((xs[i] + xs[i + 1]) * 0.5)

		else:
			# category / binary
			vals, counts = np.unique(col, return_counts=True)
			if vals.size <= 1:
				continue
			# weighted entropy of the sub-sets
			new_H = 0.0
			for v, c in zip(vals, counts):
				if c < min_samples_leaf or (n - c) < min_samples_leaf:
					# if a branch would be too small => ignore this category split
					new_H = base_h
					break
				y_sub = y[col == v]
				new_H += (c / n) * _entropy(y_sub)
			gain = base_h - new_H
			if gain > best_gain and gain > float(min_gain):
				best_gain = gain
				best_attr = j
				best_thr = None

	return best_attr, best_thr, best_gain

class DecisionTreeClassifier:
	"""
	DecisionTreeClassifier implemented in NumPy, with vectorized split search
	(using entropy) and multiple regularization safeguards.

	This model recursively builds a binary tree (or multi-branch tree for
	categorical features). At each node, it selects the split that maximizes
	information gain (i.e., reduces entropy the most). Each leaf predicts
	the majority class among the samples it contains.

	Parameters
	----------
	max_depth : int or None
		Maximum depth of the tree. If None, the tree grows until stopping
		criteria are met.
	min_samples_split : int
		Minimum number of samples required at a node to attempt a split.
	min_samples_leaf : int
		Minimum number of samples required in each leaf after a split.
	max_thresholds : int or None
		For numerical features, limits the number of threshold values evaluated
		at each node (subsampling candidate split points) to speed up training.
		If None, all valid thresholds are tested.
	min_gain : float
		Minimum entropy gain required to accept a split; otherwise, a leaf is created.
	random_state : int or None
		Random seed for reproducibility (useful if stochastic elements are added).

	Notes
	-----
	- Criterion: entropy. The best split minimizes the weighted entropy
	  of the child nodes.
	- Numerical features: candidate thresholds are midpoints between
	  consecutive distinct values (optionally subsampled via `max_thresholds`).
	- Categorical features: one branch is created per unique category.
	- Vectorized implementation:
	  - stable sorting by feature values per node,
	  - cumulative class counts (prefix sums) for O(n) evaluation
		of all valid splits per feature,
	  - `min_samples_leaf` and `min_gain` constraints to prevent weak splits
		and limit overfitting.
	- Stopping conditions: maximum depth reached, insufficient samples,
	  gain below `min_gain`, or pure node (identical labels).

	Attributes
	----------
	_tree : _Node or None
		Root of the trained tree (recursive node structure).
	typ : str
		'c' indicating a classification task.
	"""


	typ = 'c'

	def __init__(self,
				 max_depth: int | None = 10,
				 min_samples_split: int = 2,
				 min_samples_leaf: int = 1,
				 max_thresholds: int | None = 50,
				 min_gain: float = 1e-6,
				 random_state: int | None = None):
		self.max_depth = max_depth
		self.min_samples_split = int(min_samples_split)
		self.min_samples_leaf = int(min_samples_leaf)
		self.max_thresholds = None if max_thresholds is None else int(max_thresholds)
		self.min_gain = float(min_gain)
		self.random_state = random_state
		self._tree: _Node | None = None

	def _plurality(self, y: np.ndarray):
		return Counter(y).most_common(1)[0][0]

	def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
		n = y.size
		if n == 0:
			return _Node(is_leaf=True, prediction=None)
		if np.unique(y).size == 1:
			return _Node(is_leaf=True, prediction=y[0])
		if (self.max_depth is not None and depth >= self.max_depth) or n < self.min_samples_split:
			return _Node(is_leaf=True, prediction=self._plurality(y))

		attr, thr, gain = _best_split_vectorized(
			X, y,
			min_samples_leaf=self.min_samples_leaf,
			max_thresholds=self.max_thresholds,
			min_gain=self.min_gain
		)
		if attr is None or gain <= 0.0:
			return _Node(is_leaf=True, prediction=self._plurality(y))

		if thr is None:
			# split catégoriel
			branches = {}
			vals = np.unique(X[:, attr])
			for v in vals:
				mask = (X[:, attr] == v)
				if mask.sum() < self.min_samples_leaf or (n - mask.sum()) < self.min_samples_leaf:
					branches[v] = _Node(is_leaf=True, prediction=self._plurality(y))
				else:
					branches[v] = self._build(X[mask], y[mask], depth + 1)
			return _Node(is_leaf=False, attr=int(attr), threshold=None, branches=branches)
		else:
			# split numérique
			mask_le = X[:, attr] <= thr
			if mask_le.sum() < self.min_samples_leaf or (n - mask_le.sum()) < self.min_samples_leaf:
				return _Node(is_leaf=True, prediction=self._plurality(y))
			left = self._build(X[mask_le], y[mask_le], depth + 1)
			right = self._build(X[~mask_le], y[~mask_le], depth + 1)
			return _Node(is_leaf=False, attr=int(attr), threshold=float(thr), branches={"le": left, "gt": right})

	# ---------- API ----------
	def fit(self, X, y):
		X = np.asarray(X)
		y = np.asarray(y).reshape(-1)
		self._tree = self._build(X, y, depth=0)
		return self

	def _majority_subtree(self, node: _Node):
		counts = Counter()
		stack = [node]
		while stack:
			n = stack.pop()
			if n.is_leaf:
				counts[n.prediction] += 1
			else:
				for child in n.branches.values():
					stack.append(child)
		pred = counts.most_common(1)[0][0] if counts else None
		return _Node(is_leaf=True, prediction=pred)

	def _predict_one(self, node: _Node, x: np.ndarray):
		while not node.is_leaf:
			if node.threshold is None:
				v = x[node.attr]
				node = node.branches.get(v, self._majority_subtree(node))
				if node.is_leaf:
					break
			else:
				node = node.branches["le"] if x[node.attr] <= node.threshold else node.branches["gt"]
		return node.prediction

	def predict(self, X):
		if self._tree is None:
			raise RuntimeError("DecisionTreeClassifier non entraîné : appelez fit d'abord.")
		X = np.asarray(X)
		return np.array([self._predict_one(self._tree, X[i]) for i in range(X.shape[0])])
