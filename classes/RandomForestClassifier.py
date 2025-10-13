#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np
from typing import Optional, Union, Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from classes.DecisionTreeClassifier import DecisionTreeClassifier as BaseTree


# ---------- helpers shared with regressor ----------

def _resolve_max_features(max_features: Union[str, int, float, None], p: int) -> int:
	if max_features is None:
		k = p
	elif isinstance(max_features, str):
		if max_features == "sqrt":
			k = int(np.sqrt(p))
		elif max_features == "log2":
			k = int(np.log2(p)) if p > 1 else 1
		else:
			raise ValueError(f"max_features unsupported: {max_features}")
	elif isinstance(max_features, (int, np.integer)):
		k = int(max_features)
	elif isinstance(max_features, float):
		if not (0 < max_features <= 1):
			raise ValueError("max_features float must be in (0,1].")
		k = int(np.ceil(max_features * p))
	else:
		raise ValueError(f"max_features type unsupported: {type(max_features)}")
	return max(1, min(k, p))


def _fit_one_tree(args: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
							  Optional[int], int, int]) -> Tuple[BaseTree, np.ndarray]:
	"""
	Fonction executed as sub-process (CPU-bound) when n_jobs != 1.
	"""
	X, y, row_idx, feat_idx, max_depth, min_samples_split, min_samples_leaf = args
	X_boot_sub = X[row_idx][:, feat_idx]
	y_boot = y[row_idx]
	tree = BaseTree(max_depth=max_depth,
					min_samples_split=min_samples_split,
					min_samples_leaf=min_samples_leaf)
	tree.fit(X_boot_sub, y_boot)
	return tree, feat_idx


class RandomForestClassifier:
	"""
	RandomForestClassifier implemented in NumPy with support for parallel training.

	This model builds an ensemble of independent decision trees.
	Each tree is trained on:
	  1) a bootstrap sample of the training rows (sampling with replacement),
	  2) a random subset of features selected at each split (controlled by `max_features`).

	The final prediction is obtained by majority voting among all trees.
	This combination of bagging and random feature selection greatly reduces variance.

	Principle
	---------
	A random forest is an ensemble of decision trees trained independently.
	Each tree learns from a random subset of data and features, and the ensemble
	aggregates their predictions by majority vote.

	Parameters
	----------
	n_estimators : int
		Number of trees in the forest.
	max_depth : int or None
		Maximum depth of each tree. If None, trees are expanded until a stopping
		criterion is met.
	min_samples_split : int
		Minimum number of samples required to attempt a split.
	min_samples_leaf : int
		Minimum number of samples required to form a leaf.
	max_features : {"sqrt", "log2", int, float, None}
		Number of features to consider when looking for the best split:
		  - "sqrt"  -> square root of the number of features (default for classification)
		  - "log2"  -> base-2 logarithm of the number of features
		  - int     -> fixed number of features
		  - float   -> proportion of features (0 < f ≤ 1)
		  - None    -> all features are considered
	bootstrap : bool
		If True, each tree is trained on a bootstrap sample of the training data.
	random_state : int or None
		Random seed for reproducibility.
	n_jobs : int
		Number of parallel processes used for training.
		If -1, all CPU cores are used.
	dtype : str
		Internal numeric type ("float32" by default for efficiency and speed).

	Notes
	-----
	Training:
		- For each tree t:
		    - Draw n rows (bootstrap) and k features (according to `max_features`).
		    - Train a DecisionTreeClassifier on (X_boot[:, feats], y_boot).
		- Sampling indices are pre-generated to reduce Python overhead.
		- Optionally, tree training is parallelized using `ProcessPoolExecutor`.

	Prediction:
		- `predict`: majority vote among the T tree predictions
		  (tie-breaking handled implicitly via argmax).
		- `predict_proba`: class probabilities computed as the average of per-tree
		  indicator votes.

	Attributes
	----------
	classes_ : np.ndarray
		List of unique class labels observed during training (stable order).
	_estimators : list
		List of (tree, feature_indices) pairs representing each trained tree and
		its selected features.
	_class_to_idx : dict
		Mapping from class label to index, used to vectorize voting operations.
	typ : str
		'c' indicating a classification task.
	"""



	typ = 'c'

	def __init__(self,
				 n_estimators: int = 100,
				 max_depth: Optional[int] = None,
				 min_samples_split: int = 2,
				 min_samples_leaf: int = 1,
				 max_features: Union[str, int, float, None] = "sqrt",
				 bootstrap: bool = True,
				 random_state: Optional[int] = None,
				 n_jobs: int = 1,
				 dtype: str = "float32"):
		self.n_estimators = int(n_estimators)
		self.max_depth = max_depth
		self.min_samples_split = int(min_samples_split)
		self.min_samples_leaf = int(min_samples_leaf)
		self.max_features = max_features
		self.bootstrap = bool(bootstrap)
		self.random_state = None if random_state is None else int(random_state)
		self.n_jobs = int(n_jobs)
		self.dtype = dtype

		# learned
		self._estimators: List[Tuple[BaseTree, np.ndarray]] = []
		self._rng = np.random.default_rng(self.random_state)
		self.classes_: np.ndarray | None = None
		self._class_to_idx: Dict[object, int] | None = None

	def _bootstrap_indices(self, n: int) -> np.ndarray:
		if self.bootstrap:
			return self._rng.integers(0, n, size=n)
		else:
			return self._rng.permutation(n)

	def fit(self, X: np.ndarray, y: np.ndarray):
		X = np.asarray(X, dtype=self.dtype, order="C")
		y = np.asarray(y).reshape(-1)
		n, p = X.shape

		self.classes_ = np.unique(y)
		self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}

		k = _resolve_max_features(self.max_features, p)
		self._estimators = []

		# Pre-generate all the tries
		row_indices = [self._bootstrap_indices(n) for _ in range(self.n_estimators)]
		feat_indices = [self._rng.choice(p, size=k, replace=False) for _ in range(self.n_estimators)]

		if self.n_jobs == 1:
			# fast sequential
			for r_idx, f_idx in zip(row_indices, feat_indices):
				X_boot_sub = X[r_idx][:, f_idx]
				y_boot = y[r_idx]
				tree = BaseTree(max_depth=self.max_depth,
								min_samples_split=self.min_samples_split,
								min_samples_leaf=self.min_samples_leaf).fit(X_boot_sub, y_boot)
				self._estimators.append((tree, f_idx.astype(int)))
		else:
			# parallel (CPU-bound)
			tasks = []
			with ProcessPoolExecutor(max_workers=None if self.n_jobs < 0 else self.n_jobs) as ex:
				for r_idx, f_idx in zip(row_indices, feat_indices):
					tasks.append(ex.submit(
						_fit_one_tree,
						(X, y, r_idx, f_idx, self.max_depth,
						 self.min_samples_split, self.min_samples_leaf)
					))
				for fut in as_completed(tasks):
					tree, f_idx = fut.result()
					self._estimators.append((tree, f_idx.astype(int)))

		return self

	def predict(self, X: np.ndarray):
		if not self._estimators:
			raise RuntimeError("The model isn't trained, call `fit` first")
		X = np.asarray(X, dtype=self.dtype, order="C")
		n = X.shape[0]
		T = len(self._estimators)

		# votes as class indices
		votes = np.empty((T, n), dtype=np.int32)
		for t, (tree, feat_idx) in enumerate(self._estimators):
			pred = tree.predict(X[:, feat_idx])
			# map to class indices
			idxs = np.vectorize(self._class_to_idx.get)(pred)
			votes[t] = idxs

		# vectorised biggest vote
		# trick: for a few classes, we count hist on each line
		n_classes = len(self.classes_)
		counts = np.zeros((n, n_classes), dtype=np.int32)
		for c in range(n_classes):
			counts[:, c] = np.sum(votes == c, axis=0)
		winners = np.argmax(counts, axis=1)
		return self.classes_[winners]

	def predict_proba(self, X: np.ndarray):
		if self.classes_ is None:
			raise RuntimeError("The model isn't trained, call `fit` first")
		X = np.asarray(X, dtype=self.dtype, order="C")
		n = X.shape[0]
		n_classes = len(self.classes_)
		T = len(self._estimators)

		probs = np.zeros((n, n_classes), dtype=np.float32)
		for (tree, feat_idx) in self._estimators:
			pred = tree.predict(X[:, feat_idx])
			# accumulate votes per class (vectorial)
			for c_idx, c in enumerate(self.classes_):
				probs[:, c_idx] += (pred == c)

		probs /= float(T)
		return probs
