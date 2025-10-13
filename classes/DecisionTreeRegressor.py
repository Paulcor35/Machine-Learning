#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class _Node:
    is_leaf: bool
    value: float | None = None       # value predicted if this node is a leaf
    attr: int | None = None          # index of the feature used for split
    threshold: float | None = None   # threshold value for numeric split
    left: Optional["_Node"] = None   # left child (_Node) if numeric split
    right: Optional["_Node"] = None  # right child (_Node) if numeric split

def _variance(y: np.ndarray) -> float:
    """
    Computes the variance of target values y.

    Variance is used as the splitting criterion (MSE reduction).
    """
    m = y.mean()                      # mean of y
    return float((y * y).mean() - m * m)  # Var(y) = E[y^2] - (E[y])^2

def _best_split_vectorized(X: np.ndarray,
                           y: np.ndarray,
                           min_samples_leaf: int = 1,
                           max_splits_per_feature: int | None = None,
                           min_gain: float = 1e-6) -> tuple[int, float]:
    """
    Finds the best split across all features using a vectorized approach.

    Steps:
    1. For each feature, sort the values (stable sort for reproducibility).
    2. Compute all possible split positions between distinct values.
    3. Apply `min_samples_leaf` constraints.
    4. Optionally subsample splits if too many (`max_splits_per_feature`).
    5. Compute left/right variance efficiently using prefix sums.
    6. Choose the split that maximizes variance reduction.

    Returns:
        (best_feature_index, best_threshold) or (-1, 0.0) if no valid split.
    """
    n, p = X.shape
    base_var = _variance(y)            # variance of current node
    best_gain = -1.0                    # best variance reduction seen
    best_attr = -1                       # best feature index
    best_thr = 0.0                       # best threshold value
    min_gain = float(min_gain)

    for j in range(p):
        xj = X[:, j]                   # take column j
        order = np.argsort(xj, kind="mergesort")  # stable sort
        xs = xj[order]                 # sorted values of feature j
        ys = y[order]                  # target values reordered accordingly

        # Only consider splits where feature value changes
        diff = np.diff(xs)
        valid_pos = np.nonzero(diff != 0)[0]  # candidate split positions

        if valid_pos.size == 0:
            continue  # skip features with no variation

        # Apply min_samples_leaf constraint
        mask_leaf = (valid_pos + 1 >= min_samples_leaf) & (valid_pos + 1 <= n - min_samples_leaf)
        valid_pos = valid_pos[mask_leaf]
        if valid_pos.size == 0:
            continue  # skip if no valid splits remain

        # Subsample splits if too many candidate thresholds
        if max_splits_per_feature is not None and valid_pos.size > max_splits_per_feature:
            step = int(np.ceil(valid_pos.size / max_splits_per_feature))
            valid_pos = valid_pos[::step]

        # Compute prefix sums for fast variance calculation
        csum = np.cumsum(ys, dtype=np.float64)
        csum2 = np.cumsum(ys * ys, dtype=np.float64)

        left_n = valid_pos + 1          # number of samples on left
        right_n = n - left_n            # number of samples on right

        left_sum = csum[valid_pos]
        left_sum2 = csum2[valid_pos]
        left_mean = left_sum / left_n
        left_var = left_sum2 / left_n - left_mean * left_mean

        right_sum = csum[-1] - left_sum
        right_sum2 = csum2[-1] - left_sum2
        right_mean = right_sum / right_n
        right_var = right_sum2 / right_n - right_mean * right_mean

        # Weighted variance after split
        new_var = (left_n / n) * left_var + (right_n / n) * right_var
        gains = base_var - new_var        # variance reduction

        k = int(np.argmax(gains))
        gain = float(gains[k])
        if gain > best_gain and gain > min_gain:
            best_gain = gain
            best_attr = j
            # threshold = midpoint between two consecutive values
            i = int(valid_pos[k])
            best_thr = float((xs[i] + xs[i + 1]) * 0.5)

    if best_attr < 0:
        return -1, 0.0
    return best_attr, best_thr


class DecisionTreeRegressor:
    """
    DecisionTreeRegressor implemented with NumPy.

    - Recursively builds a binary tree partitioning the feature space.
    - Each node chooses a split to maximize variance reduction (MSE reduction).
    - Leaves predict the mean of target values in that node.

    Stopping conditions:
    - Maximum depth reached.
    - Insufficient samples.
    - Variance gain below min_gain.
    - Constant target values.
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
        """
        Recursively build the tree node.

        - idx: indices of samples belonging to current node
        - depth: current depth in the tree
        """
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

        # min_samples_leaf safeguard
        if left_idx.size < self.min_samples_leaf or right_idx.size < self.min_samples_leaf:
            return _Node(is_leaf=True, value=float(y_sub.mean()))

        left = self._build(X, y, left_idx, depth + 1)
        right = self._build(X, y, right_idx, depth + 1)
        return _Node(is_leaf=False, attr=attr, threshold=thr, left=left, right=right)

    def fit(self, X, y):
        """
        Fit the decision tree regressor.

        Converts inputs to NumPy arrays and initializes recursive building.
        """
        X = np.asarray(X, dtype=np.float32, order="C")
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        idx = np.arange(y.size, dtype=np.int32)
        self.tree = self._build(X, y, idx, depth=0)
        return self

    def _predict_one(self, x: np.ndarray, node: _Node) -> float:
        """
        Traverse the tree to make a prediction for a single sample x.
        """
        while not node.is_leaf:
            node = node.left if x[node.attr] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        """
        Predict target values for multiple samples X.
        """
        if self.tree is None:
            raise RuntimeError("The model isn't trained, call `fit` first")
        X = np.asarray(X, dtype=np.float32, order="C")
        return np.array([self._predict_one(X[i], self.tree) for i in range(X.shape[0])],
                        dtype=np.float32)
