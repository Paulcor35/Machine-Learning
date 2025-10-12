# classes/RandomForestClassifier.py
#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-

import numpy as np
from typing import Optional, Union, Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

# base learner (ta version numpy optimisée)
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
    Fonction exécutée en sous-processus (CPU-bound) quand n_jobs != 1.
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
    RandomForestClassifier (optimisé NumPy + parallélisable)
    - Base learner: DecisionTreeClassifier (maison)
    - Bootstrap des lignes + sous-échantillonnage des features
    - Vote majoritaire / proba = fréquence des votes
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

        # appris
        self._estimators: List[Tuple[BaseTree, np.ndarray]] = []
        self._rng = np.random.default_rng(self.random_state)
        self.classes_: np.ndarray | None = None
        self._class_to_idx: Dict[object, int] | None = None

    # --------- internal ---------

    def _bootstrap_indices(self, n: int) -> np.ndarray:
        if self.bootstrap:
            return self._rng.integers(0, n, size=n)
        else:
            return self._rng.permutation(n)

    # --------- API ---------

    def fit(self, X, y):
        X = np.asarray(X, dtype=self.dtype, order="C")
        y = np.asarray(y).reshape(-1)
        n, p = X.shape

        self.classes_ = np.unique(y)
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        k = _resolve_max_features(self.max_features, p)
        self._estimators = []

        # Pré-générer tous les tirages → moins d’overhead Python
        row_indices = [self._bootstrap_indices(n) for _ in range(self.n_estimators)]
        feat_indices = [self._rng.choice(p, size=k, replace=False) for _ in range(self.n_estimators)]

        if self.n_jobs == 1:
            # séquentiel rapide
            for r_idx, f_idx in zip(row_indices, feat_indices):
                X_boot_sub = X[r_idx][:, f_idx]
                y_boot = y[r_idx]
                tree = BaseTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf).fit(X_boot_sub, y_boot)
                self._estimators.append((tree, f_idx.astype(int)))
        else:
            # parallèle (CPU-bound)
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

    def predict(self, X):
        if not self._estimators:
            raise RuntimeError("Modèle non entraîné : appelez fit(X, y) d'abord.")
        X = np.asarray(X, dtype=self.dtype, order="C")
        n = X.shape[0]
        T = len(self._estimators)

        # votes en indices de classes (plus rapide que objets)
        votes = np.empty((T, n), dtype=np.int32)
        for t, (tree, feat_idx) in enumerate(self._estimators):
            pred = tree.predict(X[:, feat_idx])
            # map to class indices
            idxs = np.vectorize(self._class_to_idx.get)(pred)
            votes[t] = idxs

        # vote majoritaire vectorisé
        # trick: pour peu de classes, on compte via hist sur chaque ligne
        n_classes = len(self.classes_)
        counts = np.zeros((n, n_classes), dtype=np.int32)
        for c in range(n_classes):
            counts[:, c] = np.sum(votes == c, axis=0)
        winners = np.argmax(counts, axis=1)
        return self.classes_[winners]

    def predict_proba(self, X):
        if self.classes_ is None:
            raise RuntimeError("Modèle non entraîné.")
        X = np.asarray(X, dtype=self.dtype, order="C")
        n = X.shape[0]
        n_classes = len(self.classes_)
        T = len(self._estimators)

        probs = np.zeros((n, n_classes), dtype=np.float32)
        for (tree, feat_idx) in self._estimators:
            pred = tree.predict(X[:, feat_idx])
            # accumulate votes per class (vectoriel)
            for c_idx, c in enumerate(self.classes_):
                probs[:, c_idx] += (pred == c)

        probs /= float(T)
        return probs

    # --------- scikit-like ---------

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "dtype": self.dtype,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k in ("n_estimators", "min_samples_split", "min_samples_leaf", "n_jobs") and v is not None:
                v = int(v)
            elif k in ("max_depth",) and v is not None:
                v = int(v)
            elif k in ("dtype", "max_features", "bootstrap", "random_state"):
                # laissons tels quels (resolve/validation ailleurs)
                pass
            if hasattr(self, k):
                setattr(self, k, v)
        return self