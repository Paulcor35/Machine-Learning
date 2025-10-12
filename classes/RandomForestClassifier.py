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
    RandomForestClassifier (classification par forêts aléatoires),
    implémenté en NumPy avec entraînement parallélisable.

    Principe
    --------
    Une forêt est un ensemble d’arbres de décision indépendants.
    Chaque arbre est entraîné sur :
      1) un bootstrap des lignes (échantillonnage avec remise),
      2) un sous-ensemble aléatoire de variables à chaque split (max_features).
    La prédiction finale se fait par vote majoritaire des arbres.
    Ce bagging + sous-échantillonnage de variables réduit fortement la variance.

    Paramètres
    ----------
    n_estimators : int
        Nombre d’arbres dans la forêt.
    max_depth : int | None
        Profondeur maximale des arbres (None = croissance jusqu’au critère d’arrêt).
    min_samples_split : int
        Minimum d’échantillons requis pour tenter un split.
    min_samples_leaf : int
        Minimum d’échantillons dans une feuille.
    max_features : {"sqrt","log2", int, float, None}
        Nombre de variables candidates aux splits :
          - "sqrt"  → √p (classique en classification)
          - "log2"  → log₂(p)
          - int     → nombre fixe
          - float   → proportion (0 < f ≤ 1)
          - None    → toutes les variables
    bootstrap : bool
        Si True, échantillonnage avec remise des lignes pour chaque arbre.
    random_state : int | None
        Graine aléatoire (reproductibilité).
    n_jobs : int
        Nombre de processus parallèles pour l’entraînement (>1 ou -1 pour tous les cœurs).
    dtype : str
        Type numérique interne (par défaut "float32" pour vitesse/mémoire).

    Notes
    -----
    - Entraînement :
        • Pour chaque arbre t :
            - Tirer n lignes (bootstrap) et k features (max_features).
            - Entraîner un DecisionTreeClassifier sur (X_boot[:, feats], y_boot).
        • Les tirages sont pré-générés pour limiter l’overhead Python.
        • Optionnellement, l’apprentissage des arbres est parallélisé (ProcessPoolExecutor).

    - Prédiction :
        • predict : vote majoritaire des T prédictions d’arbres (gestion tie-break implicite via argmax).
        • predict_proba : fréquence des votes par classe (moyenne des indicatrices).

    Avantages
    ---------
    - Très robuste au sur-apprentissage grâce à l’agrégation d’arbres variés.
    - Supporte naturellement les interactions non linéaires et les variables bruitées.
    - Parallélisable et efficace sur des données tabulaires.

    Attributs principaux
    --------------------
    classes_ : np.ndarray
        Liste des classes vues à l’entraînement (ordre stable).
    _estimators : list[tuple(tree, feat_idx)]
        Arbres entraînés et indices de variables utilisés par chacun.
    _class_to_idx : dict
        Mapping classe → index (utile pour vectoriser les votes).
    typ : str
        'c' pour indiquer une tâche de classification.
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

        # Pré-générer tous les tirages
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

        # votes en indices de classes
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