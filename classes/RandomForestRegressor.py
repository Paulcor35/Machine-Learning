# classes/RandomForestRegressor.py
#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-

import numpy as np
from typing import Optional, Union, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

# Arbre base (ta version numpy)
from classes.DecisionTreeRegressor import DecisionTreeRegressor as BaseTree


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
                              Optional[int], int]) -> Tuple[BaseTree, np.ndarray]:
    """Fonction exécutée en sous-processus quand n_jobs != 1."""
    X, y, row_idx, feat_idx, max_depth, min_samples_split = args
    # vues NumPy, pas de pandas
    X_boot_sub = X[row_idx][:, feat_idx]
    y_boot = y[row_idx]
    tree = BaseTree(max_depth=max_depth, min_samples_split=min_samples_split)
    tree.fit(X_boot_sub, y_boot)
    return tree, feat_idx


class RandomForestRegressor:
    """
    RandomForestRegressor (régression par forêts aléatoires) 
    implémenté de manière vectorisée avec NumPy et support de la parallélisation CPU.

    Principe
    --------
    Une forêt aléatoire est un ensemble d’arbres de décision indépendants,
    chacun entraîné sur un sous-échantillon des données et un sous-ensemble aléatoire de variables.
    La prédiction finale est la moyenne des prédictions de tous les arbres.

    Paramètres
    ----------
    n_estimators : int
        Nombre total d’arbres de la forêt.
    max_depth : int | None
        Profondeur maximale de chaque arbre. None = croissance complète (jusqu’à pureté totale).
    min_samples_split : int
        Nombre minimal d’échantillons requis pour effectuer un split.
    max_features : {"sqrt", "log2", int, float, None}
        Nombre de variables considérées à chaque split :
          - "sqrt"  → √p (par défaut pour la régression)
          - "log2"  → log₂(p)
          - int     → nombre fixe de variables
          - float   → proportion (0 < f ≤ 1)
          - None    → toutes les variables
    bootstrap : bool
        Si True, chaque arbre est entraîné sur un échantillon bootstrap (avec remise).
    random_state : int | None
        Graine aléatoire pour la reproductibilité.
    n_jobs : int
        Nombre de processus parallèles utilisés (si > 1). 
        -1 → tous les cœurs disponibles.
    dtype : str
        Type de données interne ("float32" par défaut, pour efficacité mémoire).

    Notes
    -----
    - Pour chaque arbre :
        1. Un échantillon de lignes est tiré aléatoirement (bootstrap).
        2. Un sous-ensemble de colonnes (features) est sélectionné aléatoirement.
        3. Un arbre de décision est entraîné sur ces sous-échantillons.

    - La prédiction finale est obtenue par moyenne :
        ŷ = (1 / T) Σ f_t(x)
      où T = n_estimators et f_t(x) = prédiction de l’arbre t.

    - L’entraînement peut être parallélisé grâce à `ProcessPoolExecutor` pour accélérer le fit
      sur les machines multi-cœurs (utile pour de grands ensembles d’arbres).

    - Attributs principaux :
        • _estimators : liste des paires (arbre, indices_features)
        • _rng         : générateur pseudo-aléatoire NumPy
        • typ = 'r'    : indique une tâche de régression

    Avantages
    ----------
    - Robuste au sur-apprentissage (grâce à l’agrégation).
    - Gère naturellement la variance élevée des arbres individuels.
    """

    typ = 'r'

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 max_features: Union[str, int, float, None] = "sqrt",
                 bootstrap: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 dtype: str = "float32"):   # float32: plus compact et souvent plus rapide
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.random_state = None if random_state is None else int(random_state)
        self.n_jobs = int(n_jobs)
        self.dtype = dtype

        self._estimators: List[Tuple[BaseTree, np.ndarray]] = []
        self._rng = np.random.default_rng(self.random_state)
        self._p = None  # nb de features apprises

    # -------------------- helpers --------------------

    def _bootstrap_indices(self, n: int) -> np.ndarray:
        if self.bootstrap:
            return self._rng.integers(0, n, size=n)
        else:
            return self._rng.permutation(n)

    # -------------------- API --------------------

    def fit(self, X, y):
        X = np.asarray(X, dtype=self.dtype, order="C")
        y = np.asarray(y, dtype=self.dtype).reshape(-1)
        n, p = X.shape
        self._p = p
        k = _resolve_max_features(self.max_features, p)

        # Pré-génère tous les tirages pour limiter le coût d’orchestration
        row_indices = [self._bootstrap_indices(n) for _ in range(self.n_estimators)]
        feat_indices = [self._rng.choice(p, size=k, replace=False) for _ in range(self.n_estimators)]

        self._estimators = []

        if self.n_jobs == 1:
            # Entraînement séquentiel rapide (NumPy only)
            for r_idx, f_idx in zip(row_indices, feat_indices):
                X_boot_sub = X[r_idx][:, f_idx]
                y_boot = y[r_idx]
                tree = BaseTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split).fit(X_boot_sub, y_boot)
                self._estimators.append((tree, f_idx.astype(int)))
        else:
            # Parallélisation par sous-processus (CPU-bound)
            tasks = []
            with ProcessPoolExecutor(max_workers=None if self.n_jobs < 0 else self.n_jobs) as ex:
                for r_idx, f_idx in zip(row_indices, feat_indices):
                    tasks.append(ex.submit(
                        _fit_one_tree,
                        (X, y, r_idx, f_idx, self.max_depth, self.min_samples_split)
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
        out = np.zeros(n, dtype=np.float64)  # accumulate en float64 pour la stabilité

        for tree, feat_idx in self._estimators:
            out += tree.predict(X[:, feat_idx]).astype(np.float64)

        out /= float(len(self._estimators))
        return out.astype(self.dtype, copy=False)