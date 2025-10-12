# classes/DecisionTreeClassifier.py
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
    attr: int | None = None               # index de la feature
    threshold: float | None = None        # seuil si split numérique
    branches: dict | None = None          # {"le": _Node, "gt": _Node} ou {val: _Node}

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
    Recherche le meilleur split ENTROPIE sur toutes les colonnes.
    - Pour les colonnes numériques: midpoints entre valeurs consécutives distinctes,
      sous-échantillonnés à max_thresholds si nécessaire.
    - Pour les colonnes catégorielles (valeurs discrètes non-numériques ou peu de valeurs):
      gain par partition {valeur}.
    Retourne (attr, threshold, gain). Si split catégoriel: threshold=None.
    """
    n, p = X.shape
    base_h = _entropy(y)
    best_gain, best_attr, best_thr = -1.0, None, None

    for j in range(p):
        col = X[:, j]
        # numérique (float/int avec >2 valeurs)
        if np.issubdtype(col.dtype, np.number) and np.unique(col).size > 2:
            order = np.argsort(col, kind="mergesort")
            xs = col[order].astype(np.float64, copy=False)
            ys = y[order]

            # positions où la valeur change (split entre i et i+1)
            diff = np.diff(xs)
            pos = np.nonzero(diff != 0)[0]
            if pos.size == 0:
                continue

            # contrainte min_samples_leaf
            mask_leaf = (pos + 1 >= min_samples_leaf) & (pos + 1 <= n - min_samples_leaf)
            pos = pos[mask_leaf]
            if pos.size == 0:
                continue

            # sous-échantillonnage des splits si nécessaire
            if max_thresholds is not None and pos.size > max_thresholds:
                step = int(np.ceil(pos.size / max_thresholds))
                pos = pos[::step]

            # préfixes pour accélérer les entropies
            # on calcule entropies gauche/droite via histogrammes cumulés sur y
            # (fonctionne très bien si y est binaire ; pour multi-classes on agrège par classe)
            classes, y_idx = np.unique(ys, return_inverse=True)
            C = classes.size

            # cumuls par classe
            one_hot = np.zeros((ys.size, C), dtype=np.int32)
            one_hot[np.arange(ys.size), y_idx] = 1
            csum = np.cumsum(one_hot, axis=0)            # (n, C)
            left_cnt = csum[pos]                          # (k, C)
            left_n = left_cnt.sum(axis=1).astype(np.float64)  # (k,)
            right_cnt = csum[-1] - left_cnt               # (k, C)
            right_n = right_cnt.sum(axis=1).astype(np.float64)

            # entropies pondérées
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
            # catégoriel / binaire
            vals, counts = np.unique(col, return_counts=True)
            if vals.size <= 1:
                continue
            # entropie pondérée des sous-ensembles
            new_H = 0.0
            for v, c in zip(vals, counts):
                if c < min_samples_leaf or (n - c) < min_samples_leaf:
                    # si une branche serait trop petite => ignorer ce split catégoriel
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
    DecisionTreeClassifier (arbre de décision pour la classification) en NumPy,
    avec recherche de split vectorisée (entropie) et garde-fous de régularisation.

    Principe
    --------
    On construit récursivement un arbre binaire (ou multi-branches pour les
    variables catégorielles). À chaque nœud, on choisit le split qui maximise
    le gain d’information (réduction d’entropie). Une feuille prédit la classe
    majoritaire des échantillons qu’elle contient.

    Paramètres
    ----------
    max_depth : int | None
        Profondeur maximale de l’arbre. None → croissance jusqu’aux critères d’arrêt.
    min_samples_split : int
        Nombre minimal d’échantillons dans un nœud pour tenter un split.
    min_samples_leaf : int
        Nombre minimal d’échantillons requis dans chaque feuille après le split.
    max_thresholds : int | None
        Pour les features numériques, limite le nombre de seuils évalués par nœud
        (sous-échantillonnage des positions de coupure) afin d’accélérer l’entraînement.
        None → toutes les positions valides.
    min_gain : float
        Gain d’entropie minimal requis pour accepter un split (sinon on crée une feuille).
    random_state : int | None
        Graine aléatoire (utile si vous ajoutez des éléments stochastiques).

    Notes
    -----
    - Critère : entropie. Le meilleur split est celui qui minimise
      l’entropie pondérée des sous-nœuds.
    - Numérique : les seuils candidats sont les milieux entre valeurs
      consécutives distinctes (éventuellement sous-échantillonnés via `max_thresholds`).
    - Catégoriel : partition par valeur (une branche par modalité).
    - Implémentation vectorisée :
        • tri stable par feature au nœud,
        • comptages cumulés par classe (préfixes) pour évaluer rapidement
          toutes les coupures valides en O(n) par feature,
        • contraintes `min_samples_leaf` et `min_gain` pour éviter des splits faibles
          et limiter le sur-apprentissage.
    - Arrêts : profondeur atteinte, effectif insuffisant, gain < min_gain,
      ou pureté (toutes les étiquettes identiques).

    Avantages
    ---------
    - Interprétable (règles if/else lisibles).
    - Capte naturellement des interactions et non-linéarités.
    - Version vectorisée bien plus rapide que des boucles Python naïves.

    Attributs
    ---------
    _tree : _Node | None
        Racine de l’arbre entraîné (structure récursive de nœuds).
    typ : str
        'c' pour indiquer une tâche de classification.
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

    # ---------- construction ----------
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