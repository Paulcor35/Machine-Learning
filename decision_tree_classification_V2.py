import pandas as pd
import numpy as np
import time
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------- Chargement et préparation des données -------------------------

df_prepared = pd.read_csv("Data-20251001/Carseats_prepared.csv")
target_col = "High"

# Séparer les features et la target
X = df_prepared.drop(target_col, axis=1).to_numpy()
y = df_prepared[target_col].to_numpy()

n_samples, n_features = X.shape

# Encodage des variables catégorielles en int si nécessaire
for i in range(n_features):
    if X[:, i].dtype.kind not in 'fbi':  # si ce n'est pas float, bool ou int
        values, X[:, i] = np.unique(X[:, i], return_inverse=True)

# Split train/test (80% / 20%)
np.random.seed(42)
indices = np.random.permutation(n_samples)
train_size = int(0.8 * n_samples)

train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# ------------------------- Fonctions utilitaires -------------------------

def entropy(y_subset):
    """
    Calcule l'entropie d'un tableau 1D de labels.
    Utile pour mesurer la "pureté" d'un sous-ensemble dans un arbre de décision.
    """
    if len(y_subset) == 0:
        return 0.0
    counts = np.bincount(y_subset) if np.issubdtype(y_subset.dtype, np.integer) else \
             np.array([np.sum(y_subset == val) for val in np.unique(y_subset)])
    probs = counts / counts.sum()
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

def plurality_val(y_subset):
    """
    Retourne la classe majoritaire dans un sous-ensemble de labels.
    Sert pour prédire une feuille ou gérer une valeur inconnue.
    """
    if len(y_subset) == 0:
        return 0
    counts = Counter(y_subset)
    return counts.most_common(1)[0][0]

def best_split(X_subset, y_subset, feature_indices, max_thresholds=50):
    """
    Trouve le meilleur attribut et seuil pour maximiser le gain d'information.
    - X_subset : sous-échantillon des features
    - y_subset : labels correspondants
    - feature_indices : colonnes à considérer
    - max_thresholds : nombre maximal de seuils à tester pour les attributs numériques
    Retour : (best_attr, best_thresh)
    """
    base_entropy = entropy(y_subset)
    best_gain = -1
    best_attr = None
    best_thresh = None

    for attr in feature_indices:
        col = X_subset[:, attr]
        unique_vals = np.unique(col)

        # Attribut numérique
        if len(unique_vals) > 2 and col.dtype.kind in 'fbi':
            if len(unique_vals) > max_thresholds:
                thresholds = np.percentile(unique_vals, np.linspace(0, 100, max_thresholds))
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
            for thresh in thresholds:
                mask_left = col <= thresh
                mask_right = ~mask_left
                if np.sum(mask_left) == 0 or np.sum(mask_right) == 0:
                    continue
                new_entropy = (mask_left.sum()/len(y_subset)) * entropy(y_subset[mask_left]) + \
                              (mask_right.sum()/len(y_subset)) * entropy(y_subset[mask_right])
                gain = base_entropy - new_entropy
                if gain > best_gain:
                    best_gain = gain
                    best_attr = attr
                    best_thresh = thresh
        else:  # Attribut binaire ou catégoriel
            new_entropy = 0
            for val in unique_vals:
                mask = col == val
                new_entropy += (mask.sum()/len(y_subset)) * entropy(y_subset[mask])
            gain = base_entropy - new_entropy
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
                best_thresh = None
    return best_attr, best_thresh

# ------------------------- Construction récursive de l'arbre -------------------------

def dt_learning(X_subset, y_subset, feature_indices=None, depth=0, max_depth=10, min_samples_split=2):
    """
    Construction récursive d'un arbre de décision.
    Retourne :
        - (attr, thresh, left_subtree, right_subtree) pour attribut numérique
        - (attr, {val: subtree}) pour attribut catégoriel
        - classe majoritaire pour feuille
    """
    if feature_indices is None:
        feature_indices = list(range(X_subset.shape[1]))

    # Conditions d'arrêt
    if len(y_subset) < min_samples_split or len(np.unique(y_subset)) == 1 \
       or depth >= max_depth or len(feature_indices) == 0:
        return plurality_val(y_subset)

    attr, thresh = best_split(X_subset, y_subset, feature_indices)
    if attr is None:
        return plurality_val(y_subset)

    col = X_subset[:, attr]

    if thresh is not None:
        # Attribut numérique : scinder gauche / droite
        mask_left = col <= thresh
        mask_right = ~mask_left
        left_subtree = dt_learning(X_subset[mask_left], y_subset[mask_left], feature_indices,
                                   depth+1, max_depth, min_samples_split)
        right_subtree = dt_learning(X_subset[mask_right], y_subset[mask_right], feature_indices,
                                    depth+1, max_depth, min_samples_split)
        return (attr, thresh, left_subtree, right_subtree)
    else:
        # Attribut catégoriel : créer un dictionnaire de sous-arbres
        tree = {}
        for val in np.unique(col):
            mask = col == val
            tree[val] = dt_learning(X_subset[mask], y_subset[mask],
                                    [f for f in feature_indices if f != attr],
                                    depth+1, max_depth, min_samples_split)
        return (attr, tree)

# ------------------------- Prédiction -------------------------

def predict_single(tree, x):
    """
    Prédit la classe pour un exemple unique x.
    - tree : arbre construit par dt_learning
    - x : vecteur 1D d'une observation
    """
    if not isinstance(tree, tuple) and not isinstance(tree, dict):
        return tree

    # Noeud numérique
    if isinstance(tree, tuple) and len(tree) == 4:
        attr, thresh, left_subtree, right_subtree = tree
        if x[attr] <= thresh:
            return predict_single(left_subtree, x)
        else:
            return predict_single(right_subtree, x)

    # Noeud catégoriel
    if isinstance(tree, tuple) and len(tree) == 2:
        attr, subtrees = tree
        val = x[attr]
        if val not in subtrees:
            # Retourne la classe majoritaire si valeur inconnue
            values = []
            stack = [subtrees]
            while stack:
                node = stack.pop()
                for v in node.values():
                    if isinstance(v, dict):
                        stack.append(v)
                    else:
                        values.append(v)
            return plurality_val(np.array(values))
        return predict_single(subtrees[val], x)

    return tree

def predict(tree, X_array):
    """
    Prédit les classes pour toutes les observations X_array.
    - tree : arbre construit par dt_learning
    - X_array : tableau 2D (n_samples, n_features)
    """
    return np.array([predict_single(tree, X_array[i]) for i in range(len(X_array))])

# ------------------------- Entraînement -------------------------

start_train = time.time()
tree = dt_learning(X_train, y_train, max_depth=6, min_samples_split=5)
end_train = time.time()
print(f"Temps d'apprentissage optimisé (from scratch) : {end_train - start_train:.4f} s")

# ------------------------- Prédiction -------------------------

start_pred = time.time()
y_pred = predict(tree, X_test)
end_pred = time.time()
print(f"Temps de prédiction optimisé (from scratch) : {end_pred - start_pred:.4f} s")

accuracy = np.mean(y_pred == y_test)
print(f"Accuracy (from scratch) : {accuracy*100:.2f}%")

# Matrice de confusion
conf_matrix = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
print("Matrice de confusion :\n", conf_matrix)

# ------------------------- Comparaison avec scikit-learn -------------------------

clf = DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=42)

start_train = time.time()
clf.fit(X_train, y_train)
end_train = time.time()
print(f"\nTemps apprentissage (sklearn) : {end_train - start_train:.4f} s")

start_pred = time.time()
y_pred_test = clf.predict(X_test)
end_pred = time.time()
print(f"Temps prédiction (sklearn) : {end_pred - start_pred:.4f} s")

accuracy = accuracy_score(y_test, y_pred_test)
print(f"Accuracy (sklearn) : {accuracy*100:.2f}%")
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred_test))
print("Rapport de classification :\n", classification_report(y_test, y_pred_test))
