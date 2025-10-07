import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------- Chargement et préparation des données -------------------------
df = pd.read_csv("Data-20251001/ozone_prepared.csv")

# Colonne cible
target_col = "maxO3"

# Vérifier qu’elle existe
assert target_col in df.columns, f"La colonne cible '{target_col}' est absente du fichier !"

# Séparer features et target (exclure l'id si présent)
X = df.drop(columns=[target_col, "id"], errors='ignore')
y = df[target_col]

# ------------------------- Séparer train/test -------------------------
train_ratio = 0.8
n = len(df)
#np.random.seed(42)
shuffled_indices = np.random.permutation(n)
train_size = int(train_ratio * n)

train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:]

X_train = X.iloc[train_indices].reset_index(drop=True)
X_test = X.iloc[test_indices].reset_index(drop=True)
y_train = y.iloc[train_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)

# Vérification que les colonnes du train et du test sont identiques
assert list(X_train.columns) == list(X_test.columns), "Les colonnes de X_train et X_test ne correspondent pas !"

# ------------------------- Arbre de régression from scratch -------------------------
def variance(df, target_attr):
    return np.var(df[target_attr])

def importance_regression(attributes, df, target_attr):
    base_var = variance(df, target_attr)
    best_gain = -1
    best_attr = None
    best_threshold = None

    for attr in attributes:
        values = np.sort(df[attr].unique())
        for i in range(1, len(values)):
            threshold = (values[i - 1] + values[i]) / 2
            left = df[df[attr] <= threshold]
            right = df[df[attr] > threshold]
            if len(left) == 0 or len(right) == 0:
                continue
            new_var = (len(left)/len(df))*variance(left, target_attr) + (len(right)/len(df))*variance(right, target_attr)
            gain = base_var - new_var
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
                best_threshold = threshold
    return best_attr, best_threshold

def mean_val(df, target_attr):
    return df[target_attr].mean()

def dt_learning_regression(df, attributes, parent_df, target_attr, depth=0, max_depth=None):
    if df.empty:
        return mean_val(parent_df, target_attr)
    if len(df[target_attr].unique()) == 1:
        return df[target_attr].iloc[0]
    if not attributes or (max_depth is not None and depth >= max_depth):
        return mean_val(df, target_attr)

    A, threshold = importance_regression(attributes, df, target_attr)
    if A is None:
        return mean_val(df, target_attr)

    tree = {A: {}}
    left = df[df[A] <= threshold]
    right = df[df[A] > threshold]

    tree[A][f"<= {threshold:.3f}"] = dt_learning_regression(left, attributes, df, target_attr, depth+1, max_depth)
    tree[A][f"> {threshold:.3f}"] = dt_learning_regression(right, attributes, df, target_attr, depth+1, max_depth)

    return tree

def predict_regression(tree, example):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    for condition, subtree in tree[attr].items():
        op, threshold = condition.split(" ")
        threshold = float(threshold)
        value = example[attr]
        if op == "<=" and value <= threshold:
            return predict_regression(subtree, example)
        elif op == ">" and value > threshold:
            return predict_regression(subtree, example)
    return np.nan

# ------------------------- Apprentissage et prédiction -------------------------
print("Évaluation model from scratch:\n")
attributes = [col for col in X_train.columns]

start_train = time.time()
tree = dt_learning_regression(pd.concat([X_train, y_train], axis=1), attributes, pd.DataFrame(), target_col, max_depth=5)
end_train = time.time()
print(f"Temps d'apprentissage : {end_train - start_train:.4f} s")

start_pred = time.time()
y_pred_scratch = [predict_regression(tree, row) for _, row in X_test.iterrows()]
end_pred = time.time()
print(f"Temps de prédiction : {end_pred - start_pred:.4f} s\n")

mse = mean_squared_error(y_test, y_pred_scratch)
r2 = r2_score(y_test, y_pred_scratch)
print(f"MSE: {mse:.4f}\nR²: {r2:.4f}\n")

# ------------------------- Avec sklearn -------------------------
print("Évaluation avec sklearn:\n")
regressor = DecisionTreeRegressor(criterion="squared_error", max_depth=5, random_state=42)


start_train = time.time()
regressor.fit(X_train, y_train)
end_train = time.time()
print(f"Temps d'apprentissage : {end_train - start_train:.4f} s")

start_pred = time.time()
y_pred_sklearn = regressor.predict(X_test)
end_pred = time.time()
print(f"Temps de prédiction : {end_pred - start_pred:.4f} s")

mse = mean_squared_error(y_test, y_pred_sklearn)
r2 = r2_score(y_test, y_pred_sklearn)
print(f"\nMSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
