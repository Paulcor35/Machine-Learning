import pandas as pd
import numpy as np
from collections import Counter
import time

# ------------------------- Charger les données -------------------------
df = pd.read_csv("Data-20251001/ozone_complet.txt", sep=";", quotechar='"')

# Vérification rapide
print(df.head())
print(df.info())

# Colonne cible
target_col = "maxO3"

# Remplacer les valeurs manquantes par la moyenne
df = df.fillna(df.mean())

# Séparer features et target
X = df.drop(target_col, axis=1)
y = df[target_col]



# ------------------------- Séparer train/test -------------------------
train_ratio = 0.7
n = len(df)
np.random.seed(42)
shuffled_indices = np.random.permutation(n)
train_size = int(train_ratio * n)

train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:]

X_train = X.iloc[train_indices].reset_index(drop=True)
X_test = X.iloc[test_indices].reset_index(drop=True)
y_train = y.iloc[train_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)

# Reconstituer DataFrames complets
df_train = X_train.copy()
df_train[target_col] = y_train
df_test = X_test.copy()
df_test[target_col] = y_test



# ------------------------- Arbre de régression -------------------------

# Variance pour mesure de qualité
def variance(df, target_attr):
    return np.var(df[target_attr])

# Choisir le meilleur attribut (réduction de variance)
def importance_regression(attributes, df, target_attr):
    base_var = variance(df, target_attr)
    best_gain = -1
    best_attr = None

    for attr in attributes:
        values = df[attr].unique()
        new_var = 0
        for v in values:
            subset = df[df[attr] == v]
            new_var += (len(subset) / len(df)) * variance(subset, target_attr)
        gain = base_var - new_var
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    return best_attr

# Moyenne pour feuille
def mean_val(df, target_attr):
    return df[target_attr].mean()

# Construction de l'arbre
def dt_learning_regression(df, attributes, parent_df, target_attr):
    if df.empty:
        return mean_val(parent_df, target_attr)
    if len(df[target_attr].unique()) == 1:
        return df[target_attr].iloc[0]
    if not attributes:
        return mean_val(df, target_attr)

    A = importance_regression(attributes, df, target_attr)
    tree = {A: {}}
    for v in df[A].unique():
        exs = df[df[A] == v]
        subtree = dt_learning_regression(
            exs,
            [attr for attr in attributes if attr != A],
            df,
            target_attr
        )
        tree[A][v] = subtree
    return tree

# Fonction de prédiction
def predict_regression(tree, example):
    if not isinstance(tree, dict):
        return tree

    attr = next(iter(tree))
    attr_value = example[attr]

    if attr_value not in tree[attr]:
        # Moyenne des valeurs du sous-arbre si valeur inconnue
        values = []
        for subtree in tree[attr].values():
            if isinstance(subtree, dict):
                stack = [subtree]
                while stack:
                    node = stack.pop()
                    if isinstance(node, dict):
                        for v in node.values():
                            stack.append(v)
                    else:
                        values.append(node)
            else:
                values.append(subtree)
        return np.mean(values)

    return predict_regression(tree[attr][attr_value], example)




# ------------------------- Apprentissage et prédiction -------------------------
attributes = [col for col in df_train.columns if col != target_col]

# Temps d'apprentissage
start_train = time.time()
tree = dt_learning_regression(df_train, attributes, pd.DataFrame(), target_col)
end_train = time.time()
print(f"Temps d'apprentissage : {end_train - start_train:.4f} s")

# Temps de prédiction
start_pred = time.time()
y_pred = [predict_regression(tree, row) for _, row in df_test.iterrows()]
end_pred = time.time()
print(f"Temps de prédiction : {end_pred - start_pred:.4f} s")

# ------------------------- Évaluation -------------------------
y_true = df_test[target_col].values
y_pred = np.array(y_pred)

mse = np.mean((y_true - y_pred)**2)
ss_tot = np.sum((y_true - np.mean(y_true))**2)
ss_res = np.sum((y_true - y_pred)**2)
r2 = 1 - (ss_res / ss_tot)

print(f"MSE: {mse:.4f}, R²: {r2:.4f}")






# ------------------------- Avec sklearn -------------------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# 🔹 Créer le modèle
regressor = DecisionTreeRegressor(criterion="squared_error", random_state=42)

# 🔹 Mesurer le temps d'entraînement
start_train = time.time()
regressor.fit(X_train, y_train)
end_train = time.time()
print(f"Temps d'apprentissage : {end_train - start_train:.4f} s")

# 🔹 Mesurer le temps de prédiction
start_pred = time.time()
y_pred = regressor.predict(X_test)
end_pred = time.time()
print(f"Temps de prédiction : {end_pred - start_pred:.4f} s")

# 🔹 Évaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
