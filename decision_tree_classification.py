import pandas as pd
import numpy as np
import math
from collections import Counter
import time


# =====================================================================================================================
# Charger les données préparées
# =====================================================================================================================
df_prepared = pd.read_csv("Data-20251001/Carseats_prepared.csv")

# Séparer train/test
train_ratio = 0.7
n = len(df_prepared)
shuffled_indices = np.random.permutation(n)
train_size = int(train_ratio * n)

train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:]

X_train = df_prepared.drop('High', axis=1).iloc[train_indices].reset_index(drop=True)
X_test = df_prepared.drop('High', axis=1).iloc[test_indices].reset_index(drop=True)
y_train = df_prepared['High'].iloc[train_indices].reset_index(drop=True)
y_test = df_prepared['High'].iloc[test_indices].reset_index(drop=True)



# =====================================================================================================================
# Arbre de décision from scratch
# =====================================================================================================================

# --- Fonctions utilitaires ---
def entropy(df, target_attr):
    values = df[target_attr]
    counter = Counter(values)
    total = len(df)
    return -sum((count / total) * math.log2(count / total) for count in counter.values())

def importance(attributes, df, target_attr):
    base_entropy = entropy(df, target_attr)
    best_gain = -1
    best_attr = None
    
    for attr in attributes:
        values = df[attr].unique()
        new_entropy = 0
        for v in values:
            subset = df[df[attr] == v]
            new_entropy += (len(subset) / len(df)) * entropy(subset, target_attr)
        gain = base_entropy - new_entropy
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    return best_attr

def plurality_val(df, target_attr):
    values = df[target_attr]
    return Counter(values).most_common(1)[0][0]

def dt_learning(df, attributes, parent_df, target_attr):
    if df.empty:
        return plurality_val(parent_df, target_attr)
    if len(df[target_attr].unique()) == 1:
        return df[target_attr].iloc[0]
    if not attributes:
        return plurality_val(df, target_attr)
    
    A = importance(attributes, df, target_attr)
    tree = {A: {}}
    for v in df[A].unique():
        exs = df[df[A] == v]
        subtree = dt_learning(
            exs,
            [attr for attr in attributes if attr != A],
            df,
            target_attr
        )
        tree[A][v] = subtree
    return tree

# --- Définir les attributs ---
attributes = list(X_train.columns)

# --- Mesurer le temps d'apprentissage ---
start_train = time.time()
tree = dt_learning(pd.concat([X_train, y_train], axis=1), attributes, pd.DataFrame(), "High")
end_train = time.time()
print(f"Temps d'apprentissage (from scratch) : {end_train - start_train:.4f} secondes")

# --- Fonction de prédiction ---
def predict(tree, example):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    attr_value = example[attr]

    if attr_value not in tree[attr]:
        # Descente pour récupérer toutes les feuilles
        values = []
        stack = [tree[attr]]
        while stack:
            node = stack.pop()
            for v in node.values():
                if isinstance(v, dict):
                    stack.append(v)
                else:
                    values.append(v)
        return Counter(values).most_common(1)[0][0]
    
    return predict(tree[attr][attr_value], example)

# --- Mesurer le temps de prédiction ---
start_pred = time.time()
y_pred = [predict(tree, X_test.iloc[i]) for i in range(len(X_test))]
end_pred = time.time()
print(f"Temps de prédiction (from scratch) : {end_pred - start_pred:.4f} secondes")

# --- Évaluation ---
y_pred = pd.Series(y_pred, name="Predicted")
y_true = y_test.reset_index(drop=True)

accuracy = (y_true == y_pred).mean()
print(f"\nAccuracy (test from scratch) : {accuracy*100:.2f}%")
print("Matrice de confusion :\n", pd.crosstab(y_true, y_pred, rownames=["Actual"], colnames=["Predicted"]))





# =====================================================================================================================
# Arbre de décision avec scikit-learn
# =====================================================================================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

clf = DecisionTreeClassifier(criterion="entropy", random_state=42)

# Temps d'apprentissage
start_train = time.time()
clf.fit(X_train, y_train)
end_train = time.time()
print(f"\nTemps d'apprentissage (scikit-learn) : {end_train - start_train:.4f} secondes")

# Temps de prédiction
start_pred = time.time()
y_pred_test = clf.predict(X_test)
end_pred = time.time()
print(f"Temps de prédiction (scikit-learn) : {end_pred - start_pred:.4f} secondes")

# Évaluation
accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nAccuracy (test scikit-learn) : {accuracy*100:.2f}%")
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred_test))
print("Rapport de classification :\n", classification_report(y_test, y_pred_test))
