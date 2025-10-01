import pandas as pd
import numpy as np

# Charger le CSV
df = pd.read_csv("Data-20251001/Carseats.csv", index_col=0)

# Transformer les colonnes qualitatives en numériques
for col in df.columns:
    if df[col].dtype == 'object':
        df[col], _ = pd.factorize(df[col])

# Séparer features et target
X = df.drop('High', axis=1)
y = df['High']

# Paramètres
train_ratio = 0.8
n = len(df)

# Mélanger les indices
np.random.seed(42)  # pour reproductibilité
shuffled_indices = np.random.permutation(n)

# Indices train/test
train_size = int(train_ratio * n)
train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:]

# Créer les sets
X_train = X.iloc[train_indices].reset_index(drop=True)
X_test = X.iloc[test_indices].reset_index(drop=True)
y_train = y.iloc[train_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)

# Vérification
print("Train size:", len(X_train))
print("Test size:", len(X_test))


# StandardScaler from scratch
def standard_scaler_fit(X):
    """Calcule la moyenne et l'écart-type pour chaque colonne"""
    mean = X.mean()
    std = X.std(ddof=0)  # ddof=0 pour population standard deviation
    return mean, std

def standard_scaler_transform(X, mean, std):
    """Applique la normalisation"""
    return (X - mean) / std

# Fit sur le training
mean_train, std_train = standard_scaler_fit(X_train)

# Transformer train et test avec les stats du train
X_train_scaled = standard_scaler_transform(X_train, mean_train, std_train)
X_test_scaled = standard_scaler_transform(X_test, mean_train, std_train)

# Vérification
print(X_train_scaled.head())
print(X_test_scaled.head())