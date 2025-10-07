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

# StandardScaler from scratch
def standard_scaler_fit(X):
    mean = X.mean()
    std = X.std(ddof=0)
    return mean, std

def standard_scaler_transform(X, mean, std):
    return (X - mean) / std

# Fit sur toutes les données (pour préparer un dataset unique)
mean_X, std_X = standard_scaler_fit(X)
X_scaled = standard_scaler_transform(X, mean_X, std_X)

# Reconstituer le DataFrame complet (sans binning)
df_prepared = X_scaled.copy()
df_prepared["High"] = y.values

# Sauvegarder dans un fichier CSV
df_prepared.to_csv("Data-20251001/Carseats_prepared.csv", index=False)

print("Data prepared saved ✅")
