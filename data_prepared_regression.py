import pandas as pd
import numpy as np

# ------------------------- Charger les données -------------------------
df = pd.read_csv("Data-20251001/ozone_complet.txt", sep=";", quotechar='"')

# Vérification rapide
print(df.head())

# Colonne cible (valeur à prédire)
target_col = "maxO3"

# ------------------------- Gestion des valeurs manquantes -------------------------
# Remplacer les valeurs manquantes par la moyenne de chaque colonne
df = df.fillna(df.mean(numeric_only=True))


# ------------------------- StandardScaler (from scratch) -------------------------
def standard_scaler_fit(X):
    """Calcule la moyenne et l'écart-type de chaque colonne"""
    mean = X.mean()
    std = X.std(ddof=0)
    return mean, std

def standard_scaler_transform(X, mean, std):
    """Applique la standardisation : (X - mean) / std"""
    return (X - mean) / std

# ------------------------- Préparation des données -------------------------
# On enlève la colonne cible et l'indice maxO3 (car ce n’est pas une mesure brute)
X = df.drop([target_col, "id"], axis=1, errors='ignore')
y = df[target_col]

# Calcul des statistiques (moyenne, écart-type) sur X
mean_X, std_X = standard_scaler_fit(X)

# Transformation (standardisation)
X_scaled = standard_scaler_transform(X, mean_X, std_X)

# Reconstituer le DataFrame complet standardisé
df_prepared = X_scaled.copy()
df_prepared[target_col] = y.values

# Sauvegarder dans un nouveau fichier CSV
df_prepared.to_csv("Data-20251001/ozone_prepared.csv", index=False)

print("✅ Données nettoyées, standardisées et sauvegardées dans Data-20251001/ozone_prepared.csv")
