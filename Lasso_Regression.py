import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

#------------------------------------------------------------------------------
#   Implémentation Lasso Regression
#------------------------------------------------------------------------------
class LassoRegression:
    def __init__(self, learning_rate, iterations, l1_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = np.zeros(self.n)

        for j in range(self.n):
            grad = -2 * (self.X[:, j]).dot(self.Y - Y_pred) / self.m
            # Régularisation L1 : soft thresholding
            if grad > self.l1_penalty:
                dW[j] = grad - self.l1_penalty
            elif grad < -self.l1_penalty:
                dW[j] = grad + self.l1_penalty
            else:
                dW[j] = 0

        db = -2 * np.sum(self.Y - Y_pred) / self.m
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db
        return self
    
    def cost(self):
        Y_pred = self.predict(self.X)
        mse = np.mean((self.Y - Y_pred)**2)
        l1 = self.l1_penalty * np.sum(np.abs(self.W))
        return mse + l1

    def predict(self, X):
        return X.dot(self.W) + self.b


#------------------------------------------------------------------------------
#   Fonctions utilitaires
#------------------------------------------------------------------------------
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split = int(len(y) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

def regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    return {"MSE": mse, "MAE": mae, "R2": r2}


#------------------------------------------------------------------------------
#   Chargement et préparation des données
#------------------------------------------------------------------------------
df = pd.read_csv("Data-20251001/ozone_complet.txt", sep=";")
df = df.fillna(df.mean())
df = df.drop(columns=["id"], errors="ignore")
df = df.drop(columns=["maxO3v"], errors="ignore")

X = df.drop(columns=["maxO3"])
y = df["maxO3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Normalisation des features
moy = X_train.mean()
std = X_train.std()
X_train = (X_train - moy) / std
X_test = (X_test - moy) / std

# Normalisation de la cible
y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std


#------------------------------------------------------------------------------
#   Recherche des meilleurs hyperparamètres avec mesure du temps
#------------------------------------------------------------------------------
grid_lr = [0.001, 0.01, 0.1]
grid_l1 = [0.01, 0.1, 0.5, 1.0]
results = []

best_r2 = -np.inf
best_model = None
best_params = {}

print("\nRecherche des meilleurs hyperparamètres...\n")

for lr in grid_lr:
    for l1 in grid_l1:
        start = time.time()
        model = LassoRegression(learning_rate=lr, iterations=2000, l1_penalty=l1)
        model.fit(X_train.values, y_train.values)
        elapsed = time.time() - start

        y_pred = model.predict(X_test.values)
        metrics = regression_metrics(y_test.values, y_pred)
        results.append([lr, l1, metrics["MSE"], metrics["MAE"], metrics["R2"], elapsed])

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model = model
            best_params = {"learning_rate": lr, "l1_penalty": l1, "metrics": metrics, "time": elapsed}

#------------------------------------------------------------------------------
#   Affichage des résultats sous forme de tableau
#------------------------------------------------------------------------------
headers = ["Learning Rate", "L1 Penalty", "MSE", "MAE", "R²", "Temps (s)"]

#------------------------------------------------------------------------------
#   Affichage du meilleur modèle
#------------------------------------------------------------------------------
print("\n===== Meilleurs hyperparamètres trouvés =====")
print(f"Learning Rate = {best_params['learning_rate']}")
print(f"L1 Penalty = {best_params['l1_penalty']}")
print("----- Performances correspondantes -----")
print(f"MSE : {best_params['metrics']['MSE']:.4f}")
print(f"MAE : {best_params['metrics']['MAE']:.4f}")
print(f"R²  : {best_params['metrics']['R2']:.4f}")
print(f"Temps d'entraînement : {best_params['time']:.3f} s")

#------------------------------------------------------------------------------
#   Visualisation finale
#------------------------------------------------------------------------------
y_pred = best_model.predict(X_test.values)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Valeurs réelles (maxO3)")
plt.ylabel("Prédictions Lasso")
plt.title(f"Lasso Regression - Meilleur modèle (lr={best_params['learning_rate']}, L1={best_params['l1_penalty']})")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--')
plt.show()

# --------------------------------------------------------------------------
# Comparaison avec scikit-learn
# --------------------------------------------------------------------------
alpha = best_params['l1_penalty']  # L1 penalty
print("alpha: ",alpha)

start_time = time.time()
sk_model = Lasso(alpha=alpha, max_iter=2000)
sk_model.fit(X_train, y_train)
elapsed_time = time.time() - start_time

y_pred_sk = sk_model.predict(X_test)

metrics_sk = regression_metrics(y_test.values, y_pred_sk)

print("\n===== Comparaison avec scikit-learn =====")
print(f"MSE : {metrics_sk['MSE']:.4f}")
print(f"MAE : {metrics_sk['MAE']:.4f}")
print(f"R²  : {metrics_sk['R2']:.4f}")
print(f"Temps d'entraînement : {elapsed_time:.3f} s")

# --------------------------------------------------------------------------
# Visualisation des deux modèles
# --------------------------------------------------------------------------
plt.scatter(y_test, best_model.predict(X_test.values), alpha=0.6, label='Notre Lasso')
plt.scatter(y_test, y_pred_sk, alpha=0.6, label='scikit-learn Lasso')
plt.xlabel("Valeurs réelles (maxO3)")
plt.ylabel("Prédictions")
plt.title("Comparaison Lasso maison vs scikit-learn")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--')
plt.legend()
plt.show()