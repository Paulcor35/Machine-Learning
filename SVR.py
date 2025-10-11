import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.svm import LinearSVR

# ====================================
#   Implémentation SVR linéaire from scratch
# ====================================
class LinearSVR:
    def __init__(self, learning_rate=1e-3, C=1.0, epsilon=0.1, n_iters=100, shuffle=True, random_state=0):
        self.lr = learning_rate
        self.C = C
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.shuffle = shuffle
        self.random_state = random_state
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.n_iters):
            if self.shuffle:
                idx = rng.permutation(n)
                X_epoch, y_epoch = X[idx], y[idx]
            else:
                X_epoch, y_epoch = X, y

            for x_i, y_i in zip(X_epoch, y_epoch):
                y_pred = np.dot(self.w, x_i) + self.b
                err = y_pred - y_i

                if err > self.epsilon:
                    grad_w = self.w + self.C * x_i
                    grad_b = self.C
                elif err < -self.epsilon:
                    grad_w = self.w - self.C * x_i
                    grad_b = -self.C
                else:
                    grad_w = self.w
                    grad_b = 0.0

                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b
        return self

    def predict(self, X):
        return np.dot(X, self.w) + self.b


# ====================================
#   Fonctions utilitaires
# ====================================
def train_test_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standardize(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X_train - mu)/sigma, (X_test - mu)/sigma, mu, sigma

def regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    return {"MSE": mse, "MAE": mae, "R2": r2}


# ====================================
#   1. Chargement et préparation des données
# ====================================
df = pd.read_csv("Data-20251001/ozone_complet.txt", sep=";")
df = df.fillna(df.mean())
df = df.drop(columns=["id"], errors="ignore")
df = df.drop(columns=["maxO3v"], errors="ignore")

y = df["maxO3"].values
X = df.drop(columns=["maxO3"]).select_dtypes(include=[np.number]).to_numpy(dtype=float)
feature_names = df.drop(columns=["maxO3"]).select_dtypes(include=[np.number]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test, mu, sigma = standardize(X_train, X_test)

y_mean = y_train.mean()
y_std = y_train.std()
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std


# ====================================
#   2. Grid Search avec mesure du temps
# ====================================
grid_C = [0.1, 1, 5, 10]
grid_eps = [0.01, 0.1, 0.5]
results = []
best_R2 = -np.inf
best = None

print("\nRecherche des meilleurs hyperparamètres...\n")

for C in grid_C:
    for eps in grid_eps:
        start = time.time()

        svr = LinearSVR(learning_rate=1e-3, C=C, epsilon=eps, n_iters=2000, random_state=0)
        svr.fit(X_train, y_train_scaled)
        y_pred_scaled = svr.predict(X_test)
        y_pred = y_pred_scaled * y_std + y_mean

        elapsed = time.time() - start
        metrics = regression_metrics(y_test, y_pred)
        metrics["time"] = elapsed
        results.append([C, eps, metrics["MSE"], metrics["MAE"], metrics["R2"], elapsed])

        if metrics["R2"] > best_R2:
            best_R2 = metrics["R2"]
            best = {"C": C, "epsilon": eps, "metrics": metrics, "model": svr}

# ====================================
#   3. Affichage du tableau complet
# ====================================
headers = ["C", "epsilon", "MSE", "MAE", "R²", "Temps (s)"]

# ====================================
#   4. Meilleur modèle
# ====================================
print("\n===== Meilleurs hyperparamètres trouvés =====")
print(f"C = {best['C']}")
print(f"epsilon = {best['epsilon']}")
print("----- Performances correspondantes -----")
for k, v in best["metrics"].items():
    if k != "time":
        print(f"{k}: {v:.4f}")
print(f"Temps d'entraînement: {best['metrics']['time']:.3f} s")

# ====================================
#   5. Visualisation finale
# ====================================
y_pred_scaled = best["model"].predict(X_test)
y_pred = y_pred_scaled * y_std + y_mean

plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Valeurs réelles (maxO3)")
plt.ylabel("Prédictions SVR")
plt.title(f"SVR from scratch - Meilleur modèle (C={best['C']}, ε={best['epsilon']})")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--')
plt.show()

# --------------------------------------------------------------------------
# Comparaison avec scikit-learn LinearSVR
# --------------------------------------------------------------------------
start_time = time.time()
sk_svr = LinearSVR(C=best['C'], epsilon=best['epsilon'], n_iters=2000, random_state=0)
sk_svr.fit(X_train, y_train_scaled)
elapsed_time = time.time() - start_time

y_pred_scaled_sk = sk_svr.predict(X_test)
y_pred_sk = y_pred_scaled_sk * y_std + y_mean

# Utilisation de regression_metrics à la place des fonctions sklearn
metrics_sk = regression_metrics(y_test, y_pred_sk)
metrics_sk["time"] = elapsed_time

print("\n===== Comparaison avec scikit-learn LinearSVR =====")
print(f"MSE : {metrics_sk['MSE']:.4f}")
print(f"MAE : {metrics_sk['MAE']:.4f}")
print(f"R²  : {metrics_sk['R2']:.4f}")
print(f"Temps d'entraînement : {metrics_sk['time']:.3f} s")

# --------------------------------------------------------------------------
# Visualisation des deux modèles
# --------------------------------------------------------------------------
plt.scatter(y_test, best["model"].predict(X_test) * y_std + y_mean, alpha=0.6, label='SVR maison')
plt.scatter(y_test, y_pred_sk, alpha=0.6, label='scikit-learn LinearSVR')
plt.xlabel("Valeurs réelles (maxO3)")
plt.ylabel("Prédictions")
plt.title(f"Comparaison SVR maison vs scikit-learn (C={best['C']}, ε={best['epsilon']})")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--')
plt.legend()
plt.show()