# main.py
import numpy as np
import pandas as pd
from svc_scratch import SVC, benchmark_model, plot_roc, plot_pr, plot_linear_importances
from sklearn.svm import LinearSVC, SVC as SKSVC

# --------- préparation données ----------
df = pd.read_csv("data/Carseats.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df[["High","Urban","US"]] = df[["High","Urban","US"]].replace({"Yes":1,"No":0})
df = pd.get_dummies(df, columns=["ShelveLoc"], drop_first=True)

y = df["High"].values
X = (df.drop(columns=["High"])
       .select_dtypes(include=[np.number])
       .to_numpy(dtype=float, copy=True))
feature_names = (df.drop(columns=["High"]).select_dtypes(include=[np.number]).columns.tolist())

# split + standardize
rng = np.random.default_rng(42)
idx = rng.permutation(len(y))
n_test = int(0.2 * len(y))
test_idx, train_idx = idx[:n_test], idx[n_test:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

mu = X_train.mean(axis=0); sigma = X_train.std(axis=0); sigma[sigma==0] = 1.0
X_train = (X_train - mu)/sigma
X_test  = (X_test  - mu)/sigma

# --------- modèles ----------
scratch = SVC(learning_rate=1e-4, C=9.0, n_iters=100, shuffle=True, random_state=0)
lsvc    = LinearSVC(C=9.0, loss="hinge", fit_intercept=True, max_iter=10000, random_state=0)

res_scratch = benchmark_model(scratch, X_train, y_train, X_test, y_test)
res_lsvc    = benchmark_model(lsvc,    X_train, y_train, X_test, y_test)

# --------- affichage résumé ----------
for name, res in [("Scratch", res_scratch), ("LinearSVC", res_lsvc)]:
    base = res["base_metrics"]; best = res["best_metrics"]; times = res["times"]
    print(f"{name:>10} | base F1={base['f1']:.3f} | best F1={best['f1']:.3f} @thr={best['thr']:.3f} | fit={times['fit']*1000:.1f}ms pred={times['predict(scores)']*1000:.2f}ms")

# --------- visualisations ----------
plot_roc([res_scratch, res_lsvc], ["Scratch", "LinearSVC"])
plot_pr([res_scratch, res_lsvc], ["Scratch", "LinearSVC"])

# Importances (modèles linéaires uniquement)
plot_linear_importances(scratch.w, feature_names, top_k=10, title="Scratch - importances")