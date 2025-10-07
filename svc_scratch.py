# svc_scratch.py
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from contextlib import contextmanager
from typing import Dict, Tuple, List, Any


# ---------- Timing ----------
@contextmanager
def timer(name: str, store: Dict[str, float] | None = None):
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        if store is not None:
            store[name] = dt


# ---------- SVC linéaire (hinge + L2) ----------
class SVC:
    """
    SVM linéaire entraînée par SGD sur la perte hinge + L2.

    Paramètres
    ----------
    learning_rate : float
        Pas d'apprentissage pour l'update SGD.
    C : float
        Poids de la partie hinge (équivaut à 1/lambda en formulation primal).
    n_iters : int
        Nombre d'époques.
    shuffle : bool
        Mélange des exemples à chaque époque.
    random_state : int
        Graine pour la permutation.
    """
    def __init__(self, learning_rate: float = 1e-3, C: float = 1.0,
                 n_iters: int = 10, shuffle: bool = True, random_state: int = 0):
        self.lr = learning_rate
        self.C = C
        self.n_iters = n_iters
        self.shuffle = shuffle
        self.random_state = random_state
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    # --- fit / predict ---
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVC":
        y = np.where(y <= 0, -1, 1).astype(float)   # {0,1} -> {-1,+1}
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(d, dtype=float)
        self.b = 0.0

        for _ in range(self.n_iters):
            if self.shuffle:
                idx = rng.permutation(n)
                X_ep, y_ep = X[idx], y[idx]
            else:
                X_ep, y_ep = X, y
            for x_i, y_i in zip(X_ep, y_ep):
                margin = y_i * (np.dot(self.w, x_i) + self.b)
                if margin >= 1:
                    # gradient de la régularisation L2 uniquement
                    self.w -= self.lr * self.w
                else:
                    # hinge active : régul + terme de classification
                    self.w -= self.lr * (self.w - self.C * y_i * x_i)
                    self.b  += self.lr * (self.C * y_i)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Le modèle n'est pas entraîné.")
        return X @ self.w + self.b

    def predict_with_threshold(self, X: np.ndarray, threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        s = self.decision_function(X)
        return (s >= threshold).astype(int), s


# ---------- Évaluation / métriques ----------
def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = float((y_pred == y_true).mean())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def evaluate_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Tuple[Dict[str, Any], np.ndarray]:
    y_pred = (scores >= thr).astype(int)
    m = metrics_from_preds(y_true, y_pred)
    return m, y_pred

def search_best_threshold(y_true: np.ndarray, scores: np.ndarray,
                          percentiles: np.ndarray = np.linspace(1, 99, 99)) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Balaye une grille de seuils (percentiles des scores) et retourne le meilleur F1."""
    thr_list = np.percentile(scores, percentiles)
    best = {"thr": 0.0, "f1": -1.0}
    rec_list, prec_list = [], []
    for thr in thr_list:
        m, _ = evaluate_at_threshold(y_true, scores, float(thr))
        rec_list.append(m["rec"]); prec_list.append(m["prec"])
        if m["f1"] > best["f1"]:
            best = {"thr": float(thr), **m}
    return best, np.array(rec_list), np.array(prec_list), thr_list

def roc_curve_from_scores(y_true: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Courbe ROC sans sklearn."""
    thr = np.sort(np.unique(s))
    thr = np.concatenate(([-np.inf], thr, [np.inf]))
    tpr, fpr = [], []
    P, N = int((y_true == 1).sum()), int((y_true == 0).sum())
    for t in thr:
        y_hat = (s >= t).astype(int)
        tp = int(((y_true == 1) & (y_hat == 1)).sum())
        fp = int(((y_true == 0) & (y_hat == 1)).sum())
        tpr.append(tp / P if P > 0 else 0.0)
        fpr.append(fp / N if N > 0 else 0.0)
    return np.array(fpr), np.array(tpr)

def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    x, y = x[order], y[order]
    return float(np.trapz(y, x))

def pr_auc(rec: np.ndarray, prec: np.ndarray) -> float:
    order = np.argsort(rec)
    return float(np.trapz(prec[order], rec[order]))


# ---------- Benchmark helper ----------
def benchmark_model(model, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    threshold_grid: np.ndarray | None = None) -> Dict[str, Any]:
    """
    Entraîne un modèle (avec .fit et .decision_function), mesure les temps,
    renvoie scores, métriques au seuil 0 et au meilleur seuil F1.
    """
    times: Dict[str, float] = {}
    with timer("fit", store=times):
        model.fit(X_train, y_train)

    with timer("predict(scores)", store=times):
        scores = model.decision_function(X_test)
    base_metrics, _ = evaluate_at_threshold(y_test, scores, 0.0)

    # recherche du meilleur seuil
    if threshold_grid is None:
        best, rec_list, prec_list, thr_list = search_best_threshold(y_test, scores)
    else:
        best = {"thr": 0.0, "f1": -1.0}
        rec_list, prec_list, thr_list = [], [], threshold_grid
        for thr in threshold_grid:
            m, _ = evaluate_at_threshold(y_test, scores, float(thr))
            rec_list.append(m["rec"]); prec_list.append(m["prec"])
            if m["f1"] > best["f1"]:
                best = {"thr": float(thr), **m}

    fpr, tpr = roc_curve_from_scores(y_test, scores)
    roc_auc = auc_trapz(fpr, tpr)
    pr_area = pr_auc(np.array(rec_list), np.array(prec_list))

    return {
        "model": model,
        "scores": scores,
        "times": times,
        "base_metrics": base_metrics,
        "best_metrics": best,
        "roc": (fpr, tpr, roc_auc),
        "pr": (np.array(rec_list), np.array(prec_list), pr_area),
    }


# ---------- Visualisations ----------
def plot_roc(models_results: List[Dict[str, Any]], labels: List[str], title: str = "ROC - comparaison"):
    plt.figure()
    for res, lab in zip(models_results, labels):
        fpr, tpr, auc_val = res["roc"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{lab} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_pr(models_results: List[Dict[str, Any]], labels: List[str], title: str = "Precision-Recall - comparaison"):
    plt.figure()
    for res, lab in zip(models_results, labels):
        rec, prec, pr_area = res["pr"]
        bm = res["best_metrics"]
        plt.plot(rec, prec, linewidth=2, label=f"{lab} (best thr={bm['thr']:.3f}, F1={bm['f1']:.3f}, PR-AUC={pr_area:.3f})")
        # point du meilleur seuil
        plt.scatter([bm["rec"]], [bm["prec"]], s=60)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_linear_importances(w: np.ndarray, feature_names: List[str], top_k: int = 15, title: str = "Importances (linéaire) – |w|"):
    abs_w = np.abs(w)
    k = min(top_k, len(abs_w))
    idx = np.argsort(abs_w)[-k:][::-1]
    vals = abs_w[idx][::-1]
    names = [feature_names[i] for i in idx[::-1]]

    pos = np.arange(k)
    plt.figure(figsize=(8, 6))
    plt.barh(pos, vals)
    plt.yticks(pos, names)
    plt.xlabel("|poids|")
    plt.title(title)
    plt.tight_layout()
    plt.show()