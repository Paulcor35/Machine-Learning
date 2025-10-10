#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np
import matplotlib.pyplot as plt

def _safe_get(history: dict, key: str):
    """Return history[key] if present and non-empty, else None."""
    v = history.get(key, None)
    return v if (v is not None and len(v) > 0) else None

def plot_linear_importances(w: np.ndarray, feature_names: list[str], top_k: int = 15, title: str = "Importances (linéaire) - |w|"):
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

# --- Classification ---
def print_classification_report(models_results: list[dict], labels: list[str]) -> None:
    """
    Affiche Acc/Prec/Rec/F1 au seuil 0.0 (base) et au meilleur seuil (best),
    + temps d'entraînement et d'inférence (scores).
    """
    print("\n=== Classification report ===")
    for lab, res in zip(labels, models_results):
        base = res["base_metrics"]   # {'acc','prec','rec','f1',...}
        best = res["best_metrics"]   # idem + 'thr'
        times = res["times"]         # {'fit', 'predict(scores)'}
        print(
            f"{lab:>20} | "
            f"base: Acc={base['acc']:.3f} Prec={base['prec']:.3f} Rec={base['rec']:.3f} F1={base['f1']:.3f} | "
            f"best@thr={best['thr']:.3f}: Acc={best['acc']:.3f} Prec={best['prec']:.3f} Rec={best['rec']:.3f} F1={best['f1']:.3f} | "
            f"fit={times['fit']*1000:.1f} ms | pred={times['predict(scores)']*1000:.1f} ms")
            
def plot_roc(models_results: list[dict[str, any]], labels: list[str], title: str = "ROC - comparaison"):
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

def plot_pr(models_results: list[dict[str, any]], labels: list[str], title: str = "Precision-Recall - comparaison"):
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
      
def plot_classification_learning_curves(history: dict, title: str = "Classification – accuracy & loss"):
    """
    history: dict with any of the following (lists, length = #epochs or steps)
      - 'train_loss', 'val_loss'
      - 'train_acc',  'val_acc'
      Optionally: 'train_f1', 'val_f1'  (we'll show them if present)
    """
    train_loss = _safe_get(history, "train_loss")
    val_loss   = _safe_get(history, "val_loss")
    train_acc  = _safe_get(history, "train_acc")
    val_acc    = _safe_get(history, "val_acc")
    train_f1   = _safe_get(history, "train_f1")
    val_f1     = _safe_get(history, "val_f1")

    epochs = None
    for v in [train_loss, val_loss, train_acc, val_acc, train_f1, val_f1]:
        if v is not None:
            epochs = np.arange(1, len(v) + 1); break
    if epochs is None:
        raise ValueError("history is empty: provide at least one of train_loss/val_loss/train_acc/val_acc.")

    # Loss
    plt.figure()
    if train_loss is not None: plt.plot(epochs, train_loss, label="train loss", linewidth=2)
    if val_loss   is not None: plt.plot(epochs, val_loss,   label="val loss",   linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} – loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Accuracy (+ F1 if provided)
    if (train_acc is not None) or (val_acc is not None) or (train_f1 is not None) or (val_f1 is not None):
        plt.figure()
        if train_acc is not None: plt.plot(epochs, train_acc, label="train acc", linewidth=2)
        if val_acc   is not None: plt.plot(epochs, val_acc,   label="val acc",   linewidth=2)
        if train_f1  is not None: plt.plot(epochs, train_f1,  label="train F1",  linewidth=2, linestyle="--")
        if val_f1    is not None: plt.plot(epochs, val_f1,    label="val F1",    linewidth=2, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy / F1")
        plt.title(f"{title} – accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()

# --- Régression ---
def print_regression_report(models_results: list[dict], labels: list[str]) -> None:
	"""Affiche MSE / MAE / R² + temps (fit/predict) pour chaque modèle."""
	print("\n=== Regression report ===")
	for lab, res in zip(labels, models_results):
		m = res["reg_metrics"]; t = res["times"]
		print(f"{lab:>20} | MSE={m['mse']:.4f} | MAE={m['mae']:.4f} | R²={m['r2']:.4f} | "
		      f"fit={t['fit']*1000:.1f} ms | pred={t['predict']*1000:.1f} ms")

def plot_regression_parity(y_true: np.ndarray, models_results: list[dict], labels: list[str],
                           title: str = "Predicted vs Actual"):
	"""Nuage Pred vs Actual (parity plot). Idéal pour voir l'alignement avec y=x."""
	plt.figure()
	ymin, ymax = np.min(y_true), np.max(y_true)
	for res, lab in zip(models_results, labels):
		plt.scatter(y_true, res["y_pred"], alpha=0.6, label=lab)
	plt.plot([ymin, ymax], [ymin, ymax], linestyle="--", linewidth=1)
	plt.xlabel("Valeurs réelles")
	plt.ylabel("Prédictions")
	plt.title(title)
	plt.legend()
	plt.tight_layout()
	plt.show()

def plot_regression_learning_curves(history: dict, title: str = "Regression – loss & score"):
    """
    history: dict with any of the following (lists, length = #epochs or steps)
      - 'train_mse', 'val_mse'         (treated as loss)
      - 'train_mae', 'val_mae'         (optional, drawn on a second figure)
      - 'train_r2',  'val_r2'          (accuracy-like score ↑)
    """
    train_mse = _safe_get(history, "train_mse")
    val_mse   = _safe_get(history, "val_mse")
    train_mae = _safe_get(history, "train_mae")
    val_mae   = _safe_get(history, "val_mae")
    train_r2  = _safe_get(history, "train_r2")
    val_r2    = _safe_get(history, "val_r2")

    epochs = None
    for v in [train_mse, val_mse, train_mae, val_mae, train_r2, val_r2]:
        if v is not None:
            epochs = np.arange(1, len(v) + 1); break
    if epochs is None:
        raise ValueError("history is empty: provide at least one of train_mse/val_mse/train_r2/val_r2.")

    # Loss (MSE)
    if (train_mse is not None) or (val_mse is not None):
        plt.figure()
        if train_mse is not None: plt.plot(epochs, train_mse, label="train MSE", linewidth=2)
        if val_mse   is not None: plt.plot(epochs, val_mse,   label="val MSE",   linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("MSE (loss ↓)")
        plt.title(f"{title} – MSE")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # MAE
    if (train_mae is not None) or (val_mae is not None):
        plt.figure()
        if train_mae is not None: plt.plot(epochs, train_mae, label="train MAE", linewidth=2)
        if val_mae   is not None: plt.plot(epochs, val_mae,   label="val MAE",   linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("MAE (↓)")
        plt.title(f"{title} – MAE")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Score (R²)
    if (train_r2 is not None) or (val_r2 is not None):
        plt.figure()
        if train_r2 is not None: plt.plot(epochs, train_r2, label="train R²", linewidth=2)
        if val_r2   is not None: plt.plot(epochs, val_r2,   label="val R²",   linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("R² (↑)")
        plt.title(f"{title} – R²")
        plt.legend()
        plt.tight_layout()
        plt.show()