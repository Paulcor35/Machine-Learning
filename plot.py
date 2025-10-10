#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np
import matplotlib.pyplot as plt

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
	"""Nuage Pred vs Actual (parity plot). Idéal pour voir l’alignement avec y=x."""
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