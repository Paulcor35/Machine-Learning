#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

def plot_linear_importances(w: np.ndarray, feature_names: List[str], top_k: int = 15, title: str = "Importances (linear) - |w|") -> None:
	abs_w = np.abs(w)
	k = min(top_k, len(abs_w))
	idx = np.argsort(abs_w)[-k:][::-1]
	vals = abs_w[idx][::-1]
	names = [feature_names[i] for i in idx[::-1]]

	pos = np.arange(k)
	plt.figure(figsize=(8, 6))
	plt.barh(pos, vals)
	plt.yticks(pos, names)
	plt.xlabel("|weight|")
	plt.title(title)
	plt.tight_layout()
	plt.show()

# --- Classification ---
def print_classification_report(models_results: List[Dict], labels: List[str]) -> None:
	"""
	Shows Acc/Prec/Rec/F1 at threshold 0.0 (base) and at the best threshold (best),
	+ time of training and prediction (scores).
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

def plot_roc(models_results: List[Dict[str, Any]], labels: List[str], title: str = "ROC - comparison") -> None:
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

def plot_pr(models_results: List[Dict[str, Any]], labels: List[str], title: str = "Precision-Recall - comparison") -> None:
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

# --- Regression ---
def print_regression_report(models_results: List[Dict], labels: List[str]) -> None:
	"""
	Shows MSE / MAE / R² + time (fit/predict) for each model.
	"""
	print("\n=== Regression report ===")
	for lab, res in zip(labels, models_results):
		m = res["reg_metrics"]; t = res["times"]
		print(f"{lab:>20} | MSE={m['mse']:.4f} | MAE={m['mae']:.4f} | R²={m['r2']:.4f} | "
			  f"fit={t['fit']*1000:.1f} ms | pred={t['predict']*1000:.1f} ms")

def plot_regression_parity(y_true: np.ndarray, models_results: List[Dict], labels: List[str],
						   title: str = "Predicted vs Actual") -> None:
	"""
	Predicted vs Actual cloud (parity plot). Ideal to see the alignment with y=x.
	"""
	plt.figure()
	ymin, ymax = np.min(y_true), np.max(y_true)
	for res, lab in zip(models_results, labels):
		plt.scatter(y_true, res["y_pred"], alpha=0.6, label=lab)
	plt.plot([ymin, ymax], [ymin, ymax], linestyle="--", linewidth=1)
	plt.xlabel("True values")
	plt.ylabel("Predictions")
	plt.title(title)
	plt.legend()
	plt.tight_layout()
	plt.show()
