#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np
import matplotlib.pyplot as plt


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
