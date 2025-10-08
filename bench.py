#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np
from time import perf_counter
from contextlib import contextmanager
from typing import Dict, Tuple, List, Any


@contextmanager
def timer(name: str, store: Dict[str, float] | None = None):
	t0 = perf_counter()
	try:
		yield
	finally:
		dt = perf_counter() - t0
		if store is not None:
			store[name] = dt

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
