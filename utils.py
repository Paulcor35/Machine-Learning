#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np
import os
import pandas as pd
import sys
import logging
import yaml
from platform import system
from importlib import import_module

# scikit-learn imports (alias pour éviter conflit avec ta classe Ridge)
from sklearn.linear_model import Ridge as SkRidge, Lasso as SkLasso
from sklearn.svm import LinearSVC as SkSVC, SVR as SkSVR
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier, DecisionTreeRegressor as SkDecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier, RandomForestRegressor as SkRandomForestRegressor

from classes.Lasso import Lasso
from classes.Ridge import Ridge
from classes.DecisionTreeClassifier import DecisionTreeClassifier
from classes.DecisionTreeRegressor import DecisionTreeRegressor
from classes.SVC import SVC
from classes.SVR import SVR
from classes.RandomForestClassifier import RandomForestClassifier
from classes.RandomForestRegressor import RandomForestRegressor

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

def getPath(script_dir, file_dir):
	plat = system()
	if plat == "Windows":
		script_dir = script_dir.replace("/", "\\")
		file_dir = file_dir.replace("/", "\\")
	else:
		script_dir = script_dir.replace("\\", "/")
		file_dir = file_dir.replace("\\", "/")
	return script_dir, file_dir

def split(X, y, test_size=0.2, random_state=42):
	np.random.seed(random_state)
	indices = np.arange(len(y))
	np.random.shuffle(indices)
	split = int(len(y) * (1 - test_size))
	train_idx, test_idx = indices[:split], indices[split:]
	return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def calculate_mse(y_true, y_pred):
	return np.mean((y_true - y_pred) ** 2)

def read_file(fname, sep):
	# FIX: le sep était passé à os.path.join par erreur
	script_dir = os.path.dirname(os.path.abspath(__file__))
	script_dir, file_dir = getPath(script_dir, fname)
	full_path = os.path.join(script_dir, file_dir)
	log.debug(f"Lecture fichier: {full_path} (sep='{sep}')")
	return pd.read_csv(full_path, sep=sep)

def read_regression(fname):
	df = read_file(fname, ";")
	df = df.fillna(df.mean())
	df = df.drop(columns=["id"], errors="ignore")
	#df = df.drop(columns=["maxO3v"], errors="ignore")
	return df

def read_classif(fname):
	df = read_file(fname, ",")
	df = df.drop(columns=["Unnamed: 0"], errors="ignore")
	# encodage binaire simple
	for col in ["High", "Urban", "US"]:
		if col in df.columns:
			df[col] = df[col].replace({"Yes": 1, "No": 0})
	# one-hot ShelveLoc si présent
	if "ShelveLoc" in df.columns:
		df = pd.get_dummies(df, columns=["ShelveLoc"], drop_first=True)
	return df

def read_file_wtype(fname, typ):
	if typ == "r":
		return read_regression(fname)
	else:
		return read_classif(fname)

def read_params():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	script_dir, file_dir = getPath(script_dir, "params.yaml")
	with open(os.path.join(script_dir, file_dir), "r") as fp:
		params = yaml.safe_load(fp)
	return params

def apply_params(model, algo_name, typ, params, is_sci=False):
    """
    Applique les hyper-paramètres lus dans params.yaml au modèle.
    - Pour scikit: utilise model.set_params(**par) + petite traduction de clés si besoin.
    - Pour scratch: setattr (ou set_params si dispo).
    """
    # Sécuriser la lecture du dict
    par = (
        params.get(algo_name, {})
              .get("scikit" if is_sci else "scratch", {})
              .get(typ, {})
        or {}
    )
    if not par:
        return

    # --- PETIT MAPPING POUR SCIKIT SI BESOIN ---
    par = par.copy()

    # SVM (régression) scikit = sklearn.svm.SVR
    if is_sci and algo_name == "SVM" and typ == "r":
        if "n_iters" in par:
            par["max_iter"] = par.pop("n_iters")
        par.pop("random_state", None)  # ignoré par SVR

	# --- AFFICHAGE DES PARAMÈTRES CHARGÉS ---
    print(f"\nHyperparamètres appliqués à {algo_name} "
          f"({'scikit-learn' if is_sci else 'scratch'}) [{typ}] :")
    for k, v in par.items():
        print(f"   {k}: {v}")
    print("-" * 60)

    # --- APPLICATION ---
    try:
        if hasattr(model, "set_params"):
            model.set_params(**par)  # scikit (et certains scratch compatibles)
        else:
            for k, v in par.items():
                setattr(model, k, v)  # scratch
    except Exception as e:
        # Fallback: au moins on pousse ce qu'on peut sur scratch
        for k, v in par.items():
            try:
                setattr(model, k, v)
            except:
                pass

def get_class(class_name: str, typ: str):
	"""
	Retourne la classe *maison* en fonction de l'algorithme ET du type ('r' ou 'c').
	Pas de mapping dict : uniquement des conditions.
	Attend les fichiers/modules suivants dans ton dossier classes/ :
	  - SVC.py -> class SVC(…)			 (classification)
	  - SVR.py -> class SVR(…)			 (régression)
	  - DecisionTreeClassifier.py -> class DecisionTreeClassifier(…)
	  - DecisionTreeRegressor.py  -> class DecisionTreeRegressor(…)
	  - RandomForestClassifier.py -> class RandomForestClassifier(…)
	  - RandomForestRegressor.py  -> class RandomForestRegressor(…)
	  - Ridge.py -> class Ridge(…)
	  - Lasso.py -> class Lasso(…)
	"""

	if class_name == "SVM":
		if typ == "c":
			return SVC
		else:
			return SVR

	if class_name == "DecisionTree":
		if typ == "c":
			return DecisionTreeClassifier
		else:

			return DecisionTreeRegressor

	if class_name == "Lasso":
		return Lasso

	if class_name == "RandomForest":
		if typ == "c":
			return RandomForestClassifier
		else:
			return RandomForestRegressor

	if class_name == "Ridge":
		return Ridge

	# Si algo inconnu
	raise ValueError(f"Aucune classe custom gérée pour '{class_name}' avec typ='{typ}'.")

type_map = {"r": "regression", "c": "classification"}
algos_sci_map = {
	"DecisionTree": {"r": SkDecisionTreeRegressor, "c": SkDecisionTreeClassifier},
	"RandomForest": {"r": SkRandomForestRegressor, "c": SkRandomForestClassifier},
	"Ridge": {"r": SkRidge},
	"Lasso": {"r": SkLasso},
	"SVM": {"c": SkSVC, "r": SkSVR},
}

# export
__all__ = ["type_map", "algos_sci_map", "get_class", "read_file_wtype", "split", "calculate_mse"]
