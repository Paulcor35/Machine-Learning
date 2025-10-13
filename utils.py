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

# scikit-learn imports (aliases to avoid conflicts)
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

def split(X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	np.random.seed(random_state)
	indices = np.arange(len(y))
	np.random.shuffle(indices)
	split = int(len(y) * (1 - test_size))
	train_idx, test_idx = indices[:split], indices[split:]
	return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray):
	return np.mean((y_true - y_pred) ** 2)

def read_file(fname: str, sep: str) -> pd.DataFrame:
	script_dir = os.path.dirname(os.path.abspath(__file__))
	script_dir, file_dir = getPath(script_dir, fname)
	full_path = os.path.join(script_dir, file_dir)
	log.debug(f"Lecture fichier: {full_path} (sep='{sep}')")
	return pd.read_csv(full_path, sep=sep)

def read_regression(fname: str) -> pd.DataFrame:
	df = read_file(fname, ";")
	df = df.fillna(df.mean())
	df = df.drop(columns=["id"], errors="ignore")
	#df = df.drop(columns=["maxO3v"], errors="ignore")
	return df

def read_classif(fname: str) -> pd.DataFrame:
	df = read_file(fname, ",")
	df = df.drop(columns=["Unnamed: 0"], errors="ignore")
	# simple binary encoding
	for col in ["High", "Urban", "US"]:
		if col in df.columns:
			df[col] = df[col].replace({"Yes": 1, "No": 0})
	# one-hot ShelveLoc if present
	if "ShelveLoc" in df.columns:
		df = pd.get_dummies(df, columns=["ShelveLoc"], drop_first=True)
	return df

def read_file_wtype(fname: str, typ: str) -> pd.DataFrame:
	if typ == "r":
		return read_regression(fname)
	else:
		return read_classif(fname)

def read_params() -> dict[str, dict[str, any]]:
	script_dir = os.path.dirname(os.path.abspath(__file__))
	script_dir, file_dir = getPath(script_dir, "params.yaml")
	with open(os.path.join(script_dir, file_dir), "r") as fp:
		params = yaml.safe_load(fp)
	return params

def apply_params(model, algo_name: str, typ: str, params: dict, ar: list[str], is_sci=False) -> None:
	"""
	Applies all the hyperparameters read from params.yaml to the model
	- For scikit: uses model.set_params(**par)
	- For scratch: setattr (or set_params if available)
	"""
	# Secure the dict reading
	par = (
		params.get(algo_name, {})
			  .get("scikit" if is_sci else "scratch", {})
			  .get(typ, {})
		or {}
	)
	if not par:
		return

	if in_args(ar, "hyperparams"):
		print(f"\nHyperparameters applied to {algo_name} "
			  f"({'scikit-learn' if is_sci else 'scratch'}) [{typ}] :")
		for k, v in par.items():
			print(f"   {k}: {v}")
		print("-" * 60)

	try:
		if hasattr(model, "set_params"):
			model.set_params(**par)  # scikit
		else:
			for k, v in par.items():
				setattr(model, k, v)  # scratch
	except Exception as e:
		for k, v in par.items():
			try:
				setattr(model, k, v)
			except:
				pass

def in_args(ar: list[str], val: str) -> bool:
	return "all" in ar or val in ar

type_map = {"r": "regression", "c": "classification"}
algos_map = {
	"DecisionTree": {"r": DecisionTreeRegressor, "c": DecisionTreeClassifier},
	"RandomForest": {"r": RandomForestRegressor, "c": RandomForestClassifier},
	"Ridge": {"r": Ridge},
	"Lasso": {"r": Lasso},
	"SVM": {"c": SVC, "r": SVR},
}
algos_sci_map = {
	"DecisionTree": {"r": SkDecisionTreeRegressor, "c": SkDecisionTreeClassifier},
	"RandomForest": {"r": SkRandomForestRegressor, "c": SkRandomForestClassifier},
	"Ridge": {"r": SkRidge},
	"Lasso": {"r": SkLasso},
	"SVM": {"c": SkSVC, "r": SkSVR},
}
