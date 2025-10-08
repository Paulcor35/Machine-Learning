#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import sys
from platform import system


def getPath(script_dir, file_dir):
	plat = system()
	if plat == "Windows":
		script_dir = script_dir.replace("/", "\\")
		file_dir = file_dir.replace("/", "\\")
	else:
		script_dir = script_dir.replace("\\", "/")
		file_dir = file_dir.replace("\\", "/")
	return script_dir, file_dir

def split(X, y, ratio=0.8):
	idx = int(ratio * len(X))
	X_train, X_val = X[:idx], X[idx:]
	y_train, y_val = y[:idx], y[idx:]

	return X_train, X_val, y_train, y_val

def calculate_mse(y_true, y_pred):
	return np.mean((y_true - y_pred) ** 2)

def read_file(fname, sep):
	script_dir , file_dir = getPath(os.path.dirname(os.path.abspath(__file__)), fname)
	return pd.read_csv(os.path.join(script_dir+file_dir, sep=sep)

def read_regression(fname):
	df = read_file(fname, ";")
	df = df.drop("id", axis=1)
	return df

def read_classif(fname):
	df = read_file(fname, ",")
	df = df.drop(columns=["Unnamed: 0"], errors="ignore")
	df[["High","Urban","US"]] = df[["High","Urban","US"]].replace({"Yes":1,"No":0})
	df = pd.get_dummies(df, columns=["ShelveLoc"], drop_first=True)
	return df

def read_file_wtype(fname, typ):
	if typ == "r":
		return read_regression(fname)
	else:
		return read_classif(fname)

def get_class(class_name):
	return getattr(sys.modules[__name__], class_name)

type_map = {"r": "regression", "c": "classification"}
algos_sci_map = {
	"DecisionTree": {"r": DecisionTreeRegressor, "c": DecisionTreeClassifier},
	"RandomForest": {"r": , "c": },
	"Ridge": {"r": Ridge},
	"Lasso": {"r": },
	"SVM": {"c": SVC}
}
