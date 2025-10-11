#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import argparse
import pandas as pd
import numpy as np
import utils
import bench
import plot

parser = argparse.ArgumentParser(description="Machine learning algorithms implementation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", help="input file", required=True)
parser.add_argument("-t", "--type", help="algorithm type", choices=["r", "regression", "c", "classification"], default="r")
parser.add_argument("-F", "--to-find", help="dependant var to find", required=True)
parser.add_argument("-a", "--algorithm", help="algorithm to train and use", choices=["DecisionTree", "RandomForest", "Ridge", "Lasso", "SVM"], required=True)
args = parser.parse_args()

args.type = args.type[0]

def _maybe_plot_importances(m, label, feature_names):
    weights = None
    if hasattr(m, "coef_") and m.coef_ is not None:
        weights = np.ravel(m.coef_)
    elif hasattr(m, "w") and m.w is not None:
        weights = np.ravel(m.w)
    if weights is not None and weights.size == len(feature_names):
        plot.plot_linear_importances(weights, feature_names, top_k=10, title=f"{label} - importances")


def main():
	ModelClass = utils.get_class(args.algorithm, args.type)
	model = ModelClass()
	if args.type not in model.typ:
		try:
			raise TypeError("bad type")
		except Exception as e:
			e.add_note(f"{args.algorithm} doesn't support {utils.type_map[args.type]}")
	params = utils.read_params()

	# Read and prepare data
	df = utils.read_file_wtype(args.file, args.type)

	y = df[args.to_find]
	X = (df.drop(columns=[args.to_find])
		.select_dtypes(include=[np.number])
		.to_numpy(dtype=float, copy=True))
	feature_names = (df.drop(columns=[args.to_find]).select_dtypes(include=[np.number]).columns.tolist())

	X_train, X_test, y_train, y_test = utils.split(X, y)

	mu = X_train.mean(axis=0); sigma = X_train.std(axis=0); sigma[sigma==0] = 1.0
	X_train = (X_train - mu)/sigma
	X_test  = (X_test  - mu)/sigma

	model_sci = utils.algos_sci_map[args.algorithm][args.type]()

	if model.typ[0] == "c":
		res = bench.benchmark_classification(model, X_train, y_train, X_test, y_test)
		res_sci = bench.benchmark_classification(model_sci, X_train, y_train, X_test, y_test)

		plot.print_classification_report([res, res_sci], [args.algorithm, f"{args.algorithm}_scikit"])
		plot.plot_roc([res, res_sci], [args.algorithm, f"{args.algorithm}_scikit"])
		plot.plot_pr([res, res_sci], [args.algorithm, f"{args.algorithm}_scikit"])
	else:
		res = bench.benchmark_regression(model, X_train, y_train, X_test, y_test)
		res_sci = bench.benchmark_regression(model_sci, X_train, y_train, X_test, y_test)

		plot.print_regression_report([res, res_sci], [args.algorithm, f"{args.algorithm}_scikit"])
		plot.plot_regression_parity(y_test.to_numpy(dtype=float), [res, res_sci], [args.algorithm, f"{args.algorithm}_scikit"])

	_maybe_plot_importances(res["model"], args.algorithm, feature_names)
	_maybe_plot_importances(res_sci["model"], f"{args.algorithm}_scikit", feature_names)

if __name__ == "__main__":
	main()
