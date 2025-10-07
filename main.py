#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Machine learning algorithms implementation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", help="input file", required=True)
parser.add_argument("-t", "--type", help="file type", choices=["csv"], default="csv")
parser.add_argument("-F", "--to-find", help="dependant var to find", required=True)
parser.add_argument("-a", "--algorithm", help="algorithm to train and use", required=True)
parser.add_argument("-s", "--scikit", help="use the scikit-learn implementation", action="store_true")
args = parser.parse_args()

def read_file(fname, ftype):
	if ftype == "csv":
		return pd.read_csv(fname, sep=";")
	else:
		raise ValueError("Unsupported file type")

def main():
	# Read and prepare data
	df = read_file(args.file, args.type)
	df = df.drop("id", axis=1)
	df = df.fillna(0)

	# Proper train/test split
	train_size = int(0.8 * len(df))
	train_df = df.head(train_size)
	test_df = df.tail(len(df) - train_size)

	print(f"Training set size: {len(train_df)}")
	print(f"Test set size: {len(test_df)}")

	# Prepare data
	y_train = train_df[args.to_find]
	X_train = train_df.drop(args.to_find, axis=1)

	y_test = test_df[args.to_find]
	X_test = test_df.drop(args.to_find, axis=1)

	print("Feature matrix shape:", X_train.shape)
	print("Target vector shape:", y_train.shape)

	# Train
	# TODO

	# Show
	# TODO

	# Make predictions
	y_pred_train = predict(X_train, beta)
	y_pred_test = predict(X_test, beta)

	# Calculate errors
	mse_train = calculate_mse(y_train.values, y_pred_train)
	mse_test = calculate_mse(y_test.values, y_pred_test)

	print(f"\nTraining MSE: {mse_train:.6f}")
	print(f"Test MSE: {mse_test:.6f}")

	print("\nSample predictions vs actual:")
	print("Predicted\tActual")
	for i in range(min(5, len(y_test))):
		print(f"{y_pred_test[i]:.4f}\t\t{y_test.values[i]:.4f}")

if __name__ == "__main__":
	main()
