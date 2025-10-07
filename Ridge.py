#!/usr/bin/python3
import argparse
import pandas as pd
import numpy as np
# (MCO)B = (X^t X)^{-1}X^t y
# L2 = ||B||²
# RSS_{L2} = sum_{i=1}^n (Y_i-Ŷ_l)² + λ sum_{j_1}^P B²_j

parser = argparse.ArgumentParser(description="Ridge Regression Implementation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--file", help="input file", required=True)
parser.add_argument("-t", "--type", help="file type", choices=["csv"], default="csv")
parser.add_argument("-F", "--to-find", help="dependant var to find", required=True)
parser.add_argument("-l", "--lambda", help="regularization parameter (-1 to find automatically)", type=float, default=-1.0)
args = parser.parse_args()


def read_file(fname, ftype):
	if ftype == "csv":
		return pd.read_csv(fname, sep=";")
	else:
		raise ValueError("Unsupported file type")

def ridge(X, y, lambda_val=1.0):
	"""
	Ridge regression implementation
	β = (X^t X + λI)^{-1} X^t y
	"""

	X_np = X.values if hasattr(X, 'values') else X
	y_np = y.values if hasattr(y, 'values') else y

	X_np = np.column_stack([np.ones(X_np.shape[0]), X_np])

	n_features = X_np.shape[1]
	XtX = X_np.T @ X_np
	regularization = lambda_val * np.eye(n_features)
	# Don't regularize the intercept term
	regularization[0, 0] = 0

	beta = np.linalg.inv(XtX + regularization) @ X_np.T @ y_np

	return beta

def predict(X, beta):
	"""Make predictions using the learned coefficients"""
	X_np = X.values if hasattr(X, 'values') else X
	X_np = np.column_stack([np.ones(X_np.shape[0]), X_np])
	return X_np @ beta

def calculate_mse(y_true, y_pred):
	"""Calculate Mean Squared Error"""
	return np.mean((y_true - y_pred) ** 2)

def find_optimal_lambda(X, y, lambdas=np.logspace(-3, 3, 50)):
	"""Find optimal lambda using cross-validation"""
	best_lambda = 1.0
	best_mse = float('inf')

	for lambda_val in lambdas:
		# Simple holdout validation (for simplicity)
		split_idx = int(0.8 * len(X))
		X_train, X_val = X[:split_idx], X[split_idx:]
		y_train, y_val = y[:split_idx], y[split_idx:]

		beta = ridge(X_train, y_train, lambda_val)
		y_pred = predict(X_val, beta)
		mse = calculate_mse(y_val, y_pred)

		if mse < best_mse:
			best_mse = mse
			best_lambda = lambda_val

	return best_lambda

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

	# Train Ridge regression
	l = getattr(args, 'lambda')
	l = l if l != -1.0 else find_optimal_lambda(X_train, y_train)
	print("Lambda: ", l)
	beta = ridge(X_train, y_train, lambda_val=l)

	print("\nRidge Regression Coefficients:")
	print(f"Intercept: {beta[0]:.6f}")
	for i, (col, coef) in enumerate(zip(X_train.columns, beta[1:])):
		print(f"{col}: {coef:.6f}")

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
