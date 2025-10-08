#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np
import pandas as pd
import utils


class Ridge:
	typ = ["r"]

	def __init__(self, lamb=-1.0):
		self.lamb = lamb

	def fit(self, X, y, lamb=None):
		"""
		Ridge regression implementation
		β = (X^t X + λI)^{-1} X^t y
		"""
		if lamb == None and self.lamb == -1.0:
			self.find_best_lambda(X, y)

		X_np = X.values if hasattr(X, 'values') else X
		y_np = y.values if hasattr(y, 'values') else y

		X_np = np.column_stack([np.ones(X_np.shape[0]), X_np])

		n_features = X_np.shape[1]
		XtX = X_np.T @ X_np
		regularization = self.lamb * np.eye(n_features)
		# Don't regularize the intercept term
		regularization[0, 0] = 0

		self.beta = np.linalg.inv(XtX + regularization) @ X_np.T @ y_np

		return self

	def predict(self, X):
        if self.b is None:
            raise RuntimeError("Le modèle n'est pas entraîné.")
		return X @ self.beta

	def find_best_lambda(self, X, y):
		best_lambda = 1.0
		best_mse = float('inf')

		for lamb in np.logspace(-3, 3, 50):
			X_train, X_val, y_train, y_val = split(X, y)
			beta = self.fit(X_train, y_train, lamb)
			y_pred = self.predict(X, beta)
			mse = utils.calculate_mse(y_val, y_pred)
			if mse < best_mse:
				best_mse = mse
				best_labmda = lamb

		self.lamb = best_lambda
