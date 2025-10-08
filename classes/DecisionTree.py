#!/usr/bin/python3
# -*- Mode: Python; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- #
import numpy as np
import pandas as pd


class DecisionTree:
	typ = ["r", "c"]

	def __init__(self, current):
		self.cur = current

	def fit(self, X, y):
# def dt_learning(df, attributes, parent_df, target_attr)
# def dt_learning_regression(df, attributes, parent_df, target_attr, depth=0, max_depth=None)
		"""
		Construit récursivement un arbre de décision pour la classification.

		À chaque nœud :
		1. Vérifie les cas de base :
		   - DataFrame vide : retourne la classe majoritaire du parent
		   - Toutes les instances appartiennent à la même classe : retourne cette classe
		   - Plus d'attributs disponibles : retourne la classe majoritaire du noeud
		2. Sélectionne le meilleur attribut et seuil (si numérique) via la fonction importance
		3. Crée les branches :
		   - Pour un attribut numérique, deux branches : <= seuil et > seuil
		   - Pour un attribut catégoriel, une branche par valeur unique
		4. Applique la récursion sur chaque sous-ensemble

		Arguments :
			X : ...
			y : ...

		Retour :
			self : ...
		"""

#		if df.empty:
#			return plurality_val(parent_df, target_attr)
#		if len(df[target_attr].unique()) == 1:
#			return df[target_attr].iloc[0]
#		if not attributes:
#			return plurality_val(df, target_attr)

		A, threshold = importance(X, y)
		tree = {A: {}}

		if threshold is not None:
			# Attribut numérique découpé en deux branches
			subset1 = df[df[A] <= threshold]
			subset2 = df[df[A] > threshold]
			tree[A][f"<= {threshold}"] = dt_learning(subset1, [attr for attr in attributes if attr != A], df, target_attr)
			tree[A][f"> {threshold}"] = dt_learning(subset2, [attr for attr in attributes if attr != A], df, target_attr)
		else:
			# Attribut catégoriel
			for v in df[A].unique():
				exs = df[df[A] == v]
				tree[A][v] = dt_learning(exs, [attr for attr in attributes if attr != A], df, target_attr)
		self.tree = tree

		return self

	def predict_class(self, tree, example):
		"""
		Prédit la classe d'un exemple donné en utilisant un arbre de décision.

		Arguments :
			example : pandas.Series -- un exemple contenant les valeurs des attributs

		Retour :
		str/int -- la classe prédite pour l'exemple
		"""

		if not isinstance(tree, dict):
			return tree

		attr = next(iter(tree))
		node = tree[attr]

		# Vérifier si c'est un noeud avec seuil (numérique)
		if all(isinstance(k, str) and ("<=" in k or ">" in k) for k in node.keys()):
			# Extraire le seuil
			for k in node:
				if "<=" in k:
					threshold = float(k.split("<= ")[1])
					if example[attr] <= threshold:
						return self.predict_class(node[k], example)
				elif ">" in k:
					threshold = float(k.split("> ")[1])
					if example[attr] > threshold:
						return self.predict_class(node[k], example)
			# Cas improbable si aucune condition satisfaite
			return Counter([v if not isinstance(v, dict) else None for v in node.values()]).most_common(1)[0][0]

		# Sinon c'est un attribut catégoriel
		attr_value = example[attr]
		if attr_value not in node:
			# Descente pour récupérer toutes les feuilles
			values = []
			stack = [node]
			while stack:
				n = stack.pop()
				for v in n.values():
					if isinstance(v, dict):
						stack.append(v)
					else:
						values.append(v)
			return Counter(values).most_common(1)[0][0]

		return self.predict_class(node[attr_value], example)

	def predict_regression(self, tree, example):
		"""
		Prédit la valeur cible pour un exemple donné à partir d'un arbre de régression.

		Fonction récursive qui descend dans l'arbre en fonction des conditions 
		(attribut <= seuil ou > seuil) jusqu'à atteindre une feuille.

		Args:
			example (pd.Series): Exemple pour lequel on veut prédire la cible.

		Returns:
			float: Valeur prédite pour la cible.
		"""
		# Cas feuille : retourner directement la valeur
		if not isinstance(tree, dict):
			return tree

		# Récupérer l'attribut sur lequel porte le noeud actuel
		attr = next(iter(tree))

		# Parcourir toutes les conditions de ce noeud (gauche/droite)
		for condition, subtree in tree[attr].items():
			# Extraire l'opérateur et le seuil depuis la clé
			op, threshold = condition.split(" ")
			threshold = float(threshold)

			# Valeur de l'exemple pour cet attribut
			value = example[attr]

			# Vérifier quelle branche suivre
			if op == "<=" and value <= threshold:
				return self.predict_regression(subtree, example)
			elif op == ">" and value > threshold:
				return self.predict_regression(subtree, example)

		# Si aucune condition ne correspond, retourner NaN
		return np.nan

	def predict(self, X): # TODO: set X like exemple
		if self.cur == "r":
			return self.predict_regression(self.tree, X)
		else:
			return self.predict_class(self.tree, X)

def entropy(df, target_attr):
	"""
	Calcule l'entropie d'une colonne cible.

	L'entropie est une mesure du désordre ou de l'incertitude dans un ensemble de données.
	Plus l'entropie est élevée, plus les classes sont mélangées et incertaines.

	Arguments :
	df : pandas.DataFrame -- le DataFrame contenant les données
	target_attr : str -- le nom de la colonne cible pour laquelle on calcule l'entropie

	Retour :
	float -- l'entropie de la colonne cible
	"""

	values = df[target_attr]
	counter = Counter(values)
	total = len(df)
	return -sum((count / total) * math.log2(count / total) for count in counter.values() if count > 0)


def importance(X, y):
	"""
	Sélectionne le meilleur attribut pour diviser le dataset en maximisant le gain d'information.

	Pour chaque attribut :
	- Si l'attribut est catégoriel ou binaire, on calcule l'entropie moyenne pondérée
	  des sous-ensembles correspondant à chaque valeur.
	- Si l'attribut est numérique non binaire, on teste tous les seuils possibles
	  entre valeurs consécutives et on choisit le seuil qui maximise le gain d'information.

	Arguments :
		X: ...
		y: ...

	Retour :
		best_attr : str, attribut qui maximise le gain d'information
		best_threshold : float ou None, seuil optimal si numérique, None si catégoriel
	"""

	Xy = pd.concat([X, y], axis=1)
	target = y.columns[0]
	base_entropy = entropy(Xy, target)
	best_gain = -1
	best_attr = None
	best_threshold = None  # Pour stocker le seuil optimal si numérique

	for attr in Xy.columns:
		if Xy[attr].dtype.kind in 'bifc' and len(Xy[attr].unique()) > 2:
			# Attribut numérique non binaire
			sorted_values = sorted(Xy[attr].unique())
			# Tester tous les seuils entre valeurs consécutives
			for i in range(len(sorted_values) - 1):
				threshold = (sorted_values[i] + sorted_values[i+1]) / 2
				left = Xy[Xy[attr] <= threshold]
				right = Xy[Xy[attr] > threshold]
				new_entropy = (len(left) /len(df)) * entropy(left, target) + \
							  (len(right)/len(df)) * entropy(right,target)
				gain = base_entropy - new_entropy
				if gain > best_gain:
					best_gain = gain
					best_attr = attr
					best_threshold = threshold
		else:
			# Attribut catégoriel ou binaire
			values = Xy[attr].unique()
			new_entropy = 0
			for v in values:
				subset = Xy[Xy[attr] == v]
				new_entropy += (len(subset)/len(Xy)) * entropy(subset, target)
			gain = base_entropy - new_entropy
			if gain > best_gain:
				best_gain = gain
				best_attr = attr
				best_threshold = None  # pas de seuil pour les catégoriels

	return best_attr, best_threshold

def importance_regression(X, y):
	"""
	Choisit l'attribut et le seuil qui maximise la réduction de variance
	(critère utilisé pour la régression dans un arbre de décision).

	Pour chaque attribut :
		- On trie les valeurs uniques.
		- On teste tous les seuils possibles (milieu entre deux valeurs consécutives).
		- On calcule la variance pondérée des sous-ensembles gauche et droite.
		- On choisit l'attribut et le seuil qui maximisent le gain de variance.

	Args:
		X: ...
		y: ...

	Returns:
		best_attr (str): Attribut qui donne le meilleur split.
		best_threshold (float): Seuil optimal pour ce split.
	"""

	Xy = pd.concat([X, y], axis=1)
	target = y.columns[0]
	base_var = np.var(Xy[target])
	best_gain = -1
	best_attr = None
	best_threshold = None

	for attr in Xy.columns:
		if
		values = np.sort(Xy[attr].unique())
		for i in range(1, len(values)):
			# Seuil = milieu entre deux valeurs consécutives
			threshold = (values[i - 1] + values[i]) / 2

			# Séparer le DataFrame en deux sous-ensembles
			left = Xy[Xy[attr] <= threshold]
			right = Xy[Xy[attr] > threshold]

			# Ignorer si un des sous-ensembles est vide
			if len(left) == 0 or len(right) == 0:
				continue

			# Variance pondérée des sous-ensembles
			new_var = (len(left) /len(Xy)) * variance(left, target) + \
					  (len(right)/len(Xy)) * variance(right,target)

			# Gain = réduction de variance
			gain = base_var - new_var

			# Mémoriser le meilleur split
			if gain > best_gain:
				best_gain = gain
				best_attr = attr
				best_threshold = threshold

	return best_attr, best_threshold

