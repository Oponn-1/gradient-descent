
import math
import random
import numpy as np 

def gradient(x, y, w, delta, lam):
	gradient = [0] * len(w)
	for n in range(len(x)):
		if (y[n] >= np.dot(w, x[n]) + delta):
			for n in range(len(x)):
				gradient += np.multiply((-2 * (y[n] - np.dot(w, x[n]) - delta)), (x[n]))
			gradient = gradient / len(x)
		elif (y[n] <= np.dot(w, x[n]) - delta):
			for n in range(len(x)):
				gradient += np.multiply((-2 * (y[n] - np.dot(w, x[n]) + delta)), (x[n]))
			gradient = gradient / len(x)
	regularization = 0
	for d in range(len(w)):
		regularization += w[d]
	gradient += lam * 2 * regularization
	return gradient

def objective(x, y, w, delta, lam):
	objective = [0] * len(w)
	for n in range(len(x)):
		if (y[n] >= np.dot(w, x[n]) + delta):
			for n in range(len(x)):
				objective += ((y[n] - np.dot(w, x[n]) - delta) ** 2)
			objective = objective / len(x)
		elif (y[n] <= np.dot(w, x[n]) - delta):
			for n in range(len(x)):
				objective += ((y[n] - np.dot(w, x[n]) + delta) ** 2)
			objective = objective / len(x)
	regularization = 0
	for d in range(len(w)):
		regularization += w[d] ** 2
	objective += lam * regularization
	return objective

def bgd_l2(data, y, w, eta, delta, lam, num_iter):
	history_fwdef = []
	this_w = w
	for i in range(0, num_iter):
		new_w = this_w - eta*gradient(data, y, this_w, delta, lam)
		history_fwdef.append(objective(data, y, this_w, delta, lam))
		this_w = new_w
	return this_w, history_fwdef 

def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):
	history_fw = []
	this_w = w
	if (i == -1):
		for n in range(0, num_iter):
			selection = random.randint(0, len(data) -1)
			x_selection = [data[selection]]
			y_selection = [y[selection]]
			new_w = this_w - (eta / math.sqrt(n+1))*gradient(x_selection, y_selection, this_w, delta, lam)
			history_fw.append(objective(x_selection, y_selection, this_w, delta, lam))
			this_w = new_w
	else:
		selection = i
		x_selection = [data[selection]]
		y_selection = [y[selection]]
		new_w = this_w - (eta / math.sqrt(n+1))*gradient(x_selection, y_selection, this_w, delta, lam)
		history_fw.append(objective(x_selection, y_selection, this_w, delta, lam))
		this_w = new_w
	return this_w, history_fw
