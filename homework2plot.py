import random
import numpy as np
import matplotlib.pyplot as plt
from homework2 import*

data = np.load('data.npy')
x = []
y = []
for i in range(len(data)):
	x.append([data[i][0]])
	y.append(data[i][1])
w = [0] * len(x[0])

bgd = 'Batch Gradient Descent'
sgd = 'Stochastic Gradient Descent'

def ObjPlot(results, gradientType):
	history = results[1]
	plt.plot(history)
	plt.xlabel('Iterations')
	plt.ylabel('Objective Function')
	plt.title(gradientType)
	plt.show()

if __name__ == '__main__':
# Put the code for the plots here, you can use different functions for each part
	results = bgd_l2(x, y, w, 0.05, 0.1, 0.001, 50)
	ObjPlot(results, bgd)

	results = bgd_l2(x, y, w, 0.1, 0.01, 0.001, 50)
	ObjPlot(results, bgd)

	results = bgd_l2(x, y, w, 0.1, 0, 0.001, 100)
	ObjPlot(results, bgd)

	results = bgd_l2(x, y, w, 0.1, 0, 0, 100)
	ObjPlot(results, bgd)

	results = sgd_l2(x, y, w, 1, 0.1, 0.5, 800)
	ObjPlot(results, sgd)

	results = sgd_l2(x, y, w, 1, 0.01, 0.1, 800)
	ObjPlot(results, sgd)

	results = sgd_l2(x, y, w, 1, 0, 0, 40)
	ObjPlot(results, sgd)

	results = sgd_l2(x, y, w, 1, 0, 0, 800)
	ObjPlot(results, sgd)
