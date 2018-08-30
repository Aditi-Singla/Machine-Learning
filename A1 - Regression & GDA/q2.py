import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt

dataX = pd.read_csv('q3x.dat', header=None)
X = np.array(dataX.as_matrix(columns=None))
m, n = X.shape
dataY = pd.read_csv('q3y.dat', header=None)
y = np.array(dataY.as_matrix(columns=None))

# mu = np.zeros((1,n))
# sigma = np.zeros((1,n))

# for j in range(n):
# 	mu[j] = np.mean(X[:,j])
# 	sigma[j] = np.std(X[:,j])
# 	for i in range(m):
# 		X[i][j] = (X[i][j] - mu[j])/sigma[j]
# print(mu[0])
# print(sigma[0])

X = np.insert(X, 0, 1, axis=1)
m, n = X.shape
theta = np.zeros((n,1))

eta = 0.01
diff = 1

theta = np.dot(np.dot((np.linalg.inv(np.dot(X.T,X))),(X.T)),y)

## Plot the graph for data and unweighted linear regression

plt.scatter(X[:,1], y, color='red')
plt.scatter(X[:,1], np.dot(X,theta), color='blue')
plt.xlabel('Values of x')
plt.ylabel('Values of y')
plt.grid(True)

### Locally weighted regression

tau = 10
for i in xrange(m):
	W = np.zeros((m,m))
	for j in xrange(m):
		W[j][j] = np.exp(-math.pow((X[i][1] - X[j][1]),2)/(2*tau*tau))
	theta = np.dot(np.linalg.inv( np.dot(np.dot(X.T,W),X)),np.dot(np.dot(X.T,W),y))
	plt.scatter(X[i,1], np.dot(X[i],theta), color='green')

plt.show()