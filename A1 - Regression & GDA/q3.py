import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt

dataX = open( "q2x.dat", "r" )
X1 = [[float(a) for a in line.split()] for line in dataX]  #X is a list
X = np.array([np.array(x) for x in X1])
m, n = X.shape
dataY = pd.read_csv('q2y.dat', header=None)
y = np.array(dataY.as_matrix(columns=None))

# mu = np.zeros((1,n))
# sigma = np.zeros((1,n))

# for j in range(n):
# 	mu[0][j] = np.mean(X[:,j])
# 	sigma[0][j] = np.std(X[:,j])
# 	for i in range(m):
# 		X[i][j] = (X[i][j] - mu[0][j])/sigma[0][j]

X = np.insert(X, 0, 1, axis=1)
m, n = X.shape
theta = np.zeros((n,1))

### Basic functions

def sigmoid(z):
	for i in range(len(z)):
		z[i] = math.pow((1 + np.exp(-z[i])),-1)
	return z
def sigmoid1(z):
	return 	math.pow((1 + np.exp(-z)),-1)

### Logistic Regression	

niter = 0
diff = np.ones((n,1))
while (np.linalg.norm(diff) > 0.000001):
	H = np.zeros((n,n))
	b = np.zeros((1,n))
	for i in range(n):
		for j in range(i,n):
			for k in range(m):
				H[i][j] += sigmoid1(np.dot(X[k],theta)[0])*(1 - sigmoid1(np.dot(X[k],theta)[0])) * X[k][i] * X[k][j]
				if (i != j):
					H[j][i] += sigmoid1(np.dot(X[k],theta)[0])*(1 - sigmoid1(np.dot(X[k],theta)[0])) * X[k][i] * X[k][j]
	for i in range(m):
		b = np.add(b,X[i].reshape((1,n)) * (-y[i][0] + sigmoid1(np.dot(X[i],theta)[0])))	
	theta = theta - np.dot(np.linalg.inv(H),b.T)
	diff = np.dot(np.linalg.inv(H),b.T)
	niter += 1

print('No. of iterations : ' + str(niter))	
print('Normalised theta : ['+ str(theta[0][0])+','+str(theta[1][0])+','+str(theta[2][0])+']')

### Plotting the graph

ans = sigmoid(np.dot(X,theta))
for i in range(m):
	if (ans[i][0]>=0.5):
		l = plt.scatter(X[i,1], X[i,2], marker='*', label='1', color='blue')
	else:
		l1 = plt.scatter(X[i,1], X[i,2], label='0', color='red')

plt.legend((l,l1),('y = 1', 'y = 0'),scatterpoints=1, loc='upper right', ncol=3, fontsize=8)

plt.plot(X[:,1], -(theta[0][0]/theta[2][0]) - (theta[1][0]/theta[2][0])*X[:,1], color='black')

plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()