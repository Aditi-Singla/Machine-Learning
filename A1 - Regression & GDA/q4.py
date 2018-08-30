import csv
import sys
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def getParser():
	parser = argparse.ArgumentParser(description='Input data points')
	parser.add_argument('--x', help='Datapoints features')
	parser.add_argument('--y', help='Datapoints labels')
	return parser

args = vars(getParser().parse_args(sys.argv[1:]))

mpl.rcParams['lines.color'] = 'k'
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])

### Read data from files, y = 0 for Alaska & y = 1 for Canada

dataX = open(args['x'], "r" )
X1 = [[float(a) for a in line.split()] for line in dataX]  #X is a list
X = np.array([np.array(x) for x in X1])
m, n = X.shape
dataY = pd.read_csv(args['y'], header=None)
y1 = np.array(dataY.as_matrix(columns=None))
y = np.zeros((m,1))
for i in range(m):
	if (y1[i][0] == 'Alaska'):
		y[i][0] = 0
	else:
		y[i][0] = 1

# mu = np.zeros((1,n))
# sigma = np.zeros((1,n))

# for j in range(n):
# 	mu[0][j] = np.mean(X[:,j])
# 	sigma[0][j] = np.std(X[:,j])
# 	for i in range(m):
# 		X[i][j] = (X[i][j] - mu[0][j])/sigma[0][j]

## a,d) Calculate all the values of mu, phi and sigma's

sum_y1 = 0
sum_mu0 = np.zeros((1,n))
sum_mu1 = np.zeros((1,n))
for i in range(m):
	if (y[i][0]==1):
		sum_y1 += 1
		for j in range(n):
			sum_mu1[0][j] += X[i][j]
	else:
		for j in range(n):
			sum_mu0[0][j] += X[i][j]
phi = float(sum_y1)/m
mu0 = (float(1)/(m-sum_y1)) * sum_mu0
mu1 = (float(1)/(sum_y1)) * sum_mu1
cov = np.zeros((n,n))
cov0 = np.zeros((n,n))
cov1 = np.zeros((n,n))
for i in range(m):
	if (y[i][0]==1):
		cov = np.add(cov,np.dot((X[i] - mu1).T,(X[i] - mu1)))
		cov1 = np.add(cov1,np.dot((X[i] - mu1).T,(X[i] - mu1)))
	else:
		cov = np.add(cov,np.dot((X[i] - mu0).T,(X[i] - mu0)))
		cov0 = np.add(cov0,np.dot((X[i] - mu0).T,(X[i] - mu0)))
cov = cov / float(m)
cov0 = cov0 / float(m - sum_y1)
cov1 = cov1 / float(sum_y1)

print('phi = '+str(phi))
print('mu0 = '+str(mu0))
print('mu1 = '+str(mu1))
print('sigma = '+str(cov))
print('sigma0 = '+str(cov0))
print('sigma1 = '+str(cov1))

## b) Plot the given data

for i in range(m):
	if (y[i][0]==1):
		l = plt.scatter(X[i,0], X[i,1], marker='*', label='Canada', color='blue')
	else:
		l1 = plt.scatter(X[i,0], X[i,1],  label='Alaska', color='red')	

plt.legend((l,l1),('Canada', 'Alaska'),scatterpoints=1, loc='upper right', ncol=3, fontsize=8)

## c) Plot the decision boundary for sigma0 = sigma1

A = (np.dot(np.linalg.inv(cov),(np.subtract(mu0,mu1)).T))
B = 0.5*( np.dot(np.dot(mu1,np.linalg.inv(cov)),mu1.T)[0][0] -  np.dot(np.dot(mu0,np.linalg.inv(cov)),mu0.T)[0][0] ) - (math.log(phi/(1-phi)))

# plt.plot(X[:,0], -(B/A[1][0]) - (A[0][0]/A[1][0])*X[:,0], color='black')

## e) Plot the decision boundary for sigma0 != sigma1

P = 0.5 * (np.linalg.inv(cov0) - np.linalg.inv(cov1))
Q = np.subtract((np.dot(mu1,np.linalg.inv(cov1))),(np.dot(mu0,np.linalg.inv(cov0))))

x1 = np.linspace(50, 200, 400)
y2 = np.linspace(250, 550, 400)
x1, y2 = np.meshgrid(x1, y2)

a = P[0][0]
b = 2*P[0][1]
c = P[1][1]
d = Q[0][0]
e = Q[0][1]
f = (0.5*(math.log(float(np.linalg.det(cov0))/np.linalg.det(cov1))) + (math.log(phi/(1-phi))) - 0.5*np.dot(np.dot(mu1,np.linalg.inv(cov1)),mu1.T) + 0.5*np.dot(np.dot(mu0,np.linalg.inv(cov0)),mu0.T))[0][0]
# print(a,b,c,d,e,f)

# plt.contour(x1, y2,(a*x1**2 + b*x1*y2 + c*y2**2 + d*x1 + e*y2 + f), [0], colors='k')

plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()