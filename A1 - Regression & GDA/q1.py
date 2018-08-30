import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import copy
import matplotlib.animation as anim


dataX = pd.read_csv('q1x.dat', header=None)
X = np.array(dataX.as_matrix(columns=None))
m, n = X.shape
dataY = pd.read_csv('q1y.dat', header=None)
y = np.array(dataY.as_matrix(columns=None))

X_unnormalised = copy.deepcopy(X)

### Normalise data

mu = np.zeros((1,n))
sigma = np.zeros((1,n))

for j in range(n):
	mu[j] = np.mean(X[:,j])
	sigma[j] = np.std(X[:,j])
	for i in range(m):
		X[i][j] = (X[i][j] - mu[j])/sigma[j]

X_unnormalised =  np.insert(X_unnormalised, 0, 1, axis=1)
X = np.insert(X, 0, 1, axis=1)
m, n = X.shape
theta = np.zeros((n,1))

eta = 0.02
diff = 100

#### Function to compute the cost function given the theta0 and theta1

def computeJ(a,b):
	a1 = np.ones((2,1))
	a1[0][0] = a
	a1[1][0] = b
	a2 = np.square(np.subtract(np.dot(X,a1),y))
	return (np.sum(a2)/(2*m))

#### Batch Gradient Descent

theta0_history = []
theta1_history = []
J_history = []
n_iter = 0
while (abs(diff) > 0.0000001 and n_iter<=10000):
	for i in range(n):
		b = np.multiply((np.dot(X,theta)-y),X[:,i].reshape(m,1))
		diff = eta*(np.sum(b))/m
		theta[i][0] = theta[i][0] - diff
	J_history.append(computeJ(theta[0][0],theta[1][0]))
	theta0_history.append(theta[0][0])
	theta1_history.append(theta[1][0])
	n_iter += 1


#### Unnormalise Data to print the graph

theta_unnormalised = copy.deepcopy(theta)
theta_unnormalised[0][0] = theta[0][0] - ((theta[1][0] * mu[0][0])/sigma[0][0])
theta_unnormalised[1][0] = theta[1][0]/sigma[0][0]
plt.scatter(X_unnormalised[:,1], y, color='red')
plt.plot(X_unnormalised[:,1], np.dot(X_unnormalised,theta_unnormalised), color='blue')
plt.xlabel('Areas')
plt.ylabel('Prices')
plt.grid(True)
plt.show()

print('No. of iterations : ' + str(n_iter))	
print('Normalised theta : ['+ str(theta[0][0])+','+str(theta[1][0])+']')
print('Unnormalised theta : ['+ str(theta_unnormalised[0][0])+','+str(theta_unnormalised[1][0])+']')

#######     Code for surface and contours and animation from :   #######
#####   "http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html"   ###

### Surface

fig = plt.figure()
ax = fig.gca(projection='3d')

X1 = np.arange(-15, 25, 0.25)
Y1 = np.arange(-15, 25, 0.25)
X1, Y1 = np.meshgrid(X1, Y1)
Z = np.ones((len(X1),len(Y1)))
for i in range(len(X1)):
	for j in range(len(Y1)):
		Z[i,j] = computeJ(X1[i,j],Y1[i,j])

surf = ax.plot_surface(X1, Y1, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

### Contours

fig3 = plt.figure()
ax2 = fig3.add_subplot(111, projection='3d')
cset = ax2.contour(X1, Y1, Z, cmap=cm.coolwarm)
ax2.clabel(cset, fontsize=9, inline=1)

### Animation

for i in xrange(len(theta0_history)):
	ax.scatter(theta0_history[i],theta1_history[i],J_history[i])
	ax2.scatter(theta0_history[i],theta1_history[i],J_history[i])
	plt.pause(0.2)

plt.show()