import numpy as np
import sys
import csv
import math
from cvxpy import *

#######################################
########### Read the data #############
#Datapoints

dataX = open(sys.argv[1], "r" )
xdata = [np.array([float(a) for a in line.split(',')]) for line in dataX]
m = len(xdata)
n = len(xdata[0])
##Labels
dataY = open(sys.argv[2], "r" )
ylabels = [int(line) for line in dataY]
for i in range(len(ylabels)):
	if (ylabels[i] == 2):
		ylabels[i] = -1

######################################
########## CVXPY #####################

######################################
####### Linear Kernel ################

b1 = np.random.randn(m)
Q = np.zeros((m,m))
for i in range(m):
	for j in range(m):
		Q[i][j] = (-0.5)*(ylabels[i]*ylabels[j])*( np.dot( xdata[i] , xdata[j] ) )
	b1[i] = 1.0
Cvalue = 500

alpha = Variable(m)
constraints = [0 <= alpha, alpha <= Cvalue, alpha.T * np.array(ylabels).T == 0]
obj = Maximize(quad_form(alpha,Q) + b1.T*alpha)

# Form and solve problem.
prob = Problem(obj, constraints)
prob.solve()

epsilon = 0.01
ans = np.array(alpha.value)
print(ans)
count = 0
for i in range(len(ans)):
	if ((ans[i][0] > epsilon) and (ans[i][0] < (500 - epsilon))):
		print(i,"[",ans[i][0],"]")
		count += 1
print(count)

###### Calculate w and b #############

w = np.zeros(n)
for i in range(m):
	w += ans[i][0] * ylabels[i] * xdata[i]

b = 0.0
maxValue = -float("inf")
minValue = float("inf")
for i in range(m):
	if (ylabels[i] == -1 and ((ans[i][0] > epsilon) and (ans[i][0] < (500 - epsilon)))):
		p = np.dot(w,xdata[i].T)
		maxValue = max(p,maxValue)
	if (ylabels[i] == 1 and ((ans[i][0] > epsilon) and (ans[i][0] < (500 - epsilon)))):	
		p = np.dot(w,xdata[i].T)
		minValue = min(p,minValue)
b = - 0.5 * (maxValue + minValue)

print("b : ",b)

def linearKernel(X,Y):
	return np.dot(X.T,Y)

def predictLinear(X):
	sum1 = 0.0
	for i in range(m):
		sum1 += ans[i][0] * ylabels[i] * linearKernel(xdata[i],X)
	return (sum1 + b)

dataX1 = open(sys.argv[3], "r" )
xtestdata = [np.array([float(a) for a in line.split(',')]) for line in dataX1]

dataY1 = open(sys.argv[4], "r" )
ytestlabels = [int(line) for line in dataY1]

correctcountl = 0.0
for i in range(len(xtestdata)):
	if ((predictLinear(xtestdata[i])) >= 0):
		if (ytestlabels[i] == 1):
			correctcountl += 1
	else:
		if (ytestlabels[i] == 2):
			correctcountl += 1

print("Linear Kernel:")
print(correctcountl)
print(len(xtestdata))
print((correctcountl/len(xtestdata)) * 100,"%")

######################################
####### Gaussian Kernel ################

def gaussianKernel(X,Y,gamma):
	return math.exp(-gamma * pow(np.linalg.norm(X-Y),2))

b1g = np.random.randn(m)
Qg = np.zeros((m,m))
for i in range(m):
	for j in range(m):
		Qg[i][j] = (-0.5)*(ylabels[i]*ylabels[j])*gaussianKernel(xdata[i],xdata[j],2.5) 
	b1g[i] = 1.0
Cvalue = 500

alpha1 = Variable(m)
constraints1 = [0 <= alpha1, alpha1 <= Cvalue, alpha1.T * np.array(ylabels).T == 0]
obj = Maximize(quad_form(alpha1,Qg) + b1g.T*alpha1)

# Form and solve problem.
prob = Problem(obj, constraints1)
prob.solve()

epsilon = 0.01
ans1 = np.array(alpha1.value)
print(ans1)
count = 0
for i in range(len(ans1)):
	if ((ans1[i][0] > epsilon) and (ans1[i][0] < (500 - epsilon))):
		print(i,"[",ans1[i][0],"]")
		count += 1
print(count)

###### Calculate w and b #############

def wTx(X,gamma):
	sum1 = 0.0
	for i in range(m):
		sum1 += ans1[i][0] * ylabels[i] * gaussianKernel(xdata[i],X,gamma)
	return (sum1)

bg = 0.0
maxValue = -float("inf")
minValue = float("inf")
for i in range(m):
	if (ylabels[i] == -1 and ((ans1[i][0] > epsilon) and (ans1[i][0] < (500 - epsilon)))):
		p = wTx(xdata[i],2.5)
		maxValue = max(p,maxValue)
	if (ylabels[i] == 1 and ((ans1[i][0] > epsilon) and (ans1[i][0] < (500 - epsilon)))):	
		p = wTx(xdata[i],2.5)
		minValue = min(p,minValue)

bg = - 0.5 * (maxValue + minValue)
print("bg : ",bg)

def predictGaussian(X,gamma):
	sum1 = 0.0
	for i in range(m):
		sum1 += ans1[i][0] * ylabels[i] * gaussianKernel(xdata[i],X,gamma)
	return (sum1 + bg)

correctcountg = 0.0
for i in range(len(xtestdata)):
	if ((predictGaussian(xtestdata[i],2.5)) >= 0):
		if (ytestlabels[i] == 1):
			correctcountg += 1
	else:
		if (ytestlabels[i] == 2):
			correctcountg += 1

print("Gaussian Kernel:")
print(correctcountg)
print(len(xtestdata))
print((correctcountg/len(xtestdata)) * 100,"%")
