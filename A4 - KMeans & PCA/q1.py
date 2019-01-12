import sys
import time
import random
import itertools
import numpy as np
from sklearn import svm
from collections import Counter
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')
from sklearn.model_selection import cross_val_score

####################################################
##############  Read the input files  ##############

print "Reading files...."

X_data = np.matrix(np.loadtxt(sys.argv[1], delimiter=" "))
X_label = np.matrix(np.loadtxt(sys.argv[2], delimiter=" "))
m, n = X_data.shape

####################################################
###############  Helper Functions  #################

# Find the nearest center for each data point
def findClosestCenter(centers):
	K, k2 = centers.shape
	out = np.zeros((m,1))
	out = out.astype(int)
	j = 0.0
	for i in xrange(m):
		x = X_data[i,:]		## Current data point
		diff = np.sum(np.square(x - centers), axis=1)
		out[i,0] = int(np.argmin(diff)) + 1
		j += diff[out[i,0]-1]
	return (out,j)

# Modify centers
def moveCenter(idx,K):
	centers = np.zeros((K,n))
	count = np.zeros((K,1))
	count = count.astype(float)
	for i in xrange(m):
		c = idx[i,0]-1
		tmp = np.zeros((K,n))
		tmp[c,:] = (X_data[i,:])
		centers = np.add(tmp,centers)
		count[c,0] += 1.0
	# print count
	centers = np.divide(centers,count)
	return centers

# Find the optimal centers and cluster ids for each data point
def Kmeans(K,num_iter):
	arr = random.sample(xrange(0,m),K)
	centers = X_data[arr,:]
	for i in range(num_iter):
		print "Iteration Number : ",(i)
		(idx,J) = findClosestCenter(centers)
		centers = moveCenter(idx, K)
	return(idx,J)

def Kmeans1(K,num_iter):
	arr = random.sample(xrange(0,m),K)
	initial_centers = X_data[arr,:]
	centers = X_data[arr,:]
	for i in range(num_iter):
		print "Iteration Number : ",(i)
		(idx,J) = findClosestCenter(centers)
		centers = moveCenter(idx, K)
	return(idx,J,initial_centers)

def Kmeans2(K,num_iter,initcenters):
	arr = random.sample(xrange(0,m),K)
	centers = initcenters
	for i in range(num_iter):
		print "Iteration Number : ",(i)
		(idx,J) = findClosestCenter(centers)
		centers = moveCenter(idx, K)
	return(idx,J)

# Find Accuracy
def findAccuracy(idx):
	correctCount = 0
	totalCount = 0
	for i in range(1,K+1):
		arr = [x for x in range(0,m) if idx[x]==i]
		actual = X_label[:,arr]
		totalCount += actual.shape[1]
		(a,b) = Counter(actual.flat).most_common(1)[0]
		correctCount += b
	return float(correctCount)/totalCount

####################################################
#################  Main function  ##################

print "\nRunning KMeans...."

K = 6
num_iter = 30

# a)
(idx,J) = Kmeans(K,num_iter)
# count = np.zeros((6,1))
# for i in idx:
# 	count[i-1,0] += 1
# print count
acc = findAccuracy(idx)
print "Cost, J : ", J[0,0]
print "Accuracy : ",acc

# b)
print "\nKMeans for Graphs...."
print "\nGetting optimal initial centers.."
initialCenters = np.zeros((K,n))
minJ = sys.maxint
for i in range(10):
	(idx,J,initcenters) = Kmeans1(K,num_iter)
	acc = findAccuracy(idx)
	print i," : ",J[0,0]," , ",acc
	if (J<minJ):
		minJ = J
		initialCenters = initcenters

print "\nVarying number of iterations on optimal initial centers.."
numIterations = []
J_list = []
accu_list = []
for i in range(60):
	(idx,J) = Kmeans2(K,i+1,initialCenters)
	acc = findAccuracy(idx)
	numIterations.append(i+1)
	J_list.append(J[0,0])
	accu_list.append(acc)

plt.plot(numIterations, accu_list)
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

plt.plot(numIterations, J_list)
plt.xlabel('Number of Iterations')
plt.ylabel('J')
plt.grid(True)
plt.show()

####################################################
########### Scikit Cross Validation ################

print "\nCross Validation using Scikit SVM...."

def crossValidate(data,labels,n):
	mcsvm = svm.SVC()
	start = time.time()
	scores = cross_val_score(mcsvm,data,labels,cv=n)
	end = time.time()
	return (sum(scores)/n,end-start)

(acc,timetaken) = crossValidate(np.loadtxt(sys.argv[1]),np.loadtxt(sys.argv[2]),10)
print "Accuracy for cross-validation : ",acc,"%"
print "Time taken : ",timetaken,"s"