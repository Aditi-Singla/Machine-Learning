import sys
import csv
import math
import numpy as np
from utils.svmutil import *

#######################################
########### Read the data #############
#Datapoints

dataX = open(sys.argv[1], "r" )
xdata = [[float(a) for a in line.split(',')] for line in dataX]
m = len(xdata)
n = len(xdata[0])
##Labels
dataY = open(sys.argv[2], "r" )
ylabels = [int(line) for line in dataY]

dataX1 = open(sys.argv[3], "r" )
xtestdata = [[float(a) for a in line.split(',')] for line in dataX1]

dataY1 = open(sys.argv[4], "r" )
ytestlabels = [int(line) for line in dataY1]

############################################
############ Linear Kernel #################

print("Linear Kernel : ")
prob  = svm_problem(ylabels, xdata)
param = svm_parameter('-t 0 -c 500')
m = svm_train(prob, param)
svm_predict(ytestlabels, xtestdata, m)

############################################
############ Gaussian Kernel #################

print("Gaussian Kernel : ")
prob  = svm_problem(ylabels, xdata)
param = svm_parameter('-t 2 -c 500 -g 2.5')
m = svm_train(prob, param)
svm_predict(ytestlabels, xtestdata, m)