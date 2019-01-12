import os
import sys
import time
import random
import numpy as np
from scipy import misc
from sklearn import svm
from collections import Counter
import matplotlib.image as mpimg
from sklearn.model_selection import cross_val_score
# import scipy.misc

def getImage(filename):
	l1 = np.genfromtxt("pca_temp/allfiles_lfw.txt",dtype="str")
	if filename in l1:
		index = (l1.tolist()).index(filename)
		l11 = np.loadtxt("pca_temp/projections_lfw.txt")
		img = l11[index]
		eigen = np.loadtxt("pca_temp/eigen_lfw.txt")
		img = np.dot(eigen,img)
		misc.toimage(img.reshape(50,37)).save("lfw_recons.jpg")
		
	l2 = np.genfromtxt("pca_temp/allfiles_orl.txt",dtype="str")
	if filename in l2:
		index = (l2.tolist()).index(filename)
		l21 = np.loadtxt("pca_temp/projections_orl.txt")	
		img = l21[index]
		eigen = np.loadtxt("pca_temp/eigen_orl.txt")
		img = np.dot(eigen,img)
		misc.toimage(img.reshape(112,92)).save("orl_recons.png")
		
getImage(sys.argv[1])